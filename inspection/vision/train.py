"""Reproducible transfer-learning training on a prepared dataset version.

Real val/test splits are always used for model selection and final evaluation. Synthetic samples
(if any) are added to the TRAIN split only."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import Settings, require_private
from ..datasets.manifest import current_version, load_manifest
from ..datasets.registry import TASK_ANOMALY
from ..synthetic.procedural import load_run
from .metrics import classification_metrics
from .model import ImageDataset, build_model, class_order


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_enabled(device: torch.device) -> bool:
    """Mixed precision, except on GTX 16xx cards whose FP16 kernels can produce NaNs (observed here on a
    GTX 1650: finite training loss, NaN validation outputs)."""
    return device.type == "cuda" and "GTX 16" not in torch.cuda.get_device_name(device)


def provenance_of(cfg) -> str:
    if cfg.synthetic_only:
        return "synthetic_trained_experimental"
    if cfg.synthetic_runs:
        return "trained_real_plus_synthetic"
    return "trained_real"


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs, ys, loss_sum, n = [], [], 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled(device)):
            logits = model(x)
        loss_sum += float(crit(logits.float(), y))
        n += len(y)
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys), loss_sum / max(n, 1)


def train(settings: Settings, ctx) -> dict:
    """Train + record the outcome: a run that crashes or is cancelled is marked as such in run_info.json,
    so it never lingers as 'running'."""
    require_private("Training")
    holder: dict = {}
    try:
        return _train(settings, ctx, holder)
    except BaseException as e:
        if "out" in holder:
            p = holder["out"] / "run_info.json"
            info = json.loads(p.read_text(encoding="utf-8"))
            info["status"] = "cancelled" if type(e).__name__ == "Cancelled" else "failed"
            info["error"] = f"{type(e).__name__}: {e}"
            p.write_text(json.dumps(info, indent=2), encoding="utf-8")
        raise


def _train(settings: Settings, ctx, holder: dict) -> dict:
    cfg = settings.train
    set_seed(cfg.seed)
    vdir = current_version(settings.datasets_dir, cfg.dataset_id)
    if vdir is None:
        raise RuntimeError(f"Dataset {cfg.dataset_id} is not imported; import it on the Dataset page first.")
    summary = json.loads((vdir / "summary.json").read_text(encoding="utf-8"))
    if summary["task"] == TASK_ANOMALY:
        raise RuntimeError("This dataset is an anomaly-detection dataset (normal-only training). Use the anomaly "
                           "detector (train-anomaly) instead of a supervised classifier.")
    df = load_manifest(vdir)
    classes = class_order(df["label"])
    cidx = {c: i for i, c in enumerate(classes)}
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    if va.empty or te.empty:
        raise RuntimeError("Real validation/test splits are empty - cannot train with honest model selection.")

    train_paths, train_labels, n_syn = list(tr["abs_path"]), [cidx[l] for l in tr["label"]], 0
    if cfg.synthetic_only:
        normals = tr[tr["label"].isin(["ok", "good"])]
        train_paths, train_labels = list(normals["abs_path"]), [cidx[l] for l in normals["label"]]
    for run in cfg.synthetic_runs:
        s = load_run(settings, run)
        if s["source_dataset"].iloc[0] != cfg.dataset_id:
            raise RuntimeError(f"Synthetic run {run} was generated from another dataset")
        if s["source_version"].iloc[0] != vdir.name:
            # sources were train images of an older split; some may now be val/test images -> leakage
            raise RuntimeError(f"Synthetic run {run} was generated from dataset version {s['source_version'].iloc[0]}, "
                               f"not the current version {vdir.name}. Generate a new synthetic run.")
        # safety net: a synthetic sample derived from a val/test image would leak; procedural runs only use train
        assert set(s["source_split"]) == {"train"}, "synthetic sources must come from the train split"
        if s["label"].iloc[0] not in cidx:
            raise RuntimeError(f"Synthetic label {s['label'].iloc[0]} is not a class of this dataset")
        train_paths += list(s["abs_path"])
        train_labels += [cidx[l] for l in s["label"]]
        n_syn += len(s)
    if len(set(train_labels)) < len(classes):
        raise RuntimeError("Training set does not contain every class (add synthetic defects or real data).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = time.strftime(f"{cfg.architecture}-%Y%m%d-%H%M%S")
    out = settings.models_dir / run_id
    out.mkdir(parents=True)
    holder["out"] = out
    counts = np.bincount(train_labels, minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
    info = {"run_id": run_id, "dataset_id": cfg.dataset_id, "dataset_version": vdir.name, "task": summary["task"],
            "classes": classes, "provenance": provenance_of(cfg), "config": asdict(cfg), "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "n_train_real": int(len(tr) if not cfg.synthetic_only else len(train_paths) - n_syn),
            "n_train_synthetic": n_syn, "n_val_real": int(len(va)), "n_test_real": int(len(te)),
            "train_class_counts": dict(zip(classes, counts.tolist())),
            "class_weights": weights.round(decimals=4).tolist() if cfg.class_weighted_loss else None,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running"}
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    mk = lambda paths, labels, train_: DataLoader(  # noqa: E731
        ImageDataset(paths, labels, cfg.image_size, train_, cfg.use_clahe, cfg.seed),
        batch_size=cfg.batch_size, shuffle=train_, num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    dl_tr = mk(train_paths, train_labels, True)
    dl_va = mk(list(va["abs_path"]), [cidx[l] for l in va["label"]], False)
    dl_te = mk(list(te["abs_path"]), [cidx[l] for l in te["label"]], False)

    ctx.progress(0.01, f"building {cfg.architecture} (ImageNet weights) on {device}")
    model = build_model(cfg.architecture, len(classes)).to(device)
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": cfg.lr_backbone}, {"params": head, "lr": cfg.lr_head}],
                            weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr_backbone, cfg.lr_head], total_steps=steps,
                                                pct_start=0.15)
    crit = nn.CrossEntropyLoss(weight=weights.to(device) if cfg.class_weighted_loss else None)
    use_amp = amp_enabled(device)
    info["mixed_precision"] = use_amp
    scaler = torch.amp.GradScaler(enabled=use_amp)
    history, best = [], (-1.0, float("inf"))
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tl, tn, t0 = 0.0, 0, time.time()
        for x, y in dl_tr:
            ctx.check_cancel()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += float(loss) * len(y)
            tn += len(y)
            step += 1
            if step % 10 == 0:
                ctx.progress(step / steps * 0.9, None, epoch=epoch, train_loss=round(tl / tn, 4))
        probs, ys, vloss = predict(model, dl_va, device)
        m = classification_metrics(ys, probs.argmax(1), probs[:, 1] if len(classes) == 2 else None, classes)
        rec = {"epoch": epoch, "train_loss": tl / tn, "val_loss": vloss, "val_accuracy": m["accuracy"],
               "val_f1": m["f1"], "val_roc_auc": m.get("roc_auc"), "epoch_seconds": round(time.time() - t0, 1)}
        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": cfg.architecture,
                    "image_size": cfg.image_size, "clahe": cfg.use_clahe, "epoch": epoch}, out / "last.pt")
        if (m["f1"], -vloss) > (best[0], -best[1]):
            best = (m["f1"], vloss)
            torch.save({"state_dict": model.state_dict(), "classes": classes, "arch": cfg.architecture,
                        "image_size": cfg.image_size, "clahe": cfg.use_clahe, "epoch": epoch}, out / "best.pt")
            rec["checkpoint"] = "best"
        ctx.progress(0.9 * epoch / cfg.epochs,
                     f"epoch {epoch}/{cfg.epochs} train_loss={rec['train_loss']:.4f} val_loss={vloss:.4f} "
                     f"val_f1={m['f1']:.4f} val_auc={m.get('roc_auc', float('nan')):.4f}")

    ctx.progress(0.92, "evaluating best checkpoint on held-out REAL test split")
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    for name, frame, dl in (("val", va, dl_va), ("test", te, dl_te)):
        probs, ys, _ = predict(model, dl, device)
        pred = pd.DataFrame({"sample_id": frame["sample_id"].values, "abs_path": frame["abs_path"].values,
                             "label": frame["label"].values, "y": ys, "pred": probs.argmax(1)})
        for k, c in enumerate(classes):
            pred[f"p_{c}"] = probs[:, k]
        pred.to_csv(out / f"predictions_{name}.csv", index=False)
        if name == "test":
            tm = classification_metrics(ys, probs.argmax(1), probs[:, 1] if len(classes) == 2 else None, classes)
            tm["evaluated_on"] = "held-out real test split (no synthetic, no augmentation)"
            (out / "test_metrics.json").write_text(json.dumps(tm, indent=2), encoding="utf-8")
    info.update({"status": "complete", "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "best_epoch": ck["epoch"],
                 "augmented_training_views": cfg.epochs * len(train_paths),
                 "test_metrics_summary": {k: tm.get(k) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}})
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    plot_run(out)
    ctx.progress(0.95, "calibrating probabilities (temperature scaling) and fitting the OOD detector")
    from .calibration import calibrate_run
    cal = calibrate_run(settings, run_id)
    return {"run_id": run_id, **info["test_metrics_summary"], "temperature": cal["temperature"]}


def list_runs(settings: Settings) -> list[dict]:
    out = []
    for p in sorted(settings.models_dir.glob("*/run_info.json"), reverse=True):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def plot_run(out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist = json.loads((out / "history.json").read_text(encoding="utf-8"))
    tm = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="train loss", color="#2a6fdb")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="val loss", color="#d9622b")
    ax[0].set_xlabel("epoch"); ax[0].set_title("Loss"); ax[0].legend(frameon=False)
    ax[1].plot(ep, [h["val_f1"] for h in hist], label="val F1", color="#2a6fdb")
    if hist[0].get("val_roc_auc") is not None:
        ax[1].plot(ep, [h["val_roc_auc"] for h in hist], label="val ROC-AUC", color="#1a9e77")
    ax[1].set_xlabel("epoch"); ax[1].set_title("Validation metrics"); ax[1].legend(frameon=False)
    cm = np.array(tm["confusion_matrix"])
    ax[2].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[2].text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax[2].set_xticks(range(len(tm["classes"])), tm["classes"]); ax[2].set_yticks(range(len(tm["classes"])), tm["classes"])
    ax[2].set_xlabel("predicted"); ax[2].set_ylabel("true"); ax[2].set_title(f"Test confusion (n={tm['n']})")
    for a in ax[:2]:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)
    if "roc_curve" in tm:
        fig, a = plt.subplots(figsize=(4, 4))
        a.plot(tm["roc_curve"]["fpr"], tm["roc_curve"]["tpr"], color="#2a6fdb", label=f"AUC={tm['roc_auc']:.4f}")
        a.plot([0, 1], [0, 1], "--", color="#999")
        a.set_xlabel("false positive rate"); a.set_ylabel("true positive rate"); a.set_title("Test ROC")
        a.legend(frameon=False, loc="lower right"); a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(out / "roc_test.png", dpi=150); plt.close(fig)
