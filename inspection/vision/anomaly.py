"""Anomaly detection for normal-only datasets (e.g. MVTec AD): a compact PatchCore-style detector.

ImageNet ResNet50 mid-level patch features of NORMAL training images form a memory bank; a test patch's
anomaly score is its distance to the nearest bank entry. No fine-tuning and no defect classes are used,
which matches what the dataset actually labels."""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torchvision import models

from ..config import Settings, require_private
from ..datasets.manifest import current_version, load_manifest
from ..datasets.registry import TASK_ANOMALY
from ..datasets.validate import read_image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class PatchFeatures(torch.nn.Module):
    def __init__(self, proj_dim: int = 256, seed: int = 0):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("proj", torch.randn(1536, proj_dim, generator=g) / proj_dim ** 0.5)

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)
        f = F.avg_pool2d(f, 3, 1, 1)
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(-1, c) @ self.proj
        return f.reshape(b, h * w, -1), (h, w)


def _load(path: str, size: int) -> torch.Tensor:
    return _prep(read_image(__import__("pathlib").Path(path)), size)


def _prep(img: np.ndarray, size: int) -> torch.Tensor:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return torch.from_numpy(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))


def _score(net, bank, paths, size, device, bs=8):
    return _score_images(net, bank, (read_image(__import__("pathlib").Path(p)) for p in paths), size, device, bs)


def _score_images(net, bank, imgs, size, device, bs=8):
    """imgs: any iterable of images, streamed in batches."""
    scores, maps, batch = [], [], []

    def flush():
        f, (h, w) = net(torch.stack(batch).to(device))
        batch.clear()
        for fi in f:
            d = torch.cdist(fi, bank).min(1).values
            maps.append(d.reshape(h, w).cpu().numpy())
            scores.append(float(d.max()))
    for im in imgs:
        batch.append(_prep(im, size))
        if len(batch) == bs:
            flush()
    if batch:
        flush()
    return np.array(scores), maps


def fit_and_evaluate(settings: Settings, dataset_id: str, ctx, image_size: int = 224, bank_size: int = 30000) -> dict:
    require_private("Training")
    vdir = current_version(settings.datasets_dir, dataset_id)
    if vdir is None:
        raise RuntimeError(f"{dataset_id} not imported")
    summ = json.loads((vdir / "summary.json").read_text(encoding="utf-8"))
    if summ["task"] != TASK_ANOMALY:
        raise RuntimeError("Anomaly detector expects a normal-only (anomaly detection) dataset")
    df = load_manifest(vdir)
    tr = df[(df["split"] == "train") & (df["label"] == "good")]
    va, te = df[df["split"] == "val"], df[df["split"] == "test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PatchFeatures().to(device).eval()
    ctx.progress(0.05, f"extracting patch features from {len(tr)} normal training images")
    feats = []
    for i in range(0, len(tr), 8):
        ctx.check_cancel()
        x = torch.stack([_load(p, image_size) for p in tr["abs_path"].iloc[i:i + 8]]).to(device)
        f, _ = net(x)
        feats.append(f.reshape(-1, f.shape[-1]).cpu())
    bank = torch.cat(feats)
    g = torch.Generator().manual_seed(0)
    if len(bank) > bank_size:  # random subsampling (simplified coreset)
        bank = bank[torch.randperm(len(bank), generator=g)[:bank_size]]
    bank = bank.to(device)
    ctx.progress(0.4, "scoring validation images")
    sv, _ = _score(net, bank, list(va["abs_path"]), image_size, device)
    yv = (va["label"] != "good").astype(int).values
    cands = np.unique(sv)
    thr = float(max(cands, key=lambda t: f1_score(yv, sv >= t)))
    ctx.progress(0.7, "scoring held-out test images")
    st_, maps = _score(net, bank, list(te["abs_path"]), image_size, device)
    yt = (te["label"] != "good").astype(int).values
    pred = (st_ >= thr).astype(int)
    pix_true, pix_score = [], []
    for m, mp, lab in zip(maps, te["mask_path"], te["label"]):
        gt = np.zeros((image_size, image_size), np.uint8)
        if mp:
            gm = read_image(vdir / mp)
            gt = (cv2.resize(gm if gm.ndim == 2 else gm[..., 0], (image_size, image_size),
                             interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        up = cv2.GaussianBlur(cv2.resize(m, (image_size, image_size)), (0, 0), 4)
        pix_true.append(gt[::4, ::4].ravel())
        pix_score.append(up[::4, ::4].ravel())
    pt, ps = np.concatenate(pix_true), np.concatenate(pix_score)
    metrics = {"n_test": int(len(te)), "n_test_anomalous": int(yt.sum()), "image_roc_auc": float(roc_auc_score(yt, st_)),
               "threshold_from_val": thr, "precision": float(precision_score(yt, pred, zero_division=0)),
               "recall": float(recall_score(yt, pred, zero_division=0)), "f1": float(f1_score(yt, pred, zero_division=0)),
               "accuracy": float((pred == yt).mean()),
               "pixel_roc_auc": float(roc_auc_score(pt, ps)) if pt.any() else None,
               "evaluated_on": "held-out real test split (official MVTec test images not used for threshold)"}
    run_id = time.strftime("patchcore-%Y%m%d-%H%M%S")
    out = settings.models_dir / run_id
    out.mkdir(parents=True)
    torch.save({"bank": bank.cpu(), "threshold": thr, "image_size": image_size}, out / "anomaly_bank.pt")
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    info = {"run_id": run_id, "model_type": "anomaly", "status": "complete", "provenance": "pretrained_anomaly",
            "dataset_id": dataset_id, "dataset_version": vdir.name, "task": TASK_ANOMALY, "classes": ["good", "anomaly"],
            "n_train_normal": int(len(tr)), "bank_size": int(len(bank)), "test_metrics_summary": metrics,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _plot_examples(te, maps, st_, thr, out, image_size)
    ctx.progress(0.9, "calibrating anomaly probabilities for the two-tier pipeline")
    calibrate_anomaly(settings, run_id)
    return {"run_id": run_id, **{k: metrics[k] for k in ("image_roc_auc", "pixel_roc_auc", "f1")}}


class AnomalyScorer:
    """Scores single images against a fitted run's memory bank (used by the live two-tier pipeline)."""

    def __init__(self, rdir, device):
        ck = torch.load(rdir / "anomaly_bank.pt", map_location=device, weights_only=False)
        self.bank, self.threshold, self.size = ck["bank"].to(device), float(ck["threshold"]), ck["image_size"]
        self.net = PatchFeatures().to(device).eval()
        self.device = device

    def __call__(self, img: np.ndarray) -> tuple[float, np.ndarray]:
        f, (h, w) = self.net(_prep(img, self.size)[None].to(self.device))
        d = torch.cdist(f[0], self.bank).min(1).values
        return float(d.max()), d.reshape(h, w).cpu().numpy()


def anomaly_probability(score, threshold: float, slope: float):
    """p(anomaly) = sigmoid(slope * log(score / threshold)): exactly 0.5 at the validated decision threshold,
    so the uncertainty band means the same thing as for the classifier. Only the slope is fitted."""
    return 1 / (1 + np.exp(-slope * np.log(np.maximum(score, 1e-6) / threshold)))


def calibrate_anomaly(settings: Settings, run_id: str, ctx=None) -> dict:
    """Fit the score -> probability slope (NLL), the image-statistics reference for the OOD check, and write
    predictions_{val,test}.csv so evaluation and stress testing work as for classifier runs.

    As for the classifier's temperature, clean validation is perfectly separable, so a clean fit drives the slope
    to its bound (probabilities 0/1, uncertainty band never used). The slope is therefore fitted on validation plus
    label-preserving corruptions of it ("shift-aware"); the clean fit is kept for reference."""
    from scipy.optimize import minimize_scalar
    from .calibration import STAT_NAMES, calib_metrics, stat_vector
    from .corruptions import LABEL_PRESERVING, corrupt
    prog = (lambda f, m: ctx.progress(f, m)) if ctx else (lambda f, m: None)
    rdir = settings.models_dir / run_id
    info = json.loads((rdir / "run_info.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sc = AnomalyScorer(rdir, device)
    df = load_manifest(settings.datasets_dir / info["dataset_id"] / info["dataset_version"])
    split = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    scores = {}
    for k, s in enumerate(("val", "test")):
        prog(0.1 + 0.35 * k, f"scoring {s} images")
        scores[s], _ = _score(sc.net, sc.bank, list(split[s]["abs_path"]), sc.size, device)
    yv = (split["val"]["label"] != "good").astype(int).values
    prog(0.55, "scoring corrupted validation images (shift-aware fit)")
    rng = np.random.default_rng(0)

    def corrupted():  # every val image x every label-preserving corruption, one seeded severity 1-3 each
        for k, p in enumerate(split["val"]["abs_path"]):
            im = read_image(__import__("pathlib").Path(p))
            for kind in LABEL_PRESERVING:
                yield corrupt(im, kind, int(rng.integers(1, 4)), seed=k)
    s_cv, _ = _score_images(sc.net, sc.bank, corrupted(), sc.size, device)
    y_cv = np.repeat(yv, len(LABEL_PRESERVING))

    def fit(s, y):
        def nll(slope):
            p = np.clip(anomaly_probability(s, sc.threshold, slope), 1e-12, 1 - 1e-12)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(minimize_scalar(nll, bounds=(1.0, 200.0), method="bounded").x)
    slope_clean = fit(scores["val"], yv)
    slope = fit(np.r_[scores["val"], s_cv], np.r_[yv, y_cv])
    prog(0.8, "image statistics of normal training images (OOD reference)")
    stats_tr = np.stack([stat_vector(read_image(__import__("pathlib").Path(p)), STAT_NAMES)
                         for p in split["train"]["abs_path"]])
    out = {"run_id": run_id, "model_type": "anomaly", "threshold": sc.threshold, "slope": slope,
           "slopes": {"clean_val": slope_clean, "shift_aware": slope},
           "probability": "p = sigmoid(slope * log(score / threshold)); slope fitted by NLL on validation plus "
                          "label-preserving corruptions of it (shift-aware)",
           "calibration_sets": {"clean_val": int(len(yv)), "corrupted_val": int(len(y_cv))},
           "ood": {"stat_names": STAT_NAMES, "stat_mean": stats_tr.mean(0).tolist(),
                   "stat_std": (stats_tr.std(0) + 1e-6).tolist(), "stat_z_threshold": 4.0},
           "evaluation": {}}
    for s in ("val", "test"):
        p = anomaly_probability(scores[s], sc.threshold, slope)
        y = (split[s]["label"] != "good").astype(int).values
        out["evaluation"][s] = calib_metrics(np.log(np.c_[1 - p, p].clip(1e-12, 1)), y, 1.0,
                                             (settings.agent.escalation_low, settings.agent.escalation_high))
        pd.DataFrame({"sample_id": split[s]["sample_id"].values, "abs_path": split[s]["abs_path"].values,
                      "label": split[s]["label"].values, "y": y, "pred": (p >= 0.5).astype(int),
                      "p_good": 1 - p, "p_anomaly": p, "score": scores[s]}
                     ).to_csv(rdir / f"predictions_{s}.csv", index=False)
    (rdir / "anomaly_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    prog(0.99, f"slope={slope:.2f}")
    return {"slope": slope, **{f"{s}_ece": out["evaluation"][s]["ece"] for s in ("val", "test")}}


def _plot_examples(te, maps, scores, thr, out, size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = list(np.argsort(-scores)[:3]) + list(np.argsort(scores)[:1])
    # one shared colour scale (0 .. 1.5 x threshold) so maps are comparable across images
    vmax = 1.5 * thr
    fig, axes = plt.subplots(2, len(idx), figsize=(3 * len(idx), 6.4))
    for j, i in enumerate(idx):
        img = read_image(__import__("pathlib").Path(te["abs_path"].iloc[i]))
        img = cv2.cvtColor(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        axes[0, j].imshow(cv2.resize(img, (size, size)))
        axes[0, j].set_title(f"{te['meta_defect_type'].iloc[i]}\nscore={scores[i]:.2f} thr={thr:.2f}", fontsize=8)
        im = axes[1, j].imshow(cv2.GaussianBlur(cv2.resize(maps[i], (size, size)), (0, 0), 4), cmap="inferno",
                               vmin=0, vmax=vmax)
        for a in axes[:, j]:
            a.axis("off")
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes[1, :].tolist(), orientation="horizontal", fraction=0.05, pad=0.04)
    cb.ax.axvline(thr, color="white", lw=2)
    cb.set_label("patch anomaly score (white line = decision threshold, shared scale)", fontsize=9)
    fig.savefig(out / "anomaly_examples.png", dpi=120)
    plt.close(fig)
