"""Probability calibration (temperature scaling) and out-of-distribution detection, fitted after training.

Temperature scaling divides the logits by one scalar T fitted by minimising negative log-likelihood.
On a validation set the model separates perfectly, NLL keeps falling as T -> 0, so a fit on clean
validation data alone makes probabilities MORE extreme. We therefore also fit on validation images plus
label-preserving corruptions of them ("shift-aware" calibration) and report both; the pipeline uses the
shift-aware temperature.

OOD detection: Mahalanobis distance of the ResNet penultimate features (PCA-reduced, Ledoit-Wolf covariance)
to the training distribution, thresholded at the 99th percentile of clean validation images, plus simple image
statistics (mean intensity, contrast, sharpness) as z-scores against the training set."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from ..datasets.manifest import load_manifest
from ..datasets.validate import read_image
from .corruptions import LABEL_PRESERVING, corrupt
from .model import build_model, preprocess

N_BINS = 15


# ------------------------------------------------------------------------------------------------ metrics
def softmax_t(logits: np.ndarray, t: float) -> np.ndarray:
    z = logits / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = N_BINS) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def calib_metrics(logits: np.ndarray, y: np.ndarray, t: float, band=(0.15, 0.85)) -> dict:
    p = softmax_t(logits, t)
    nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
    onehot = np.eye(p.shape[1])[y]
    pdef = 1 - p[:, 0]
    return {"n": int(len(y)), "accuracy": float((p.argmax(1) == y).mean()), "nll": nll,
            "brier": float(((p - onehot) ** 2).sum(1).mean()), "ece": ece(p, y),
            "mean_confidence": float(p.max(1).mean()),
            "frac_extreme": float(((pdef < 0.01) | (pdef > 0.99)).mean()),
            "frac_in_band": float(((pdef >= band[0]) & (pdef <= band[1])).mean())}


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    def nll(logt):
        p = softmax_t(logits, np.exp(logt))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(50)), method="bounded")
    return float(np.exp(res.x))


# ------------------------------------------------------------------------------------------------ image stats
def image_stats(img: np.ndarray) -> np.ndarray:
    """All statistics in ALL_STATS order. rel_sharpness = log(Laplacian var / intensity var) measures blur
    independently of global contrast; raw log_sharpness also drops when an image is merely dim or flat."""
    g = img if img.ndim == 2 else cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
    sharp = np.log(cv2.Laplacian(g, cv2.CV_32F).var() + 1e-6)
    return np.array([g.mean(), g.std(), sharp, sharp - 2 * np.log(g.std() + 1e-6)], np.float32)


ALL_STATS = ["mean_intensity", "contrast_std", "log_sharpness", "rel_sharpness"]
STAT_NAMES = ["mean_intensity", "contrast_std", "rel_sharpness"]    # used by new fits
LEGACY_STAT_NAMES = ["mean_intensity", "contrast_std", "log_sharpness"]  # calibration files without "stat_names"


def stat_vector(img: np.ndarray, names: list[str]) -> np.ndarray:
    s = image_stats(img)
    return np.array([s[ALL_STATS.index(n)] for n in names], np.float32)


# ------------------------------------------------------------------------------------------------ feature extraction
class Extractor:
    """Logits + penultimate features in one forward pass."""

    def __init__(self, model, device, image_size, clahe):
        self.model, self.device, self.size, self.clahe = model, device, image_size, clahe
        self.feat = None
        model.avgpool.register_forward_hook(lambda m, i, o: setattr(self, "feat", o.flatten(1)))

    @torch.no_grad()
    def __call__(self, imgs, bs: int = 16):
        """imgs: any iterable of images (streamed in batches, so corrupted copies never all sit in memory)."""
        L, F, batch = [], [], []

        def flush():
            x = torch.stack(batch).to(self.device)
            L.append(self.model(x).float().cpu().numpy())
            F.append(self.feat.float().cpu().numpy())
            batch.clear()
        for im in imgs:
            batch.append(torch.from_numpy(preprocess(im, self.size, self.clahe).transpose(2, 0, 1)))
            if len(batch) == bs:
                flush()
        if batch:
            flush()
        return np.concatenate(L), np.concatenate(F)


def load_run_model(settings, run_id, device):
    rdir = settings.models_dir / run_id
    ck = torch.load(rdir / "best.pt", map_location=device, weights_only=False)
    model = build_model(ck["arch"], len(ck["classes"]), pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


# Routing policy for image-statistic deviations. Brightness and contrast shifts are largely undone by CLAHE
# preprocessing, and the stress test showed the classifier stays right on them, so a moderate deviation there is
# tolerated when the prediction is confident. Sharpness (blur) and the feature-space detector always route.
# Replayed on the stress grid (real-data model, 7,800 corrupted inspections), this policy plus rel_sharpness cut
# tier-2 routing from 58.0 % to 36.1 % (low contrast 97.6 -> 14.4 %, underexposure 89.1 -> 9.1 %) while classifier
# errors routed to tier 2 went from 429 to 427 of 470.
OOD_POLICY = {"soft_stats": ["mean_intensity", "contrast_std"], "soft_max_z": 15.0, "confident": [0.02, 0.98]}


class OODDetector:
    """Feature-space Mahalanobis check (classifier runs) + image-statistic z-scores. Anomaly runs have no
    feature part: the anomaly score itself measures distance to the normal training data."""

    def __init__(self, d: dict, policy: dict | None = None):
        self.d = d
        self.policy = policy or OOD_POLICY
        self.has_features = "pca_mean" in d
        if self.has_features:
            self.mean = np.array(d["pca_mean"], np.float32)
            self.comp = np.array(d["pca_components"], np.float32)
            self.loc = np.array(d["loc"], np.float32)
            self.prec = np.array(d["precision"], np.float32)
        self.stat_names = d.get("stat_names", LEGACY_STAT_NAMES)
        self.stat_mu, self.stat_sd = np.array(d["stat_mean"]), np.array(d["stat_std"])

    def feature_score(self, feats: np.ndarray) -> np.ndarray:
        z = (feats - self.mean) @ self.comp.T - self.loc
        return np.einsum("ij,jk,ik->i", z, self.prec, z)

    def check(self, feat: np.ndarray | None, img: np.ndarray, p: float | None = None) -> dict:
        """p: calibrated p(defect). Without it no deviation is tolerated (strict, original behaviour)."""
        zs = (stat_vector(img, self.stat_names) - self.stat_mu) / self.stat_sd
        lo, hi = self.policy["confident"]
        confident = p is not None and (p <= lo or p >= hi)
        stat_flags, tolerated = {}, {}
        for n, z in zip(self.stat_names, zs):
            if abs(z) <= self.d["stat_z_threshold"]:
                continue
            soft = confident and n in self.policy["soft_stats"] and abs(z) <= self.policy["soft_max_z"]
            (tolerated if soft else stat_flags)[n] = round(float(z), 2)
        out = {"stat_z": {n: round(float(z), 2) for n, z in zip(self.stat_names, zs)},
               "stat_flags": stat_flags, "stat_tolerated": tolerated, "flag": bool(stat_flags),
               "policy": self.policy, "method": "image-statistic z-scores"}
        if self.has_features:
            s = float(self.feature_score(feat[None])[0])
            out |= {"feature_score": round(s, 2), "feature_threshold": round(self.d["feature_threshold"], 2),
                    "feature_flag": s > self.d["feature_threshold"], "flag": s > self.d["feature_threshold"] or bool(stat_flags),
                    "method": "Mahalanobis distance of ResNet features (PCA-64) + image-statistic z-scores"}
        return out


def load_calibration(rdir: Path) -> tuple[float, OODDetector | None, dict | None]:
    p = rdir / "calibration.json"
    if not p.exists():
        return 1.0, None, None
    d = json.loads(p.read_text())
    return d["temperature"], OODDetector(d["ood"]), d


# ------------------------------------------------------------------------------------------------ fitting
def calibrate_run(settings, run_id: str, ctx=None, seed: int = 0) -> dict:
    rdir = settings.models_dir / run_id
    info = json.loads((rdir / "run_info.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_run_model(settings, run_id, device)
    ext = Extractor(model, device, ck["image_size"], ck["clahe"])
    vdir = settings.datasets_dir / info["dataset_id"] / info["dataset_version"]
    df = load_manifest(vdir)
    cidx = {c: i for i, c in enumerate(ck["classes"])}
    split = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    paths = {s: list(f["abs_path"]) for s, f in split.items()}
    ys = {s: np.array([cidx[l] for l in split[s]["label"]]) for s in split}
    prog = (lambda f, m: ctx.progress(f, m)) if ctx else (lambda f, m: None)

    def images(s):  # streamed from disk: memory stays flat regardless of dataset size
        for p in paths[s]:
            yield read_image(Path(p))

    def corrupted(s, base_seed):
        """Every image x every label-preserving corruption, one seeded random severity 1-3 each."""
        rng = np.random.default_rng(base_seed)
        for k, im in enumerate(images(s)):
            for kind in LABEL_PRESERVING:
                yield corrupt(im, kind, int(rng.integers(1, 4)), seed=base_seed * 7919 + k)

    prog(0.05, "extracting train features (OOD reference)")
    _, ftr = ext(images("train"))
    stats_tr = np.stack([stat_vector(im, STAT_NAMES) for im in images("train")])
    prog(0.25, "validation logits (clean + corrupted)")
    lv, fv = ext(images("val"))
    lcv, _ = ext(corrupted("val", seed))
    cv_y = np.repeat(ys["val"], len(LABEL_PRESERVING))
    prog(0.55, "test logits (clean + corrupted, for reporting only)")
    lt, ftest = ext(images("test"))
    lct, _ = ext(corrupted("test", seed + 10**6))
    ct_y = np.repeat(ys["test"], len(LABEL_PRESERVING))

    t_clean = fit_temperature(lv, ys["val"])
    t_shift = fit_temperature(np.concatenate([lv, lcv]), np.concatenate([ys["val"], cv_y]))
    band = (settings.agent.escalation_low, settings.agent.escalation_high)
    evals = {}
    for name, t in (("uncalibrated", 1.0), ("clean_val_T", t_clean), ("shift_aware_T", t_shift)):
        evals[name] = {"temperature": t, "test_clean": calib_metrics(lt, ys["test"], t, band),
                       "test_corrupted_sev1_3": calib_metrics(lct, ct_y, t, band)}

    prog(0.8, "fitting OOD detector")
    pca = PCA(n_components=min(64, len(ftr) - 1), random_state=0).fit(ftr)
    ztr = pca.transform(ftr)
    lw = LedoitWolf().fit(ztr)
    det_d = {"pca_mean": pca.mean_.tolist(), "pca_components": pca.components_.tolist(), "loc": lw.location_.tolist(),
             "precision": lw.precision_.tolist(), "stat_z_threshold": 4.0, "stat_names": STAT_NAMES}
    det_d["stat_mean"], det_d["stat_std"] = stats_tr.mean(0).tolist(), (stats_tr.std(0) + 1e-6).tolist()
    det_d["feature_threshold"] = 0.0
    det = OODDetector(det_d)
    det_d["feature_threshold"] = float(np.quantile(det.feature_score(fv), 0.99))
    det = OODDetector(det_d)
    ood_clean_rate = float(np.mean([det.check(f, im)["flag"] for f, im in zip(ftest, images("test"))]))

    out = {"run_id": run_id, "temperature": t_shift, "temperature_choice": "shift_aware_T",
           "temperatures": {"clean_val_T": t_clean, "shift_aware_T": t_shift},
           "calibration_sets": {"clean_val": int(len(lv)), "corrupted_val": int(len(lcv)),
                                "corruptions": LABEL_PRESERVING, "severities": "random 1-3 per image, seeded"},
           "evaluation": evals, "ood": det_d | {"false_alarm_rate_clean_test": ood_clean_rate,
                                                "threshold_rule": "99th percentile of clean validation feature scores"},
           "note": "On perfectly separated validation data NLL decreases as T -> 0, so the clean-val temperature "
                   "sharpens probabilities. The shift-aware temperature is used by the pipeline."}
    (rdir / "calibration.json").write_text(json.dumps(out), encoding="utf-8")
    _plot_reliability(lt, ys["test"], lct, ct_y, evals, rdir)
    prog(0.99, f"T_clean={t_clean:.3f} T_shift={t_shift:.3f}")
    return {"temperature": t_shift, "t_clean": t_clean,
            **{f"{k}_ece_corrupted": v["test_corrupted_sev1_3"]["ece"] for k, v in evals.items()}}


def refit_image_stats(settings, run_id: str, ctx=None) -> dict:
    """Re-fit only the image-statistic reference (train split) of an existing calibration, e.g. after the statistic
    set changed. Much faster than a full calibrate_run: no network passes."""
    rdir = settings.models_dir / run_id
    info = json.loads((rdir / "run_info.json").read_text())
    p = rdir / ("anomaly_calibration.json" if info.get("model_type") == "anomaly" else "calibration.json")
    d = json.loads(p.read_text())
    df = load_manifest(settings.datasets_dir / info["dataset_id"] / info["dataset_version"])
    tr = df[df["split"] == "train"]
    if info.get("model_type") == "anomaly":
        tr = tr[tr["label"] == "good"]
    st = np.stack([stat_vector(read_image(Path(x)), STAT_NAMES) for x in tr["abs_path"]])
    d["ood"] |= {"stat_names": STAT_NAMES, "stat_mean": st.mean(0).tolist(), "stat_std": (st.std(0) + 1e-6).tolist()}
    p.write_text(json.dumps(d), encoding="utf-8")
    return {"run_id": run_id, "n_train": int(len(tr)), "stat_names": STAT_NAMES}


def _plot_reliability(lt, yt, lct, yct, evals, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150)
    colors = {"uncalibrated": "#8a8a8a", "clean_val_T": "#eb6834", "shift_aware_T": "#2a78d6"}
    for ax, (lg, y, title) in zip(axes, ((lt, yt, "Clean test images"), (lct, yct, "Corrupted test images (severity 1-3)"))):
        ax.plot([0, 1], [0, 1], "--", color="#bbb", lw=1)
        for name, e in evals.items():
            p = softmax_t(lg, e["temperature"])
            conf, ok = p.max(1), p.argmax(1) == y
            edges = np.linspace(0.5, 1, 11)
            xs, ysb = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() >= 5:
                    xs.append(conf[m].mean())
                    ysb.append(ok[m].mean())
            key = "test_clean" if lg is lt else "test_corrupted_sev1_3"
            ax.plot(xs, ysb, "o-", ms=4, lw=2, color=colors[name],
                    label=f"{name} (T={e['temperature']:.2f}, ECE={e[key]['ece']:.3f})")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("observed accuracy")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "reliability.png")
    plt.close(fig)
