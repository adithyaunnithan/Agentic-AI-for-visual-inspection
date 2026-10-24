"""Procedural synthetic defects composited onto clean (normal) images, with pixel masks.

Design rules:
* Sources are real NORMAL images from the TRAIN split only, so val/test stay untouched.
* Every sample is reproducible: run seed + index -> per-sample RNG; all parameters are recorded.
* Labels are only what the procedure guarantees: "defective" + the inserted defect type + its mask.
* Defects are restricted to the part (foreground estimated with Otsu + largest component)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..datasets.manifest import current_version, load_manifest
from ..datasets.validate import read_image, write_image

GENERATOR_VERSION = "procedural_v2"  # v2: edge/convex-hull part mask, all defects confined to the part
DEFECT_TYPES = ["scratch", "crack", "pit", "stain", "discoloration", "dent", "irregular_texture"]
NORMAL_LABELS = {"ok", "good"}


def foreground_mask(gray: np.ndarray) -> np.ndarray:
    """Estimate the part region: strong edges -> close -> convex hull of the largest contour.
    (Intensity thresholds fail when a metallic part has both darker and brighter areas than the background.)"""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = float(np.median(blur))
    edges = cv2.Canny(blur, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
    k = max(3, (min(h, w) // 40) | 1)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full = np.ones_like(gray, dtype=bool)
    if not cnts:
        return full
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    fg = np.zeros_like(gray, np.uint8)
    cv2.fillPoly(fg, [hull], 1)
    fg = cv2.erode(fg, np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)
    return fg if 0.05 * fg.size < fg.sum() < 0.95 * fg.size else full


def _pick_point(rng, fg):
    ys, xs = np.nonzero(fg)
    i = rng.integers(len(ys))
    return int(xs[i]), int(ys[i])


def _texture_noise(rng, shape, scale):
    n = rng.normal(0, 1, shape).astype(np.float32)
    return cv2.GaussianBlur(n, (0, 0), scale)


def _scratch(rng, img, fg):
    h, w = img.shape
    x0, y0 = _pick_point(rng, fg)
    length = rng.uniform(0.15, 0.45) * min(h, w)
    ang = rng.uniform(0, np.pi)
    curv = rng.uniform(-0.3, 0.3)
    pts = []
    for t in np.linspace(0, 1, 40):
        a = ang + curv * t
        pts.append((x0 + np.cos(a) * length * t, y0 + np.sin(a) * length * t))
    pts = np.array(pts, np.int32)
    thick = int(rng.integers(1, 4))
    delta = float(rng.choice([-1, 1]) * rng.uniform(0.15, 0.35))
    m = np.zeros((h, w), np.uint8)
    cv2.polylines(m, [pts], False, 255, thick, cv2.LINE_AA)
    soft = cv2.GaussianBlur(m.astype(np.float32) / 255, (3, 3), 0)
    out = img + delta * soft
    return out, m > 64, {"length_px": round(length, 1), "angle_rad": round(ang, 3), "curvature": round(curv, 3),
                         "thickness_px": thick, "intensity_delta": round(delta, 3)}


def _crack(rng, img, fg):
    h, w = img.shape
    m = np.zeros((h, w), np.uint8)
    x, y = _pick_point(rng, fg)
    steps = int(rng.integers(30, 90))
    heading = rng.uniform(0, 2 * np.pi)
    branches = 0
    stack = [(x, y, heading, steps)]
    while stack:
        x, y, heading, steps = stack.pop()
        for _ in range(steps):
            heading += rng.normal(0, 0.18)
            nx, ny = x + 2 * np.cos(heading), y + 2 * np.sin(heading)
            cv2.line(m, (int(x), int(y)), (int(nx), int(ny)), 255, 1, cv2.LINE_AA)
            x, y = nx, ny
            if branches < 3 and rng.random() < 0.015:
                branches += 1
                stack.append((x, y, heading + rng.choice([-1, 1]) * rng.uniform(0.5, 1.2), steps // 3))
    m = cv2.dilate(m, np.ones((2, 2), np.uint8))
    depth = float(rng.uniform(0.2, 0.4))
    soft = cv2.GaussianBlur(m.astype(np.float32) / 255, (3, 3), 0)
    return img - depth * soft, m > 64, {"steps": steps, "branches": branches, "darkening": round(depth, 3)}


def _pit(rng, img, fg):
    h, w = img.shape
    m = np.zeros((h, w), np.uint8)
    out = img.copy()
    n = int(rng.integers(1, 7))
    cx, cy = _pick_point(rng, fg)
    pits = []
    for _ in range(n):
        r = float(rng.uniform(2.5, 0.03 * min(h, w) + 3))
        px, py = int(cx + rng.normal(0, 0.05 * w)), int(cy + rng.normal(0, 0.05 * h))
        ax = (max(2, int(r)), max(2, int(r * rng.uniform(0.6, 1.0))))
        a = float(rng.uniform(0, 180))
        cv2.ellipse(m, (px, py), ax, a, 0, 360, 255, -1)
        pits.append({"x": px, "y": py, "r": round(r, 1)})
    pitf = cv2.GaussianBlur(m.astype(np.float32) / 255, (5, 5), 0)
    # dark cavity with a light rim on one side (top-left lighting)
    shift = np.roll(np.roll(pitf, -1, 0), -1, 1)
    depth = float(rng.uniform(0.3, 0.55))
    out = out - depth * pitf + 0.12 * np.clip(shift - pitf, 0, 1)
    return out, m > 64, {"n_pits": n, "pits": pits, "depth": round(depth, 3)}


def _blob(rng, shape, center, radius, k=6):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.zeros(shape, np.float32)
    for _ in range(k):
        cx = center[0] + rng.normal(0, radius * 0.5)
        cy = center[1] + rng.normal(0, radius * 0.5)
        s = radius * rng.uniform(0.3, 0.7)
        field += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    return field / field.max()


def _stain(rng, img, fg):
    c = _pick_point(rng, fg)
    radius = float(rng.uniform(0.04, 0.12) * min(img.shape))
    field = _blob(rng, img.shape, c, radius)
    mask = field > 0.45
    soft = cv2.GaussianBlur(np.clip((field - 0.35) / 0.3, 0, 1), (0, 0), 2)
    delta = float(rng.choice([-1, 1]) * rng.uniform(0.1, 0.25))
    return img + delta * soft, mask, {"radius_px": round(radius, 1), "intensity_delta": round(delta, 3)}


def _discoloration(rng, img, fg):
    c = _pick_point(rng, fg)
    radius = float(rng.uniform(0.12, 0.25) * min(img.shape))
    field = _blob(rng, img.shape, c, radius, k=4)
    mask = (field > 0.5) & fg
    delta = float(rng.choice([-1, 1]) * rng.uniform(0.08, 0.18))
    return img + delta * cv2.GaussianBlur(field, (0, 0), 4) * fg, mask, {"radius_px": round(radius, 1),
                                                                         "intensity_delta": round(delta, 3)}


def _dent(rng, img, fg):
    h, w = img.shape
    cx, cy = _pick_point(rng, fg)
    r = float(rng.uniform(0.04, 0.1) * min(h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
    bowl = np.exp(-d2)
    gy, gx = np.gradient(bowl)
    light = np.array([-0.7, -0.7])  # light from top-left
    shading = (gx * light[0] + gy * light[1]) * r
    strength = float(rng.uniform(0.35, 0.7))
    return img + strength * shading, d2 < 1.0, {"radius_px": round(r, 1), "shading_strength": round(strength, 3)}


def _irregular_texture(rng, img, fg):
    c = _pick_point(rng, fg)
    radius = float(rng.uniform(0.05, 0.12) * min(img.shape))
    field = _blob(rng, img.shape, c, radius)
    mask = field > 0.5
    tex = _texture_noise(rng, img.shape, rng.uniform(0.8, 2.0))
    tex /= np.abs(tex).max() + 1e-6
    amp = float(rng.uniform(0.12, 0.25))
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2)
    return img + amp * tex * soft, mask, {"radius_px": round(radius, 1), "amplitude": round(amp, 3)}


FUNCS = {"scratch": _scratch, "crack": _crack, "pit": _pit, "stain": _stain, "discoloration": _discoloration,
         "dent": _dent, "irregular_texture": _irregular_texture}


def synthesize_one(img_u8: np.ndarray, defect: str, seed: int, min_area: float = 3e-4,
                   max_attempts: int = 8) -> tuple[np.ndarray, np.ndarray, dict]:
    color = img_u8.ndim == 3
    gray = cv2.cvtColor(img_u8[..., :3], cv2.COLOR_BGR2GRAY) if color else img_u8
    fg = foreground_mask(gray)
    base = gray.astype(np.float32) / 255
    fg_soft = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 2)
    for attempt in range(max_attempts):
        # deterministic retry: a defect mostly clipped by the part boundary is regenerated from a derived seed
        rng = np.random.default_rng(seed + attempt * 7919)
        out, mask, params = FUNCS[defect](rng, base, fg)
        # confine every modification (and therefore the mask) to the part, with a soft edge
        out = base + (out - base) * fg_soft
        mask = mask & fg
        if mask.mean() >= min_area:
            break
    params["attempt"] = attempt
    # match sensor noise inside the defect so it is not unrealistically smooth
    noise_sigma = float(np.std(base - cv2.GaussianBlur(base, (0, 0), 1.5)))
    out = out + mask * rng.normal(0, noise_sigma, base.shape).astype(np.float32)
    out_u8 = np.clip(out * 255, 0, 255).astype(np.uint8)
    if color:  # apply the same luminance change to every channel
        diff = out_u8.astype(np.int16) - gray.astype(np.int16)
        out_u8 = np.clip(img_u8[..., :3].astype(np.int16) + diff[..., None], 0, 255).astype(np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255  # marks exactly the pixels the procedure modified
    params["noise_sigma"] = round(noise_sigma, 4)
    return out_u8, mask_u8, params


def generate(settings, dataset_id: str, n: int, defect_types: list[str] | None = None, seed: int = 1234,
             ctx=None) -> dict:
    vdir = current_version(settings.datasets_dir, dataset_id)
    if vdir is None:
        raise RuntimeError(f"Dataset {dataset_id} has not been imported yet.")
    df = load_manifest(vdir)
    clean = df[(df["split"] == "train") & (df["label"].isin(NORMAL_LABELS))].reset_index(drop=True)
    if clean.empty:
        raise RuntimeError("No clean (normal) TRAIN images available as synthesis sources.")
    defect_types = defect_types or DEFECT_TYPES
    run_id = time.strftime("synth-%Y%m%d-%H%M%S")
    root = settings.synthetic_dir / run_id
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        if ctx:
            ctx.check_cancel()
            if i % 10 == 0:
                ctx.progress(i / n, f"generated {i}/{n}")
        src = clean.iloc[int(rng.integers(len(clean)))]
        defect = defect_types[i % len(defect_types)]
        sample_seed = int(seed * 100003 + i)
        img = read_image(Path(src["abs_path"]))
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        out, mask, params = synthesize_one(img, defect, sample_seed)
        sid = f"{run_id}:{i:05d}"
        ip, mp = Path("images") / f"{i:05d}_{defect}.png", Path("masks") / f"{i:05d}_{defect}_mask.png"
        write_image(root / ip, out)
        write_image(root / mp, mask)
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None
        rows.append({"sample_id": sid, "path": ip.as_posix(), "mask_path": mp.as_posix(), "label": "defective",
                     "defect_type": defect, "sample_type": "synthetic", "method": GENERATOR_VERSION,
                     "seed": sample_seed, "params": json.dumps(params), "source_sample_id": src["sample_id"],
                     "source_path": src["path"], "source_split": "train", "source_dataset": dataset_id,
                     "source_version": vdir.name, "bbox_xyxy": json.dumps(bbox),
                     "mask_area_fraction": round(float((mask > 0).mean()), 5)})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    info = {"run_id": run_id, "dataset_id": dataset_id, "dataset_version": vdir.name, "n": len(rows),
            "defect_types": defect_types, "seed": seed, "generator": GENERATOR_VERSION,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label_semantics": "label=defective and defect_type/mask are known by construction; realism is NOT "
                               "verified. Synthetic samples are excluded from all val/test evaluation."}
    (root / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def list_runs(settings) -> list[dict]:
    out = []
    for p in sorted(settings.synthetic_dir.glob("synth-*/run_info.json"), reverse=True):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def load_run(settings, run_id: str) -> pd.DataFrame:
    root = settings.synthetic_dir / run_id
    df = pd.read_csv(root / "manifest.csv", keep_default_na=False)
    df["abs_path"] = [str(root / p) for p in df["path"]]
    df["abs_mask_path"] = [str(root / p) for p in df["mask_path"]]
    return df
