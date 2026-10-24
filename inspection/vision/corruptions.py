"""Deterministic image corruptions (severity 1-5) for calibration and stress testing.

LABEL_PRESERVING corruptions degrade image quality but keep the part fully in view, so the ground-truth
label still holds. Crop corruptions can move the defect out of frame, so their labels are NOT guaranteed;
they are used for stress testing only and never for fitting calibration."""
from __future__ import annotations

import cv2
import numpy as np

PARAMS = {
    "gaussian_blur": [1.0, 2.0, 3.0, 5.0, 8.0],          # sigma (px, 512-px image)
    "motion_blur": [5, 9, 15, 21, 31],                     # kernel length (px)
    "gaussian_noise": [8, 16, 28, 42, 60],                 # sigma (0-255)
    "low_contrast": [0.6, 0.45, 0.3, 0.2, 0.12],           # contrast factor around the mean
    "underexposure": [0.75, 0.6, 0.45, 0.32, 0.22],        # intensity scale
    "jpeg": [40, 25, 15, 10, 5],                           # quality
    "crop_zoom": [0.85, 0.7, 0.55, 0.45, 0.35],            # fraction of the image kept (centre)
    "crop_shift": [0.1, 0.2, 0.3, 0.4, 0.5],               # part shifted out of frame (fraction)
}
LABEL_PRESERVING = ["gaussian_blur", "motion_blur", "gaussian_noise", "low_contrast", "underexposure", "jpeg"]
ALL = list(PARAMS)


def corrupt(img: np.ndarray, kind: str, severity: int, seed: int = 0) -> np.ndarray:
    if severity == 0 or kind == "none":
        return img
    v = PARAMS[kind][severity - 1]
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    if kind == "gaussian_blur":
        out = cv2.GaussianBlur(f, (0, 0), v)
    elif kind == "motion_blur":
        k = np.zeros((v, v), np.float32)
        k[v // 2, :] = 1.0 / v
        ang = rng.uniform(0, 180)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((v / 2 - 0.5, v / 2 - 0.5), ang, 1.0), (v, v))
        out = cv2.filter2D(f, -1, k / max(k.sum(), 1e-6))
    elif kind == "gaussian_noise":
        out = f + rng.normal(0, v, f.shape)
    elif kind == "low_contrast":
        m = f.mean()
        out = m + (f - m) * v
    elif kind == "underexposure":
        out = f * v
    elif kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(v)])
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    elif kind == "crop_zoom":
        ch, cw = int(h * v), int(w * v)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    elif kind == "crop_shift":
        ang = rng.uniform(0, 2 * np.pi)
        dx, dy = v * w * np.cos(ang), v * h * np.sin(ang)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        border = tuple(int(x) for x in np.atleast_1d(np.median(img.reshape(-1, img.shape[2] if img.ndim == 3 else 1), 0)))
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    else:
        raise ValueError(kind)
    return np.clip(out, 0, 255).astype(np.uint8)
