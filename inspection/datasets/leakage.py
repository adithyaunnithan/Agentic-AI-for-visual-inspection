"""Rotation-aware near-duplicate similarity and split-leakage audit.

Parts on a line are often re-photographed at a different rotation. Comparing plain thumbnails misses
those re-shots, so similarity here is the maximum Pearson correlation of 64x64 thumbnails over rotations
of one image (default 10-degree steps)."""
from __future__ import annotations

import cv2
import numpy as np

SMALL = 128   # images are kept at 128x128 in memory for rotation, then thumbnailed to 64x64


def small_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(gray, (SMALL, SMALL), interpolation=cv2.INTER_AREA)


def _thumb(g: np.ndarray) -> np.ndarray:
    t = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def rotation_similarity(smalls: list[np.ndarray], step: int = 10, progress=None) -> np.ndarray:
    """n x n matrix: S[i, j] = max over rotations r of corr(rotate(i, r), j); symmetrised with max(S, S.T)."""
    n = len(smalls)
    base = np.stack([_thumb(g) for g in smalls])                      # n x 4096
    S = np.full((n, n), -1.0, np.float32)
    c = (SMALL / 2, SMALL / 2)
    angles = range(0, 360, step)
    for i, g in enumerate(smalls):
        rot = np.stack([_thumb(cv2.warpAffine(g, cv2.getRotationMatrix2D(c, a, 1.0), (SMALL, SMALL),
                                              borderMode=cv2.BORDER_REFLECT)) for a in angles])
        S[i] = (rot @ base.T).max(0)
        if progress and i % 100 == 0:
            progress(i / n)
    S = np.maximum(S, S.T)
    np.fill_diagonal(S, 1.0)
    return S


def groups_from_similarity(S: np.ndarray, min_corr: float) -> list[int]:
    n = len(S)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(*np.nonzero(np.triu(S >= min_corr, 1))):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra
    return [find(i) for i in range(n)]


def audit(S: np.ndarray, splits: list[str], thresholds=(0.99, 0.98, 0.97)) -> dict:
    """Cross-split similarity statistics. For each test image: best match in train vs best match in val
    (val is independent of test, so it is the reference level of 'similar but different part')."""
    sp = np.array(splits)
    out = {"method": "max Pearson correlation of 64x64 thumbnails over 10-degree rotations", "pairs": {}}
    for a, b in (("test", "train"), ("val", "train"), ("test", "val")):
        ia, ib = np.nonzero(sp == a)[0], np.nonzero(sp == b)[0]
        if len(ia) == 0 or len(ib) == 0:
            continue
        best = S[np.ix_(ia, ib)].max(1)
        out["pairs"][f"{a}->{b}"] = {
            "n": int(len(ia)), "median_best": round(float(np.median(best)), 4),
            "p99_best": round(float(np.quantile(best, 0.99)), 4), "max_best": round(float(best.max()), 4),
            **{f"n_ge_{t}": int((best >= t).sum()) for t in thresholds}}
    return out
