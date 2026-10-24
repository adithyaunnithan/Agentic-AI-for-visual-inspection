"""Image-level validation and near-duplicate detection (difference hash + union-find)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray | None:
    """cv2.imread cannot open non-ASCII Windows paths; decode from bytes instead."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except OSError:
        return None


def write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(str(path))


def image_info(path: Path) -> dict:
    img = read_image(path)
    if img is None:
        return {"readable": False}
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    gray = img if c == 1 else cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    # A 3-channel image whose channels are identical is effectively grayscale.
    effective_gray = c == 1 or (c >= 3 and np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]))
    return {"readable": True, "width": w, "height": h, "channels": c, "effective_grayscale": bool(effective_gray),
            "dhash": dhash(gray), "mean_intensity": float(gray.mean()), "thumb": thumbnail(gray),
            "small": cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)}


def thumbnail(gray: np.ndarray, size: int = 64) -> np.ndarray:
    """Zero-mean, unit-norm thumbnail: the dot product of two thumbnails is their Pearson correlation."""
    t = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def dhash(gray: np.ndarray, size: int = 8) -> int:
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def near_duplicate_groups(thumbs: list[np.ndarray], min_corr: float = 0.99, chunk: int = 2048) -> list[int]:
    """Return a group id per item; items whose 64x64 thumbnails correlate >= min_corr share a group.
    Grouping keeps near-duplicates (e.g. the same part photographed twice) on one side of every split.
    A coarse perceptual hash is not used for this: on datasets where every part is photographed in the
    same pose it merges hundreds of distinct parts into one group."""
    n = len(thumbs)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    X = np.stack(thumbs).astype(np.float32)
    for s in range(0, n, chunk):
        sim = X[s:s + chunk] @ X.T
        for a_local, b in zip(*np.nonzero(sim >= min_corr)):
            a = s + int(a_local)
            if b > a:
                ra, rb = find(a), find(int(b))
                if ra != rb:
                    parent[rb] = ra
    return [find(i) for i in range(n)]
