"""Vision model: OpenCV preprocessing, augmentation, ResNet transfer learning and Grad-CAM."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models

from ..datasets.validate import read_image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
NORMAL_FIRST = ("ok", "good", "normal")


def class_order(labels) -> list[str]:
    labs = sorted(set(labels))
    return sorted(labs, key=lambda l: (l not in NORMAL_FIRST, l))


def preprocess(img: np.ndarray, size: int, clahe: bool = True) -> np.ndarray:
    """BGR/gray uint8 -> HxWx3 float32 normalised. Grayscale parts get CLAHE contrast enhancement."""
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if img.ndim == 2 or clahe:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return (rgb - IMAGENET_MEAN) / IMAGENET_STD


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Domain-gap augmentation on the raw uint8 image: rotation, flips, lighting, blur, sensor noise."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)
    if rng.random() < 0.5:
        img = cv2.flip(img, 0)
    ang = rng.uniform(-180, 180)
    sc = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    f = img.astype(np.float32)
    f = f * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25)          # contrast / brightness (lighting)
    if rng.random() < 0.3:                                              # uneven illumination gradient
        gx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
        gy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
        grad = (gx * rng.uniform(-20, 20) + gy * rng.uniform(-20, 20))
        f = f + (grad if f.ndim == 2 else grad[..., None])
    if rng.random() < 0.2:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.2))
    if rng.random() < 0.3:
        f = f + rng.normal(0, rng.uniform(2, 8), f.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


class ImageDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int], size: int, train: bool, clahe: bool, seed: int = 0):
        self.paths, self.labels, self.size, self.train, self.clahe = paths, labels, size, train, clahe
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        img = read_image(Path(self.paths[i]))
        if img is None:
            raise IOError(f"unreadable image {self.paths[i]}")
        if self.train:
            img = augment(img, self.rng)
        x = preprocess(img, self.size, self.clahe)
        return torch.from_numpy(x.transpose(2, 0, 1)), self.labels[i]


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    elif arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"unsupported architecture {arch}")
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
    return m


CAM_LAYERS = ("layer3", "layer4")
CAM_MODES = ("layer4", "layer3", "fused")


class GradCAM:
    """Grad-CAM on layer3 (14x14, finer) and layer4 (7x7, more semantic) plus their geometric-mean fusion.
    One forward pass also captures the penultimate features (for OOD detection).
    Used as an *approximate* defect location, never as a verified mask."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.acts: dict[str, torch.Tensor] = {}
        self.grads: dict[str, torch.Tensor] = {}
        self.features = None
        for name in CAM_LAYERS:
            layer = getattr(model, name)
            layer.register_forward_hook(lambda m, i, o, n=name: self.acts.__setitem__(n, o))
            layer.register_full_backward_hook(lambda m, gi, go, n=name: self.grads.__setitem__(n, go[0]))
        model.avgpool.register_forward_hook(lambda m, i, o: setattr(self, "features", o.flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            return self.model(x)

    def cams(self, logits: torch.Tensor, class_idx: int) -> dict[str, np.ndarray]:
        with torch.enable_grad():
            logits[0, class_idx].backward()
        out = {}
        for n in CAM_LAYERS:
            w = self.grads[n].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * self.acts[n]).sum(1))[0].detach().float().cpu().numpy()
            out[n] = cam / (cam.max() + 1e-8)
        s = out["layer3"].shape[::-1]
        fused = np.sqrt(out["layer3"] * cv2.resize(out["layer4"], s, interpolation=cv2.INTER_CUBIC).clip(0, None))
        out["fused"] = fused / (fused.max() + 1e-8)
        return out


def _region_name(cx: float, cy: float) -> str:
    vert = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
    horiz = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
    return "centre" if (vert, horiz) == ("middle", "centre") else f"{vert}-{horiz}"


def cam_regions(cam: np.ndarray, out_hw: tuple[int, int], thresh: float = 0.5, max_regions: int = 3) -> dict:
    """Up to `max_regions` connected regions above `thresh`, ranked by activation mass. The primary region keeps
    the original fields (bbox_xyxy_norm, region, area_fraction) so reports stay backwards compatible."""
    h, w = out_hw
    up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    py, px = np.unravel_index(int(np.argmax(up)), up.shape)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((up >= thresh).astype(np.uint8))
    regions = []
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        mass = float(up[lab == k].sum())
        regions.append({"bbox_xyxy_norm": [round(x / w, 3), round(y / h, 3), round((x + bw) / w, 3), round((y + bh) / h, 3)],
                        "region": _region_name((x + bw / 2) / w, (y + bh / 2) / h),
                        "area_fraction": round(float(area) / (h * w), 4),
                        "peak": round(float(up[lab == k].max()), 3), "mass": mass})
    regions.sort(key=lambda r: -r["mass"])
    total = sum(r["mass"] for r in regions) or 1.0
    for r in regions:
        r["mass_share"] = round(r.pop("mass") / total, 3)
    regions = regions[:max_regions]
    primary = regions[0] if regions else {"bbox_xyxy_norm": None, "region": "not localised", "area_fraction": 0.0}
    return {"bbox_xyxy_norm": primary["bbox_xyxy_norm"], "region": primary["region"],
            "area_fraction": primary["area_fraction"], "peak_xy_norm": [round(px / w, 3), round(py / h, 3)],
            "regions": regions, "threshold": thresh, "heatmap": up}


def render_saliency(img: np.ndarray, heat: np.ndarray, regions: list[dict] | None = None, alpha: float = 0.45,
                    cmap: int = cv2.COLORMAP_JET, contours: bool = True, boxes: bool = True, thresh: float = 0.5,
                    colorbar: bool = True) -> np.ndarray:
    """Heatmap blended over the image with region contours, numbered boxes and a colour scale. Returns RGB."""
    base = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    base = base[..., :3].copy()
    h, w = base.shape[:2]
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    col = cv2.applyColorMap((heat * 255).astype(np.uint8), cmap)
    a = (alpha * (0.35 + 0.65 * heat))[..., None]          # weak activations stay translucent
    out = (base * (1 - a) + col * a).astype(np.uint8)
    lw = max(2, w // 250)
    if contours:
        m = (heat >= thresh).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (255, 255, 255), lw, cv2.LINE_AA)
    if boxes and regions:
        for k, r in enumerate(regions, 1):
            x0, y0, x1, y1 = r["bbox_xyxy_norm"]
            p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))
            cv2.rectangle(out, p0, p1, (0, 0, 0), lw + 2, cv2.LINE_AA)
            cv2.rectangle(out, p0, p1, (255, 255, 255), lw, cv2.LINE_AA)
            fs = max(0.5, w / 900)
            tag = f"{k}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            ty = max(p0[1], th + 6)
            cv2.rectangle(out, (p0[0], ty - th - 6), (p0[0] + tw + 8, ty), (255, 255, 255), -1)
            cv2.putText(out, tag, (p0[0] + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 2, cv2.LINE_AA)
    if colorbar:
        bw = max(14, w // 30)
        grad = np.linspace(1, 0, h)[:, None].repeat(bw, 1)
        bar = cv2.applyColorMap((grad * 255).astype(np.uint8), cmap)
        pad = np.full((h, bw * 3, 3), 255, np.uint8)
        fs = max(0.4, w / 1300)
        for txt, y in (("high", int(0.04 * h) + 10), ("low", h - 8)):
            cv2.putText(pad, txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (60, 60, 60), 1, cv2.LINE_AA)
        out = np.hstack([out, np.full((h, 6, 3), 255, np.uint8), bar, pad])
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def overlay(img: np.ndarray, heat: np.ndarray, bbox_norm=None) -> np.ndarray:
    """Backwards-compatible simple overlay."""
    regions = [{"bbox_xyxy_norm": bbox_norm}] if bbox_norm else None
    return render_saliency(img, heat, regions, colorbar=False)

