"""EXPERIMENTAL / future extension: a small convolutional VAE for generative augmentation.

Status: runnable but not used by default. Trained on real NORMAL train images only, it can sample new
'normal-looking' images; it cannot produce labelled defects without defect training data, so samples are
stored with label='unlabelled' and never enter evaluation. A conditional GAN/VAE for defect synthesis
requires a larger set of real defect images (or masks) and more GPU memory than the 4 GB target."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets.manifest import current_version, load_manifest
from ..datasets.validate import read_image, write_image

SIZE = 128


class ConvVAE(nn.Module):
    def __init__(self, z: int = 64):
        super().__init__()
        ch = [1, 32, 64, 128, 256]
        self.enc = nn.Sequential(*[m for i in range(4) for m in (nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1),
                                                                   nn.BatchNorm2d(ch[i + 1]), nn.LeakyReLU(0.2))])
        self.mu, self.lv = nn.Linear(256 * 8 * 8, z), nn.Linear(256 * 8 * 8, z)
        self.fc = nn.Linear(z, 256 * 8 * 8)
        dec = []
        for i in range(4, 1, -1):
            dec += [nn.ConvTranspose2d(ch[i], ch[i - 1], 4, 2, 1), nn.BatchNorm2d(ch[i - 1]), nn.ReLU()]
        dec += [nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()]
        self.dec = nn.Sequential(*dec)
        self.z = z

    def forward(self, x):
        h = self.enc(x).flatten(1)
        mu, lv = self.mu(h), self.lv(h)
        zz = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(self.fc(zz).view(-1, 256, 8, 8)), mu, lv

    def sample(self, n, device):
        return self.dec(self.fc(torch.randn(n, self.z, device=device)).view(-1, 256, 8, 8))


def train_vae(settings, dataset_id: str, epochs: int = 30, n_samples: int = 32, ctx=None) -> dict:
    vdir = current_version(settings.datasets_dir, dataset_id)
    df = load_manifest(vdir)
    tr = df[(df["split"] == "train") & df["label"].isin(["ok", "good"])]
    imgs = []
    for p in tr["abs_path"]:
        im = read_image(Path(p))
        g = im if im.ndim == 2 else cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
        imgs.append(cv2.resize(g, (SIZE, SIZE), interpolation=cv2.INTER_AREA))
    x = torch.tensor(np.stack(imgs)[:, None], dtype=torch.float32) / 255
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = ConvVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(x))
        tot = 0.0
        for i in range(0, len(x), 32):
            if ctx:
                ctx.check_cancel()
            b = x[perm[i:i + 32]].to(device)
            rec, mu, lv = model(b)
            loss = F.binary_cross_entropy(rec, b, reduction="sum") - 0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        hist.append(tot / len(x))
        if ctx:
            ctx.progress((ep + 1) / epochs, f"VAE epoch {ep + 1}/{epochs} loss/img={hist[-1]:.1f}")
    run_id = time.strftime("vae-%Y%m%d-%H%M%S")
    root = settings.synthetic_dir / run_id
    model.eval()
    with torch.no_grad():
        s = (model.sample(n_samples, device).cpu().numpy()[:, 0] * 255).astype(np.uint8)
    for i, im in enumerate(s):
        write_image(root / "images" / f"{i:04d}.png", im)
    info = {"run_id": run_id, "kind": "vae_experimental", "dataset_id": dataset_id, "n": n_samples, "epochs": epochs,
            "loss_history": hist, "label": "unlabelled", "sample_type": "synthetic",
            "note": "Experimental; not used for training or evaluation."}
    (root / "vae_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), root / "vae.pt")
    return info
