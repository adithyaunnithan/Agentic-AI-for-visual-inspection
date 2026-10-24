"""Quantitative saliency check. The real casting data has no defect masks, but synthetic defects do, so we can
measure how well each Grad-CAM variant points at a known defect:

  pointing game   - is the CAM peak inside the (slightly dilated) defect mask?
  energy-in-mask  - share of total CAM activation inside the dilated mask (chance level = mask area share)
  box IoU         - IoU of the primary region box with the mask's bounding box

Caveat: synthetic sources are training-split images, and defect realism is not verified. This is a relative
comparison of saliency methods, not a measurement on real defects."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .datasets.validate import read_image
from .pipeline import InspectionPipeline
from .synthetic.procedural import load_run
from .vision.model import CAM_MODES, cam_regions


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def saliency_eval(settings, run_id: str, synth_run: str, ctx=None, tolerance_px: int = 15) -> dict:
    s = load_run(settings, synth_run)
    info = json.loads((settings.models_dir / run_id / "run_info.json").read_text())
    if synth_run in info["config"].get("synthetic_runs", []):
        raise RuntimeError("Use a synthetic run that the model was NOT trained on.")
    pipe = InspectionPipeline(settings, run_id)
    acc = {m: {"hit": [], "energy": [], "iou": [], "chance": []} for m in CAM_MODES}
    per_type = {}
    for k, r in enumerate(s.itertuples()):
        if ctx:
            ctx.check_cancel()
            if k % 20 == 0:
                ctx.progress(k / len(s), f"{k}/{len(s)}")
        img = read_image(Path(r.abs_path))
        mask = read_image(Path(r.abs_mask_path)) > 0
        dil = cv2.dilate(mask.astype(np.uint8), np.ones((2 * tolerance_px + 1,) * 2, np.uint8)).astype(bool)
        ys, xs = np.nonzero(mask)
        h, w = mask.shape
        mbox = [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]
        v = pipe.vision(img, r.sample_id)
        for m in CAM_MODES:
            reg = cam_regions(v["cams"][m], (h, w))
            heat = reg["heatmap"]
            py, px = np.unravel_index(int(np.argmax(heat)), heat.shape)
            hit = bool(dil[py, px])
            energy = float(heat[dil].sum() / (heat.sum() + 1e-8))
            iou = _iou(reg["bbox_xyxy_norm"], mbox) if reg["bbox_xyxy_norm"] else 0.0
            acc[m]["hit"].append(hit)
            acc[m]["energy"].append(energy)
            acc[m]["iou"].append(iou)
            acc[m]["chance"].append(float(dil.mean()))
            per_type.setdefault(r.defect_type, {}).setdefault(m, []).append(hit)
    res = {"run_id": run_id, "synthetic_run": synth_run, "n": int(len(s)), "tolerance_px": tolerance_px,
           "modes": {m: {"pointing_game": float(np.mean(a["hit"])), "energy_in_mask": float(np.mean(a["energy"])),
                         "chance_energy": float(np.mean(a["chance"])), "box_iou": float(np.mean(a["iou"]))}
                     for m, a in acc.items()},
           "pointing_game_by_type": {t: {m: round(float(np.mean(x)), 3) for m, x in d.items()} for t, d in per_type.items()},
           "caveat": "Synthetic defects on training-split images; relative comparison of CAM variants only.",
           "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = settings.eval_dir / time.strftime(f"{run_id}__saliency-%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    (out / "saliency.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return {m: {k: round(v, 3) for k, v in d.items()} for m, d in res["modes"].items()}


def list_saliency(settings) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(settings.eval_dir.glob("*__saliency-*/saliency.json"),
                                                      key=lambda p: p.stat().st_mtime, reverse=True)]
