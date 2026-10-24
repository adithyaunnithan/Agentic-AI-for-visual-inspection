"""End-to-end inspection: image -> preprocessing -> vision model -> two-tier agent -> checked report."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from .agent.agent import ClaudeAgent, RuleAgent, assemble_report, check_report, llm_credentials_present
from .agent.kb import KnowledgeBase
from .config import Settings
from .datasets.manifest import current_version, load_manifest
from .datasets.validate import read_image, write_image
from .vision.anomaly import AnomalyScorer, anomaly_probability
from .vision.calibration import OOD_POLICY, OODDetector, load_calibration
from .vision.model import GradCAM, build_model, cam_regions, preprocess, render_saliency

PROVENANCE_LABELS = {
    "trained_real": "Trained model backed by real data",
    "trained_real_plus_synthetic": "Trained on real data + synthetic defects (train split only)",
    "synthetic_trained_experimental": "EXPERIMENTAL: defects learned from synthetic data only",
    "pretrained_anomaly": "Pretrained ImageNet backbone + memory bank of real normal images (no fine-tuning)",
    "demo_deterministic": "DEMO ONLY - deterministic pseudo-prediction, not a real model",
}
DOMAINS = {"casting_kaggle": "metal casting, submersible pump impeller, top view",
           "mvtec_ad": "MVTec AD industrial objects", "mvtec_ad_hf_mirror": "MVTec AD industrial objects",
           "neu_surface": "hot-rolled steel strip surface"}
SALIENCY_DEFAULT = "fused"   # chosen by the localisation test on synthetic masks (tools: cli.py saliency-eval)
ANOMALY_REGION_THRESH = 1 / 1.5   # anomaly map is scaled so 1.0 = 1.5 x threshold; this level = the threshold


def latest_complete_run(settings: Settings, provenance: str | None = "trained_real") -> str | None:
    """Newest complete classifier run on the CURRENT dataset version (preferring the given provenance)."""
    runs = []
    for p in sorted(settings.models_dir.glob("*/run_info.json"), reverse=True):
        info = json.loads(p.read_text(encoding="utf-8"))
        if info.get("status") == "complete" and info.get("model_type", "classifier") == "classifier":
            runs.append(info)
    if not runs:
        return None
    cur = {}
    for info in runs:
        v = current_version(settings.datasets_dir, info["dataset_id"])
        cur[info["run_id"]] = v is not None and v.name == info["dataset_version"]
    ranked = sorted(runs, key=lambda i: (not cur[i["run_id"]], i.get("provenance") != provenance))
    return ranked[0]["run_id"]


class InspectionPipeline:
    def __init__(self, settings: Settings, model_run: str | None = None, use_kb: bool | None = None,
                 escalate_on_provenance: bool = True, saliency_mode: str = SALIENCY_DEFAULT,
                 use_calibration: bool = True):
        self.s = settings
        # False only for offline what-if analysis of experimental models; never in the live app
        self.escalate_on_provenance = escalate_on_provenance
        self.saliency_mode = saliency_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_kb = settings.agent.use_knowledge_base if use_kb is None else use_kb
        self.kb = KnowledgeBase() if self.use_kb else None
        self.rule = RuleAgent(self.kb)
        self.run_id = model_run or latest_complete_run(settings)
        self.model, self.anomaly = None, None
        self.temperature, self.ood_detector, self.calibration = 1.0, None, None
        self.info: dict = {"provenance": "demo_deterministic", "dataset_id": settings.train.dataset_id,
                           "task": "binary_classification", "classes": ["ok", "defective"]}
        if self.run_id:
            rdir = settings.models_dir / self.run_id
            self.info = json.loads((rdir / "run_info.json").read_text(encoding="utf-8"))
            if self.info.get("model_type") == "anomaly":
                cal = rdir / "anomaly_calibration.json"
                if not cal.exists():
                    raise RuntimeError(f"Anomaly run {self.run_id} is not calibrated: run `python cli.py calibrate {self.run_id}`")
                self.calibration = json.loads(cal.read_text(encoding="utf-8"))
                self.anomaly = AnomalyScorer(rdir, self.device)
                self.ood_detector = OODDetector(self.calibration["ood"])   # image statistics only
                self.legacy_ood = None
                self._llm, self._llm_error = None, None
                return
            ck = torch.load(rdir / "best.pt", map_location=self.device, weights_only=False)
            self.model = build_model(ck["arch"], len(ck["classes"]), pretrained=False).to(self.device)
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()
            self.ck = ck
            self.cam = GradCAM(self.model)
            t, det, cal = load_calibration(rdir)
            self.temperature = t if use_calibration else 1.0
            self.ood_detector, self.calibration = det, cal
            self.legacy_ood = None if det else self._legacy_ood_stats(rdir)
        self._llm = None
        self._llm_error = None

    # ------------------------------------------------------------------ helpers
    def _legacy_ood_stats(self, rdir: Path) -> dict:
        """Mean-intensity z-score, only for runs fitted before calibration.json existed."""
        p = rdir / "ood_stats.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        vdir = current_version(self.s.datasets_dir, self.info["dataset_id"])
        stats = {"mean": None, "std": None}
        if vdir:
            df = load_manifest(vdir)
            tr = df[df["split"] == "train"].sample(n=min(300, (df["split"] == "train").sum()), random_state=0)
            means = [float(im.mean()) for im in (read_image(Path(ap)) for ap in tr["abs_path"]) if im is not None]
            stats = {"mean": float(np.mean(means)), "std": float(np.std(means) + 1e-6), "n": len(means)}
        p.write_text(json.dumps(stats), encoding="utf-8")
        return stats

    @property
    def llm(self):
        if self._llm is None and self._llm_error is None:
            if not llm_credentials_present():
                self._llm_error = "no Anthropic credentials configured (set ANTHROPIC_API_KEY)"
            else:
                try:
                    a = self.s.agent
                    self._llm = ClaudeAgent(self.kb, a.model, a.effort, a.send_image_to_llm)
                except Exception as e:  # noqa: BLE001
                    self._llm_error = f"LLM client unavailable: {e}"
        return self._llm

    @property
    def provenance(self) -> str:
        return self.info["provenance"]

    # ------------------------------------------------------------------ vision tool
    def vision(self, img: np.ndarray, part_id: str) -> dict:
        t0 = time.time()
        classes = self.info["classes"]
        cams = None
        if self.anomaly is not None:
            score, amap = self.anomaly(img)
            thr = self.anomaly.threshold
            p = p_raw = float(anomaly_probability(score, thr, self.calibration["slope"]))
            pred = classes[int(p >= 0.5)]
            # patch-score map on a fixed scale (1.0 = 1.5 x decision threshold); regions = patches above threshold
            cams = {"anomaly map": np.clip(amap / (1.5 * thr), 0, 1)}
            region = cam_regions(cams["anomaly map"], img.shape[:2], thresh=ANOMALY_REGION_THRESH)
            heat = region.pop("heatmap")
            loc = region | {"method": "PatchCore patch anomaly map (approximate)"}
            if p < self.s.agent.escalation_low:
                loc = {"bbox_xyxy_norm": None, "region": "n/a (no anomaly predicted)", "area_fraction": None,
                       "method": loc["method"]}
            ood = self.ood_detector.check(None, img, p)
            ood["anomaly_score"], ood["anomaly_threshold"] = round(score, 3), round(thr, 3)
        elif self.model is None:
            h = int(hashlib.sha256(img.tobytes()).hexdigest()[:8], 16)
            p = p_raw = (h % 1000) / 1000
            loc = {"bbox_xyxy_norm": None, "region": "not localised", "area_fraction": None, "method": "none (demo mode)"}
            pred = classes[1] if p >= 0.5 else classes[0]
            heat = None
            ood = {"flag": False, "method": "none (demo mode)"}
        else:
            x = torch.from_numpy(preprocess(img, self.ck["image_size"], self.ck["clahe"]).transpose(2, 0, 1))[None]
            logits = self.cam.forward(x.to(self.device))
            lg = logits.detach().float().cpu().numpy()[0]
            raw = np.exp(lg - lg.max()); raw /= raw.sum()                     # noqa: E702
            z = lg / self.temperature
            probs = np.exp(z - z.max()); probs /= probs.sum()                 # noqa: E702
            target = int(np.argmax(probs[1:]) + 1) if len(classes) > 1 else 0   # explain the most likely defect class
            cams = self.cam.cams(logits, target)
            region = cam_regions(cams[self.saliency_mode], img.shape[:2])
            heat = region.pop("heatmap")
            p, p_raw = float(1 - probs[0]), float(1 - raw[0])
            pred = classes[int(np.argmax(probs))]
            loc = region | {"method": f"grad-cam saliency, {self.saliency_mode} (approximate)"}
            if p < self.s.agent.escalation_low:   # saliency of a confidently normal part is not a defect location
                loc = {"bbox_xyxy_norm": None, "region": "n/a (no defect predicted)", "area_fraction": None,
                       "method": loc["method"]}
            if self.ood_detector is not None:
                ood = self.ood_detector.check(self.cam.features.detach().float().cpu().numpy()[0], img, p)
            else:
                zz = abs(float(img.mean()) - self.legacy_ood["mean"]) / self.legacy_ood["std"]
                ood = {"flag": bool(zz > 4.0), "z": round(zz, 2), "method": "legacy: z-score of mean intensity"}
        conf = max(p, 1 - p)
        return {"part_id": part_id, "classes": classes, "predicted_class": pred, "p_defect": round(p, 4),
                "p_defect_uncalibrated": round(p_raw, 4), "temperature": round(self.temperature, 4),
                "confidence": round(conf, 4), "location": loc, "ood": ood, "model_provenance": self.provenance,
                "model_run": self.run_id, "dataset_id": self.info["dataset_id"], "task": self.info["task"],
                "subtypes_available": self.info["task"] == "multiclass_defect_classification",
                "vision_latency_s": round(time.time() - t0, 4), "heatmap": heat, "cams": cams}

    # ------------------------------------------------------------------ routing
    def route(self, v: dict) -> tuple[dict, list[str]]:
        """Which tier-2 triggers fire for a vision result, and the human-readable reasons."""
        band = (self.s.agent.escalation_low, self.s.agent.escalation_high)
        p = v["p_defect"]
        ood = bool(v["ood"].get("flag"))
        if ood and self.anomaly is not None and p >= OOD_POLICY["confident"][1]:
            # An unusual image can only raise a nearest-normal-patch distance, i.e. push towards FLAG. The OOD check
            # guards against a wrong PASS, so it adds nothing to a confident anomaly (e.g. MVTec 'flip' parts).
            v["ood"]["not_routed"] = "confident anomaly: OOD only guards against a wrong PASS"
            ood = False
        routing = {"in_band": band[0] <= p <= band[1], "ood": ood,
                   "provenance": self.provenance == "demo_deterministic" or (
                       self.provenance == "synthetic_trained_experimental" and self.escalate_on_provenance)}
        reasons = []
        if routing["in_band"]:
            reasons.append(f"defect probability {p:.3f} is inside the uncertainty band [{band[0]}, {band[1]}]")
        if routing["ood"]:
            o = v["ood"]
            what = []
            if o.get("feature_flag"):
                what.append("image features unlike the training data")
            if o.get("stat_flags"):
                names = {"mean_intensity": "brightness", "contrast_std": "contrast", "log_sharpness": "sharpness",
                         "rel_sharpness": "sharpness"}
                what.append("unusual " + " and ".join(names.get(k, k) for k in o["stat_flags"]))
            if not what:
                what.append("image statistics unlike the training data")
            reasons.append("out-of-distribution input: " + "; ".join(what))
        if routing["provenance"]:
            reasons.append(f"model provenance is '{self.provenance}'")
        return routing, reasons

    # ------------------------------------------------------------------ main entry
    def inspect_image(self, img: np.ndarray, part_id: str, source_path: str | None = None,
                      use_llm: bool = True, save: bool = True, meta: dict | None = None) -> dict:
        t0 = time.time()
        a = self.s.agent
        band = (a.escalation_low, a.escalation_high)
        v = self.vision(img, part_id)
        routing, reasons = self.route(v)
        tier, llm_usage, llm_error, agent_name = "tier1_rules", None, None, self.rule.name
        fields = None
        if (reasons or a.llm_for_all) and use_llm and self.provenance != "demo_deterministic" and self.llm:
            tier = "tier2_llm"
            try:
                fields, llm_usage = self.llm.decide(v, img, "; ".join(reasons) or "full-review mode",
                                                    DOMAINS.get(v["dataset_id"], "industrial parts"))
                agent_name = self.llm.name
            except Exception as e:  # noqa: BLE001 - any LLM failure falls back to the safe rule path
                llm_error = f"{type(e).__name__}: {e}"
                tier = "tier2_fallback_rules"
        elif reasons and self.provenance != "demo_deterministic":
            llm_error = self._llm_error if use_llm else "LLM disabled for this inspection"
            tier = "tier2_fallback_rules"
        if fields is None:
            fields = self.rule.decide(v, band, "; ".join(reasons) if reasons else None)
        report = assemble_report(v, fields, agent_name, tier)
        checks = check_report(report, v, self.kb, v["dataset_id"], band)
        if checks["hard_fail"] and agent_name != self.rule.name:
            # An LLM report that violates hard grounding rules is not trusted: hold the part for a human.
            fields = self.rule.decide(v, band, "LLM report failed grounding checks: " + ", ".join(checks["hard_fail"]))
            rejected = report
            report = assemble_report(v, fields, self.rule.name, "tier2_llm_rejected")
            report["rejected_llm_report"] = rejected
            checks = check_report(report, v, self.kb, v["dataset_id"], band)
        rec = {"report_id": uuid.uuid4().hex[:12], "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "report": report, "checks": checks, "routing": routing,
               "vision": {k: x for k, x in v.items() if k not in ("heatmap", "cams")},
               "source_path": source_path, "llm_usage": llm_usage, "llm_error": llm_error,
               "kb_enabled": self.kb is not None, "meta": meta or {}, "total_latency_s": round(time.time() - t0, 4)}
        if save:
            rec["assets"] = self._save_assets(rec["report_id"], img, v)
            day = time.strftime("%Y-%m-%d")
            with open(self.s.reports_dir / f"reports_{day}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        rec["heatmap"] = v["heatmap"]
        rec["cams"] = v["cams"]
        return rec

    def _save_assets(self, report_id: str, img: np.ndarray, v: dict) -> dict:
        """Persist the inspected image and its saliency overlay so the report history is self-contained."""
        day = time.strftime("%Y-%m-%d")
        d = self.s.reports_dir / "assets" / day
        h, w = img.shape[:2]
        s = min(1.0, 512 / max(h, w))
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
        out = {"input": f"assets/{day}/{report_id}_input.png"}
        write_image(self.s.reports_dir / out["input"], small)
        if v["heatmap"] is not None:
            regions = v["location"].get("regions") if v["location"].get("bbox_xyxy_norm") else None
            sal = render_saliency(small, v["heatmap"], regions)
            out["saliency"] = f"assets/{day}/{report_id}_saliency.png"
            write_image(self.s.reports_dir / out["saliency"], cv2.cvtColor(sal, cv2.COLOR_RGB2BGR))
        return out

    def inspect_path(self, path: Path, **kw) -> dict:
        img = read_image(path)
        if img is None:
            raise IOError(f"Cannot read image {path}")
        rec = self.inspect_image(img, path.stem, str(path), **kw)
        rec.pop("heatmap", None)
        rec.pop("cams", None)
        return rec


def load_history(settings: Settings) -> list[dict]:
    """All persisted inspection reports, newest first (older records without ids get a stable derived id)."""
    recs = []
    for f in sorted(settings.reports_dir.glob("reports_*.jsonl"), reverse=True):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            d.setdefault("report_id", f"{f.stem}-{i:05d}")
            d.setdefault("saved_at", d["report"]["timestamp"])
            recs.append(d)
    recs.sort(key=lambda d: d["saved_at"], reverse=True)
    return recs
