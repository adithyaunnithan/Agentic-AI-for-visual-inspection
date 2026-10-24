"""Stress test: push held-out REAL test images through corruptions (severity 1-5) and measure how the
two-tier pipeline routes them - does degraded input reach tier 2 instead of being passed or flagged blindly?"""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .datasets.validate import read_image
from .pipeline import InspectionPipeline
from .vision.calibration import ece
from .vision.corruptions import ALL, LABEL_PRESERVING, corrupt

NORMAL = {"ok", "good", "normal"}


def run_stress(settings: Settings, run_id: str, ctx, kinds: list[str] | None = None,
               severities=(1, 2, 3, 4, 5), use_llm: bool = False, llm_limit: int = 20, max_images: int | None = None,
               use_calibration: bool = True) -> dict:
    kinds = kinds or ALL
    rdir = settings.models_dir / run_id
    test = pd.read_csv(rdir / "predictions_test.csv", keep_default_na=False)
    if max_images:
        test = test.groupby("label", group_keys=False).apply(lambda g: g.head(max_images // 2))
    pipe = InspectionPipeline(settings, run_id, use_kb=True, use_calibration=use_calibration)
    sid = time.strftime(f"{run_id}__stress-%Y%m%d-%H%M%S") + ("" if use_calibration else "-uncal")
    out = settings.eval_dir / sid
    out.mkdir(parents=True)
    rows, llm_used = [], 0
    jobs = [("none", 0)] + [(k, s) for k in kinds for s in severities]
    total = len(jobs) * len(test)
    done = 0
    for r in test.itertuples():
        img = read_image(Path(r.abs_path))
        defective = r.label not in NORMAL
        for k, (kind, sev) in enumerate(jobs):
            ctx.check_cancel()
            im = corrupt(img, kind, sev, seed=zlib.crc32(f"{r.sample_id}|{kind}|{sev}".encode()))
            allow = use_llm and llm_used < llm_limit
            rec = pipe.inspect_image(im, r.sample_id, r.abs_path, use_llm=allow, save=False,
                                     meta={"corruption": kind, "severity": sev})
            if rec["report"]["tier"].startswith("tier2_llm"):
                llm_used += 1
            v, rep, rt = rec["vision"], rec["report"], rec["routing"]
            rows.append({"sample_id": r.sample_id, "label": r.label, "defective": defective, "corruption": kind,
                         "severity": sev, "label_preserving": kind in LABEL_PRESERVING or kind == "none",
                         "p_cal": v["p_defect"], "p_raw": v["p_defect_uncalibrated"], "decision": rep["decision"],
                         "tier": rep["tier"], "in_band": rt["in_band"], "ood": rt["ood"],
                         "ood_feature": bool(v["ood"].get("feature_flag")), "ood_stats": bool(v["ood"].get("stat_flags")),
                         "feature_score": v["ood"].get("feature_score"),
                         **{f"z_{n}": z for n, z in (v["ood"].get("stat_z") or {}).items()},
                         "latency_s": rec["total_latency_s"]})
            done += 1
        if done % (len(jobs) * 10) == 0:
            ctx.progress(done / total, f"{done}/{total} inspections")
    df = pd.DataFrame(rows)
    df.to_csv(out / "rows.csv", index=False)
    summary = summarise(df)
    res = {"stress_id": sid, "run_id": run_id, "n_images": int(len(test)), "n_inspections": int(len(df)),
           "temperature": pipe.temperature, "calibrated": use_calibration,
           "band": [settings.agent.escalation_low, settings.agent.escalation_high],
           "llm_used": llm_used, "by_condition": summary, "overall_corrupted": overall(df[df.corruption != "none"]),
           "clean": overall(df[df.corruption == "none"]),
           "note": "Crop corruptions can move the defect out of frame, so their ground-truth labels are not guaranteed.",
           "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "stress.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    plot(df, out)
    return {"stress_id": sid, **{k: res["overall_corrupted"][k] for k in ("tier2_rate", "baseline_missed", "agentic_missed")}}


def overall(g: pd.DataFrame) -> dict:
    d = g[g.defective]
    o = g[~g.defective]
    y = g.defective.astype(int).values
    pr = np.c_[1 - g.p_raw, g.p_raw]
    pc = np.c_[1 - g.p_cal, g.p_cal]
    return {"n": int(len(g)),
            "accuracy": float(((g.p_raw >= 0.5) == g.defective).mean()),
            "ece_raw": ece(pr, y), "ece_calibrated": ece(pc, y),
            "tier2_rate": float((g.tier != "tier1_rules").mean()),
            "band_rate": float(g.in_band.mean()), "ood_rate": float(g.ood.mean()),
            "baseline_missed": float((d.p_raw < 0.5).mean()) if len(d) else None,
            "agentic_missed": float((d.decision == "PASS").mean()) if len(d) else None,
            "baseline_false_reject": float((o.p_raw >= 0.5).mean()) if len(o) else None,
            "agentic_false_reject": float((o.decision == "FLAG_FOR_REWORK").mean()) if len(o) else None,
            "autonomous_accuracy": float(((g.decision == "FLAG_FOR_REWORK") == g.defective)[g.decision != "ESCALATE"].mean())
            if (g.decision != "ESCALATE").any() else None,
            "errors_caught": _errors_caught(g)}


def _errors_caught(g: pd.DataFrame) -> dict:
    """Of the images the classifier gets wrong (threshold 0.5), how many does the router send to tier 2?"""
    wrong = (g.p_raw >= 0.5) != g.defective
    n = int(wrong.sum())
    return {"classifier_errors": n, "routed_to_tier2": int((g.tier[wrong] != "tier1_rules").sum()),
            "share": float((g.tier[wrong] != "tier1_rules").mean()) if n else None}


def summarise(df: pd.DataFrame) -> list[dict]:
    out = []
    for (k, s), g in df.groupby(["corruption", "severity"]):
        out.append({"corruption": k, "severity": int(s), **overall(g)})
    return out


def plot(df: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = [k for k in ALL if k in set(df.corruption)]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    clean = df[df.corruption == "none"]
    for ax, (metric, title) in zip(axes, (("acc", "Classifier accuracy"), ("tier2", "Routed to tier 2"),
                                          ("missed", "Missed defects: classifier (dashed) vs two-tier (solid)"))):
        for c, k in zip(colors, kinds):
            xs = [0] + sorted(df[df.corruption == k].severity.unique())
            ys, yb = [], []
            for s in xs:
                g = clean if s == 0 else df[(df.corruption == k) & (df.severity == s)]
                d = g[g.defective]
                if metric == "acc":
                    ys.append(((g.p_raw >= 0.5) == g.defective).mean())
                elif metric == "tier2":
                    ys.append((g.tier != "tier1_rules").mean())
                else:
                    ys.append((d.decision == "PASS").mean())
                    yb.append((d.p_raw < 0.5).mean())
            ax.plot(xs, ys, "o-", color=c, lw=2, ms=4, label=k.replace("_", " "))
            if metric == "missed":
                ax.plot(xs, yb, "--", color=c, lw=1.2, alpha=0.8)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("severity (0 = clean)")
        ax.set_xticks(range(6))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylim(-0.02, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eee")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "stress.png")
    plt.close(fig)


def list_stress(settings: Settings) -> list[dict]:
    """Newest first, calibrated (deployed configuration) before uncalibrated comparison runs."""
    out = [json.loads(p.read_text()) for p in sorted(settings.eval_dir.glob("*__stress-*/stress.json"),
                                                     key=lambda p: p.stat().st_mtime, reverse=True)]
    return sorted(out, key=lambda s: not s.get("calibrated", True))
