"""System evaluation on the held-out REAL test split.

Variants compared on identical images:
  baseline_classifier_only  : pass/fail at p=0.5, no agent
  agentic_rules_kb          : two-tier agent (rule tier + escalation), with knowledge base
  agentic_rules_no_kb       : ablation without the knowledge base
  agentic_llm_kb / _no_kb   : only when Anthropic credentials are available (borderline cases go to the LLM)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .datasets.validate import read_image
from .pipeline import InspectionPipeline
from .synthetic.procedural import list_runs as list_synth, load_run
from .vision.metrics import classification_metrics

NORMAL = {"ok", "good", "normal"}


def gt_action(label: str) -> str:
    return "PASS" if label in NORMAL else "FLAG_FOR_REWORK"


def decision_metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    esc = df["decision"] == "ESCALATE"
    auto = df[~esc]
    defective = df["gt"] == "FLAG_FOR_REWORK"
    out = {
        "n": n,
        "agreement_overall": float((df["decision"] == df["gt"]).mean()),
        "escalation_rate": float(esc.mean()),
        "autonomous_coverage": float(1 - esc.mean()),
        "agreement_autonomous": float((auto["decision"] == auto["gt"]).mean()) if len(auto) else None,
        "missed_defect_rate": float(((df["decision"] == "PASS") & defective).sum() / max(defective.sum(), 1)),
        "false_reject_rate": float(((df["decision"] == "FLAG_FOR_REWORK") & ~defective).sum() / max((~defective).sum(), 1)),
        "escalated_defective": int((esc & defective).sum()), "escalated_ok": int((esc & ~defective).sum()),
        "counts": df.groupby(["gt", "decision"]).size().rename("n").reset_index().to_dict("records"),
    }
    if "groundedness" in df:
        out.update({"mean_groundedness": float(df["groundedness"].mean()),
                    "mean_completeness": float(df["completeness"].mean()),
                    "reports_with_ungrounded_numbers": int((~df["numbers_grounded"]).sum()),
                    "hard_check_failures": int(df["hard_fail"].sum())})
    if "total_latency_s" in df:
        lat = df["total_latency_s"]
        out.update({"latency_mean_s": float(lat.mean()), "latency_p95_s": float(lat.quantile(0.95)),
                    "throughput_img_per_s": float(1 / lat.mean()),
                    "vision_latency_mean_s": float(df["vision_latency_s"].mean())})
    return out


def run_variant(pipe: InspectionPipeline, test: pd.DataFrame, use_llm: bool, llm_limit: int, ctx, label: str,
                reports_out: Path | None = None) -> pd.DataFrame:
    rows, llm_used = [], 0
    for k, r in enumerate(test.itertuples()):
        ctx.check_cancel()
        img = read_image(Path(r.abs_path))
        allow = use_llm and llm_used < llm_limit
        rec = pipe.inspect_image(img, r.sample_id, r.abs_path, use_llm=allow, save=False)
        rec.pop("heatmap", None)
        rec.pop("cams", None)
        if rec["report"]["tier"].startswith("tier2_llm"):
            llm_used += 1
        rep, chk = rec["report"], rec["checks"]
        rows.append({"sample_id": r.sample_id, "label": r.label, "gt": gt_action(r.label), "p_defect": rep["p_defect"],
                     "decision": rep["decision"], "tier": rep["tier"], "severity": rep["severity"],
                     "groundedness": chk["groundedness"], "completeness": chk["completeness"],
                     "numbers_grounded": chk["checks"]["numbers_grounded"], "hard_fail": bool(chk["hard_fail"]),
                     "total_latency_s": rec["total_latency_s"], "vision_latency_s": rec["vision"]["vision_latency_s"],
                     "llm_error": rec["llm_error"],
                     "llm_tokens_in": (rec["llm_usage"] or {}).get("input_tokens", 0),
                     "llm_tokens_out": (rec["llm_usage"] or {}).get("output_tokens", 0)})
        if reports_out is not None:
            with open(reports_out, "a", encoding="utf-8") as f:
                f.write(json.dumps({"variant": label, "label": r.label, **rec}, default=str) + "\n")
        if k % 20 == 0:
            ctx.progress(None, f"{label}: {k}/{len(test)}")
    return pd.DataFrame(rows)


def evaluate_run(settings: Settings, run_id: str, ctx, use_llm: bool = False, llm_limit: int = 40,
                 what_if_deployed: bool = False) -> dict:
    """what_if_deployed: for experimental (synthetic-only) models, disable the blanket provenance escalation
    so the uncertainty-band policy can be analysed. The live app always escalates such models."""
    rdir = settings.models_dir / run_id
    info = json.loads((rdir / "run_info.json").read_text(encoding="utf-8"))
    test = pd.read_csv(rdir / "predictions_test.csv", keep_default_na=False)
    classes = info["classes"]
    eval_id = time.strftime(f"{run_id}__eval-%Y%m%d-%H%M%S")
    out = settings.eval_dir / eval_id
    out.mkdir(parents=True)
    result = {"eval_id": eval_id, "run_id": run_id, "model_provenance": info["provenance"],
              "dataset_id": info["dataset_id"], "dataset_version": info["dataset_version"],
              "test_set": f"held-out REAL test split, n={len(test)} "
                          f"({test['label'].value_counts().to_dict()}); no synthetic or augmented images",
              "escalation_band": [settings.agent.escalation_low, settings.agent.escalation_high],
              "what_if_deployed": what_if_deployed,
              "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if what_if_deployed:
        result["analysis_note"] = ("WHAT-IF ANALYSIS: blanket escalation of experimental models disabled to study the "
                                   "uncertainty-band policy. The live application always escalates this model.")
    p_pos = test[f"p_{classes[1]}"] if len(classes) == 2 else None
    result["vision"] = classification_metrics(test["y"], test["pred"], p_pos, classes)
    cal = rdir / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text(encoding="utf-8"))
        result["calibration"] = {"temperature": c["temperature"], "temperatures": c["temperatures"],
                                 "evaluation": c["evaluation"],
                                 "ood_false_alarm_rate_clean_test": c["ood"]["false_alarm_rate_clean_test"]}

    base = pd.DataFrame({"gt": [gt_action(l) for l in test["label"]],
                         "decision": ["FLAG_FOR_REWORK" if p >= 0.5 else "PASS" for p in 1 - test[f"p_{classes[0]}"]]})
    result["variants"] = {"baseline_classifier_only": decision_metrics(base)}

    variants = [("agentic_rules_kb", True, False), ("agentic_rules_no_kb", False, False)]
    from .agent.agent import llm_credentials_present
    if use_llm and llm_credentials_present():
        variants += [("agentic_llm_kb", True, True), ("agentic_llm_no_kb", False, True)]
    elif use_llm:
        result["llm_note"] = "LLM variants skipped: no Anthropic credentials configured."
    for i, (name, kb, llm) in enumerate(variants):
        ctx.progress(0.05 + 0.85 * i / len(variants), f"running variant {name}")
        pipe = InspectionPipeline(settings, run_id, use_kb=kb, escalate_on_provenance=not what_if_deployed)
        df = run_variant(pipe, test, llm, llm_limit, ctx, name, out / "reports.jsonl")
        df.to_csv(out / f"decisions_{name}.csv", index=False)
        m = decision_metrics(df)
        if llm:
            m["llm_calls"] = int(df["tier"].str.startswith("tier2_llm").sum())
            m["llm_errors"] = int(df["llm_error"].astype(bool).sum())
            m["llm_tokens_in"], m["llm_tokens_out"] = int(df["llm_tokens_in"].sum()), int(df["llm_tokens_out"].sum())
        result["variants"][name] = m

    # Synthetic sanity check - reported separately, never mixed with real-world evidence.
    synth = [r for r in list_synth(settings) if r["dataset_id"] == info["dataset_id"]]
    if synth:
        s = load_run(settings, synth[0]["run_id"])
        used_in_training = synth[0]["run_id"] in info["config"].get("synthetic_runs", [])
        pipe = InspectionPipeline(settings, run_id)
        ps = [pipe.vision(read_image(Path(p)), "synthetic")["p_defect"] for p in s["abs_path"][:200]]
        result["synthetic_sanity_check"] = {
            "run_id": synth[0]["run_id"], "n": len(ps), "used_in_training": used_in_training,
            "fraction_flagged_defective": float(np.mean(np.array(ps) >= 0.5)),
            "by_defect_type": pd.DataFrame({"t": s["defect_type"][:len(ps)], "f": np.array(ps) >= 0.5})
                                .groupby("t")["f"].mean().round(3).to_dict(),
            "caveat": "Synthetic images are derived from TRAIN images. This is a sanity check of the generator / "
                      "model interaction, NOT evidence of real-world generalisation."}

    (out / "evaluation.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    plot_eval(result, out)
    ctx.progress(0.99, f"saved {out}")
    return {"eval_id": eval_id, **{k: {kk: v.get(kk) for kk in ("agreement_overall", "escalation_rate",
                                                                "missed_defect_rate", "false_reject_rate")}
                                   for k, v in result["variants"].items()}}


def plot_eval(result: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(result["variants"])
    keys = [("agreement_overall", "Decision agreement"), ("escalation_rate", "Escalation rate"),
            ("missed_defect_rate", "Missed-defect rate"), ("false_reject_rate", "False-reject rate")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    colors = ["#8a8a8a", "#2a6fdb", "#7fa7ea", "#d9622b", "#f0a77f"]
    for ax, (k, t) in zip(axes, keys):
        vals = [result["variants"][n].get(k) or 0 for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors[:len(names)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=7)
        ax.set_title(t, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "decision_metrics.png", dpi=150)
    plt.close(fig)


def list_evals(settings: Settings) -> list[dict]:
    out = []
    for p in sorted(settings.eval_dir.glob("*/evaluation.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out
