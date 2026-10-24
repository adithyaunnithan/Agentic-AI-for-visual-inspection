"""Inspection agents.

* RuleAgent  - deterministic tier-1 agent (always available, no network). Also the fallback.
* ClaudeAgent - LLM agent with tool calling (vision tool, knowledge-base tool, report-submission tool).

Division of labour (hallucination control): numeric / factual fields (probabilities, location, model
provenance, subtype_verified) are copied from tool outputs by code. The LLM only decides the
decision, severity, hypotheses, action and the written rationale - and those are checked afterwards."""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

from .kb import KnowledgeBase

DECISIONS = ["PASS", "FLAG_FOR_REWORK", "ESCALATE"]
SEVERITIES = ["none", "low", "medium", "high", "unknown"]
CASTING_SUBTYPES = ["blow hole", "pinhole", "burr", "shrinkage", "mould material defect", "pouring metal defect",
                    "metallurgical defect", "crack", "rough texture"]


# --------------------------------------------------------------------------------------- report assembly
def assemble_report(vision: dict, llm_fields: dict, generated_by: str, tier: str) -> dict:
    decision = llm_fields["decision"]
    defective_class = vision["predicted_class"] not in ("ok", "good", "normal")
    if vision["task"] == "binary_classification":
        defect_type = "casting_defect (subtype not identified by model)" if defective_class else "none"
    elif vision["task"] == "image_anomaly_detection":
        defect_type = "anomaly (type not identified by model)" if defective_class else "none"
    else:
        defect_type = vision["predicted_class"]
    return {
        "part_id": vision["part_id"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "decision": decision,
        "defect_detected": {"PASS": False, "FLAG_FOR_REWORK": True, "ESCALATE": None}[decision],
        "defect_type": defect_type,
        "defect_subtype_hypotheses": llm_fields.get("defect_subtype_hypotheses", []),
        "subtype_verified": bool(vision.get("subtypes_available", False)),
        "location": {k: vision["location"].get(k) for k in ("bbox_xyxy_norm", "region", "area_fraction")}
        | {"method": vision["location"].get("method", "grad-cam saliency (approximate)")},
        "severity": llm_fields["severity"],
        "confidence": vision["confidence"],
        "p_defect": vision["p_defect"],
        "recommended_action": llm_fields["recommended_action"],
        "rationale": llm_fields["rationale"],
        "evidence": llm_fields.get("evidence", []),
        "kb_entries_used": llm_fields.get("kb_entries_used", []),
        "escalation_reason": llm_fields.get("escalation_reason", ""),
        "tier": tier,
        "generated_by": generated_by,
        "model_provenance": vision["model_provenance"],
        "model_run": vision.get("model_run"),
    }


# --------------------------------------------------------------------------------------- checks
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(%?)")


def check_report(report: dict, vision: dict, kb: KnowledgeBase | None, dataset_id: str, band: tuple[float, float]) -> dict:
    """Automated groundedness + completeness checks. Returns per-check booleans and scores."""
    p = vision["p_defect"]
    valid_ids = {e["id"] for e in kb.for_dataset(dataset_id)} if kb else set()
    checks = {
        "decision_valid": report["decision"] in DECISIONS,
        "no_pass_on_likely_defect": not (report["decision"] == "PASS" and p >= 0.5),
        "no_flag_on_confident_ok": not (report["decision"] == "FLAG_FOR_REWORK" and p < band[0]),
        "kb_refs_exist": set(report["kb_entries_used"]) <= valid_ids,
        "subtypes_marked_unverified": not (report["defect_subtype_hypotheses"] and report["subtype_verified"]
                                           and not vision.get("subtypes_available")),
        "subtypes_grounded": (not report["defect_subtype_hypotheses"]) or (
            kb is not None and all(any(s2 in h.lower() for s2 in CASTING_SUBTYPES + ["anomaly"])
                                   for h in report["defect_subtype_hypotheses"])),
        "severity_consistent": (report["severity"] == "none") == (report["decision"] == "PASS"),
        "escalation_has_reason": report["decision"] != "ESCALATE" or bool(report["escalation_reason"].strip()),
    }
    if kb and report["decision"] == "FLAG_FOR_REWORK":
        checks["severity_matches_policy"] = report["severity"] == kb.severity(vision["location"].get("area_fraction"))
    # Numbers quoted in free text must come from tool outputs (probabilities, area, bbox) or policy thresholds.
    allowed = [p, 1 - p, vision["confidence"], vision["location"].get("area_fraction") or 0, 0.10, 0.03, 0.5,
               *band, *(vision["location"].get("bbox_xyxy_norm") or [])]
    bad = []
    text = report["rationale"] + " " + " ".join(report["evidence"])
    for num, pct in _NUM.findall(text):
        v = float(num) / (100 if pct else 1)
        if v > 1 and not pct:
            continue  # counts / ids / years are not probability claims
        if not any(abs(v - a) <= 0.011 for a in allowed):
            bad.append(num + pct)
    checks["numbers_grounded"] = not bad
    required = {
        "decision": bool(report["decision"]),
        "severity_known": report["severity"] not in ("unknown", ""),
        "recommended_action_specific": len(report["recommended_action"]) > 25,
        "rationale": len(report["rationale"]) > 30,
        "location_when_defect": report["decision"] == "PASS" or report["location"]["bbox_xyxy_norm"] is not None,
        "kb_cited_when_defect": report["decision"] == "PASS" or bool(report["kb_entries_used"]),
        "evidence": bool(report["evidence"]),
    }
    hard = ("decision_valid", "no_pass_on_likely_defect", "no_flag_on_confident_ok", "kb_refs_exist",
            "subtypes_marked_unverified")
    return {"checks": checks, "ungrounded_numbers": bad,
            "groundedness": round(sum(checks.values()) / len(checks), 4),
            "completeness_fields": required, "completeness": round(sum(required.values()) / len(required), 4),
            "hard_fail": [k for k in hard if not checks[k]]}


# --------------------------------------------------------------------------------------- rule agent
class RuleAgent:
    name = "rule_agent"

    def __init__(self, kb: KnowledgeBase | None):
        self.kb = kb

    def decide(self, vision: dict, band: tuple[float, float], reason: str | None = None) -> dict:
        p, loc = vision["p_defect"], vision["location"]
        ev = [f"vision model p(defect)={p:.3f} ({vision['model_provenance']})"]
        if loc.get("bbox_xyxy_norm"):
            ev.append(f"Grad-CAM saliency peak at {loc['region']} (approximate), area_fraction={loc['area_fraction']}")
        if reason:  # escalation requested by the pipeline (uncertainty band / OOD / demo model)
            return {"decision": "ESCALATE", "severity": "unknown",
                    "defect_subtype_hypotheses": [],
                    "recommended_action": "Hold the part and route it to a human inspector for review",
                    "rationale": f"Automatic decision withheld: {reason}.", "evidence": ev,
                    "kb_entries_used": ["KB-POL-003"] if self.kb else [], "escalation_reason": reason}
        if p < 0.5:
            return {"decision": "PASS", "severity": "none", "defect_subtype_hypotheses": [],
                    "recommended_action": "Release the part to the next process step; no defect detected",
                    "rationale": f"The vision model reports a low defect probability ({p:.3f}), below the "
                                 f"uncertainty band, so the part passes.", "evidence": ev,
                    "kb_entries_used": ["KB-POL-002"] if self.kb else [], "escalation_reason": ""}
        if self.kb:
            sev = self.kb.severity(loc.get("area_fraction"))
            action = self.kb.ACTIONS[sev]
            refs = ["KB-POL-001", "KB-POL-002"] + (["KB-CAST-001"] if vision["dataset_id"] == "casting_kaggle" else [])
            rat = (f"The vision model reports a defect with p={p:.3f}. Severity '{sev}' follows the saliency-area "
                   f"policy (approximate area_fraction={loc.get('area_fraction')}). The model does not identify the "
                   f"defect subtype.")
        else:  # ablation: no knowledge base -> no severity policy, generic action
            sev, action, refs = "unknown", "Route to quality team", []
            rat = f"The vision model reports a defect with p={p:.3f}."
        return {"decision": "FLAG_FOR_REWORK", "severity": sev, "defect_subtype_hypotheses": [],
                "recommended_action": action, "rationale": rat, "evidence": ev, "kb_entries_used": refs,
                "escalation_reason": ""}


# --------------------------------------------------------------------------------------- LLM agent
def llm_credentials_present() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_PROFILE"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


SUBMIT_TOOL = {
    "name": "submit_inspection_report",
    "description": "Submit the final inspection report for this part. Call exactly once, after gathering evidence.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["decision", "severity", "defect_subtype_hypotheses", "recommended_action", "rationale",
                     "evidence", "kb_entries_used", "escalation_reason"],
        "properties": {
            "decision": {"type": "string", "enum": DECISIONS},
            "severity": {"type": "string", "enum": SEVERITIES},
            "defect_subtype_hypotheses": {"type": "array", "items": {"type": "string"},
                                          "description": "Unverified subtype hypotheses, empty if none."},
            "recommended_action": {"type": "string"},
            "rationale": {"type": "string", "description": "2-4 sentences grounded in tool outputs / image."},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "kb_entries_used": {"type": "array", "items": {"type": "string"}},
            "escalation_reason": {"type": "string", "description": "Empty unless decision is ESCALATE."},
        },
    },
}
VISION_TOOL = {"name": "get_vision_result",
               "description": "Return the vision model's output for the current part: predicted class, defect "
                              "probability, approximate Grad-CAM location, out-of-distribution check and model provenance.",
               "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}}
KB_TOOL = {"name": "search_knowledge_base",
           "description": "Search the defect knowledge base (label semantics, defect types, root-cause hints, "
                          "severity/action/escalation policy). Returns entries with ids you can cite.",
           "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"],
                            "additionalProperties": False}}

SYSTEM = """You are the reasoning stage of an automated visual-inspection line ({domain}).
A vision model has already scored the part. You were consulted because: {why}.

Call get_vision_result first.{kb_line} Then call submit_inspection_report exactly once.

Grounding rules:
- Every statement must come from the vision tool output,{kb_src} or what is visible in the attached image.
- Quote probabilities and areas exactly as the tool returns them; do not introduce other numbers.
- {subtype_line}
- PASS only when you are confident the part is defect-free. Escalated parts are held for a human, so ESCALATE is the safe choice under genuine uncertainty or conflicting evidence.
- Grad-CAM location is approximate saliency, not a measured defect boundary."""


class ClaudeAgent:
    def __init__(self, kb: KnowledgeBase | None, model: str, effort: str, send_image: bool):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=2, timeout=120)
        self.kb, self.model, self.effort, self.send_image = kb, model, effort, send_image
        self.name = f"llm_agent:{model}"

    def _image_block(self, img: np.ndarray) -> dict:
        h, w = img.shape[:2]
        s = 512 / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(buf.tobytes()).decode()}}

    def decide(self, vision: dict, image: np.ndarray | None, why: str, domain: str) -> tuple[dict, dict]:
        binary = vision["task"] != "multiclass_defect_classification"
        system = SYSTEM.format(
            domain=domain, why=why,
            kb_line=" Use search_knowledge_base for label semantics and the severity, action and escalation "
                    "policies, and cite entry ids." if self.kb else "",
            kb_src=" the knowledge base," if self.kb else "",
            subtype_line=("The vision model does NOT identify defect subtypes. Any subtype goes in "
                          "defect_subtype_hypotheses and is an unverified hypothesis." if binary else
                          "The predicted class comes from the model; do not substitute another class."))
        tools = [VISION_TOOL, SUBMIT_TOOL] + ([KB_TOOL] if self.kb else [])
        content = []
        if self.send_image and image is not None:
            content.append(self._image_block(image))
        content.append({"type": "text", "text": f"Inspect part {vision['part_id']}."})
        messages = [{"role": "user", "content": content}]
        vis_public = {k: v for k, v in vision.items() if k not in ("heatmap",)}
        usage = {"input_tokens": 0, "output_tokens": 0, "turns": 0, "latency_s": 0.0}
        t0 = time.time()
        nudged = False
        for _ in range(6):
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=8000, system=system, tools=tools, messages=messages,
                thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            usage["turns"] += 1
            usage["input_tokens"] += resp.usage.input_tokens
            usage["output_tokens"] += resp.usage.output_tokens
            usage["served_by"] = resp.model
            if resp.stop_reason == "refusal":
                raise RuntimeError("LLM declined the request (refusal)")
            messages.append({"role": "assistant", "content": resp.content})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                if nudged:
                    raise RuntimeError("LLM ended without submitting a report")
                nudged = True
                messages.append({"role": "user", "content": "Please call submit_inspection_report now."})
                continue
            results, submitted = [], None
            for c in calls:
                if c.name == "get_vision_result":
                    out = json.dumps(vis_public, default=str)
                elif c.name == "search_knowledge_base" and self.kb:
                    out = json.dumps(self.kb.search(c.input.get("query", ""), vision["dataset_id"]))
                elif c.name == "submit_inspection_report":
                    submitted = dict(c.input)
                    out = "report received"
                else:
                    out = f"unknown tool {c.name}"
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": out})
            if submitted is not None:
                usage["latency_s"] = round(time.time() - t0, 2)
                return submitted, usage
            messages.append({"role": "user", "content": results})
        raise RuntimeError("LLM agent exceeded the turn limit")
