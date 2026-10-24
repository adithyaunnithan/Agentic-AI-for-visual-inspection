"""Agentic Visual Inspection - main page (run with: streamlit run app/Inspect.py)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from common import DECISION_COLORS, PROV_COLORS, settings

from inspection.agent.agent import llm_credentials_present
from inspection.config import PUBLIC_DEMO
from inspection.datasets.manifest import current_version, load_manifest
from inspection.datasets.validate import read_image
from inspection.pipeline import (ANOMALY_REGION_THRESH, PROVENANCE_LABELS, SALIENCY_DEFAULT, InspectionPipeline,
                                 latest_complete_run)
from inspection.vision.corruptions import ALL as CORRUPTIONS, corrupt
from inspection.vision.model import cam_regions, render_saliency
from inspection.vision.train import list_runs

# Tried in order by the "I'm not sure" button: quality loss the model genuinely suffers from, mildest first.
UNSURE_CANDIDATES = [("gaussian_blur", 3), ("motion_blur", 3), ("gaussian_noise", 4), ("gaussian_blur", 4),
                     ("motion_blur", 4), ("crop_zoom", 3), ("gaussian_noise", 5), ("gaussian_blur", 5)]
TIER_LABELS = {"tier1_rules": "1 · rules", "tier2_llm": "2 · LLM", "tier2_fallback_rules": "2 · held",
               "tier2_llm_rejected": "2 · LLM rejected"}
MAX_UPLOAD_MB = 10                 # also enforced by server.maxUploadSize in the public deployment
MAX_UPLOAD_PIXELS = 40_000_000     # a 10 MB PNG can still decode to a huge bitmap
LLM_CALLS_PER_SESSION = 5          # public demo only: the Space owner's API key pays for these

st.set_page_config(page_title="Agentic Visual Inspection", page_icon="🔎", layout="wide")
S = settings()
st.title("🔎 Agentic Visual Inspection")
st.caption("Vision model (tier 1) → LLM agent for uncertain or unusual parts (tier 2) → structured, checked inspection report")

# ------------------------------------------------------------------ sidebar: model + agent config
runs = [r for r in list_runs(S) if r.get("status") == "complete" and (
    r.get("model_type", "classifier") == "classifier"
    or (S.models_dir / r["run_id"] / "anomaly_calibration.json").exists())]
default_run = latest_complete_run(S)
runs.sort(key=lambda r: r["run_id"] != default_run)   # newest real-data model on the current split first
with st.sidebar:
    st.header("Model")
    options = [r["run_id"] for r in runs] + ([] if PUBLIC_DEMO else ["(demo mode - no model)"])
    choice = st.selectbox("Vision model run", options, format_func=lambda o: o if o.startswith("(") else
                          f"{o} · {next(r['dataset_id'] + ' · ' + r['provenance'] for r in runs if r['run_id'] == o)}")
    run_id = None if choice.startswith("(demo") else choice
    use_cal = st.toggle("Calibrated probabilities (temperature scaling)", value=True)
    st.header("Agent")
    # Settings are a process-wide cached object shared by every session, so a public demo must not let visitors
    # change them: the widgets are shown but locked.
    a = S.agent
    a.escalation_low, a.escalation_high = st.slider("Uncertainty band p(defect) → tier 2", 0.0, 1.0,
                                                    (a.escalation_low, a.escalation_high), 0.01, disabled=PUBLIC_DEMO)
    use_llm = st.toggle("Use LLM agent for tier 2", value=True)
    a.llm_for_all = st.toggle("LLM reviews every part (slow)", value=a.llm_for_all, disabled=PUBLIC_DEMO)
    a.send_image_to_llm = st.toggle("Send image to LLM (VLM mode)", value=a.send_image_to_llm, disabled=PUBLIC_DEMO)
    use_kb = st.toggle("Defect knowledge base tool", value=a.use_knowledge_base)
    st.caption(f"LLM: `{a.model}` · effort `{a.effort}`")
    if llm_credentials_present():
        st.success("Anthropic credentials found")
    else:
        st.warning("No Anthropic credentials — tier-2 cases fall back to rule-based escalation. "
                   "Set `ANTHROPIC_API_KEY` to enable the LLM agent.")
    if PUBLIC_DEMO:
        llm_left = LLM_CALLS_PER_SESSION - st.session_state.get("llm_calls", 0)
        if llm_left <= 0:
            use_llm = False
            st.info(f"LLM limit of {LLM_CALLS_PER_SESSION} calls for this session reached — tier-2 parts are now "
                    "held by the rule agent.")
        else:
            st.caption(f"Public demo: {llm_left} of {LLM_CALLS_PER_SESSION} LLM calls left in this session.")
    else:
        st.page_link("pages/6_Report_History.py", label="Open report history", icon="🗂️")


@st.cache_resource(max_entries=4)
def get_pipeline(run_id, use_kb, use_cal):
    return InspectionPipeline(S, run_id, use_kb=use_kb, use_calibration=use_cal)


pipe = get_pipeline(run_id, use_kb, use_cal)
prov = pipe.provenance
st.markdown(f"**Model provenance:** :{PROV_COLORS.get(prov, 'gray')}[{PROVENANCE_LABELS[prov]}]"
            + (f" · run `{pipe.run_id}` · dataset `{pipe.info['dataset_id']}` {pipe.info.get('dataset_version', '')}"
               if pipe.run_id else ""))
if prov == "demo_deterministic":
    st.error("DEMO MODE: no trained model is available. Predictions are deterministic pseudo-values derived from "
             "the image hash and every part is escalated. Train a model on the Training page.")
if pipe.anomaly is not None:
    tm = pipe.info.get("test_metrics_summary") or {}
    st.caption(f"Second domain: anomaly detector trained on normal parts only · held-out test image ROC-AUC "
               f"{tm.get('image_roc_auc', 0):.3f} · F1 {tm.get('f1', 0):.3f} · score → p(defect) slope "
               f"{pipe.calibration['slope']:.1f} (shift-aware)")
elif pipe.run_id:
    tm = pipe.info.get("test_metrics_summary") or {}
    cal_txt = (f" · temperature T={pipe.temperature:.2f}" + ("" if pipe.ood_detector else " · legacy OOD check (re-calibrate run)"))
    st.caption("Held-out real test performance: " +
               " · ".join(f"{k}={v:.4f}" for k, v in tm.items() if isinstance(v, (int, float))) + cal_txt)


def find_unsure(img: np.ndarray) -> tuple[str, int] | None:
    """First candidate corruption that makes the pipeline route this image to tier 2."""
    for kind, sev in UNSURE_CANDIDATES:
        routing, reasons = pipe.route(pipe.vision(corrupt(img, kind, sev, seed=0), "probe"))
        if reasons:
            return kind, sev
    return None


to_rgb = lambda x: cv2.cvtColor(x if x.ndim == 3 else cv2.cvtColor(x, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)  # noqa: E731

# ------------------------------------------------------------------ input
st.session_state.setdefault("corr_kind", "none")
st.session_state.setdefault("corr_sev", 3)
if st.session_state.get("prev_run", run_id) != run_id:   # a new model starts from an undegraded image
    st.session_state.corr_kind = "none"
st.session_state.prev_run = run_id
c_in, c_prev = st.columns([1, 1])
with c_in:
    src = st.radio("Image source", ["Upload", "Example image" if PUBLIC_DEMO else "Random real test image"],
                   horizontal=True)
    img, part_id, gt, path = None, None, None, None
    if src == "Upload":
        up = st.file_uploader("Part image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
                              help=f"Up to {MAX_UPLOAD_MB} MB.")
        if up:
            if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
            else:
                img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_UNCHANGED)
                if img is None or img.ndim not in (2, 3) or img.size == 0:
                    st.error("This file could not be decoded as an image.")
                    img = None
                elif img.shape[0] * img.shape[1] > MAX_UPLOAD_PIXELS:
                    st.error(f"Image is too large ({img.shape[1]}×{img.shape[0]} px).")
                    img = None
                else:
                    part_id = Path(up.name).stem
    elif src == "Example image":
        samples = pd.read_csv(S.data_root / "samples" / "samples.csv")
        pick = st.selectbox("Example", range(len(samples)),
                            format_func=lambda i: f"{samples.file[i]} · ground truth: {samples.label[i]}")
        row = samples.iloc[pick]
        img, part_id, gt = read_image(S.data_root / "samples" / row["file"]), row["sample_id"], row["label"]
        st.caption(f"Ground truth (hidden from the model): **{gt}** · {row['attribution']}")
    else:
        vdir = current_version(S.datasets_dir, pipe.info["dataset_id"])
        if vdir is None:
            st.info("No dataset imported yet — see the Dataset Management page.")
        else:
            test = load_manifest(vdir).query("split == 'test'")
            if st.button("🎲 Draw a test image") or "sample_idx" not in st.session_state:
                st.session_state.sample_idx = int(np.random.randint(len(test)))
                st.session_state.corr_kind = "none"
            row = test.iloc[st.session_state.sample_idx % len(test)]
            img, part_id, gt, path = read_image(Path(row["abs_path"])), row["sample_id"], row["label"], row["abs_path"]
            st.caption(f"Ground truth (hidden from the model): **{gt}** · {row['sample_id']}")
    if st.button("🤔 I'm not sure — degrade this image", disabled=img is None or pipe.provenance == "demo_deterministic",
                 help="Applies the mildest blur, noise or crop that pushes THIS image out of the model's comfort "
                      "zone, then inspects it, so you can watch the part go to tier 2 instead of being decided by rules."):
        with st.spinner("Looking for a degradation that makes the model unsure…"):
            found = find_unsure(img)
        if found:
            st.session_state.corr_kind, st.session_state.corr_sev = found
            st.session_state.auto_run = True
        else:
            st.toast("None of the candidate degradations sent this image to tier 2 — the model stays confident.")
    with st.expander("🧪 Stress test: degrade the image before inspection", expanded=st.session_state.corr_kind != "none"):
        kind = st.selectbox("Corruption", ["none"] + CORRUPTIONS, format_func=lambda k: k.replace("_", " "), key="corr_kind")
        sev = st.slider("Severity", 1, 5, key="corr_sev", disabled=kind == "none")
        if kind.startswith("crop"):
            st.caption("Crops can move the defect out of frame — the ground-truth label may no longer hold.")
    meta = {}
    if img is not None and kind != "none":
        img = corrupt(img, kind, sev, seed=0)
        meta = {"corruption": kind, "severity": sev}
    go = st.button("Run inspection", type="primary", disabled=img is None) or st.session_state.pop("auto_run", False)
with c_prev:
    if img is not None:
        st.image(to_rgb(img), caption=part_id + (f" · {kind.replace('_', ' ')} severity {sev}" if meta else ""),
                 width="stretch")

if go and img is not None:
    with st.spinner("Inspecting…"):
        # the public demo keeps no copy of visitors' uploads or reports
        rec = pipe.inspect_image(img, part_id, path, use_llm=use_llm, meta=meta, save=not PUBLIC_DEMO)
    st.session_state.last = rec
    st.session_state.last_img = img
    if PUBLIC_DEMO and use_llm and pipe.llm is not None and rec["report"]["tier"] != "tier1_rules":
        st.session_state.llm_calls = st.session_state.get("llm_calls", 0) + 1   # failed calls count too
        st.rerun()   # refresh the remaining-calls counter in the sidebar

# ------------------------------------------------------------------ result: original | heatmap | decision
rec = st.session_state.get("last")
if rec:
    r, v, chk, rt = rec["report"], rec["vision"], rec["checks"], rec.get("routing", {})
    im = st.session_state.last_img
    cams = rec.get("cams")
    is_anomaly = bool(cams) and "anomaly map" in cams
    st.divider()
    st.subheader("Result")
    if rec.get("meta"):
        st.caption(f"Inspected with {rec['meta']['corruption'].replace('_', ' ')} at severity {rec['meta']['severity']}.")
    over, reg = None, None
    if cams:
        with st.expander("Heatmap display options"):
            c1, c2, c3, c4 = st.columns(4)
            modes = list(cams)
            mode = c1.selectbox("Map", modes, index=modes.index(SALIENCY_DEFAULT) if SALIENCY_DEFAULT in modes else 0,
                                help="Anomaly runs: PatchCore patch scores. Classifier runs — layer4: coarse 7×7, "
                                     "most class-specific · layer3: finer 14×14 · fused: geometric mean")
            alpha = c2.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)
            thresh = c3.slider("Region threshold", 0.2, 0.9, round(ANOMALY_REGION_THRESH if is_anomaly else 0.5, 2), 0.05,
                               key=f"thresh_{is_anomaly}")
            cmap_name = c4.selectbox("Colormap", ["JET", "INFERNO", "TURBO", "VIRIDIS"])
            t1, t2 = st.columns(2)
            show_contours = t1.toggle("Contours", value=True)
            show_boxes = t2.toggle("Numbered region boxes", value=True)
        reg = cam_regions(cams[mode], im.shape[:2], thresh=thresh)
        heat = reg.pop("heatmap")
        over = render_saliency(im, heat, reg["regions"], alpha=alpha, cmap=getattr(cv2, f"COLORMAP_{cmap_name}"),
                               contours=show_contours, boxes=show_boxes, thresh=thresh)
    g1, g2, g3 = st.columns([1, 1.08, 1.15])
    g1.image(to_rgb(im), caption="original (as inspected)", width="stretch")
    if over is not None:
        what = "anomaly map" if is_anomaly else f"Grad-CAM saliency ({mode})"
        g2.image(over, caption=f"{what} — white contours = level ≥ {thresh}", width="stretch")
    else:
        g2.info("No heatmap in demo mode.")
    with g3:
        st.markdown(f"### :{DECISION_COLORS[r['decision']]}[{r['decision']}]")
        m1, m2 = st.columns(2)
        m1.metric("p(defect)", f"{r['p_defect']:.3f}",
                  help=f"Calibrated with T={v.get('temperature', 1)}; uncalibrated {v.get('p_defect_uncalibrated', r['p_defect']):.4f}")
        m2.metric("Tier", TIER_LABELS.get(r["tier"], r["tier"]), help=r["tier"])
        m3, m4 = st.columns(2)
        m3.metric("Severity", r["severity"])
        m4.metric("Latency", f"{rec['total_latency_s'] * 1000:.0f} ms")
        badges = [("in uncertainty band", rt.get("in_band")), ("out-of-distribution", rt.get("ood")),
                  ("model provenance", rt.get("provenance"))]
        fired = [b for b, on in badges if on]
        st.markdown("**Routing:** " + (" · ".join(f":red[{b}]" for b in fired) + " → tier 2" if fired else
                                       ":green[confident and in-distribution] → tier 1"))
        tolerated = v.get("ood", {}).get("stat_tolerated")
        if tolerated and not fired:
            st.caption("Image statistics are off (" + ", ".join(f"{k.replace('_', ' ')} z={z}" for k, z in tolerated.items())
                       + ") but within the tolerance for a confident prediction, so the part stays in tier 1.")
        st.markdown(f"**Recommended action:** {r['recommended_action']}")
        st.markdown(f"**Rationale:** {r['rationale']}")
        if r["escalation_reason"]:
            st.warning(f"Escalation reason: {r['escalation_reason']}")
        if rec.get("llm_error"):
            st.info(f"LLM not used: {rec['llm_error']}")
    if not is_anomaly and cams:
        st.caption("Saliency is approximate evidence for the *defect* class, not a measured defect boundary. "
                   "For confidently normal parts it is not a defect location.")
    elif is_anomaly:
        st.caption("The anomaly map shows patch distances to the nearest normal training patch, on a fixed scale "
                   "(1.0 = 1.5 × the decision threshold). It is approximate, not a measured defect boundary.")
    if reg and reg["regions"]:
        st.dataframe(pd.DataFrame([{"#": k, "region": x["region"], "area": f"{100 * x['area_fraction']:.2f}%",
                                    "peak activation": x["peak"], "share of activation": f"{100 * x['mass_share']:.0f}%",
                                    "box (x0,y0,x1,y1, normalised)": x["bbox_xyxy_norm"]}
                                   for k, x in enumerate(reg["regions"], 1)]), hide_index=True, width="stretch")
    k1, k2 = st.columns(2)
    k1.metric("Groundedness checks", f"{chk['groundedness'] * 100:.0f}%")
    k2.metric("Report completeness", f"{chk['completeness'] * 100:.0f}%")
    with st.expander("Structured report (JSON)"):
        st.json(r)
    with st.expander("Grounding / completeness checks"):
        st.json(chk)
    with st.expander("Vision tool output (incl. OOD detector)"):
        st.json(v)
