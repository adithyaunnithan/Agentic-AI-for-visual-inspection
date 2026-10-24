from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inspection.config import load_settings  # noqa: E402
from inspection.jobs import JobManager  # noqa: E402

PROV_COLORS = {"trained_real": "green", "trained_real_plus_synthetic": "blue", "pretrained_anomaly": "blue",
               "synthetic_trained_experimental": "orange", "demo_deterministic": "red"}
DECISION_COLORS = {"PASS": "green", "FLAG_FOR_REWORK": "orange", "ESCALATE": "red"}


@st.cache_resource
def settings():
    return load_settings()


@st.cache_resource
def jobs() -> JobManager:
    return JobManager(settings().jobs_dir)


def fmt_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def jobs_panel(kinds: tuple[str, ...] | None = None, limit: int = 8) -> None:
    @st.fragment(run_every=2)
    def _panel():
        js = [j for j in jobs().list(40) if kinds is None or j["kind"] in kinds][:limit]
        if not js:
            st.caption("No operations yet.")
            return
        for j in js:
            icon = {"running": "⏳", "succeeded": "✅", "failed": "❌", "cancelled": "⏹️", "interrupted": "⚠️"}.get(j["status"], "•")
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"{icon} **{j['title']}** — `{j['status']}` · {j.get('message', '')}")
                if j["status"] == "running":
                    c1.progress(float(j.get("progress", 0)))
                    if c2.button("Cancel", key=f"cancel-{j['id']}"):
                        jobs().cancel(j["id"])
                ex = j.get("extra") or {}
                if ex:
                    c1.caption(" · ".join(f"{k}={v}" for k, v in ex.items()))
                if j.get("error"):
                    c1.error(j["error"])
                with c1.expander("log", expanded=False):
                    st.code("\n".join(j.get("log", [])[-40:]) or "(empty)")
                    if j.get("result") is not None:
                        st.json(j["result"])
                started = j.get("started_at")
                if started:
                    c2.caption(time.strftime("%H:%M:%S", time.localtime(started)))
    _panel()
