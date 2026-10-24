"""Public demo entry point (Hugging Face Docker Space): only the Inspect page.

Run from the assembled deploy/ folder (see build_deploy.py):  streamlit run public_app.py"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
# Both must be set before inspection.config is imported (it reads them at import time).
os.environ.setdefault("PUBLIC_DEMO", "1")
os.environ.setdefault("AGENTIC_INSPECTION_DATA", str(ROOT / "data"))
for p in (ROOT, ROOT / "app"):   # the Inspect page imports `common` from app/ and `inspection` from the root
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# st.navigation replaces the app/pages/ auto-discovery: Dataset Management, Training etc. are not reachable.
st.navigation([st.Page("app/Inspect.py", title="Inspect", icon="🔎", default=True)], position="hidden").run()
