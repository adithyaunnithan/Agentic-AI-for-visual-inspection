---
title: Agentic Visual Inspection
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Agentic Visual Inspection — public demo

Two-tier inspection of cast pump impellers: a ResNet50 classifier (with Grad-CAM, calibration and an
out-of-distribution check) decides confident parts by rules; uncertain or unusual parts go to an LLM agent
(Claude) or are held for a human. Each inspection produces a structured, automatically checked report.

**What this demo does and does not do**
- Inspect only: upload an image (up to 10 MB) or pick one of 10 example images. Dataset downloads, imports and
  training are disabled (`PUBLIC_DEMO=1`).
- Uploads and reports are processed in memory and not stored.
- The LLM agent runs only if the Space has an `ANTHROPIC_API_KEY` secret, at most 5 calls per browser session.
  Without it, tier-2 parts are held by the rule agent.
- The model was trained on one camera, one part and one lighting setup. It is not validated for other parts.

**Data and attribution.** The model was trained on the *Real-life industrial dataset of casting product*
(R. Dabhi, Kaggle, PILOT TECHNOCAST; CC BY-NC-ND 4.0). The dataset is not included; the 10 example images are
unmodified copies of held-out test images, listed in `data/samples/ATTRIBUTION.txt`. Non-commercial use only.

## Rebuilding this folder

From the project repository: `python deploy/build_deploy.py [--run <model_run_id>]`. It copies the inspection
code, one model run (`best.pt`, `run_info.json`, `calibration.json`) and the example images into this folder.
