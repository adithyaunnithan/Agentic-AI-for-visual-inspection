"""Central configuration. Paths are configurable via environment variables so that
large data (datasets, checkpoints) can live outside synced folders such as OneDrive."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = Path(os.environ.get("AGENTIC_INSPECTION_DATA", Path.home() / "agentic_inspection_data"))
# Hosted public demo (deploy/): inspection only - no dataset downloads, imports or training.
PUBLIC_DEMO = os.environ.get("PUBLIC_DEMO") == "1"


def require_private(action: str) -> None:
    if PUBLIC_DEMO:
        raise RuntimeError(f"{action} is disabled in the public demo (PUBLIC_DEMO=1)")


@dataclass
class AgentConfig:
    model: str = os.environ.get("INSPECTION_LLM_MODEL", "claude-opus-5-5")
    effort: str = "low"               # low effort is enough for a short, grounded report
    # Two-tier policy: the LLM agent is only consulted when p(defect) falls in [low, high].
    escalation_low: float = 0.15
    escalation_high: float = 0.85
    llm_for_all: bool = False         # set True to run the LLM on every part (slow, costly)
    send_image_to_llm: bool = True    # VLM mode for ambiguous cases
    use_knowledge_base: bool = True   # ablation switch


@dataclass
class TrainConfig:
    dataset_id: str = "casting_kaggle"
    architecture: str = "resnet50"
    image_size: int = 224
    epochs: int = 8
    batch_size: int = 16
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    class_weighted_loss: bool = True
    use_clahe: bool = True
    synthetic_runs: list[str] = field(default_factory=list)  # synthetic runs mixed into TRAIN only
    synthetic_only: bool = False      # experimental: train on synthetic defects + real normals
    seed: int = 42
    num_workers: int = 0              # Windows-friendly default


@dataclass
class Settings:
    data_root: Path = DEFAULT_DATA_ROOT
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Derived directories -------------------------------------------------
    @property
    def raw_dir(self) -> Path: return self.data_root / "raw"
    @property
    def extracted_dir(self) -> Path: return self.data_root / "extracted"
    @property
    def incoming_dir(self) -> Path: return self.data_root / "incoming"
    @property
    def datasets_dir(self) -> Path: return self.data_root / "datasets"
    @property
    def synthetic_dir(self) -> Path: return self.data_root / "synthetic"
    @property
    def models_dir(self) -> Path: return self.data_root / "models"
    @property
    def reports_dir(self) -> Path: return self.data_root / "reports"
    @property
    def jobs_dir(self) -> Path: return self.data_root / "jobs"
    @property
    def eval_dir(self) -> Path: return self.data_root / "evaluations"
    @property
    def human_eval_dir(self) -> Path: return self.data_root / "human_eval"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.extracted_dir, self.incoming_dir, self.datasets_dir, self.synthetic_dir,
                  self.models_dir, self.reports_dir, self.jobs_dir, self.eval_dir, self.human_eval_dir):
            d.mkdir(parents=True, exist_ok=True)


_SETTINGS_FILE = PROJECT_ROOT / "settings.json"


def load_settings() -> Settings:
    s = Settings()
    if _SETTINGS_FILE.exists():
        raw = json.loads(_SETTINGS_FILE.read_text(encoding="utf-8"))
        if raw.get("data_root"):
            s.data_root = Path(raw["data_root"])
        for k, v in raw.get("agent", {}).items():
            if hasattr(s.agent, k):
                setattr(s.agent, k, v)
        for k, v in raw.get("train", {}).items():
            if hasattr(s.train, k):
                setattr(s.train, k, v)
    s.ensure_dirs()
    return s


def save_settings(s: Settings) -> None:
    _SETTINGS_FILE.write_text(json.dumps({
        "data_root": str(s.data_root),
        "agent": asdict(s.agent),
        "train": asdict(s.train),
    }, indent=2), encoding="utf-8")
