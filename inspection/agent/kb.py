from __future__ import annotations

import json
from pathlib import Path

KB_PATH = Path(__file__).with_name("knowledge_base.json")


class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH):
        self.entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
        self.by_id = {e["id"]: e for e in self.entries}

    def for_dataset(self, dataset_id: str) -> list[dict]:
        return [e for e in self.entries if "*" in e["applies_to"] or dataset_id in e["applies_to"]]

    def search(self, query: str, dataset_id: str) -> list[dict]:
        q = set(query.lower().replace("_", " ").split())
        scored = []
        for e in self.for_dataset(dataset_id):
            words = set((e["topic"] + " " + e["text"]).lower().replace("_", " ").split())
            scored.append((len(q & words), e))
        return [e for s, e in sorted(scored, key=lambda t: -t[0]) if s > 0][:4] or self.for_dataset(dataset_id)[:4]

    def severity(self, area_fraction: float | None) -> str:
        r = self.by_id["KB-POL-001"]["rule"]
        if area_fraction is None or area_fraction <= 0:
            return "medium"
        return "high" if area_fraction >= r["high"] else "medium" if area_fraction >= r["medium"] else "low"

    ACTIONS = {
        "high": "Remove from line and quarantine; scrap/rework disposition by quality engineer; log for root-cause analysis",
        "medium": "Remove from line for rework assessment; re-inspect the indicated region under raking light",
        "low": "Flag for rework assessment; re-inspect the indicated region to confirm",
    }
