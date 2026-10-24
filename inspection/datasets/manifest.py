"""Normalise adapter output into a versioned dataset: copied files, manifest, leakage-resistant splits, stats.

Layout:  <data_root>/datasets/<dataset_id>/<version>/{images,masks}/<label>/...  manifest.csv  summary.json
A dataset version is never overwritten; <dataset_id>/CURRENT names the active version."""
from __future__ import annotations

import json
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .download import sha256_file
from .registry import TASK_ANOMALY, DatasetSource
from .leakage import audit, groups_from_similarity, rotation_similarity
from .validate import image_info

NEAR_DUP_CORR = 0.99  # rotation-aware thumbnail correlation at/above which two images are treated as one part


@dataclass
class Candidate:
    src: Path
    label: str                         # class name ("ok", "defective", "good", "<anomaly type>", ...)
    official_subset: str = ""          # e.g. train/test as published, when the source defines one
    mask_src: Path | None = None
    n_boxes: int | None = None         # verified bounding boxes parsed from annotation files
    extra: dict = field(default_factory=dict)


def stratified_group_split(labels: list[str], groups: list[int], fractions=(0.7, 0.15, 0.15),
                           seed: int = 42) -> list[str]:
    """Assign whole near-duplicate groups to train/val/test so that no near-duplicate pair straddles
    two splits, while keeping class ratios close to the requested fractions."""
    names = ("train", "val", "test")
    by_group: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    group_label = {g: Counter(labels[i] for i in idx).most_common(1)[0][0] for g, idx in by_group.items()}
    rng = random.Random(seed)
    out = [""] * len(labels)
    for lab in sorted(set(group_label.values())):
        gs = sorted(g for g, l in group_label.items() if l == lab)
        rng.shuffle(gs)
        total = sum(len(by_group[g]) for g in gs)
        targets = [f * total for f in fractions]
        filled = [0, 0, 0]
        for g in gs:
            # put the group in the split that is furthest below its target
            k = max(range(3), key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1e-9))
            for i in by_group[g]:
                out[i] = names[k]
            filled[k] += len(by_group[g])
    return out


def build_version(src: DatasetSource, task: str, candidates: list[Candidate], datasets_dir: Path,
                  progress=None, check_cancel=None, seed: int = 42, notes: str = "") -> dict:
    version = time.strftime("v%Y%m%d-%H%M%S")
    root = datasets_dir / src.id / version
    root.mkdir(parents=True, exist_ok=False)
    rows, errors, smalls = [], [], []
    seen_sha: dict[str, int] = {}
    exact_dupes = 0

    for k, c in enumerate(candidates):
        if check_cancel:
            check_cancel()
        if progress and k % 50 == 0:
            progress(k / max(len(candidates), 1), f"validating {k}/{len(candidates)}")
        info = image_info(c.src)
        if not info["readable"]:
            errors.append({"file": str(c.src), "error": "unreadable or corrupt image"})
            continue
        sha = sha256_file(c.src)
        if sha in seen_sha:
            prev = rows[seen_sha[sha]]
            exact_dupes += 1
            if prev["label"] != c.label:
                errors.append({"file": str(c.src), "error": f"exact duplicate of {prev['source_file']} with a "
                                                             f"different label ({prev['label']} vs {c.label})"})
            continue
        mask_rel = ""
        if c.mask_src is not None:
            if image_info(c.mask_src).get("readable"):
                mrel = Path("masks") / c.label / f"{len(rows):06d}_mask{c.mask_src.suffix.lower()}"
                (root / mrel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(c.mask_src, root / mrel)
                mask_rel = mrel.as_posix()
            else:
                errors.append({"file": str(c.mask_src), "error": "mask listed but unreadable"})
        rel = Path("images") / c.label / f"{len(rows):06d}{c.src.suffix.lower()}"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c.src, root / rel)
        seen_sha[sha] = len(rows)
        smalls.append(info["small"])
        rows.append({
            "sample_id": f"{src.id}:{len(rows):06d}", "path": rel.as_posix(), "label": c.label,
            "official_subset": c.official_subset, "mask_path": mask_rel,
            "n_boxes": c.n_boxes if c.n_boxes is not None else "",
            "sample_type": "real", "source_dataset": src.id, "source_url": src.official_url,
            "source_file": str(c.src), "license": src.license, "attribution": src.attribution,
            "sha256": sha, "width": info["width"], "height": info["height"], "channels": info["channels"],
            "effective_grayscale": info["effective_grayscale"], "dhash": str(info["dhash"]),
            **{f"meta_{k2}": v for k2, v in c.extra.items()},
        })

    if not rows:
        raise RuntimeError(f"No readable images were found for {src.id}; nothing was imported.")
    df = pd.DataFrame(rows)
    if progress:
        progress(0.85, "rotation-aware near-duplicate grouping")
    # Similarity over rotations catches the same part photographed at a different angle.
    S = rotation_similarity(smalls)
    df["group_id"] = groups_from_similarity(S, NEAR_DUP_CORR)

    if task == TASK_ANOMALY:
        # Keep the official normal-only training set; split the official test set into val/test.
        df["split"] = ""
        tr = df["official_subset"] == "train"
        df.loc[tr, "split"] = "train"
        rest = df[~tr]
        sp = stratified_group_split(rest["label"].tolist(), rest["group_id"].tolist(), (0.0, 1 / 3, 2 / 3), seed)
        df.loc[~tr, "split"] = sp
    else:
        df["split"] = stratified_group_split(df["label"].tolist(), df["group_id"].tolist(), seed=seed)

    # Leakage check: a near-duplicate group must never appear in more than one split.
    leak = df.groupby("group_id")["split"].nunique()
    leaking_groups = int((leak > 1).sum())
    leakage_audit = audit(S, df["split"].tolist())
    leakage_audit["near_duplicate_threshold"] = NEAR_DUP_CORR

    df.to_csv(root / "manifest.csv", index=False)
    summary = summarise(df, root)
    summary.update({
        "dataset_id": src.id, "version": version, "task": task, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": {"name": src.name, "url": src.official_url, "license": src.license, "attribution": src.attribution},
        "validation": {"candidates": len(candidates), "imported": len(df), "errors": errors,
                       "exact_duplicates_removed": exact_dupes,
                       "near_duplicate_groups": int(df["group_id"].nunique()),
                       "largest_near_duplicate_group": int(df["group_id"].value_counts().max()),
                       "groups_spanning_splits": leaking_groups, "leakage_audit": leakage_audit},
        "split_seed": seed, "notes": notes,
    })
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (datasets_dir / src.id / "CURRENT").write_text(version, encoding="utf-8")
    return summary


def summarise(df: pd.DataFrame, root: Path) -> dict:
    counts = df["label"].value_counts().to_dict()
    by_split = {s: g["label"].value_counts().to_dict() for s, g in df.groupby("split")}
    mx, mn = max(counts.values()), min(counts.values())
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    dims = df.groupby(["width", "height", "channels"]).size().reset_index(name="n").to_dict("records")
    return {"n_real": int(len(df)), "classes": counts, "n_classes": len(counts), "split_counts": by_split,
            "imbalance_ratio": round(mx / max(mn, 1), 3), "n_masks": int((df["mask_path"] != "").sum()),
            "n_boxes": int(pd.to_numeric(df["n_boxes"], errors="coerce").fillna(0).sum()),
            "image_dims": dims, "storage_bytes": size}


def current_version(datasets_dir: Path, dataset_id: str) -> Path | None:
    p = datasets_dir / dataset_id / "CURRENT"
    if not p.exists():
        return None
    v = datasets_dir / dataset_id / p.read_text(encoding="utf-8").strip()
    return v if v.exists() else None


def list_versions(datasets_dir: Path, dataset_id: str) -> list[dict]:
    out = []
    for v in sorted((datasets_dir / dataset_id).glob("v*"), reverse=True):
        s = v / "summary.json"
        if s.exists():
            out.append(json.loads(s.read_text(encoding="utf-8")))
    return out


def load_manifest(version_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(version_dir / "manifest.csv", keep_default_na=False)
    df["abs_path"] = [str(version_dir / p) for p in df["path"]]
    return df
