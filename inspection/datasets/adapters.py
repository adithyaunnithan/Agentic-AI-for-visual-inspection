"""Source adapters: acquire (download or locate manual drop) -> verify -> safe extract -> candidates."""
from __future__ import annotations

import json
import re
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from ..config import Settings, require_private
from ..jobs import JobContext
from . import registry
from .archive import extract, verify_zip
from .download import UA, download, ensure_disk_space, sha256_file
from .manifest import Candidate, build_version, current_version
from .validate import IMAGE_EXTS

ACQ_FILE = "acquisition.json"


def _write_acq(settings: Settings, ds_id: str, data: dict) -> None:
    d = settings.raw_dir / ds_id
    d.mkdir(parents=True, exist_ok=True)
    (d / ACQ_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")


def acquisition_record(settings: Settings, ds_id: str) -> dict | None:
    p = settings.raw_dir / ds_id / ACQ_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _images_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


# --------------------------------------------------------------------------------------------- casting
class CastingKaggleAdapter:
    source = registry.BY_ID["casting_kaggle"]
    task = registry.TASK_BINARY

    def acquire(self, settings: Settings, ctx: JobContext) -> dict:
        ref = self.source.download["ref"]
        ctx.progress(0.01, f"querying Kaggle metadata for {ref}")
        meta = requests.get(f"https://www.kaggle.com/api/v1/datasets/view/{ref}", headers=UA, timeout=30).json()
        if meta.get("licenseName") and "NoDerivatives" not in meta["licenseName"]:
            ctx.progress(message=f"note: licence on Kaggle now reads '{meta['licenseName']}' - review it")
        est = int(meta.get("totalBytes") or self.source.approx_bytes)
        ensure_disk_space(settings.raw_dir, est)
        dest = settings.raw_dir / self.source.id / "archive.zip"
        url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"

        def prog(done, total):
            ctx.progress(0.05 + 0.75 * (done / total if total else 0),
                         None, downloaded_mb=round(done / 1e6, 1), total_mb=round((total or 0) / 1e6, 1))
        ctx.progress(0.05, f"downloading {url} (resumable)")
        download(url, dest, progress=prog, check_cancel=ctx.check_cancel)
        ctx.progress(0.82, "verifying zip CRCs")
        z = verify_zip(dest)
        sha = sha256_file(dest)
        rec = {"dataset_id": self.source.id, "url": url, "official_page": self.source.official_url, "file": str(dest),
               "bytes": dest.stat().st_size, "sha256": sha, "zip_members": z["members"],
               "uncompressed_bytes": z["uncompressed_bytes"], "kaggle_version": meta.get("currentVersionNumber"),
               "license": meta.get("licenseName", self.source.license), "verified": True,
               "verified_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        out = settings.extracted_dir / self.source.id
        if out.exists():
            shutil.rmtree(out)  # extraction dir is a derived cache; the verified archive is the source of truth
        ctx.progress(0.85, "extracting safely")
        rec["extraction"] = extract(dest, out, check_cancel=ctx.check_cancel)
        _write_acq(settings, self.source.id, rec)
        ctx.progress(0.99, f"verified {z['members']} archive members, sha256={sha[:12]}...")
        return rec

    def candidates(self, settings: Settings, subset: str = "512") -> tuple[list[Candidate], str]:
        root = settings.extracted_dir / self.source.id
        cands = []
        label_map = {"def_front": "defective", "ok_front": "ok"}
        for p in _images_under(root):
            parts = [x.lower() for x in p.parts]
            cls = p.parent.name.lower()
            if cls not in label_map:
                continue
            if subset == "512" and any("casting_512x512" in x for x in parts):
                cands.append(Candidate(p, label_map[cls], extra={"origin_subset": "casting_512x512"}))
            elif subset == "300_aug" and any(x == "casting_data" for x in parts):
                off = "train" if "train" in parts else "test" if "test" in parts else ""
                cands.append(Candidate(p, label_map[cls], official_subset=off, extra={"origin_subset": "casting_data_300"}))
        notes = ("Using the 1,300 UNAUGMENTED 512x512 images (519 ok / 781 defective per the dataset author). "
                 "The 300x300 'casting_data' folder was augmented by the author before publication, so its "
                 "train/test folders can contain augmented copies of the same parts; it is not used as a benchmark."
                 if subset == "512" else
                 "WARNING: 300x300 subset is pre-augmented by the author; near-duplicate grouping is applied but "
                 "residual leakage between augmented copies is possible.")
        return cands, notes


# --------------------------------------------------------------------------------------------- MVTec
def _mvtec_candidates(root: Path) -> list[Candidate]:
    cands = []
    for p in _images_under(root):
        parts = p.parts
        if "ground_truth" in parts:
            continue
        try:
            i = max(k for k, x in enumerate(parts) if x in ("train", "test"))
        except ValueError:
            continue
        subset, dtype = parts[i], parts[i + 1] if i + 1 < len(parts) - 1 else ""
        category = parts[i - 1] if i > 0 else ""
        mask = None
        if subset == "test" and dtype != "good":
            m = Path(*parts[:i]) / "ground_truth" / dtype / f"{p.stem}_mask.png"
            mask = m if m.exists() else None
        label = "good" if dtype == "good" else "anomaly"
        cands.append(Candidate(p, label, official_subset=subset, mask_src=mask,
                               extra={"category": category, "defect_type": dtype}))
    return cands


class MVTecManualAdapter:
    source = registry.BY_ID["mvtec_ad"]
    task = registry.TASK_ANOMALY

    def acquire(self, settings: Settings, ctx: JobContext) -> dict:
        inbox = settings.incoming_dir / self.source.id
        inbox.mkdir(parents=True, exist_ok=True)
        archives = sorted([*inbox.glob("*.tar.xz"), *inbox.glob("*.tar"), *inbox.glob("*.zip")])
        if not archives:
            raise FileNotFoundError(
                f"MVTec AD requires accepting the licence on the official page ({self.source.official_url}). "
                f"Download the full archive or a category archive and place it in {inbox}, then retry Import.")
        out = settings.extracted_dir / self.source.id
        recs = []
        for k, a in enumerate(archives):
            ctx.progress(k / len(archives), f"hashing + extracting {a.name}")
            info = extract(a, out, check_cancel=ctx.check_cancel)
            recs.append({"file": str(a), "bytes": a.stat().st_size, "sha256": sha256_file(a), **info})
        rec = {"dataset_id": self.source.id, "archives": recs, "verified": True, "official_page": self.source.official_url,
               "verified_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "note": "MVTec does not publish checksums; archive integrity verified by successful full decompression."}
        _write_acq(settings, self.source.id, rec)
        return rec

    def candidates(self, settings: Settings, category: str | None = None) -> tuple[list[Candidate], str]:
        cands = _mvtec_candidates(settings.extracted_dir / self.source.id)
        if category:
            cands = [c for c in cands if c.extra.get("category") == category]
        return cands, "Official MVTec split: train = good only; test = good + anomalies (masks)."


class MVTecHFMirrorAdapter:
    source = registry.BY_ID["mvtec_ad_hf_mirror"]
    task = registry.TASK_ANOMALY

    def _list_files(self, repo: str, category: str) -> list[dict]:
        url = f"https://huggingface.co/api/datasets/{repo}/tree/main/{category}"
        files, nxt = [], url + "?recursive=true"
        while nxt:
            r = requests.get(nxt, headers=UA, timeout=60)
            r.raise_for_status()
            files += [e for e in r.json() if e.get("type") == "file"]
            nxt = r.links.get("next", {}).get("url")
        return files

    def acquire(self, settings: Settings, ctx: JobContext, category: str | None = None) -> dict:
        repo = self.source.download["repo"]
        category = category or self.source.download["category"]
        ctx.progress(0.01, f"listing {repo}/{category}")
        files = self._list_files(repo, category)
        if not files:
            raise FileNotFoundError(f"No files for category '{category}' in {repo}")
        total = sum(f.get("size", 0) for f in files)
        ensure_disk_space(settings.extracted_dir, total, safety_factor=1.5)
        out = settings.extracted_dir / self.source.id
        verified = 0
        for k, f in enumerate(files):
            ctx.check_cancel()
            rel = f["path"]
            dest = out / rel
            if ".." in Path(rel).parts:
                raise RuntimeError(f"Unsafe path from mirror listing: {rel}")
            sha = (f.get("lfs") or {}).get("oid")
            download(f"https://huggingface.co/datasets/{repo}/resolve/main/{rel}", dest,
                     expected_size=f.get("size"), expected_sha256=sha, check_cancel=ctx.check_cancel)
            verified += 1 if sha else 0
            if k % 20 == 0:
                ctx.progress(0.02 + 0.95 * k / len(files), f"{k}/{len(files)} files")
        rec = {"dataset_id": self.source.id, "repo": repo, "category": category, "files": len(files), "bytes": total,
               "sha256_verified_files": verified, "verified": True, "official_page": registry.BY_ID["mvtec_ad"].official_url,
               "mirror_page": self.source.official_url, "verified_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        _write_acq(settings, self.source.id, rec)
        return rec

    def candidates(self, settings: Settings, category: str | None = None) -> tuple[list[Candidate], str]:
        cands = _mvtec_candidates(settings.extracted_dir / self.source.id)
        if category:
            cands = [c for c in cands if c.extra.get("category") == category]
        return cands, "MVTec AD via licence-permitted mirror; official train/test split preserved."


# --------------------------------------------------------------------------------------------- NEU
NEU_PREFIX = {"cr": "crazing", "in": "inclusion", "pa": "patches", "ps": "pitted_surface",
              "rs": "rolled-in_scale", "sc": "scratches"}


class NEUManualAdapter:
    source = registry.BY_ID["neu_surface"]
    task = registry.TASK_MULTICLASS

    def acquire(self, settings: Settings, ctx: JobContext) -> dict:
        inbox = settings.incoming_dir / self.source.id
        inbox.mkdir(parents=True, exist_ok=True)
        items = list(inbox.iterdir())
        if not items:
            raise FileNotFoundError(
                f"Download NEU-CLS or NEU-DET from the official page ({self.source.official_url}) and place the "
                f"zip or extracted folder in {inbox}, then retry Import.")
        out = settings.extracted_dir / self.source.id
        recs = []
        for it in items:
            if it.is_file() and it.suffix.lower() in (".zip", ".tar", ".xz", ".gz"):
                recs.append({"file": str(it), "sha256": sha256_file(it), **extract(it, out, check_cancel=ctx.check_cancel)})
            elif it.is_dir():
                shutil.copytree(it, out / it.name, dirs_exist_ok=True)
                recs.append({"folder": str(it)})
        rec = {"dataset_id": self.source.id, "items": recs, "verified": True,
               "verified_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        _write_acq(settings, self.source.id, rec)
        return rec

    def candidates(self, settings: Settings) -> tuple[list[Candidate], str]:
        root = settings.extracted_dir / self.source.id
        boxes: dict[str, int] = {}
        for x in root.rglob("*.xml"):
            try:
                boxes[x.stem.lower()] = len(ET.parse(x).getroot().findall("object"))
            except ET.ParseError:
                pass
        cands = []
        for p in _images_under(root):
            folder = p.parent.name.lower().replace("-", "_").replace(" ", "_")
            label = next((v for v in NEU_PREFIX.values() if v.replace("-", "_") == folder), None)
            if label is None:
                m = re.match(r"([a-z]{2})[_-]", p.stem.lower())
                label = NEU_PREFIX.get(m.group(1)) if m else None
            if label:
                cands.append(Candidate(p, label, n_boxes=boxes.get(p.stem.lower())))
        return cands, "Class label from folder name or NEU filename prefix (Cr/In/Pa/PS/RS/Sc); boxes from VOC XML."


ADAPTERS = {
    "casting_kaggle": CastingKaggleAdapter(),
    "mvtec_ad": MVTecManualAdapter(),
    "mvtec_ad_hf_mirror": MVTecHFMirrorAdapter(),
    "neu_surface": NEUManualAdapter(),
}


def acquire_and_import(settings: Settings, ds_id: str, ctx: JobContext, **opts) -> dict:
    require_private("Dataset download")
    ad = ADAPTERS[ds_id]
    acq = acquisition_record(settings, ds_id)
    force = opts.pop("force_download", False)
    if force or not (acq and acq.get("verified")):
        acq_opts = {k: v for k, v in opts.items() if k == "category" and ds_id == "mvtec_ad_hf_mirror"}
        acq = ad.acquire(settings, ctx, **acq_opts)
    else:
        ctx.progress(0.5, "verified acquisition already present - skipping download")
    return import_only(settings, ds_id, ctx, **opts)


def import_only(settings: Settings, ds_id: str, ctx: JobContext, **opts) -> dict:
    require_private("Dataset import")
    ad = ADAPTERS[ds_id]
    cands, notes = ad.candidates(settings, **opts)
    if not cands:
        raise RuntimeError(f"No candidate images found for {ds_id} under {settings.extracted_dir / ds_id}")
    ctx.progress(0.6, f"normalising {len(cands)} images")
    summary = build_version(ad.source, ad.task, cands, settings.datasets_dir,
                            progress=lambda f, m=None: ctx.progress(0.6 + 0.38 * f, m),
                            check_cancel=ctx.check_cancel, seed=settings.train.seed, notes=notes)
    return {"version": summary["version"], "n_real": summary["n_real"], "classes": summary["classes"],
            "errors": len(summary["validation"]["errors"])}


def status(settings: Settings, ds_id: str) -> dict:
    acq = acquisition_record(settings, ds_id)
    v = current_version(settings.datasets_dir, ds_id)
    summ = json.loads((v / "summary.json").read_text(encoding="utf-8")) if v else None
    inbox = settings.incoming_dir / ds_id
    part = settings.raw_dir / ds_id / "archive.zip.part"
    return {"acquisition": acq, "current_version": summ,
            "partial_download_bytes": part.stat().st_size if part.exists() else 0,
            "incoming_dir": str(inbox), "incoming_items": [p.name for p in inbox.iterdir()] if inbox.exists() else []}
