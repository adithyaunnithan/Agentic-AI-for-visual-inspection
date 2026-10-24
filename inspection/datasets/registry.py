"""Curated catalogue of industrial defect datasets + live discovery against public catalogue APIs.

Every curated entry records the official source, task type, licence and access method. The access
method decides what the app may do automatically:
  auto_official  - documented public API/endpoint of the official host, downloaded automatically
  auto_mirror    - third-party redistribution that the dataset licence explicitly permits
  manual         - requires a form / registration / licence acceptance -> user is guided, then imports
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import requests

# Task taxonomy used throughout the app. Datasets and models declare which of these they support.
TASK_ANOMALY = "image_anomaly_detection"        # normal-only training, anomaly score at test time
TASK_BINARY = "binary_classification"           # normal vs defective, both labelled
TASK_MULTICLASS = "multiclass_defect_classification"
TASK_DETECTION = "object_detection"
TASK_SEGMENTATION = "pixel_segmentation"

PROJECT_TASK = TASK_BINARY  # The case study (casting inspection) is a normal-vs-defective decision.


@dataclass
class DatasetSource:
    id: str
    name: str
    official_url: str
    domain: str
    tasks: list[str]
    classes: list[str]
    approx_size: str
    approx_bytes: int
    annotations: str
    access: str
    access_notes: str
    license: str
    license_url: str
    attribution: str
    redistribution: str
    relevance: float           # 0..1 fit to the casting-inspection case study
    relevance_reason: str
    download: dict = field(default_factory=dict)

    @property
    def access_score(self) -> float:
        return {"auto_official": 1.0, "auto_mirror": 0.8, "manual": 0.4}.get(self.access, 0.0)

    @property
    def rank_score(self) -> float:
        return round(0.7 * self.relevance + 0.3 * self.access_score, 3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rank_score"] = self.rank_score
        return d


CATALOGUE: list[DatasetSource] = [
    DatasetSource(
        id="casting_kaggle",
        name="Real-life industrial dataset of casting product (submersible pump impeller)",
        official_url="https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product",
        domain="Metal casting - pump impeller, top view, grayscale",
        tasks=[TASK_BINARY],
        classes=["ok_front", "def_front"],
        approx_size="~105 MB zip; 1,300 unaugmented 512x512 images + 7,348 pre-augmented 300x300 images",
        approx_bytes=104_708_903,
        annotations="Image-level labels only (defective / ok). No defect subtype, box or mask labels.",
        access="auto_official",
        access_notes="Public Kaggle dataset; downloaded through Kaggle's public dataset download API endpoint "
                     "(api/v1/datasets/download/...). Supports HTTP range requests, so downloads resume.",
        license="CC BY-NC-ND 4.0",
        license_url="https://creativecommons.org/licenses/by-nc-nd/4.0/",
        attribution="Ravirajsinh Dabhi, 'Real-life industrial dataset of casting product', Kaggle (PILOT TECHNOCAST).",
        redistribution="Non-commercial use with attribution; no redistribution of modified versions. "
                       "Synthetic derivatives are kept local and never redistributed.",
        relevance=0.95,
        relevance_reason="Exact domain of the project (casting inspection); binary task matches pass/flag decision.",
        download={"kind": "kaggle", "ref": "ravirajsinh45/real-life-industrial-dataset-of-casting-product"},
    ),
    DatasetSource(
        id="mvtec_ad",
        name="MVTec Anomaly Detection (MVTec AD)",
        official_url="https://www.mvtec.com/company/research/datasets/mvtec-ad",
        domain="15 object/texture categories (e.g. metal_nut, screw, grid)",
        tasks=[TASK_ANOMALY, TASK_SEGMENTATION],
        classes=["good", "<category-specific anomaly types, test set only>"],
        approx_size="~4.9 GB full; single categories ~0.1-1 GB",
        approx_bytes=5_200_000_000,
        annotations="Pixel-precise ground-truth masks for anomalous test images; training set contains only good images.",
        access="manual",
        access_notes="Official download requires accepting the licence via a form on mvtec.com. After download, "
                     "place the .tar.xz (full or per-category) in <data_root>/incoming/mvtec_ad/ and click Import.",
        license="CC BY-NC-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        attribution="Bergmann et al., 'MVTec AD - A Comprehensive Real-World Dataset for Unsupervised Anomaly "
                    "Detection', CVPR 2019.",
        redistribution="Non-commercial, share-alike, attribution required.",
        relevance=0.70,
        relevance_reason="Industrial anomaly detection with masks; different parts and an anomaly (not classifier) task.",
        download={"kind": "manual"},
    ),
    DatasetSource(
        id="mvtec_ad_hf_mirror",
        name="MVTec AD - single category via Hugging Face mirror (foersben/mvtec-ad)",
        official_url="https://huggingface.co/datasets/foersben/mvtec-ad",
        domain="MVTec AD category (default: metal_nut)",
        tasks=[TASK_ANOMALY, TASK_SEGMENTATION],
        classes=["good", "<anomaly types>"],
        approx_size="~100-250 MB per category",
        approx_bytes=250_000_000,
        annotations="Original MVTec folder layout incl. ground_truth masks.",
        access="auto_mirror",
        access_notes="Third-party redistribution published under the dataset's CC BY-NC-SA 4.0 licence, which "
                     "permits redistribution. Per-file SHA-256 (git-LFS) checksums are verified. Prefer the "
                     "official source for any published benchmark numbers.",
        license="CC BY-NC-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        attribution="Bergmann et al., CVPR 2019 (MVTec Software GmbH). Mirror: huggingface.co/datasets/foersben/mvtec-ad.",
        redistribution="Non-commercial, share-alike, attribution required.",
        relevance=0.65,
        relevance_reason="Same as MVTec AD; automatic but via a mirror rather than the official host.",
        download={"kind": "hf_mirror", "repo": "foersben/mvtec-ad", "category": "metal_nut"},
    ),
    DatasetSource(
        id="neu_surface",
        name="NEU surface defect database (NEU-CLS / NEU-DET)",
        official_url="http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm",
        domain="Hot-rolled steel strip surface, 200x200 grayscale",
        tasks=[TASK_MULTICLASS, TASK_DETECTION],
        classes=["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
        approx_size="~1,800 images (300 per class), ~30-60 MB",
        approx_bytes=60_000_000,
        annotations="Class labels (NEU-CLS); bounding boxes in Pascal VOC XML (NEU-DET).",
        access="manual",
        access_notes="Distributed by Northeastern University (Prof. Kechen Song) through links on the official page. "
                     "Kaggle copies list an 'Unknown' licence, so they are not downloaded automatically. Download "
                     "from the official page, place the zip/folder in <data_root>/incoming/neu_surface/ and Import.",
        license="Academic use (terms on the official page; cite the authors)",
        license_url="http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm",
        attribution="K. Song and Y. Yan, 'A noise robust method based on completed local binary patterns for "
                    "hot-rolled steel strip surface defects', Applied Surface Science, 2013.",
        redistribution="Not redistributed by this application.",
        relevance=0.55,
        relevance_reason="Real multiclass defect types, but steel strip rather than castings.",
        download={"kind": "manual"},
    ),
]

BY_ID = {d.id: d for d in CATALOGUE}


def ranked(task: str = PROJECT_TASK) -> list[DatasetSource]:
    """Rank by relevance+accessibility; datasets that cannot serve the requested task go last."""
    return sorted(CATALOGUE, key=lambda d: (task in d.tasks, d.rank_score), reverse=True)


def live_discovery(query: str = "casting defect", timeout: float = 15) -> dict[str, list[dict]]:
    """Query public catalogue APIs (Kaggle, Hugging Face) so the user sees what exists right now.
    Results are *unvetted*: licences must be reviewed before any of them is added as an adapter."""
    out: dict[str, list[dict]] = {"kaggle": [], "huggingface": [], "errors": []}
    try:
        r = requests.get("https://www.kaggle.com/api/v1/datasets/list", params={"search": query}, timeout=timeout)
        r.raise_for_status()
        for d in r.json()[:15]:
            out["kaggle"].append({"ref": d.get("ref"), "title": d.get("title"), "license": d.get("licenseName"),
                                  "bytes": d.get("totalBytes"), "url": f"https://www.kaggle.com/datasets/{d.get('ref')}"})
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"Kaggle: {e}")
    try:
        r = requests.get("https://huggingface.co/api/datasets", params={"search": query, "limit": 15, "full": "true"},
                         timeout=timeout)
        r.raise_for_status()
        for d in r.json():
            lic = next((t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("license:")), "unspecified")
            out["huggingface"].append({"ref": d.get("id"), "license": lic, "downloads": d.get("downloads"),
                                       "url": f"https://huggingface.co/datasets/{d.get('id')}"})
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"Hugging Face: {e}")
    return out
