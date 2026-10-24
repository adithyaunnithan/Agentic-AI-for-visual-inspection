"""Resumable HTTP download with progress, retries, cancellation and disk-space checks."""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Callable

import requests

from ..config import require_private

CHUNK = 1 << 20
UA = {"User-Agent": "agentic-inspection-dataset-autopilot/1.0 (academic project)"}


class DownloadError(RuntimeError):
    pass


def free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def ensure_disk_space(path: Path, needed: int, safety_factor: float = 2.5) -> None:
    """Archives must fit twice (archive + extraction) plus headroom."""
    free = free_bytes(path)
    if needed and free < needed * safety_factor:
        raise DownloadError(f"Insufficient disk space at {path}: {free / 1e9:.2f} GB free, "
                            f"~{needed * safety_factor / 1e9:.2f} GB required (archive + extraction + headroom).")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, *, expected_size: int | None = None, expected_sha256: str | None = None,
             progress: Callable[[int, int | None], None] | None = None,
             check_cancel: Callable[[], None] | None = None, retries: int = 5, timeout: float = 60) -> Path:
    """Download url -> dest. Partial data lives in dest.part and is resumed with HTTP Range requests.
    The final file is only moved into place after size (and checksum, if known) verification."""
    require_private("Dataset download")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if dest.exists() and (expected_sha256 is None or sha256_file(dest) == expected_sha256):
        return dest

    attempt = 0
    while True:
        attempt += 1
        have = part.stat().st_size if part.exists() else 0
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout, allow_redirects=True) as r:
                if r.status_code == 416:            # range beyond end -> part is already complete
                    break
                if r.status_code not in (200, 206):
                    raise DownloadError(f"HTTP {r.status_code} for {url}")
                if have and r.status_code == 200:    # server ignored Range -> restart cleanly
                    have = 0
                    part.unlink(missing_ok=True)
                total = None
                if r.headers.get("Content-Length"):
                    total = have + int(r.headers["Content-Length"])
                elif expected_size:
                    total = expected_size
                with open(part, "ab") as f:
                    done = have
                    for chunk in r.iter_content(CHUNK):
                        if check_cancel:
                            check_cancel()
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
            break
        except (requests.RequestException, DownloadError) as e:
            if attempt > retries:
                raise DownloadError(f"Download failed after {retries} retries: {e}") from e
            time.sleep(min(30, 2 ** attempt))

    size = part.stat().st_size
    if expected_size and size != expected_size:
        raise DownloadError(f"Size mismatch for {dest.name}: got {size}, expected {expected_size}. "
                            f"The partial file was kept for resumption/inspection: {part}")
    if expected_sha256:
        got = sha256_file(part)
        if got != expected_sha256:
            part.unlink(missing_ok=True)
            raise DownloadError(f"SHA-256 mismatch for {dest.name}: got {got}, expected {expected_sha256}")
    part.replace(dest)
    return dest
