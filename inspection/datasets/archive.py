"""Safe archive extraction: rejects absolute paths, drive letters, '..' traversal, links and zip bombs."""
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path, PurePosixPath

MAX_TOTAL_BYTES = 20 * 1024**3   # refuse archives that expand beyond 20 GB
MAX_RATIO = 200                  # refuse members with an absurd compression ratio


class UnsafeArchiveError(RuntimeError):
    pass


def _safe_member_path(name: str, root: Path) -> Path:
    if not name or "\x00" in name:
        raise UnsafeArchiveError(f"Invalid member name: {name!r}")
    norm = name.replace("\\", "/")
    p = PurePosixPath(norm)
    if p.is_absolute() or (len(norm) > 1 and norm[1] == ":") or ".." in p.parts:
        raise UnsafeArchiveError(f"Unsafe path in archive: {name!r}")
    target = (root / Path(*p.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise UnsafeArchiveError(f"Path escapes extraction root: {name!r}")
    return target


def verify_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad:
            raise UnsafeArchiveError(f"Corrupt member in zip (CRC error): {bad}")
        infos = z.infolist()
        return {"members": len(infos), "uncompressed_bytes": sum(i.file_size for i in infos)}


def extract(path: Path, dest: Path, progress=None, check_cancel=None) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    rejected: list[str] = []
    n = 0
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            total = sum(i.file_size for i in infos)
            if total > MAX_TOTAL_BYTES:
                raise UnsafeArchiveError(f"Archive expands to {total / 1e9:.1f} GB (> limit)")
            for k, info in enumerate(infos):
                if check_cancel:
                    check_cancel()
                if info.compress_size and info.file_size / max(info.compress_size, 1) > MAX_RATIO:
                    raise UnsafeArchiveError(f"Suspicious compression ratio for {info.filename}")
                # zip symlinks are encoded in the upper bits of external_attr
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    rejected.append(info.filename)
                    continue
                target = _safe_member_path(info.filename, dest)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(target, "wb") as out:
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                n += 1
                if progress and k % 200 == 0:
                    progress(k / len(infos))
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            members = t.getmembers()
            total = sum(m.size for m in members)
            if total > MAX_TOTAL_BYTES:
                raise UnsafeArchiveError(f"Archive expands to {total / 1e9:.1f} GB (> limit)")
            for k, m in enumerate(members):
                if check_cancel:
                    check_cancel()
                if m.issym() or m.islnk() or m.isdev():
                    rejected.append(m.name)
                    continue
                target = _safe_member_path(m.name, dest)
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                src = t.extractfile(m)
                if src is None:
                    rejected.append(m.name)
                    continue
                with src, open(target, "wb") as out:
                    while chunk := src.read(1 << 20):
                        out.write(chunk)
                n += 1
                if progress and k % 200 == 0:
                    progress(k / len(members))
    else:
        raise UnsafeArchiveError(f"Unsupported archive format: {path.name}")
    return {"extracted_files": n, "rejected_members": rejected}
