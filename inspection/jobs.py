"""Minimal background job manager: threads + persisted status files + cooperative cancellation.

Status is written to <data_root>/jobs/<job_id>.json so the UI (and the CLI) can show the
exact state of every operation, including after a page reload."""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable


class Cancelled(Exception):
    pass


def pid_alive(pid: int | None) -> bool:
    """Whether another process (e.g. a CLI job) is still running. Avoids os.kill(pid, 0), which on Windows
    would send CTRL_C_EVENT."""
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class JobContext:
    def __init__(self, job_id: str, path: Path, cancel_event: threading.Event):
        self.job_id = job_id
        self.path = path
        self._cancel = cancel_event
        self.state: dict[str, Any] = {}

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def progress(self, fraction: float | None = None, message: str | None = None, **extra: Any) -> None:
        if fraction is not None:
            self.state["progress"] = max(0.0, min(1.0, float(fraction)))
        if message is not None:
            self.state["message"] = message
            self.state.setdefault("log", []).append(f"{time.strftime('%H:%M:%S')} {message}")
            self.state["log"] = self.state["log"][-300:]
        if extra:
            self.state.setdefault("extra", {}).update(extra)
        self.state["updated_at"] = time.time()
        self._write()

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()


class JobManager:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._events: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}

    def submit(self, kind: str, title: str, fn: Callable[[JobContext], Any]) -> str:
        job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{uuid.uuid4().hex[:6]}"
        ev = threading.Event()
        ctx = JobContext(job_id, self.jobs_dir / f"{job_id}.json", ev)
        ctx.state = {"id": job_id, "kind": kind, "title": title, "status": "running", "progress": 0.0,
                     "message": "started", "log": [], "started_at": time.time(), "updated_at": time.time(),
                     "pid": os.getpid()}
        ctx._write()

        def run() -> None:
            try:
                result = fn(ctx)
                ctx.state["status"] = "succeeded"
                ctx.state["result"] = result
                ctx.progress(1.0, "finished")
            except Cancelled:
                ctx.state["status"] = "cancelled"
                ctx.progress(message="cancelled by user")
            except Exception as e:  # noqa: BLE001 - surface every failure in the UI
                ctx.state["status"] = "failed"
                ctx.state["error"] = f"{type(e).__name__}: {e}"
                ctx.state["traceback"] = traceback.format_exc()
                ctx.progress(message=f"failed: {e}")

        t = threading.Thread(target=run, name=job_id, daemon=True)
        self._events[job_id] = ev
        self._threads[job_id] = t
        t.start()
        return job_id

    def cancel(self, job_id: str) -> None:
        if job_id in self._events:
            self._events[job_id].set()

    def is_alive(self, job_id: str) -> bool:
        t = self._threads.get(job_id)
        return bool(t and t.is_alive())

    def list(self, limit: int = 30) -> list[dict]:
        out = []
        for p in sorted(self.jobs_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                st = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            # A job marked running whose thread/process is gone was interrupted (e.g. app restart).
            if st.get("status") == "running":
                if st.get("pid") not in (None, os.getpid()):
                    alive = pid_alive(st["pid"])           # started by the CLI or another app instance
                else:
                    alive = self.is_alive(st["id"])
                if not alive:
                    st["status"] = "interrupted"
                elif st.get("pid") != os.getpid():
                    st["title"] += " (external process; cancel it there)"
            out.append(st)
        return out


def run_inline(kind: str, fn: Callable[[JobContext], Any], jobs_dir: Path, verbose: bool = True) -> Any:
    """Run a job function synchronously (CLI use) while still persisting status."""
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-cli"
    ctx = JobContext(job_id, jobs_dir / f"{job_id}.json", threading.Event())
    ctx.state = {"id": job_id, "kind": kind, "title": f"{kind} (CLI)", "status": "running", "progress": 0.0,
                 "log": [], "started_at": time.time(), "pid": os.getpid()}
    if verbose:
        orig = ctx.progress

        def progress(fraction=None, message=None, **extra):
            orig(fraction, message, **extra)
            if message:
                pct = f"{100 * ctx.state.get('progress', 0):5.1f}%"
                print(f"[{kind} {pct}] {message}", flush=True)
        ctx.progress = progress  # type: ignore[method-assign]
    try:
        result = fn(ctx)
        ctx.state["status"] = "succeeded"
        ctx.state["result"] = result
        ctx.progress(1.0, "finished")
        return result
    except Exception as e:
        ctx.state["status"] = "failed"
        ctx.state["error"] = f"{type(e).__name__}: {e}"
        ctx.progress(message=f"failed: {e}")
        raise
