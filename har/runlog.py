"""Run bookkeeping shared by the HAR runners: machine / device / git info, logging every line of
terminal output to log files as well, atomic file writes and config hashes."""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

from .paths import REPO_ROOT


def machine_name() -> str:
    return (platform.node() or socket.gethostname() or "unknown").replace(" ", "_")


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def device_info(device: str) -> dict:
    import torch

    info = {"device": device, "gpu_name": "", "cuda_version": torch.version.cuda or "",
            "torch_version": torch.__version__, "cpu": platform.processor() or platform.machine(),
            "torch_threads": torch.get_num_threads(), "python": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}"}
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False "
                               f"(torch {torch.__version__}, built for CUDA {torch.version.cuda}); see "
                               "environment/requirements.md, GPU setup")
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain", "--", "har", "lci"))}


def code_sha256() -> str:
    """Hash of the har/ and lci/ Python sources, so a result can be tied to exact code even when the
    working tree is not committed."""
    h = hashlib.sha256()
    for p in sorted(list((REPO_ROOT / "har").glob("*.py")) + list((REPO_ROOT / "lci").glob("*.py"))):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def settings_hash(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj) -> None:
    write_text_atomic(path, json.dumps(obj, indent=1, sort_keys=False, default=str))


def write_csv(path: Path, df) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


class _Tee:
    """sys.stdout / sys.stderr replacement writing to the terminal and to every open log file."""

    def __init__(self, stream):
        self.stream, self.files = stream, []

    def write(self, s):
        self.stream.write(s)
        for f in self.files:
            f.write(s)
            f.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        for f in self.files:
            f.flush()

    def isatty(self):
        return False


def _install() -> tuple[_Tee, _Tee]:
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout)
    if not isinstance(sys.stderr, _Tee):
        sys.stderr = _Tee(sys.stderr)
    return sys.stdout, sys.stderr


@contextlib.contextmanager
def tee_to(path: Path, mode: str = "a"):
    """Everything printed (stdout and stderr) inside the block is also appended to `path`."""
    out, err = _install()
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, mode, encoding="utf-8")
    out.files.append(f)
    err.files.append(f)
    try:
        yield path
    finally:
        out.files.remove(f)
        err.files.remove(f)
        f.close()
