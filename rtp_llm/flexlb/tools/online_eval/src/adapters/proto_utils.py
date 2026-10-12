"""Generate and cache RPC modules and encode request metadata."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from runtime.paths import REPO_ROOT


PROTO_DIR = REPO_ROOT / "rtp_llm" / "cpp" / "model_rpc" / "proto"

_PROTO_CACHE: dict = {}


def _proto_out_dir() -> Path:
    out = os.environ.get("FLEXLB_EVAL_PROTO_OUT")
    if out:
        return Path(out)
    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
    return Path(tmp) / "flexlb_eval_proto"


def _generate_proto(proto_name: str) -> tuple:
    out_dir = _proto_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = PROTO_DIR / proto_name
    if not src.is_file():
        raise FileNotFoundError(f"proto source not found: {src}")
    py_name = proto_name.replace(".proto", "_pb2.py")
    grpc_name = proto_name.replace(".proto", "_pb2_grpc.py")
    need_regen = True
    if (out_dir / py_name).is_file() and (out_dir / grpc_name).is_file():
        need_regen = src.stat().st_mtime > (out_dir / py_name).stat().st_mtime
    if need_regen:
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={out_dir}",
            f"--grpc_python_out={out_dir}",
            proto_name,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    if str(out_dir) not in sys.path:
        sys.path.insert(0, str(out_dir))
    import importlib

    base = proto_name.replace(".proto", "")
    pb2 = importlib.import_module(f"{base}_pb2")
    pb2_grpc = importlib.import_module(f"{base}_pb2_grpc")
    return pb2, pb2_grpc


def ensure_proto_modules() -> tuple:
    """Engine-side protos (rpc_service.proto)."""
    if "rpc" not in _PROTO_CACHE:
        _PROTO_CACHE["rpc"] = _generate_proto("model_rpc_service.proto")
    return _PROTO_CACHE["rpc"]


def ensure_schedule_proto_modules() -> tuple:
    """Master-side protos (flexlb_service.proto)."""
    if "schedule" not in _PROTO_CACHE:
        _PROTO_CACHE["schedule"] = _generate_proto("flexlb_schedule_service.proto")
    return _PROTO_CACHE["schedule"]


def encode_unique_key(meta: dict) -> str:
    return "flexlb_eval:" + json.dumps(meta, separators=(",", ":"))
