"""Runtime generation/loading for rtp-llm model_rpc_service protobufs."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[5]
PROTO_DIR = REPO_ROOT / "rtp_llm" / "cpp" / "model_rpc" / "proto"
PROTO_FILE = PROTO_DIR / "model_rpc_service.proto"
DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "FLEXLB_EVAL_PROTO_OUT",
        str(Path(tempfile.gettempdir()) / "flexlb_eval_proto"),
    )
)


def ensure_proto_modules(out_dir: Path | None = None) -> Tuple[ModuleType, ModuleType]:
    """Generate and import model_rpc_service_pb2/_grpc modules.

    The repo intentionally does not check in generated Python protobuf files.
    Online evaluation tools generate them into a temporary directory at runtime.
    """

    out = Path(out_dir or DEFAULT_OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    pb2_path = out / "model_rpc_service_pb2.py"
    grpc_path = out / "model_rpc_service_pb2_grpc.py"

    if _needs_regen(pb2_path, grpc_path):
        _generate(out)

    out_str = str(out)
    if out_str not in sys.path:
        sys.path.insert(0, out_str)

    # Import by generated top-level module names. Reloading avoids stale modules
    # when a developer regenerates after changing the proto in the same process.
    pb2 = importlib.import_module("model_rpc_service_pb2")
    pb2_grpc = importlib.import_module("model_rpc_service_pb2_grpc")
    return pb2, pb2_grpc


def _needs_regen(pb2_path: Path, grpc_path: Path) -> bool:
    if not pb2_path.exists() or not grpc_path.exists():
        return True
    proto_mtime = PROTO_FILE.stat().st_mtime
    return (
        pb2_path.stat().st_mtime < proto_mtime
        or grpc_path.stat().st_mtime < proto_mtime
    )


def _generate(out: Path) -> None:
    try:
        import grpc_tools.protoc  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "grpc_tools is required for FlexLB online evaluation. "
            "Run inside luoli_gpu or install grpcio-tools/protobuf."
        ) from exc

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={out}",
        f"--grpc_python_out={out}",
        str(PROTO_FILE),
    ]
    subprocess.run(cmd, check=True)
