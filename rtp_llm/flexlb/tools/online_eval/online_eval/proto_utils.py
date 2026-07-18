"""Runtime generation and loading for FlexLB protobuf modules."""

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
SCHEDULE_PROTO_FILE = PROTO_DIR / "flexlb_schedule_service.proto"
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

    return _ensure_proto_modules(PROTO_FILE, "model_rpc_service", out_dir)


def ensure_schedule_proto_modules(
    out_dir: Path | None = None,
) -> Tuple[ModuleType, ModuleType]:
    """Generate and import the standalone FlexLB schedule protocol."""

    return _ensure_proto_modules(
        SCHEDULE_PROTO_FILE, "flexlb_schedule_service", out_dir
    )


def _ensure_proto_modules(
    proto_file: Path,
    module_name: str,
    out_dir: Path | None,
) -> Tuple[ModuleType, ModuleType]:
    """Generate and import one protobuf module pair."""

    out = Path(out_dir or DEFAULT_OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    pb2_path = out / f"{module_name}_pb2.py"
    grpc_path = out / f"{module_name}_pb2_grpc.py"
    if _needs_regen(proto_file, pb2_path, grpc_path):
        _generate(out, proto_file)

    out_str = str(out)
    if out_str not in sys.path:
        sys.path.insert(0, out_str)
    return (
        importlib.import_module(f"{module_name}_pb2"),
        importlib.import_module(f"{module_name}_pb2_grpc"),
    )


def _needs_regen(proto_file: Path, pb2_path: Path, grpc_path: Path) -> bool:
    if not pb2_path.exists() or not grpc_path.exists():
        return True
    proto_mtime = proto_file.stat().st_mtime
    return (
        pb2_path.stat().st_mtime < proto_mtime
        or grpc_path.stat().st_mtime < proto_mtime
    )


def _generate(out: Path, proto_file: Path) -> None:
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
        str(proto_file),
    ]
    subprocess.run(cmd, check=True)
