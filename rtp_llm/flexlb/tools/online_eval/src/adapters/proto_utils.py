"""Generate and cache RPC modules and encode request metadata."""

from __future__ import annotations

import json
import fcntl
import hashlib
from importlib.metadata import version
import shutil
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
    cache = _proto_out_dir()
    cache.mkdir(parents=True, exist_ok=True)
    src = PROTO_DIR / proto_name
    if not src.is_file():
        raise FileNotFoundError(f"proto source not found: {src}")
    sources = sorted(PROTO_DIR.glob("*.proto"))
    digest = hashlib.sha256((sys.version + version("grpcio-tools") + version("protobuf")).encode())
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    out_dir = cache / digest.hexdigest()
    with (cache / (digest.hexdigest() + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = out_dir / 'generated.json'
        try:
            hashes = json.loads(manifest.read_text(encoding='utf-8'))
            valid = isinstance(hashes, dict) and bool(hashes) and all(
                (out_dir / name).is_file() and hashlib.sha256((out_dir / name).read_bytes()).hexdigest() == expected
                for name, expected in hashes.items())
            valid = valid and (out_dir / proto_name.replace('.proto', '_pb2.py')).is_file() and (out_dir / proto_name.replace('.proto', '_pb2_grpc.py')).is_file()
        except (OSError, ValueError, TypeError):
            valid = False
        if not valid:
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                stage = Path(temporary) / 'generated'
                stage.mkdir()
                command = [sys.executable, '-m', 'grpc_tools.protoc', f'-I{PROTO_DIR}',
                           f'--python_out={stage}', f'--grpc_python_out={stage}',
                           *[source.name for source in sources]]
                try:
                    subprocess.run(command, check=True, capture_output=True)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError('protobuf generation failed: ' + exc.stderr.decode('utf-8', errors='replace')) from exc
                generated = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                             for file in stage.glob('*.py')}
                (stage / 'generated.json').write_text(json.dumps(generated), encoding='utf-8')
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                stage.rename(out_dir)
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
