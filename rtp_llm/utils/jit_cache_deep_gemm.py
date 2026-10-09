"""Complete DeepJIT disk entries and wheel-specific remote cache identities."""

import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path

SNAPSHOT_MANIFEST = ".jit_deep_gemm_checksums.json"
_ENTRY_FILES = ("kernel.cu", "kernel.cubin", "meta.json", ".committed")
_COMPILER_ENV = (
    "CC",
    "CXX",
    "CUDAHOSTCXX",
    "NVCC_CCBIN",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "CFLAGS",
    "CXXFLAGS",
    "DG_JIT_NVCC_FLAGS",
    "DG_JIT_CPP_STANDARD",
    "DG_JIT_WITH_LINEINFO",
)


def deep_gemm_build_scope(runtime_scope: str | None) -> str | None:
    try:
        dist = importlib.metadata.distribution("deep_gemm")
    except importlib.metadata.PackageNotFoundError:
        return None
    version = re.match(r"(\d+)\.(\d+)", dist.version)
    if version is None or tuple(map(int, version.groups())) < (2, 8):
        return None
    record = dist.read_text("RECORD")
    if not runtime_scope or not record:
        raise ValueError("DeepJIT cache requires runtime identity and wheel RECORD")
    # RECORD contains the installed wheel's binary/header hashes without importing
    # DeepGEMM, whose runtime must see its managed cache env before initialization.
    identity = {
        "version": dist.version,
        "wheel_record": record,
        "runtime": runtime_scope,
        "compiler_env": {key: os.environ.get(key, "") for key in _COMPILER_ENV},
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return "deepjit-v1-" + digest


def deepjit_entry_files(entry: Path) -> tuple[Path, ...]:
    if entry.is_symlink() or not entry.is_dir():
        raise ValueError(f"invalid DeepJIT entry: {entry}")
    for name in _ENTRY_FILES:
        path = entry / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"incomplete DeepJIT entry: {path}")
        size = path.stat().st_size
        if (name == ".committed" and size != 0) or (name != ".committed" and size == 0):
            raise ValueError(f"invalid DeepJIT entry size: {path}")
    if not isinstance(json.loads((entry / "meta.json").read_text()), dict):
        raise ValueError(f"invalid DeepJIT metadata: {entry}")
    files = tuple(sorted(entry.iterdir()))
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError(f"invalid DeepJIT entry member: {entry}")
    return files


def deepjit_snapshot_files(root: Path, *, strict: bool = False) -> dict[str, Path]:
    result = {}
    component = root / "deep_gemm"
    cache = component / "cache"
    try:
        if not component.exists():
            return result
        if component.is_symlink() or not component.is_dir():
            raise ValueError(f"invalid DeepGEMM component directory: {component}")
        if not cache.exists():
            return result
        if cache.is_symlink() or not cache.is_dir():
            raise ValueError(f"invalid DeepJIT cache: {cache}")
        for entry in cache.iterdir():
            try:
                result.update(
                    (path.relative_to(root).as_posix(), path)
                    for path in deepjit_entry_files(entry)
                )
            except (OSError, ValueError):
                if strict:
                    raise
    except (OSError, ValueError):
        if strict:
            raise
    return result


def deepjit_checksums(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in deepjit_snapshot_files(root, strict=True).items()
    }


def validate_deepjit_snapshot(root: Path) -> None:
    actual = deepjit_checksums(root)
    manifest = root / SNAPSHOT_MANIFEST
    if not actual and not manifest.exists():
        return
    expected = json.loads(manifest.read_text())
    if (
        not isinstance(expected, dict)
        or expected.get("schema_version") != 1
        or expected.get("files") != actual
    ):
        raise ValueError("DeepJIT snapshot payload checksum mismatch")
    manifest.unlink()
