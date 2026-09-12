"""DeepJIT entry and build identity rules for the existing DeepGEMM component."""

import hashlib
import importlib.metadata
import json
import os
import re
from contextlib import suppress
from pathlib import Path

from rtp_llm.utils.deep_gemm_compat import deep_gemm_uses_deepjit

BUILD_MANIFEST = "deep_gemm/rtp_build_manifest.json"
ENTRY_FORMAT = "deepjit-v1"
SNAPSHOT_MANIFEST = ".jit_deep_gemm_checksums.json"
_SCOPE_PATTERN = re.compile(r"deep_gemm-.+-deepjit-v1-[0-9a-f]{64}\Z")
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
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
    "DJ_JIT_NVCC_FLAGS",
)


class DeepGemmBuildIdentityError(ValueError):
    pass


def deep_gemm_build_scope(runtime_scope: str | None) -> str | None:
    """Identify a wheel without importing DeepGEMM or constructing its runtime.

    The builder installs the source/patch/toolchain manifest as wheel data. The
    wheel's own SHA belongs in the external release manifest, avoiding recursion.
    Older wheels retain their original scope when they have no such manifest.
    """
    try:
        distribution = importlib.metadata.distribution("deep_gemm")
    except importlib.metadata.PackageNotFoundError:
        return None
    path = Path(distribution.locate_file(BUILD_MANIFEST))
    uses_deepjit = deep_gemm_uses_deepjit(distribution.version)
    if not uses_deepjit and not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported schema_version")
        if manifest.get("cache_entry_format") != ENTRY_FORMAT:
            raise ValueError("unsupported cache_entry_format")
        sources = manifest.get("sources", {})
        for name in ("deepgemm", "deepjit", "cutlass"):
            if not _SHA_PATTERN.fullmatch(str(sources.get(name, ""))):
                raise ValueError(f"missing full {name} source SHA")
        if not _SHA256_PATTERN.fullmatch(str(manifest.get("patch_sha256", ""))):
            raise ValueError("missing patch_sha256")
        build = manifest.get("build", {})
        for name in (
            "python_soabi",
            "torch_version",
            "host_arch",
            "cuda_version",
            "nvcc_version",
            "host_compiler",
        ):
            if not isinstance(build.get(name), str) or not build[name]:
                raise ValueError(f"missing build.{name}")
        if not isinstance(build.get("cxx11_abi"), bool):
            raise ValueError("missing build.cxx11_abi")
        if not _SHA256_PATTERN.fullmatch(str(build.get("libstdcxx_sha256", ""))):
            raise ValueError("missing build.libstdcxx_sha256")
        for name in ("target_archs", "flags"):
            value = build.get(name)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(f"invalid build.{name}")
        if not build["target_archs"]:
            raise ValueError("missing build.target_archs")
        if not runtime_scope:
            raise ValueError("cannot identify the actual Torch/C++ runtime")
        identity = {
            "build_manifest": manifest,
            "runtime_scope": runtime_scope,
            "compiler_env": {name: os.environ.get(name, "") for name in _COMPILER_ENV},
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return f"{ENTRY_FORMAT}-{hashlib.sha256(encoded).hexdigest()}"
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        raise DeepGemmBuildIdentityError(
            f"DeepGEMM {distribution.version} requires a valid {BUILD_MANIFEST}: {exc}"
        ) from exc


def is_deepjit_scope(name: str) -> bool:
    return _SCOPE_PATTERN.fullmatch(name) is not None


def is_deepjit_path(relative: str) -> bool:
    return is_deepjit_scope(relative.split("/", 1)[0])


def deepjit_entry_files(entry: Path) -> tuple[Path, ...]:
    """Validate a completed CUDA entry; loader/real warmup still checks loadability."""
    if entry.is_symlink() or not entry.is_dir():
        raise ValueError(f"invalid DeepJIT entry: {entry}")
    for name in _ENTRY_FILES:
        path = entry / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"incomplete DeepJIT entry: {path}")
        size = path.stat().st_size
        if (name == ".committed" and size != 0) or (name != ".committed" and size == 0):
            raise ValueError(f"invalid DeepJIT entry size: {path}")
    metadata = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid DeepJIT metadata: {entry}")
    files = tuple(sorted(entry.iterdir()))
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError(f"invalid DeepJIT entry member: {entry}")
    return files


def deepjit_snapshot_files(root: Path, *, strict: bool = False) -> dict[str, Path]:
    """Collect complete new-format entries, leaving legacy DG rules to the caller."""
    result = {}
    component = root / "deep_gemm"
    try:
        if not component.exists():
            return result
        if component.is_symlink() or not component.is_dir():
            raise ValueError(f"invalid DeepGEMM component directory: {component}")
        scopes = tuple(component.iterdir())
    except (OSError, ValueError):
        if strict:
            raise
        return result
    for scope in scopes:
        if not is_deepjit_scope(scope.name):
            continue
        try:
            if scope.is_symlink() or not scope.is_dir():
                raise ValueError(f"invalid DeepJIT scope: {scope}")
            if strict and any(child.name != "cache" for child in scope.iterdir()):
                raise ValueError(f"uncommitted DeepJIT snapshot member: {scope}")
            cache = scope / "cache"
            if not cache.exists():
                continue
            if cache.is_symlink() or not cache.is_dir():
                raise ValueError(f"invalid DeepJIT cache: {cache}")
            for entry in cache.iterdir():
                try:
                    files = deepjit_entry_files(entry)
                    result.update(
                        (path.relative_to(root).as_posix(), path) for path in files
                    )
                except (OSError, ValueError):
                    if strict:
                        raise
        except (OSError, ValueError):
            if strict:
                raise
    return result


def is_complete_deepjit_marker(name: str, path: Path) -> bool:
    parts = name.split("/")
    if not (
        len(parts) == 5
        and parts[0] == "deep_gemm"
        and is_deepjit_scope(parts[1])
        and parts[2] == "cache"
        and parts[4] == ".committed"
    ):
        return False
    with suppress(OSError, ValueError):
        deepjit_entry_files(path.parent)
        return True
    return False


def deepjit_checksums(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in deepjit_snapshot_files(root, strict=True).items()
    }


def validate_deepjit_snapshot(root: Path) -> None:
    actual = deepjit_checksums(root)
    manifest_path = root / SNAPSHOT_MANIFEST
    if not actual and not manifest_path.exists():
        return
    expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(expected, dict)
        or expected.get("schema_version") != 1
        or expected.get("files") != actual
    ):
        raise ValueError("DeepJIT snapshot payload checksum mismatch")
    manifest_path.unlink()
