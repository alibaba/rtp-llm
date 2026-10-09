"""Opt-in isolated FlashInfer runtime for K3; run before attention imports."""

import importlib.metadata
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

from rtp_llm.utils.jit_cache_identity import (
    isolated_package_identity,
    register_component_identity,
    selected_package_identity,
)


def setup_flashinfer_runtime():
    register_jit_cache_identities()
    value = os.environ.get("KIMI_K3_FLASHINFER_ROOT")
    if not value:
        return
    if os.environ.get("MODEL_TYPE") not in ("kimi_k3", "kimi_k3_mtp"):
        raise RuntimeError("KIMI_K3_FLASHINFER_ROOT is only supported for K3")
    root = Path(value).resolve(strict=True)
    expected = (root / "flashinfer").resolve(strict=True)
    loaded = sys.modules.get("flashinfer")
    if loaded is not None and Path(loaded.__file__).resolve().parent != expected:
        raise RuntimeError("FlashInfer was imported before K3 runtime selection; restart the process")
    sys.path.insert(0, str(root))
    # FlashInfer caches its architecture probe on first use. Select CUTLASS
    # before any FlashInfer import can cache an unavailable/older DSL.
    from .models_py.utils.cutlass import setup_cutlass_import_path

    setup_cutlass_import_path()


def _native_deep_gemm_identity() -> str:
    spec = importlib.util.find_spec("k3_native_deep_gemm")
    if spec is None or not spec.origin:
        raise ValueError("K3 native DeepGEMM is not importable")
    package = Path(spec.origin).resolve().parent
    manifest = json.loads((package / "BUILD_INFO.json").read_text())
    binaries = tuple(package.glob("_C*.so"))
    if len(binaries) != 1 or not manifest.get("binary_sha256"):
        raise ValueError("K3 native DeepGEMM build manifest is incomplete")
    if sha256(binaries[0].read_bytes()).hexdigest() != manifest["binary_sha256"]:
        raise ValueError("K3 native DeepGEMM binary differs from its build manifest")
    source = json.dumps(manifest, sort_keys=True).encode()
    return sha256(source).hexdigest()


def _installed_deep_gemm_identity() -> str:
    spec = importlib.util.find_spec("deep_gemm")
    if spec is None or not spec.origin:
        raise ValueError("installed DeepGEMM is not importable")
    package = Path(spec.origin).resolve().parent
    digest = sha256()
    try:
        digest.update(importlib.metadata.version("deep_gemm").encode())
    except importlib.metadata.PackageNotFoundError:
        # Source and Bazel installations need not have a wheel dist-info.
        pass
    source_files = sorted(
        path
        for path in package.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".cu", ".cuh", ".h", ".cpp", ".so"}
        and not any(
            part.startswith("tmp") or part == "__pycache__"
            for part in path.relative_to(package).parts
        )
    )
    if not source_files:
        raise ValueError("installed DeepGEMM has no identifiable source files")
    for path in source_files:
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _flashinfer_cache_identity(scopes: Mapping[str, str]) -> tuple[str, ...] | None:
    root = os.environ.get("KIMI_K3_FLASHINFER_ROOT")
    if not root:
        return None
    return (
        scopes["torch"],
        isolated_package_identity(root, Path("flashinfer"), "flashinfer-python"),
    )


def _cute_dsl_cache_identity(scopes: Mapping[str, str]) -> tuple[str, ...]:
    root = os.environ.get("KIMI_K3_CUTLASS_DSL_ROOT")
    cutlass = (
        isolated_package_identity(
            root,
            Path("nvidia_cutlass_dsl/dsl_packages/cutlass"),
            "nvidia-cutlass-dsl",
            companion_record_prefixes=("nvidia_cutlass_dsl_libs_",),
        )
        if root
        else importlib.metadata.version("nvidia-cutlass-dsl")
    )
    return (
        scopes["accelerator"],
        cutlass,
        selected_package_identity("tokenspeed_mla", "tokenspeed-mla"),
        selected_package_identity(
            "tokenspeed_triton", "tokenspeed-triton", include_compiler_binaries=True
        ),
    )


def _deep_gemm_cache_identity(scopes: Mapping[str, str]) -> tuple[str, ...]:
    producers = []
    native_backend = os.environ.get("KIMI_K3_MOE_BACKEND", "rtp") == "vllm_native"
    try:
        producers.append("installed:" + _installed_deep_gemm_identity())
    except (OSError, ValueError):
        if not native_backend:
            raise
    if native_backend:
        producers.append("native:" + _native_deep_gemm_identity())
    return (scopes["accelerator"], *producers)


def register_jit_cache_identities() -> None:
    """Bind selected dependencies during model runtime bootstrap, before JIT."""
    if os.environ.get("MODEL_TYPE") not in ("kimi_k3", "kimi_k3_mtp"):
        return
    register_component_identity("flashinfer", _flashinfer_cache_identity)
    register_component_identity("trtllm_deep_gemm", _flashinfer_cache_identity)
    register_component_identity("cute_dsl", _cute_dsl_cache_identity)
    register_component_identity("deep_gemm", _deep_gemm_cache_identity)
