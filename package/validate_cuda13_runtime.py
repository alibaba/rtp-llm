"""Reject incompatible CUDA13 operator versions and real import failures."""

import argparse
import importlib
import inspect
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# These are explicit DeepEP runtime overrides shared with the resolver and image.
COMMUNICATION_OVERRIDES = {
    "nvidia-nccl-cu13": "2.30.4",
    "nvidia-nvshmem-cu13": "3.5.19",
    "nvidia-cuda-cccl": "13.2.86",
}


def validate_dependency_metadata():
    roots = (
        "torch",
        "deep-ep",
        "deep-gemm",
        "flash-mla",
        "tilelang",
        "xgrammar",
        "flashinfer-python",
        "cuda-linear-attention",
        "flash-linear-attention",
        "tokenspeed-mla",
        "nvidia-cutlass-dsl",
    )
    pending = [
        (name, ("", "cu13") if name == "nvidia-cutlass-dsl" else ("",))
        for name in roots
    ]
    visited = set()
    while pending:
        package, extras = pending.pop()
        key = (canonicalize_name(package), tuple(sorted(extras)))
        if key in visited:
            continue
        visited.add(key)
        for declaration in metadata.requires(package) or ():
            requirement = Requirement(declaration)
            if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra}) for extra in extras
            ):
                continue
            actual = metadata.version(requirement.name)
            name = canonicalize_name(requirement.name)
            overridden = (
                name in COMMUNICATION_OVERRIDES
                and actual == COMMUNICATION_OVERRIDES[name]
            )
            if not overridden and not requirement.specifier.contains(
                actual, prereleases=True
            ):
                raise RuntimeError(
                    f"{package} requires {declaration}, installed {actual}"
                )
            pending.append((requirement.name, tuple({"", *requirement.extras})))


def incompatible_cache_packages():
    runtime_version = metadata.version("flashinfer-python")
    incompatible = []
    for package in ("flashinfer-cubin", "flashinfer-jit-cache"):
        try:
            cache_version = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
        compatible = cache_version == runtime_version
        if package == "flashinfer-jit-cache":
            compatible |= cache_version.startswith(
                (runtime_version + "+", runtime_version + ".")
            )
        if not compatible:
            incompatible.append(package)
    return incompatible


def validate_runtime():
    expected = {
        "apache-tvm-ffi": "0.1.13",
        "flashinfer-python": "0.6.18.post1",
        "nvidia-cutlass-dsl": "4.7.1",
        "nvidia-cutlass-dsl-libs-base": "4.7.1",
        "nvidia-cutlass-dsl-libs-core": "4.7.1",
        "nvidia-cutlass-dsl-libs-cu13": "4.7.1",
    }
    for package, required_version in expected.items():
        actual_version = metadata.version(package)
        if actual_version != required_version:
            raise RuntimeError(
                f"CUDA13 {package}: expected {required_version}, got {actual_version}"
            )
    for package, required_version in COMMUNICATION_OVERRIDES.items():
        if metadata.version(package) != required_version:
            raise RuntimeError(f"DeepEP requires {package}=={required_version}")
    validate_dependency_metadata()
    incompatible = incompatible_cache_packages()
    if incompatible:
        raise RuntimeError(
            "FlashInfer has incompatible precompiled cache packages: "
            + ", ".join(incompatible)
        )
    rtp = importlib.import_module("rtp_llm")
    # Match start_backend_server's parent-path alias in the installed wheel.
    sys.path.append(str(Path(rtp.__file__).parent / ".."))
    helper = importlib.import_module("rtp_llm.models_py.utils.cutlass")
    helper.setup_cutlass_import_path()
    print("Validated RTP CUTLASS import with backend parent-path alias")
    runtime = importlib.import_module("flashinfer")
    decode = importlib.import_module("flashinfer.decode")
    required = {"cum_seq_lens_q", "max_q_len", "backend", "out"}
    parameters = inspect.signature(
        decode.trtllm_batch_decode_with_kv_cache_mla
    ).parameters
    missing = required.difference(parameters)
    if missing:
        raise RuntimeError(
            "FlashInfer compact MLA parameters are missing: "
            + ", ".join(sorted(missing))
        )
    cute = importlib.import_module("cutlass.cute")
    if not callable(getattr(cute.arch, "map_dsmem_ptr", None)):
        raise RuntimeError("K3 router requires cutlass.cute.arch.map_dsmem_ptr")
    importlib.import_module("tilelang")
    tvm = importlib.import_module("tvm")
    attrs = tvm.ir.make_node("ir.DictAttrs", runtime_probe=1)
    if attrs["runtime_probe"] != 1 or attrs.__dict__["runtime_probe"] != 1:
        raise RuntimeError("TileLang TVM DictAttrs field access failed")
    for module in ("tokenspeed_mla.mla_decode", "tokenspeed_mla.mla_prefill", "cula"):
        loaded = importlib.import_module(module)
        print(f"Validated CUDA13 operator import: {module} at {loaded.__file__}")
    print(
        "Validated FlashInfer compact MLA and CuTe router API: " + runtime.__version__
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incompatible-caches", action="store_true")
    args = parser.parse_args()
    if args.incompatible_caches:
        print(" ".join(incompatible_cache_packages()))
    else:
        validate_runtime()
