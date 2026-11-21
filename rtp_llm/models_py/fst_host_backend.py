"""Check the native SHM reader before selecting it on CUDA-compatible runtimes."""

import ctypes
import importlib
import importlib.metadata
import importlib.util
import re
from pathlib import Path
from typing import Optional


def _package_build():
    # Bazel may put distinct wheels on separate sys.path entries. Do not
    # attest an imported extension using metadata from a shadowed wheel.
    spec = importlib.util.find_spec("fast_safetensors")
    if spec is None or spec.origin is None:
        raise ImportError("fast_safetensors is not installed")
    package = Path(spec.origin).parent
    distributions = importlib.metadata.Distribution.discover(
        name="fast-safetensors", path=[str(package.parent)]
    )
    dist = next(iter(distributions), None)
    local = dist.version.lower().partition("+")[2] if dist else ""
    markers = re.split(r"[._-]", local)
    for runtime, pattern in (
        ("Ppu", r"ppu[0-9]*"),
        ("ROCm", r"(?:rocm|hip)[0-9]*"),
        ("Cuda", r"(?:cu|cuda)[0-9]*"),
    ):
        if any(re.fullmatch(pattern, marker) for marker in markers):
            return package, runtime
    return package, None


def _has_hip_copy_abi(path: str) -> bool:
    # Untagged ROCm builds need not use a particular local version suffix.
    # Inspect symbols on the loaded reader, never the global process handle.
    handle = ctypes.CDLL(path)
    return all(
        hasattr(handle, symbol)
        for symbol in ("hipMemcpyAsync", "hipHostRegister", "hipStreamCreate")
    ) and not hasattr(handle, "cudaMemcpyAsync")


def fst_host_backend_reason(
    backend: str, *, use_shm: bool, use_nogds: bool
) -> Optional[str]:
    """Explain an unsafe/missing backend, without invoking a GPU or reading weights."""
    if backend not in ("Ppu", "ROCm"):
        return None
    if not use_shm or use_nogds:
        return (
            f"{backend} fastsafetensors requires its native SHM reader; "
            "CUDA GDS/nogds is not supported"
        )
    try:
        package, declared = _package_build()
        if declared and declared != backend:
            return f"fast-safetensors was built for {declared}, not {backend}"
        if backend == "Ppu" and declared != backend:
            # The PPU SDK exports CUDA-compatible symbols too. They cannot
            # distinguish a vendor reader from an ordinary NVIDIA build.
            return (
                "cannot identify a PPU build of fast-safetensors; "
                "install a PPU SDK compatible SHM wheel"
            )
        wrapper = importlib.import_module("fast_safetensors")
        native = importlib.import_module("fast_safetensors._C")
        extension = Path(native.__file__)
        if (package / extension.name).resolve() != extension.resolve():
            return "fast-safetensors metadata and native extension come from different packages"
        for module in (wrapper, native):
            cls = getattr(module, "LoadWithShm", None)
            if not callable(cls) or not callable(
                getattr(cls, "load_safetensors_to_device", None)
            ):
                return f"{module.__name__} lacks LoadWithShm.load_safetensors_to_device"
        if (
            backend == "ROCm"
            and declared is None
            and not _has_hip_copy_abi(native.__file__)
        ):
            return "cannot identify the HIP copy ABI of the installed fast-safetensors SHM reader"
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        return f"cannot use the native {backend} fast-safetensors SHM reader: {exc}"
    return None
