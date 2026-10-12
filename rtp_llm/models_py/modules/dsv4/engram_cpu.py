"""Explicitly startup-warmed CPU gather/dequant for experimental Engram prefill.

The caller owns CPU indices and a BF16 output (pinned for asynchronous H2D).
ctypes.CDLL releases the GIL while the native bounded OpenMP team runs. Neither
the operator nor its startup loader changes Torch's global CPU thread settings.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import torch

_LIBRARY = None
_LOCK = threading.Lock()
_BACKEND = None


def _build_library():
    source = Path(__file__).with_suffix(".cpp")
    compiler = shlex.split(os.environ.get("CXX", ""))
    if not compiler:
        compiler = [shutil.which("c++") or shutil.which("g++") or "c++"]
    version = subprocess.check_output([*compiler, "--version"], timeout=30)
    flags = [
        "-std=c++17",
        "-O3",
        "-shared",
        "-fPIC",
        "-fopenmp",
        "-fno-fast-math",
        "-ffp-contract=off",
    ]
    digest = hashlib.sha256(source.read_bytes() + version)
    digest.update(repr((compiler, flags, platform.machine())).encode())
    base = Path(
        os.environ.get("RTP_ENGRAM_CPU_CACHE_DIR")
        or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "rtp_llm/engram_cpu"
    )
    cache = base / digest.hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / "engram_cpu.so"
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not library.is_file():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                output = Path(temporary) / library.name
                with (cache / "build.log").open("w") as log:
                    process = subprocess.run(
                        [*compiler, *flags, str(source), "-o", str(output)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=120,
                    )
                if process.returncode:
                    raise RuntimeError(
                        f"Engram CPU compilation failed; see {cache / 'build.log'}"
                    )
                os.replace(output, library)
    return library


def prepare_cpu_lookup():
    """Startup only. Compile/load once, initialize the small immutable LUT."""
    global _LIBRARY, _BACKEND
    with _LOCK:
        if _LIBRARY is None:
            library = ctypes.CDLL(str(_build_library()))
            library.engram_cpu_prepare.argtypes = []
            library.engram_cpu_prepare.restype = ctypes.c_int
            library.engram_cpu_lookup.argtypes = (
                [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 5 + [ctypes.c_int]
            )
            library.engram_cpu_lookup.restype = ctypes.c_int
            backend = library.engram_cpu_prepare()
            if backend not in (1, 2):
                raise RuntimeError(f"Engram CPU preparation failed: {backend}")
            _BACKEND = "avx512" if backend == 2 else "scalar"
            _LIBRARY = library
    return True


def _cpu_tensor(tensor):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.device.type == "cpu"
        and tensor.layout == torch.strided
        and not tensor.requires_grad
        and not tensor.is_conj()
        and not tensor.is_neg()
    )


def is_supported(weight, scales, indices, out, *, num_threads=16):
    """Metadata-only qualification; inputs must remain alive until return."""
    if not (
        type(num_threads) is int
        and 1 <= num_threads <= 16
        and all(_cpu_tensor(t) for t in (weight, scales, indices, out))
        and weight.dtype in (torch.uint8, torch.float8_e4m3fn)
        and scales.dtype == torch.uint8
        and weight.ndim == scales.ndim == indices.ndim == out.ndim == 2
        and indices.dtype == torch.int64
        and out.dtype == torch.bfloat16
        and indices.is_contiguous()
        and out.is_contiguous()
    ):
        return False
    n, dim = weight.shape
    if not (
        0 < dim <= 4096
        and dim % 32 == 0
        and scales.shape == (n, dim // 32)
        and weight.stride(1) == scales.stride(1) == 1
        and weight.stride(0) >= dim
        and scales.stride(0) >= dim // 32
        and out.shape == (indices.shape[0], indices.shape[1] * dim)
    ):
        return False
    output_storage = out.untyped_storage().data_ptr()
    return all(
        not t.numel() or t.untyped_storage().data_ptr() != output_storage
        for t in (weight, scales, indices)
    )


def lookup_host_rows_cpu(weight, scales, indices, out, *, num_threads=16):
    """Synchronously fill caller-owned output, without compilation/allocation."""
    if _LIBRARY is None:
        raise RuntimeError("Engram CPU lookup requires startup prepare_cpu_lookup")
    if not is_supported(weight, scales, indices, out, num_threads=num_threads):
        raise ValueError("Unsupported Engram CPU lookup tensor geometry or threads")
    status = _LIBRARY.engram_cpu_lookup(
        weight.data_ptr(),
        scales.data_ptr(),
        indices.data_ptr(),
        out.data_ptr(),
        weight.shape[0],
        weight.shape[1],
        weight.stride(0),
        scales.stride(0),
        indices.numel(),
        num_threads,
    )
    if status:
        raise RuntimeError(f"Engram CPU lookup failed: {status}")
    return out
