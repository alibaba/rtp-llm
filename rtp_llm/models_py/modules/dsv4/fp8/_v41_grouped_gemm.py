"""Opt-in, startup-warmed variable-M BF16 index GEMM for SM100 prefill.

Call warmup(weight) at startup, then try_grouped_index_gemm(x, weight, rows).
Rows are host integers partitioning a contiguous [sum(rows), 512] tensor.
Each GEMM retains its own M and produces BF16 [M, 128], using FP32 accumulation.
Unsupported/cold/captured calls return None before enqueuing work. The caller
must provide its original per-segment fallback. Eager current streams are
supported; callers establish input dependencies before entering a side stream.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import torch

_MAX_SEGMENTS = 1024
_MAX_ROWS = 65536
_READY = {}
_LIBRARY = None
_LOCK = threading.Lock()


def _enabled():
    return os.environ.get("DSV41_GROUPED_INDEX_GEMM", "0") == "1"


def _rows_supported(segment_rows, total):
    return (
        isinstance(segment_rows, (tuple, list))
        and len(segment_rows) <= _MAX_SEGMENTS
        and all(type(m) is int and 0 <= m <= _MAX_ROWS for m in segment_rows)
        and sum(segment_rows) == total
        and 0 <= total <= _MAX_ROWS
    )


def _weight_supported(weight):
    return (
        isinstance(weight, torch.Tensor)
        and weight.is_cuda
        and weight.shape == (128, 512)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and not weight.requires_grad
        and weight.data_ptr() % 16 == 0
    )


def is_supported(x, weight, segment_rows):
    """Metadata-only gate; a Tensor of segment lengths is deliberately rejected."""
    return (
        isinstance(x, torch.Tensor)
        and x.ndim == 2
        and x.shape[1] == 512
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and not x.requires_grad
        and x.data_ptr() % 16 == 0
        and _weight_supported(weight)
        and x.device == weight.device
        and _rows_supported(segment_rows, x.shape[0])
    )


def _build_library():
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(cuda) / "bin/nvcc") if cuda else shutil.which("nvcc")
    cxx = os.environ.get("CXX") or shutil.which("c++")
    if not nvcc or not cxx:
        return None
    cuda = Path(nvcc).resolve().parent.parent
    if not (cuda / "include/cublas_v2.h").is_file():
        return None
    source = Path(__file__).with_suffix(".cc")
    digest = hashlib.sha256(source.read_bytes())
    digest.update(platform.machine().encode())
    digest.update(str(cuda).encode())
    digest.update(subprocess.check_output([cxx, "--version"]))
    digest.update(subprocess.check_output([nvcc, "--version"]))
    cache = (
        Path(
            os.environ.get("DG_JIT_CACHE_DIR")
            or os.environ.get("DEEP_GEMM_CACHE_DIR")
            or Path.home() / ".deep_gemm/cache"
        )
        / "rtp_v41_grouped_index"
        / digest.hexdigest()[:24]
    )
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / "driver.so"
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not library.is_file():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                output = Path(temporary) / library.name
                command = [
                    cxx,
                    str(source),
                    "-shared",
                    "-fPIC",
                    "-O2",
                    "-std=c++17",
                    "-I" + str(cuda / "include"),
                    "-L" + str(cuda / "lib64"),
                    "-lcublas",
                    "-o",
                    str(output),
                ]
                with (cache / "build.log").open("w") as log:
                    completed = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT
                    )
                if completed.returncode:
                    raise RuntimeError(
                        f"V41 grouped index JIT failed; see {cache / 'build.log'}"
                    )
                os.replace(output, library)
    return library


def _load():
    global _LIBRARY
    if _LIBRARY is None:
        path = _build_library()
        if path is None:
            return None
        lib = ctypes.CDLL(str(path))
        lib.v41_grouped_index_gemm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
        ] + [ctypes.c_void_p] * 4
        lib.v41_grouped_index_gemm.restype = ctypes.c_int
        _LIBRARY = lib
    return _LIBRARY


def _validate_output(out, x, weight):
    if not (
        isinstance(out, torch.Tensor)
        and out.shape == (x.shape[0], 128)
        and out.device == x.device
        and out.dtype == torch.bfloat16
        and out.is_contiguous()
        and not out.requires_grad
        and out.data_ptr() % 16 == 0
    ):
        raise ValueError(
            "expected aligned contiguous CUDA BF16 output [sum(rows), 128]"
        )
    if torch._C._overlaps(out, x) or torch._C._overlaps(out, weight):
        raise ValueError("grouped index output must not overlap input or weight")


def _execute(lib, x, weight, rows, out):
    rows = tuple(m for m in rows if m)
    if not rows:
        return out
    starts, offset = [], 0
    for m in rows:
        starts.append(offset)
        offset += m
    # One bounded H2D copy, independent of distinct M. The pinned allocator
    # records copy completion; the CUDA allocator orders device-table reuse
    # after GEMM on this stream. No table or activation survives in a cache.
    host = torch.tensor(
        [
            [weight.data_ptr()] * len(rows),
            [x.data_ptr() + start * 512 * 2 for start in starts],
            [out.data_ptr() + start * 128 * 2 for start in starts],
        ],
        dtype=torch.int64,
        device="cpu",
        pin_memory=True,
    )
    pointers = host.to(x.device, non_blocking=True)
    host_rows = (ctypes.c_int * len(rows))(*rows)
    stream = torch.cuda.current_stream(x.device)
    stride = len(rows) * 8
    status = lib.v41_grouped_index_gemm(
        torch.cuda.current_blas_handle(),
        stream.cuda_stream,
        len(rows),
        ctypes.cast(host_rows, ctypes.c_void_p),
        pointers.data_ptr(),
        pointers.data_ptr() + stride,
        pointers.data_ptr() + 2 * stride,
    )
    # These tensors can have been allocated on a different stream. Track their
    # external use even if the library reports an asynchronous launch error.
    for tensor in (x, weight, out):
        tensor.record_stream(stream)
    if status:
        raise RuntimeError(f"V41 grouped index GEMM failed: cuBLAS status {status}")
    return out


def try_grouped_index_gemm(x, weight, segment_rows, *, out=None):
    """Return contiguous output, or None for caller fallback; never compile here.

    Graph capture always falls back before allocation or upload. The host/device
    pointer tables are forward-local and each is at most 24 KiB (1024 segments).
    Zero-row segments are allowed and skipped. Total rows must be <= 65536.
    """
    if not _enabled() or not is_supported(x, weight, segment_rows):
        return None
    lib = _READY.get(x.device.index)
    if lib is None:
        return None
    with torch.cuda.device(x.device):
        if torch.cuda.is_current_stream_capturing():
            return None
        if out is None:
            out = torch.empty((x.shape[0], 128), dtype=torch.bfloat16, device=x.device)
        else:
            _validate_output(out, x, weight)
        return _execute(lib, x, weight, segment_rows, out)


@torch.inference_mode()
def warmup(weight):
    """Compile and exercise the fixed index interface at startup, never capture."""
    if not _enabled() or not _weight_supported(weight):
        return False
    device = weight.device
    with torch.cuda.device(device), _LOCK:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("grouped index warmup must precede CUDA graph capture")
        props = torch.cuda.get_device_properties(device)
        if (props.major, props.minor) != (10, 0) or not torch.version.cuda:
            return False
        if int(torch.version.cuda.split(".")[0]) < 13:
            return False
        if device.index in _READY:
            return True
        lib = _load()
        if lib is None:
            logging.info("V41 grouped index unavailable; retaining per-segment GEMM")
            return False
        x = torch.zeros((24, 512), dtype=torch.bfloat16, device=device)
        out = torch.empty((24, 128), dtype=torch.bfloat16, device=device)
        _execute(lib, x, weight, (1, 7, 16), out)
        torch.cuda.current_stream(device).synchronize()
        _READY[device.index] = lib
    return True
