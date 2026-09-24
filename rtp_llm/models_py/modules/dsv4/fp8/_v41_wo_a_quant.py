"""Startup-only CUDA JIT for wo_a BF16-round/group32 FP8 output on SM100.

No compilation, CUDA initialization, or vendor-library import occurs at import.
Unsupported or unprepared calls return None so the caller retains its BF16 path.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import torch

_READY = {}
_LIBRARIES = {}
_LOCK = threading.Lock()
_SOURCES = ("_v41_wo_a_quant.cu", "_v41_wo_a_quant.cuh", "_v41_wo_a_driver.cc")
# The shim and two schedules depend on this private DeepGEMM ABI. A different
# implementation falls back until its epilogue/layout has been verified.
_VENDOR_HASHES = {
    "impls/sm100_fp8_fp4_gemm_1d1d.cuh": "0b42d290fbcaea544a7b1c2c786429e52453fc60a74a939c20fa73ce960abf69",
    "epilogue/sm100_store_cd.cuh": "34264511eadb165477cfd92059e03f7cc7dbb5b694298f7e4e9f1074294dc4ad",
    "epilogue/sm100_store_cd_swap_ab.cuh": "77c280afab2a284d7a105323f3a878549d1e7f358ab0a28190c43958aed247d2",
    "common/math.cuh": "576e9db1a1889087f77b3f74d74d011bf452f16c0e5c030bdd661e961c074ddc",
}


def _device_supported(device):
    if device.type != "cuda":
        return False
    prop = torch.cuda.get_device_properties(device)
    # These two schedules have been validated on the 152-SM SM100 device.
    return (prop.major, prop.minor, prop.multi_processor_count) == (10, 0, 152)


def _weight_supported(weight, scale):
    return (
        weight.is_cuda
        and weight.shape == (8, 1024, 4096)
        and weight.dtype == torch.float8_e4m3fn
        and weight.is_contiguous()
        and scale.device == weight.device
        and scale.dtype == torch.int32
        and scale.shape == (8, 1024, 32)
        and scale.stride() == (32768, 1, 1024)
        and all(not t.requires_grad and t.data_ptr() % 16 == 0 for t in (weight, scale))
        and _device_supported(weight.device)
    )


def is_supported(a, weight):
    aq, sa = a
    w, sw = weight
    if aq.ndim != 3 or not 0 < aq.shape[0] < 2**31 - 128:
        return False
    m = aq.shape[0]
    stride = (m + 3) // 4 * 4
    return (
        aq.device == sa.device == w.device
        and aq.dtype == torch.float8_e4m3fn
        and aq.shape == (m, 8, 4096)
        and aq.stride() == (4096, m * 4096, 1)
        and sa.dtype == torch.int32
        and sa.shape == (m, 8, 32)
        and sa.stride() == (1, stride * 32, stride)
        and all(not t.requires_grad and t.data_ptr() % 16 == 0 for t in (aq, sa))
        and _weight_supported(w, sw)
    )


def _legacy(m, mode):
    mode = (
        (os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto") if mode is None else mode)
        .strip()
        .lower()
    )
    if mode not in ("auto", "legacy", "v2"):
        raise ValueError("DSV4_FP8_QUANT_KERNEL must be auto, legacy, or v2")
    return mode == "legacy" or (mode == "auto" and m * 8192 < 4 * 1024 * 1024)


def _dependencies():
    import deep_gemm

    include = Path(deep_gemm.__file__).resolve().parent / "include"
    for name, expected in _VENDOR_HASHES.items():
        path = include / "deep_gemm" / name
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            return None
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(cuda) / "bin/nvcc") if cuda else shutil.which("nvcc")
    cxx = os.environ.get("CXX") or shutil.which("c++")
    if not nvcc or not cxx or not Path(nvcc).is_file():
        return None
    version = subprocess.check_output([nvcc, "--version"], text=True)
    release = re.search(r"release (\d+)\.(\d+)", version)
    if release is None or int(release[1]) < 13:
        return None
    return include, Path(nvcc).resolve().parent.parent, nvcc, cxx


def _build_library(sms, dependencies):
    include, cuda, nvcc, cxx = dependencies
    source = Path(__file__).resolve().parent
    digest = hashlib.sha256(str(sms).encode())
    # driver.so is host code: shared caches must not mix ARM and x86 builds
    # even when CUDA, compiler version strings and GPU geometry are identical.
    digest.update(platform.machine().encode())
    for name in _SOURCES:
        digest.update((source / name).read_bytes())
    # Include all transitive vendor headers in the disk-cache identity.
    for path in sorted(include.rglob("*")):
        if path.suffix in (".h", ".hpp", ".cuh"):
            digest.update(str(path.relative_to(include)).encode())
            digest.update(path.read_bytes())
    for compiler in (nvcc, cxx):
        digest.update(subprocess.check_output([compiler, "--version"]))
    cache = Path(
        os.environ.get("DG_JIT_CACHE_DIR")
        or os.environ.get("DEEP_GEMM_CACHE_DIR")
        or Path.home() / ".deep_gemm/cache"
    )
    cache = cache / "rtp_v41_wo_a" / digest.hexdigest()[:24]
    cache.mkdir(parents=True, exist_ok=True)
    library, cubin = cache / "driver.so", cache / "kernel.cubin"
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not (library.is_file() and cubin.is_file()):
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                tmp = Path(temporary)
                commands = [
                    [
                        nvcc,
                        str(source / _SOURCES[0]),
                        "--cubin",
                        "-o",
                        str(tmp / cubin.name),
                        "--gpu-architecture=sm_100f",
                        "-O3",
                        "-std=c++20",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "--ptxas-options=--register-usage-level=10",
                        "--ptxas-options=--verbose",
                        "--diag-suppress=39,161,174,177,186,940",
                        f"-DV41_NUM_SMS={sms}",
                        "-I" + str(include),
                    ],
                    [
                        cxx,
                        str(source / _SOURCES[2]),
                        "-o",
                        str(tmp / library.name),
                        "-shared",
                        "-fPIC",
                        "-O2",
                        "-std=c++17",
                        "-I" + str(cuda / "include"),
                        "-L" + str(cuda / "lib64/stubs"),
                        "-lcuda",
                    ],
                ]
                with (cache / "build.log").open("w") as log:
                    for command in commands:
                        completed = subprocess.run(
                            command, stdout=log, stderr=subprocess.STDOUT
                        )
                        if completed.returncode:
                            raise RuntimeError(
                                f"V41 wo_a JIT failed; see {cache / 'build.log'}"
                            )
                os.replace(tmp / cubin.name, cubin)
                os.replace(tmp / library.name, library)
    return library, cubin


def _load(device):
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if sms not in _LIBRARIES:
        dependencies = _dependencies()
        if dependencies is None:
            logging.info(
                "V41 wo_a quant epilogue unavailable; retaining BF16 projection"
            )
            return None
        library, cubin = _build_library(sms, dependencies)
        lib = ctypes.CDLL(str(library))
        lib.v41_wo_a_init.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.v41_wo_a_init.restype = ctypes.c_int
        lib.v41_wo_a_is_current.argtypes = [ctypes.c_void_p]
        lib.v41_wo_a_is_current.restype = ctypes.c_int
        lib.v41_wo_a_launch.argtypes = [ctypes.c_void_p] * 7 + [
            ctypes.c_uint,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        lib.v41_wo_a_launch.restype = ctypes.c_int
        _LIBRARIES[sms] = (lib, cubin)
    lib, cubin = _LIBRARIES[sms]
    handle = ctypes.c_void_p()
    status = lib.v41_wo_a_init(os.fsencode(cubin), device.index, ctypes.byref(handle))
    if status:
        raise RuntimeError(f"V41 wo_a module initialization failed: CUDA {status}")
    return lib, handle


def _output(m, device):
    q = torch.empty((m, 8192), dtype=torch.float8_e4m3fn, device=device)
    s = torch.empty((64, (m + 3) // 4 * 4), dtype=torch.int32, device=device).T[:m]
    return q, s


def is_ready(weight, scale):
    """Check metadata and current context; never compile/load a missing module."""
    entry = _READY.get(weight.device.index)
    if entry is None or not _weight_supported(weight, scale):
        return False
    return bool(entry[0].v41_wo_a_is_current(entry[1]))


def _validate_output(out, a, weight):
    q, s = out
    m, device = a[0].shape[0], a[0].device
    if not (
        q.device == s.device == device
        and q.dtype == torch.float8_e4m3fn
        and q.shape == (m, 8192)
        and q.is_contiguous()
        and s.dtype == torch.int32
        and s.shape == (m, 64)
        and s.stride() == (1, (m + 3) // 4 * 4)
        and all(not t.requires_grad and t.data_ptr() % 16 == 0 for t in out)
    ):
        raise ValueError(
            "expected disjoint aligned CUDA E4M3 [M,8192] and packed int32 [M,64]"
        )
    if torch._C._overlaps(q, s) or any(
        torch._C._overlaps(t, x) for t in out for x in (*a, *weight)
    ):
        raise ValueError("quantized output must not overlap inputs, weights or scales")


def try_grouped_quant(a, weight, *, out=None, quant_kernel=None):
    """Return caller-owned (Q, packed scales), or None; never compile here.

    Inputs are finite. Cold capture and unsupported layouts keep the BF16 path.
    The existing quantization policy is evaluated on each eager/capture call.
    """
    if not is_supported(a, weight):
        return None
    aq, sa = a
    entry = _READY.get(aq.device.index)
    if entry is None:
        return None
    lib, handle = entry
    with torch.cuda.device(aq.device):
        if not lib.v41_wo_a_is_current(handle):
            return None
        mode = _legacy(aq.shape[0], quant_kernel)
        if out is None:
            out = _output(aq.shape[0], aq.device)
        else:
            _validate_output(out, a, weight)
        stream = torch.cuda.current_stream(aq.device)
        status = lib.v41_wo_a_launch(
            handle,
            *(t.data_ptr() for t in (*a, *weight, *out)),
            aq.shape[0],
            aq.stride(0),
            aq.stride(1),
            mode,
            stream.cuda_stream,
        )
        if status:
            raise RuntimeError(f"V41 wo_a quant epilogue failed: CUDA {status}")
        # External CUDA launches need allocator lifetime tracking on non-default streams.
        for tensor in (*a, *weight, *out):
            tensor.record_stream(stream)
    return out


@torch.inference_mode()
def warmup(weight, scale):
    """Compile/load two fixed kernels once, then launch both with private inputs."""
    if not _weight_supported(weight, scale):
        return False
    device = weight.device
    with torch.cuda.device(device), _LOCK:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V41 wo_a JIT warmup must precede CUDA graph capture")
        if device.index in _READY:
            # Retain one warmed context per device; other contexts use BF16.
            lib, handle = _READY[device.index]
            return bool(lib.v41_wo_a_is_current(handle))
        entry = _load(device)
        if entry is None:
            return False
        _READY[device.index] = entry
        try:
            for m in (128, 256):
                q = torch.zeros(
                    (8, m, 4096), dtype=torch.float8_e4m3fn, device=device
                ).transpose(0, 1)
                s = (
                    torch.full((8, 32, m), 0x7F7F7F7F, dtype=torch.int32, device=device)
                    .transpose(1, 2)
                    .transpose(0, 1)
                )
                try_grouped_quant((q, s), (weight, scale))
            torch.cuda.current_stream(device).synchronize()
        except BaseException:
            _READY.pop(device.index, None)
            raise
    return True
