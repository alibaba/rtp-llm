"""Startup-warmed candidate-only FP32 K2048 selection on CUDA major 10.

CUDA major 10 defaults to the runtime domain: 1..8192 rows and 2049..16384
FP32 scores per row, with K=2048, after startup compilation and a semantic
check against the running Torch wheel. Other devices/shapes retain fallback.

try_select never compiles or reads device data. Unsupported/cold calls return
None before allocation or output writes. warmup owns the fixed CUDA variants.
Graph callers retain captured input/output tensors through the graph lifetime;
forward-local scratch allocated during capture belongs to the graph pool.
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
_AUTO_READY = {}
# cudaErrorInvalidDeviceFunction / InvalidKernelImage / NoKernelImageForDevice.
_PREPARE_IMAGE_ERRORS = frozenset((98, 200, 209))
_LIBRARY = None
_LOCK = threading.Lock()
_SOURCE = Path(__file__).resolve().parent
_NATIVE = Path(__file__).resolve().parents[3] / "bindings/cuda/kernels"
_TORCH_HEADERS = (
    "ATen/native/cuda/SortingRadixSelect.cuh",
    "ATen/cuda/ScanUtils.cuh",
)
STATUS = {
    1: "all_coarse_ties",
    2: "small32",
    3: "small64",
    4: "small128",
    5: "radix",
    7: "exception_exact_count_le_k",
    -1: "edge_cutoff",
    -2: "coarse_overflow",
    -3: "final_key_tie",
    -4: "count_mismatch",
}


def _enabled():
    return os.environ.get("DSV41_CANDIDATE_NATIVE_TOPK", "auto") in ("1", "auto")


def _shape_supported(rows, width, k):
    return (
        type(rows) is int
        and type(width) is int
        and type(k) is int
        and k == 2048
        and 1 <= rows <= 8192
        and 2048 < width <= 16384
    )


def _tensor_supported(tensor, dtype, device=None):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.layout == torch.strided
        and tensor.is_cuda
        and tensor.dtype == dtype
        and not tensor.requires_grad
        and (device is None or tensor.device == device)
    )


def is_supported(scores, out, *, k=2048, scratch=None):
    """Metadata-only gate; scratch is optional contiguous int32[M] row status."""
    if not (
        _tensor_supported(scores, torch.float32)
        and scores.ndim == 2
        and _shape_supported(scores.shape[0], scores.shape[1], k)
        and scores.stride(1) == 1
        and scores.shape[1] <= scores.stride(0) <= (2**63 - 1) // scores.shape[0] // 4
        and _tensor_supported(out, torch.int32, scores.device)
        and out.ndim == 2
        and out.shape[0] == scores.shape[0]
        and 2048 <= out.shape[1] <= 4096
        and out.stride(1) == 1
        and out.shape[1] <= out.stride(0) <= (2**63 - 1) // out.shape[0] // 4
    ):
        return False
    tensors = [scores, out]
    if scratch is not None:
        if not (
            _tensor_supported(scratch, torch.int32, scores.device)
            and scratch.shape == (scores.shape[0],)
            and scratch.is_contiguous()
        ):
            return False
        tensors.append(scratch)
    return len({t.untyped_storage().data_ptr() for t in tensors}) == len(tensors)


def _find_nvcc():
    """CPU-only discovery for explicit startup compilation."""
    cuda = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda:
        return str(Path(cuda) / "bin/nvcc")
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    standard = Path("/usr/local/cuda/bin/nvcc")
    return (
        str(standard) if standard.is_file() and os.access(standard, os.X_OK) else None
    )


def _build_library():
    """CPU-only compilation; called exclusively from explicit startup warmup."""
    if torch.version.cuda is None or torch.version.hip is not None:
        return None
    include = Path(torch.__file__).resolve().parent / "include"
    if not all((include / name).is_file() for name in _TORCH_HEADERS):
        return None
    nvcc = _find_nvcc()
    cxx = os.environ.get("CXX") or shutil.which("c++")
    if not nvcc or not cxx:
        return None
    version = subprocess.check_output([nvcc, "--version"])
    match = re.search(rb"release (\d+)\.", version)
    if match is None or int(match[1]) < 13:
        return None
    cuda = Path(nvcc).resolve().parent.parent
    sources = [
        _SOURCE / "_v41_candidate_topk.cu",
        _SOURCE / "_v41_candidate_topk_sb.cuh",
        _SOURCE / "_v41_candidate_topk_impl.cuh",
        _NATIVE / "topk_v3_compat.cuh",
    ]
    digest = hashlib.sha256(platform.machine().encode() + version)
    digest.update(subprocess.check_output([cxx, "--version"]))
    digest.update((str(cuda) + str(include) + torch.__version__).encode())
    for source in sources:
        digest.update(source.name.encode() + source.read_bytes())
    # Installed Torch header dependencies participate in cache identity, too.
    for directory in (include / "ATen", include / "c10"):
        for path in sorted(directory.rglob("*")):
            if path.suffix in (".h", ".cuh", ".hpp"):
                digest.update(
                    str(path.relative_to(include)).encode() + path.read_bytes()
                )
    flags = [
        "-shared",
        "-Xcompiler=-fPIC",
        "-O3",
        "-std=c++20",
        "--gpu-architecture=sm_100",
        "--expt-relaxed-constexpr",
        "--ptxas-options=-v",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    digest.update(repr(flags).encode())
    cache = (
        Path(
            os.environ.get("DG_JIT_CACHE_DIR")
            or os.environ.get("DEEP_GEMM_CACHE_DIR")
            or Path.home() / ".deep_gemm/cache"
        )
        / "rtp_v41_candidate_topk"
        / digest.hexdigest()[:24]
    )
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / "candidate.so"
    with (cache / "build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not library.is_file():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                output = Path(temporary) / library.name
                command = [
                    nvcc,
                    str(sources[0]),
                    "-o",
                    str(output),
                    "-ccbin",
                    cxx,
                    *flags,
                    "-I" + str(_NATIVE),
                    "-I" + str(include),
                    "-L" + str(cuda / "lib64/stubs"),
                    "-lcuda",
                ]
                with (cache / "build.log").open("w") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT
                    )
                if result.returncode:
                    raise RuntimeError(
                        f"Candidate TopK JIT failed; see {cache / 'build.log'}"
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
        lib.v41_candidate_topk_launch.argtypes = (
            [ctypes.c_void_p] * 3
            + [ctypes.c_int] * 3
            + [ctypes.c_int64] * 2
            + [ctypes.c_void_p]
        )
        lib.v41_candidate_topk_launch.restype = ctypes.c_int
        lib.v41_candidate_topk_prepare.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.v41_candidate_topk_prepare.restype = ctypes.c_int
        lib.v41_candidate_topk_is_current.argtypes = [ctypes.c_void_p]
        lib.v41_candidate_topk_is_current.restype = ctypes.c_int
        _LIBRARY = lib
    return _LIBRARY


def _execute(lib, scores, out, scratch):
    stream = torch.cuda.current_stream(scores.device)
    try:
        error = lib.v41_candidate_topk_launch(
            scores.data_ptr(),
            out.data_ptr(),
            scratch.data_ptr(),
            scores.shape[0],
            scores.shape[1],
            out.shape[1],
            scores.stride(0),
            out.stride(0),
            stream.cuda_stream,
        )
        if error:
            raise RuntimeError(f"Candidate TopK launch failed: CUDA {error}")
    finally:
        for tensor in (scores, out, scratch):
            tensor.record_stream(stream)
    return out


def try_select(scores, out, *, k=2048, scratch=None):
    """Return out, or None for caller fallback. Never build/load on this path.

    Optional scratch is 4*M bytes, at most 32KiB, with complete per-call writes.
    No device tensor is cached. A negative row status means same-CTA SB ran.
    """
    if not _enabled() or not is_supported(scores, out, k=k, scratch=scratch):
        return None
    entry = _READY.get(scores.device.index)
    if entry is None:
        return None
    if os.environ.get(
        "DSV41_CANDIDATE_NATIVE_TOPK", "auto"
    ) != "1" and not _AUTO_READY.get(scores.device.index, False):
        return None
    lib, context = entry
    with torch.cuda.device(scores.device):
        if not lib.v41_candidate_topk_is_current(context):
            return None
        if scratch is None:
            scratch = torch.empty(
                (scores.shape[0],), dtype=torch.int32, device=scores.device
            )
        return _execute(lib, scores, out, scratch)


def _warmup_scores(width, device):
    # Build deterministic edge cases on the CPU, then copy once at startup.
    # Repeated rows exercise Torch's multi-block top-k path as well as both
    # register widths in our kernel. No RNG state or forward buffers are used.
    rows = torch.full((16, width), -torch.inf, dtype=torch.float32)
    rows[1].zero_()
    rows[2].zero_()
    rows[2, ::2] = -0.0
    rows[3].fill_(-1e30)
    rows[4].fill_(torch.inf)
    rows[5].fill_(1.0)
    rows[5, :2047] = 2.0
    rows[6] = torch.arange(width, dtype=torch.float32).remainder(4093)
    rows[7, :2047] = 1.0
    rows[7, 2047] = torch.nan
    for index, count in enumerate((2047, 2048, 2049)):
        rows[8 + index].fill_(1.0)
        rows[8 + index, :count] = torch.nan
        rows[11 + index, :count] = torch.inf
    rows[14].fill_(-0.500244140625)
    rows[15].fill_(-torch.finfo(torch.float32).max)
    return rows.repeat(16, 1).to(device=device)


def _semantic_matches(samples):
    # Torch does not guarantee tie-index stability across implementations.
    # Validate against the running wheel instead of pinning its release/hash.
    expected = []
    for scores, _ in samples:
        values, ids = torch.topk(scores, 2048, dim=-1, sorted=False)
        expected.append(torch.where(values > -torch.inf, ids, -1).int())
    actual = torch.cat([out for _, out in samples]).sort(dim=-1).values
    reference = torch.cat(expected).sort(dim=-1).values
    return torch.equal(actual, reference)


@torch.inference_mode()
def warmup(device):
    """Compile/load and prime qualified variants outside graph capture."""
    if not _enabled() or torch.version.cuda is None or torch.version.hip is not None:
        return False
    device = torch.device(device)
    if device.type != "cuda":
        return False
    with torch.cuda.device(device), _LOCK:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Candidate TopK warmup must precede graph capture")
        device = torch.device("cuda", torch.cuda.current_device())
        prop = torch.cuda.get_device_properties(device)
        capability = (prop.major, prop.minor)
        if capability[0] != 10:
            return False
        if device.index in _READY:
            lib, context = _READY[device.index]
            return bool(lib.v41_candidate_topk_is_current(context))
        try:
            lib = _load()
        except (OSError, subprocess.SubprocessError, RuntimeError):
            if os.environ.get("DSV41_CANDIDATE_NATIVE_TOPK", "auto") != "auto":
                raise
            logging.exception(
                "Candidate TopK JIT unavailable; retaining existing selection"
            )
            return False
        if lib is None:
            logging.warning("Candidate TopK unavailable; retaining existing selection")
            return False
        context = ctypes.c_void_p()
        error = lib.v41_candidate_topk_prepare(ctypes.byref(context))
        if (
            error in _PREPARE_IMAGE_ERRORS
            and os.environ.get("DSV41_CANDIDATE_NATIVE_TOPK", "auto") == "auto"
        ):
            logging.warning(
                "Candidate TopK image unavailable (CUDA %s); retaining existing selection",
                error,
            )
            return False
        if error or not context.value:
            raise RuntimeError(f"Candidate TopK prepare failed: CUDA {error}")
        # Prime both register-width families and aligned/unaligned inputs. Row
        # count is only grid.x; runtime lengths/strides add no CUDA variants.
        samples = []
        for width in (4096, 4099, 8196, 8199):
            scores = _warmup_scores(width, device)
            out = torch.empty((256, 2048), dtype=torch.int32, device=device)
            scratch = torch.empty((256,), dtype=torch.int32, device=device)
            _execute(lib, scores, out, scratch)
            samples.append((scores, out))
        torch.cuda.current_stream(device).synchronize()
        if not _semantic_matches(samples):
            logging.warning(
                "Candidate TopK semantics differ from torch.topk (%s, CUDA %s); "
                "retaining existing selection without publishing ready state",
                torch.__version__,
                torch.version.cuda,
            )
            return False
        _AUTO_READY[device.index] = True
        _READY[device.index] = (lib, context)
    return True
