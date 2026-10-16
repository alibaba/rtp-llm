"""Isolated SGLang DeepGEMM SM90 group32 dense candidate.

This module does not replace or patch the installed internal DeepGEMM package.
It preserves FP8 payloads and per-token / per-32x32-block FP32 scales exactly.
Call once outside CUDA graph capture to compile and warm each N/K/config.
"""

from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parent
_EXPECTED_DEEP_GEMM = "2.8.0+122e18b.cu132"
CONFIGS = ("m64n64", "m64n128", "m128n128")


def align_activation_scales(scales: torch.Tensor) -> torch.Tensor:
    """Lossless [M,K/32] FP32 -> M-major with a 16-byte aligned K stride.

    Physical padding rows are initialized to zero. No reduction, rounding or
    requantization is performed. Layout is the internal DeepGEMM FP32 TMA layout.
    """
    if scales.ndim != 2 or scales.dtype != torch.float32:
        raise ValueError("activation scales must be a 2-D FP32 tensor")
    m, groups = scales.shape
    if m <= 0 or groups <= 0:
        raise ValueError("activation scales must have positive dimensions")
    padded_m = (m + 3) // 4 * 4
    available = scales.untyped_storage().nbytes() - scales.storage_offset() * scales.element_size()
    if (scales.stride() == (1, padded_m) and scales.data_ptr() % 16 == 0
            and available >= padded_m * groups * 4):
        return scales
    storage = torch.zeros((groups, padded_m), dtype=scales.dtype, device=scales.device)
    result = storage.t()[:m]
    result.copy_(scales)
    return result


@lru_cache(maxsize=None)
def _device_properties(device: int):
    return torch.cuda.get_device_properties(device)


@lru_cache(maxsize=None)
def _extension(device: int, n: int, k: int):
    # Import lazily: normal Humming / SM100 execution must not build this sidecar.
    from torch.utils.cpp_extension import load

    spec = importlib.util.find_spec("deep_gemm")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("the pinned internal DeepGEMM package is required")
    version = importlib.metadata.version("deep-gemm")
    if version != _EXPECTED_DEEP_GEMM:
        raise RuntimeError(
            f"sidecar headers were audited against DeepGEMM {_EXPECTED_DEEP_GEMM}; "
            f"found {version}, revalidate before use"
        )
    include_dir = Path(next(iter(spec.submodule_search_locations))) / "include"
    for relative in ("cute/tensor.hpp", "deep_gemm/mma/sm90.cuh"):
        if not (include_dir / relative).is_file():
            raise RuntimeError(f"missing internal DeepGEMM device header: {relative}")
    sms = _device_properties(device).multi_processor_count
    digest = hashlib.sha256()
    for relative in ("binding.cpp", "launcher.cu", "vendor/sm90_fp8_gemm_1d2d.cuh"):
        digest.update((_ROOT / relative).read_bytes())
    digest.update(f"{version}:{torch.__version__}:{torch.version.cuda}:{device}:{n}:{k}:{sms}".encode())
    name = "rtp_sm90_dg_g32_" + digest.hexdigest()[:20]
    # torch's extension builder owns an isolated module-name cache, never the
    # installed DeepGEMM headers/csrc or its compiler cache. MAX_JOBS is respected.
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up the SM90 group32 sidecar before CUDA graph capture")
        return load(
            name=name,
            sources=[str(_ROOT / "binding.cpp"), str(_ROOT / "launcher.cu")],
            extra_include_paths=[str(include_dir), str(_ROOT)],
            extra_cflags=["-O2", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3", "-std=c++20", "--expt-relaxed-constexpr",
                "--expt-extended-lambda", "-lineinfo",
                "-gencode=arch=compute_90a,code=sm_90a",
                f"-DRTP_DG_N={n}", f"-DRTP_DG_K={k}", f"-DRTP_DG_SMS={sms}",
            ],
            extra_ldflags=["-lcuda"],
            with_cuda=True,
            verbose=False,
        )


def fp8_gemm_nt(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    config: str = "auto",
) -> torch.Tensor:
    """Write BF16 ``out = (A * As) @ (B * Bs).T`` without requantization.

    A/B: contiguous E4M3 [M,K]/[N,K]. As: FP32 [M,K/32], exact strides
    (1, ceil(M/4)*4). Bs: contiguous FP32 [N/32,K/32]. Initial bounded support:
    SM90, N divisible by 32, K divisible by 128, 128 <= K <= 16384.
    ``auto`` is a provisional tile choice, not a measured performance claim;
    explicit CONFIGS allow an acceptance-gated small/large-M comparison.
    """
    tensors = (a, a_scale, b, b_scale, out)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("all GEMM arguments must be torch tensors")
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("group32 dense GEMM only accepts 2-D tensors")
    if not a.is_cuda or any(tensor.device != a.device for tensor in tensors):
        raise ValueError("all GEMM tensors must be on the same CUDA device")
    m, k = a.shape
    n, b_k = b.shape
    if m <= 0 or n <= 0 or k < 128 or k > 16384 or n % 32 or k % 128 or b_k != k:
        raise ValueError("requires M,N>0, N%32=0, K%128=0 and 128<=K<=16384")
    if max(m * k, n * k, m * n) >= 2**31:
        raise ValueError("candidate bounds tensor indexing to fewer than 2^31 elements")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("A and B must preserve E4M3 FP8 payloads")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("A and B must be contiguous K-major matrices")
    if a_scale.dtype != torch.float32 or a_scale.shape != (m, k // 32):
        raise ValueError("A scales must be FP32 [M,K/32]")
    if a_scale.stride() != (1, (m + 3) // 4 * 4):
        raise ValueError("A scales require exact TMA-aligned M-major strides")
    available = a_scale.untyped_storage().nbytes() - a_scale.storage_offset() * 4
    if available < ((m + 3) // 4 * 4) * (k // 32) * 4:
        raise ValueError("A scales must physically allocate the final TMA padding rows")
    if (b_scale.dtype != torch.float32 or b_scale.shape != (n // 32, k // 32)
            or not b_scale.is_contiguous()):
        raise ValueError("B scales must be contiguous FP32 [N/32,K/32]")
    if out.dtype != torch.bfloat16 or out.shape != (m, n) or not out.is_contiguous():
        raise ValueError("output must be contiguous BF16 [M,N]")
    if any(tensor.data_ptr() % 16 for tensor in tensors):
        raise ValueError("all tensors require 16-byte aligned data pointers")
    if any(torch._C._overlaps(out, tensor) for tensor in tensors[:-1]):
        raise ValueError("output must not overlap any input")
    props = _device_properties(a.device.index)
    if (props.major, props.minor) != (9, 0):
        raise ValueError("this isolated candidate supports SM90 only")
    if config == "auto":
        config = "m64n64" if m <= 64 else "m128n128"
    if config not in CONFIGS:
        raise ValueError(f"unknown tile config {config!r}; expected auto or {CONFIGS}")
    module = _extension(a.device.index, n, k)
    module.fp8_gemm_nt(a, a_scale, b, b_scale, out, CONFIGS.index(config))
    return out


__all__ = ["CONFIGS", "align_activation_scales", "fp8_gemm_nt"]
