"""DeepSeek-V4 quant constants + activation cast helpers.

Two block sizes used across the V4 MoE pipeline:
  - ``FP8_BLOCK = 128``: per-token-group block size for FP8 (E4M3) activation
    quantization (uses UE8M0 scale-factor packing on SM100).
  - ``FP4_BLOCK = 32``: per-row block size for FP4 weight scale factors
    (DeepGEMM ``m_grouped_fp8_fp4_*`` recipe).

Plus ``_per_token_cast_to_fp8_packed_ue8m0``: a CUDA-graph-safe replacement
for ``deep_gemm.utils.per_token_cast_to_fp8(use_ue8m0=True, use_packed_ue8m0=True)``
(the upstream helper does a ``.all()`` debug assertion that triggers a
CUDA->CPU sync illegal during stream capture).
"""

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional, Tuple

import torch

FP4_BLOCK = 32
FP8_BLOCK = 128


def _deep_gemm_process_rank() -> int:
    """Return a stable process rank for rank-local DeepGEMM JIT scratch."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    for env_name in ("RANK", "LOCAL_RANK"):
        try:
            return int(os.environ[env_name])
        except (KeyError, ValueError):
            pass
    return os.getpid()


def _deep_gemm_rank_nvcc_tmpdir(rank: Optional[int] = None) -> str:
    """Return a stable per-rank temp directory for DeepGEMM's nvcc JIT."""

    base_dir = (
        os.environ.get("DSV4_MEGA_MOE_NVCC_TMPDIR")
        or os.environ.get("DG_JIT_CACHE_DIR")
        or os.environ.get("TRITON_CACHE_DIR")
        or "/tmp"
    )
    process_rank = _deep_gemm_process_rank() if rank is None else int(rank)
    return os.path.join(
        base_dir,
        "rtp_llm_dsv4_mega_moe_nvcc",
        f"rank_{process_rank}",
    )


@contextmanager
def _activate_deep_gemm_rank_nvcc_tmpdir() -> Iterator[str]:
    """Isolate nvcc scratch so one rank cannot remove another rank's files."""

    previous_tmpdir = os.environ.get("TMPDIR")
    tmpdir = _deep_gemm_rank_nvcc_tmpdir()
    try:
        os.makedirs(tmpdir, exist_ok=True)
    except OSError:
        tmpdir = os.path.join(
            "/tmp",
            "rtp_llm_dsv4_mega_moe_nvcc",
            f"rank_{_deep_gemm_process_rank()}",
        )
        os.makedirs(tmpdir, exist_ok=True)
    os.environ["TMPDIR"] = tmpdir
    try:
        yield tmpdir
    finally:
        if previous_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = previous_tmpdir


def prepare_fp4_weight_scale_for_deepgemm(
    scale: torch.Tensor,
    mn: int,
    k: int,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    """Convert V4 FP4 UE8M0 weight scale to DeepGEMM's SM100 layout.

    Routed expert checkpoints store weight scale as raw UE8M0
    ``float8_e8m0fnu``. DeepGEMM's FP8xFP4 kernels on SM100 consume the
    TMA-aligned packed ``int32`` layout. Do this once while binding weights,
    not in the GEMM hot path.
    """
    if scale.dtype == torch.int32:
        return scale
    if scale.dtype != torch.float8_e8m0fnu:
        raise TypeError(f"expected FP4 UE8M0 scale, got {scale.dtype}")

    os.environ.setdefault(
        "DG_JIT_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), f"deep_gemm_jit_{os.getuid()}"),
    )
    os.makedirs(os.environ["DG_JIT_CACHE_DIR"], exist_ok=True)

    import deep_gemm

    scale_fp32 = scale.float()
    with _activate_deep_gemm_rank_nvcc_tmpdir():
        if num_groups is None:
            return deep_gemm.transform_sf_into_required_layout(
                scale_fp32, mn, k, (1, FP4_BLOCK)
            )
        return deep_gemm.transform_sf_into_required_layout(
            scale_fp32, mn, k, (1, FP4_BLOCK), num_groups
        )


def _per_token_cast_to_fp8_packed_ue8m0(
    x: torch.Tensor,
    gran_k: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inline ``deep_gemm.utils.per_token_cast_to_fp8(use_ue8m0=True,
    use_packed_ue8m0=True)`` without the ``pack_ue8m0_to_int`` ``.all()``
    debug assertion. That assertion does a CUDA->CPU sync, which is illegal
    during ``cudaStreamCapture``.
    """
    assert x.dim() == 2, f"expected 2D input, got {x.shape}"
    m, n = x.shape
    padded_n = ((n + gran_k - 1) // gran_k) * gran_k
    if padded_n != n:
        x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
        x_padded[:, :n] = x
    else:
        x_padded = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / 448.0
    bits = sf.abs().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    sf_u = (exp.clamp(1, 254) << 23).view(torch.float)
    x_fp8 = (
        (x_view * (1.0 / sf_u.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    sf_packed = (sf_u.view(torch.int) >> 23).to(torch.uint8).view(torch.int)
    return x_fp8, sf_packed
