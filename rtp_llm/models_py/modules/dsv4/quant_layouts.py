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
from typing import Optional, Tuple

import torch

FP4_BLOCK = 32
FP8_BLOCK = 128


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
