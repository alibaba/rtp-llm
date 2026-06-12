"""MXFP8 (1x32 microscaling FP8) primitives for MiniMax-M3.

Weights are e4m3 with a UE8M0 (uint8) ``weight_scale_inv`` on a fixed
``[1, 32]`` micro-block. Activations are dynamically quantized to e4m3 with a
per-(row, 32-col) UE8M0 scale. The GEMMs go through DeepGEMM's
``fp8_fp4_gemm_nt`` / ``m_grouped_fp8_fp4_gemm_nt_contiguous`` with
``recipe=(1, 32)`` (these handle FP8xFP8 with this recipe). SM100 only.
"""

import os
from typing import Optional, Tuple

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    fp8_fp4_gemm_nt,
    m_grouped_fp8_fp4_gemm_nt_contiguous,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    sgl_per_token_group_quant_fp8,
)

MX_BLOCK = 32
_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def ue8m0_uint8_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """On-disk UE8M0 (uint8 exponent, bias 127) -> fp32 power-of-two scale."""
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def _mxfp8_quant_act_v2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast MXFP8 activation quant using the v2 CUDA kernel with UE8M0 scales.

    Uses ``sgl_per_token_group_quant_fp8`` with ``scale_ue8m0=True`` to produce
    e4m3 activations and int32-packed UE8M0 scales in the exact layout that
    DeepGEMM expects (column-major, TMA-aligned). The output can be passed
    directly to ``fp8_fp4_gemm_nt`` / ``m_grouped_fp8_fp4_gemm_nt_contiguous``
    with ``disable_ue8m0_cast=True`` — no additional packing or conversion needed.

    This matches SGLang's approach: the v2 kernel's int32 output IS DeepGEMM's
    native scale format.

    Returns (e4m3 ``[M, K]``, int32 packed scale ``[M_padded, K // 128]``).
    """
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    assert x.dim() == 2, f"expected 2D activation, got {x.shape}"
    K = x.shape[1]
    assert K % MX_BLOCK == 0, f"K={K} must be a multiple of {MX_BLOCK}"
    assert x.is_contiguous(), "input must be contiguous"

    q, s_packed = sgl_per_token_group_quant_fp8(
        x,
        group_size=MX_BLOCK,
        scale_ue8m0=True,
        column_major_scales=True,
        scale_tma_aligned=True,
    )
    # s_packed is int32 packed UE8M0, column-major layout — directly consumable
    # by DeepGEMM with disable_ue8m0_cast=True
    return q, s_packed


def mxfp8_quant_act_eager(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-(row, 32-col) MXFP8 quant of a 2D activation.

    Returns (e4m3 ``[M, K]``, fp32 power-of-two scale ``[M, K // 32]``).
    The scale is a pure power of two so it is exactly representable as UE8M0.

    .. deprecated::
        Use ``_mxfp8_quant_act_v2`` for better performance (single fused
        CUDA kernel vs ~6 PyTorch kernel launches).
    """
    assert x.dim() == 2, f"expected 2D activation, got {x.shape}"
    M, K = x.shape
    assert K % MX_BLOCK == 0, f"K={K} must be a multiple of {MX_BLOCK}"
    xf = x.to(torch.float32).view(M, K // MX_BLOCK, MX_BLOCK)
    amax = xf.abs().amax(dim=-1).clamp(min=1e-20)
    exp = torch.ceil(torch.log2(amax / _FP8_E4M3_MAX))
    scale = torch.exp2(exp)
    q = (xf / scale.unsqueeze(-1)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return q.view(M, K).to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def mxfp8_quant_act(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-(row, 32-col) MXFP8 quant of a 2D activation.

    Returns (e4m3 ``[M, K]``, fp32 power-of-two scale ``[M, K // 32]``).
    The scale is a pure power of two so it is exactly representable as UE8M0.
    Uses a single fused Triton kernel; set ``MXFP8_QUANT_EAGER=1`` to fall back
    to the eager PyTorch reference.
    """
    if os.environ.get("MXFP8_QUANT_EAGER") == "1" or not x.is_cuda:
        return mxfp8_quant_act_eager(x)
    from rtp_llm.models_py.triton_kernels.moe.mxfp8_kernels import (
        mxfp8_quant_act_triton,
    )

    return mxfp8_quant_act_triton(x)


def pack_mxfp8_scale(
    scale_fp32: torch.Tensor,
    mn: int,
    k: int,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    """Pack an fp32 (power-of-two) scale into DeepGEMM's int32 TMA layout.

    ``scale_fp32`` is ``[mn, k // 32]`` (or ``[num_groups, mn, k // 32]`` when
    ``num_groups`` is given). Uses the (1, 32) recipe; output dtype is int32.
    """
    import deep_gemm

    kwargs = dict(mn=mn, k=k, recipe=(1, MX_BLOCK))
    if num_groups is not None:
        kwargs["num_groups"] = num_groups
    sf = scale_fp32.contiguous()
    # DeepGEMM's JIT kernel launches on the *current* CUDA device. During
    # weight loading each TP rank's tensors live on its own device (e.g.
    # cuda:5) while the current device may still be cuda:0, which makes the
    # launch fail with CUDA_ERROR_INVALID_VALUE. Pin the current device to the
    # tensor's device for the launch.
    if sf.is_cuda:
        with torch.cuda.device(sf.device):
            return deep_gemm.transform_sf_into_required_layout(sf, **kwargs)
    return deep_gemm.transform_sf_into_required_layout(sf, **kwargs)


def mxfp8_linear(
    x: torch.Tensor,
    weight_e4m3: torch.Tensor,
    weight_scale_packed: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """y = x @ weight_e4m3.T   (weight is [N, K] e4m3, scale prepacked int32)."""
    M, N = x.shape[0], weight_e4m3.shape[0]
    a_q, a_s_packed = _mxfp8_quant_act_v2(x)
    out = torch.empty(M, N, device=x.device, dtype=out_dtype)
    with torch.cuda.device(x.device):
        fp8_fp4_gemm_nt(
            (a_q, a_s_packed),
            (weight_e4m3, weight_scale_packed),
            out,
            recipe_a=(1, MX_BLOCK),
            recipe_b=(1, MX_BLOCK),
            disable_ue8m0_cast=True,
        )
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out


def mxfp8_grouped_gemm(
    x: torch.Tensor,
    weight_e4m3: torch.Tensor,
    weight_scale_packed: torch.Tensor,
    m_indices: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Grouped (contiguous) MoE GEMM. ``weight_e4m3`` is ``[E, N, K]``.

    Rows of ``x`` must already be permuted so each expert's tokens are
    contiguous and each expert block is padded to
    ``deep_gemm.get_m_alignment_for_contiguous_layout()`` (128). ``m_indices``
    maps each row to its expert id.
    """
    T, N = x.shape[0], weight_e4m3.shape[1]
    a_q, a_s_packed = _mxfp8_quant_act_v2(x)
    out = torch.empty(T, N, device=x.device, dtype=out_dtype)
    with torch.cuda.device(x.device):
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a_q, a_s_packed),
            (weight_e4m3, weight_scale_packed),
            out,
            m_indices,
            recipe_a=(1, MX_BLOCK),
            recipe_b=(1, MX_BLOCK),
            disable_ue8m0_cast=True,
        )
    return out
