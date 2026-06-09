"""Fused MXFP8 (1x32 microscaling FP8) activation quant for MiniMax-M3.

Replaces the eager-PyTorch ``mxfp8_quant_act`` (``abs``/``amax``/``log2``/
``ceil``/``exp2``/``div``/``clamp``/``to(fp8)`` ~= 7 kernels) with a single
Triton kernel: dynamic per-(row, 32-col) max-abs -> UE8M0 power-of-two scale
-> e4m3 cast. The power-of-two rounding reuses :func:`_ue8m0_pow2_round` from
the existing FP8 quant kernels so the scale math is shared, and the output
(fp32 ``[M, K//32]`` scale) feeds DeepGEMM's native ``pack_mxfp8_scale``.

There is no existing drop-in for this: the FP8 group-quant kernels
(``sgl_per_token_group_quant_fp8`` / ``per_token_group_quant_fp8_v2`` /
``trt_fp8_quantize_128``) are hardwired to the 1x128 recipe and do not emit a
1x32 power-of-two scale.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
    _ue8m0_pow2_round,
)

MX_BLOCK = 32
_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


@triton.jit
def _mxfp8_quant_act_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    n_groups,
    x_row_stride,
    q_row_stride,
    s_row_stride,
    FP8_MAX: tl.constexpr,
    GROUP: tl.constexpr,
    NG: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    gblk = tl.program_id(1)
    g0 = gblk * NG
    g_idx = tl.arange(0, NG)
    groups = g0 + g_idx
    gmask = groups < n_groups
    col = tl.arange(0, GROUP)
    # [NG, GROUP] element offsets inside the row
    elem = groups[:, None] * GROUP + col[None, :]

    x = tl.load(
        x_ptr + row * x_row_stride + elem,
        mask=gmask[:, None],
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    amax = tl.maximum(amax, 1e-20)
    # UE8M0 power-of-two scale == exp2(ceil(log2(amax / FP8_MAX))), shared with
    # the existing FP8 quant kernels via the bit-hack helper.
    scale, _ = _ue8m0_pow2_round(amax / FP8_MAX)
    q = x / scale[:, None]
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)

    tl.store(
        q_ptr + row * q_row_stride + elem,
        q.to(q_ptr.dtype.element_ty),
        mask=gmask[:, None],
    )
    tl.store(s_ptr + row * s_row_stride + groups, scale, mask=gmask)


def mxfp8_quant_act_triton(x: torch.Tensor):
    """Fused per-(row, 32) MXFP8 quant. Returns (e4m3 ``[M, K]``, fp32 scale ``[M, K//32]``)."""
    assert x.dim() == 2, f"expected 2D activation, got {tuple(x.shape)}"
    M, K = x.shape
    assert K % MX_BLOCK == 0, f"K={K} must be a multiple of {MX_BLOCK}"
    x = x.contiguous()
    n_groups = K // MX_BLOCK
    q = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, n_groups, device=x.device, dtype=torch.float32)
    if M == 0:
        return q, s
    NG = 16
    grid = (M, triton.cdiv(n_groups, NG))
    _mxfp8_quant_act_kernel[grid](
        x,
        q,
        s,
        n_groups,
        x.stride(0),
        q.stride(0),
        s.stride(0),
        FP8_MAX=float(_FP8_E4M3_MAX),
        GROUP=MX_BLOCK,
        NG=NG,
        num_warps=4,
    )
    return q, s
