"""SM90 post-convolution preparation with the native Q/K reduction.

The V-copy/gate split follows vLLM's fused_gdn_prefill_post_conv implementation.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.scatter_qkv import (
    try_scatter_qkv_l2norm_sm90,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import _gdn_gate_values

SM90_POST_CONV = True


@triton.jit
def _copy_v_gating(
    packed,
    a,
    b,
    A_log,
    bias,
    v,
    g,
    beta,
    rows,
    stride_ab,
    H: tl.constexpr,
    HV: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    head = tl.program_id(1)
    col = tl.arange(0, 128)
    valid = row < rows
    source = (
        row[:, None].to(tl.int64) * ((2 * H + HV) * 128)
        + 2 * H * 128
        + head * 128
        + col[None, :]
    )
    value = tl.load(packed + source, valid[:, None], other=0)
    target = row[:, None].to(tl.int64) * (HV * 128) + head * 128 + col[None, :]
    tl.store(v + target, value, valid[:, None])
    offset = row.to(tl.int64) * stride_ab + head
    av = tl.load(a + offset, valid, other=0)
    bv = tl.load(b + offset, valid, other=0)
    gate, update = _gdn_gate_values(
        tl.load(A_log + head), av, bv, tl.load(bias + head), 1.0, 20.0
    )
    tl.store(g + row * HV + head, gate, valid)
    tl.store(beta + row * HV + head, update, valid)


def try_post_conv_gating(
    packed,
    a,
    b,
    A_log,
    bias,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    if not (
        SM90_POST_CONV
        and a.shape == b.shape == (packed.shape[0], num_v_heads)
        and a.dtype == b.dtype == packed.dtype == torch.bfloat16
        and a.device == b.device == A_log.device == bias.device == packed.device
        and a.stride(-1) == b.stride(-1) == 1
        and a.stride(0) == b.stride(0)
        and A_log.shape == bias.shape == (num_v_heads,)
        and A_log.is_contiguous()
        and bias.is_contiguous()
    ):
        return None
    prepared = try_scatter_qkv_l2norm_sm90(
        packed, num_k_heads, num_v_heads, head_k_dim, head_v_dim, copy_v=False
    )
    if prepared is None:
        return None
    q, k, v = prepared
    rows = packed.shape[0]
    g = torch.empty((1, rows, num_v_heads), device=packed.device, dtype=torch.float32)
    beta = torch.empty((1, rows, num_v_heads), device=packed.device, dtype=b.dtype)
    _copy_v_gating[(triton.cdiv(rows, 16), num_v_heads)](
        packed,
        a,
        b,
        A_log,
        bias,
        v,
        g,
        beta,
        rows,
        a.stride(0),
        num_k_heads,
        num_v_heads,
        16,
        num_warps=4,
    )
    return q, k, v, g, beta
