"""Small K3 Prefill fusions selected only by the explicit performance switch.

The accuracy path deliberately keeps the source-like eager implementations.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _interleave_tp_hidden_kernel(
    gathered,
    output,
    tokens,
    local_hidden: tl.constexpr,
    tp_size: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    total = tokens * local_hidden * tp_size
    mask = offsets < total
    output_hidden = local_hidden * tp_size
    token = offsets // output_hidden
    output_column = offsets % output_hidden
    rank = output_column // local_hidden
    local_column = output_column % local_hidden
    source_offset = (rank * tokens + token) * local_hidden + local_column
    value = tl.load(gathered + source_offset, mask=mask)
    tl.store(output + offsets, value, mask=mask)


@torch.compiler.disable
def kimi_k3_interleave_tp_hidden(
    gathered: torch.Tensor,
    tokens: int,
    tp_size: int,
) -> torch.Tensor:
    """Convert rank-major hidden shards to token-major hidden without aten copy."""

    if not gathered.is_cuda or gathered.ndim != 2 or not gathered.is_contiguous():
        raise ValueError(
            "K3 TP hidden interleave requires contiguous rank-2 CUDA input"
        )
    if gathered.shape[0] != tokens * tp_size:
        raise ValueError(
            "K3 TP hidden interleave row mismatch: "
            f"gathered={gathered.shape[0]} tokens={tokens} tp={tp_size}"
        )
    local_hidden = gathered.shape[1]
    output = torch.empty(
        (tokens, tp_size * local_hidden),
        dtype=gathered.dtype,
        device=gathered.device,
    )
    block = 1024
    _interleave_tp_hidden_kernel[(triton.cdiv(output.numel(), block),)](
        gathered,
        output,
        tokens,
        local_hidden=local_hidden,
        tp_size=tp_size,
        block=block,
        num_warps=4,
    )
    return output


@triton.jit
def _rms_norm_strided_kernel(
    x,
    weight,
    output,
    stride_x_m,
    stride_x_n,
    stride_o_m,
    stride_o_n,
    hidden_size: tl.constexpr,
    block_hidden: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_hidden)
    mask = offsets < hidden_size
    values = tl.load(
        x + row * stride_x_m + offsets * stride_x_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values = tl.where(mask, values, 0.0)
    inverse_rms = tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
    gamma = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output + row * stride_o_m + offsets * stride_o_n,
        values * inverse_rms * gamma,
        mask=mask,
    )


@torch.compiler.disable
def kimi_k3_rms_norm_strided(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm a row-strided view without materializing ``x.contiguous()``."""

    if not x.is_cuda or x.ndim != 2:
        raise ValueError("K3 strided RMSNorm requires a rank-2 CUDA tensor")
    if x.stride(-1) != 1:
        raise ValueError("K3 strided RMSNorm requires unit inner stride")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError("K3 strided RMSNorm weight width mismatch")
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    block_hidden = triton.next_power_of_2(x.shape[-1])
    _rms_norm_strided_kernel[(x.shape[0],)](
        x,
        weight,
        output,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        hidden_size=x.shape[-1],
        block_hidden=block_hidden,
        eps=float(eps),
        num_warps=8,
        num_stages=2,
    )
    return output


@triton.jit
def _situ_kernel(
    gate,
    up,
    output,
    elements,
    beta: tl.constexpr,
    linear_beta: tl.constexpr,
    has_linear_beta: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    gate_value = tl.load(gate + offsets, mask=mask).to(tl.float32)
    up_value = tl.load(up + offsets, mask=mask).to(tl.float32)

    gate_tanh = 2.0 * tl.sigmoid(2.0 * gate_value / beta) - 1.0
    activated = beta * gate_tanh * tl.sigmoid(gate_value)
    if has_linear_beta:
        up_tanh = 2.0 * tl.sigmoid(2.0 * up_value / linear_beta) - 1.0
        up_value = linear_beta * up_tanh
    tl.store(output + offsets, activated * up_value, mask=mask)


@torch.compiler.disable
def kimi_k3_situ(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
) -> torch.Tensor:
    if gate.shape != up.shape or gate.dtype != up.dtype or gate.device != up.device:
        raise ValueError("K3 fused SiTU requires matching gate/up tensors")
    if not gate.is_cuda:
        raise ValueError("K3 fused SiTU is a CUDA performance-only kernel")
    gate = gate.contiguous()
    up = up.contiguous()
    output = torch.empty_like(gate)
    block = 1024
    _situ_kernel[(triton.cdiv(gate.numel(), block),)](
        gate,
        up,
        output,
        gate.numel(),
        beta=float(beta),
        linear_beta=0.0 if linear_beta is None else float(linear_beta),
        has_linear_beta=linear_beta is not None,
        block=block,
        num_warps=4,
    )
    return output


@triton.jit
def _attn_res_score_kernel(
    anchor,
    prefix,
    norm_weight,
    projection_weight,
    scores,
    hidden_size: tl.constexpr,
    block_hidden: tl.constexpr,
    eps: tl.constexpr,
):
    token = tl.program_id(0)
    candidate = tl.program_id(1)
    offsets = tl.arange(0, block_hidden)
    mask = offsets < hidden_size
    anchor_value = tl.load(
        anchor + token * hidden_size + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    prefix_value = tl.load(
        prefix + token * hidden_size + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    value = tl.where(candidate == 0, anchor_value, prefix_value)
    value = tl.where(mask, value, 0.0)
    variance = tl.sum(value * value, axis=0) / hidden_size
    inverse_rms = 1.0 / tl.sqrt(variance + eps)
    norm = tl.load(norm_weight + offsets, mask=mask, other=0.0).to(tl.float32)
    projection = tl.load(projection_weight + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    score = tl.sum(value * inverse_rms * norm * projection, axis=0)
    tl.store(scores + token * 2 + candidate, score)


@triton.jit
def _attn_res_combine_kernel(
    anchor,
    prefix,
    scores,
    output,
    hidden_size: tl.constexpr,
    block: tl.constexpr,
):
    token = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * block + tl.arange(0, block)
    mask = offsets < hidden_size
    score0 = tl.load(scores + token * 2)
    score1 = tl.load(scores + token * 2 + 1)
    probability1 = tl.sigmoid(score1 - score0)
    value0 = tl.load(anchor + token * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    value1 = tl.load(prefix + token * hidden_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    result = value0 + probability1 * (value1 - value0)
    tl.store(output + token * hidden_size + offsets, result, mask=mask)


@torch.compiler.disable
def kimi_k3_two_way_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    projection_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if not prefix_sum.is_cuda:
        raise ValueError("K3 fused AttnRes is a CUDA performance-only kernel")
    if prefix_sum.ndim != 2 or block_residual.shape != (
        prefix_sum.shape[0],
        1,
        prefix_sum.shape[1],
    ):
        raise ValueError(
            "K3 fused AttnRes requires one anchor and one prefix candidate"
        )
    hidden_size = prefix_sum.shape[1]
    if norm_weight.numel() != hidden_size or projection_weight.numel() != hidden_size:
        raise ValueError("K3 fused AttnRes weight width mismatch")
    anchor = block_residual[:, 0, :]
    if anchor.stride(-1) != 1 or prefix_sum.stride(-1) != 1:
        raise ValueError("K3 fused AttnRes requires contiguous hidden rows")

    scores = torch.empty(
        (prefix_sum.shape[0], 2), dtype=torch.float32, device=prefix_sum.device
    )
    output = torch.empty_like(prefix_sum)
    block_hidden = triton.next_power_of_2(hidden_size)
    _attn_res_score_kernel[(prefix_sum.shape[0], 2)](
        anchor,
        prefix_sum,
        norm_weight,
        projection_weight.reshape(-1),
        scores,
        hidden_size=hidden_size,
        block_hidden=block_hidden,
        eps=float(eps),
        num_warps=8,
        num_stages=2,
    )
    block = 1024
    _attn_res_combine_kernel[(prefix_sum.shape[0], triton.cdiv(hidden_size, block))](
        anchor,
        prefix_sum,
        scores,
        output,
        hidden_size=hidden_size,
        block=block,
        num_warps=4,
    )
    return output


@triton.jit
def _linear_cache_store_kernel(
    recurrent,
    q_state,
    k_state,
    v_state,
    ssm_cache,
    conv_cache,
    stride_r_h,
    stride_r_v,
    stride_r_k,
    stride_q_c,
    stride_q_w,
    stride_k_c,
    stride_k_w,
    stride_v_c,
    stride_v_w,
    stride_ssm_h,
    stride_ssm_v,
    stride_ssm_k,
    stride_conv_w,
    stride_conv_c,
    heads: tl.constexpr,
    state_dim: tl.constexpr,
    conv_channels: tl.constexpr,
    history_size: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)

    ssm_elements = heads * state_dim * state_dim
    ssm_mask = offsets < ssm_elements
    state_k = offsets % state_dim
    state_v = (offsets // state_dim) % state_dim
    state_h = offsets // (state_dim * state_dim)
    state_value = tl.load(
        recurrent + state_h * stride_r_h + state_v * stride_r_v + state_k * stride_r_k,
        mask=ssm_mask,
        other=0.0,
    )
    tl.store(
        ssm_cache
        + state_h * stride_ssm_h
        + state_v * stride_ssm_v
        + state_k * stride_ssm_k,
        state_value,
        mask=ssm_mask,
    )

    packed_channels = 3 * conv_channels
    conv_elements = history_size * packed_channels
    conv_mask = offsets < conv_elements
    history = offsets // packed_channels
    packed_channel = offsets % packed_channels
    channel = packed_channel % conv_channels
    q_mask = conv_mask & (packed_channel < conv_channels)
    k_mask = (
        conv_mask
        & (packed_channel >= conv_channels)
        & (packed_channel < 2 * conv_channels)
    )
    v_mask = conv_mask & (packed_channel >= 2 * conv_channels)
    conv_value = tl.load(
        q_state + channel * stride_q_c + history * stride_q_w,
        mask=q_mask,
        other=0.0,
    )
    conv_value += tl.load(
        k_state + channel * stride_k_c + history * stride_k_w,
        mask=k_mask,
        other=0.0,
    )
    conv_value += tl.load(
        v_state + channel * stride_v_c + history * stride_v_w,
        mask=v_mask,
        other=0.0,
    )
    tl.store(
        conv_cache + history * stride_conv_w + packed_channel * stride_conv_c,
        conv_value,
        mask=conv_mask,
    )


@torch.compiler.disable
def kimi_k3_store_linear_cache_state(
    recurrent: torch.Tensor,
    q_state: torch.Tensor,
    k_state: torch.Tensor,
    v_state: torch.Tensor,
    ssm_cache: torch.Tensor,
    conv_cache: torch.Tensor,
) -> None:
    """Store one V-first KDA state without eager cat/transpose/copy operators."""

    if (
        recurrent.ndim != 3
        or q_state.ndim != 2
        or k_state.shape != q_state.shape
        or v_state.shape != q_state.shape
        or ssm_cache.shape != recurrent.shape
    ):
        raise ValueError("invalid K3 linear-cache state shapes")
    if conv_cache.shape != (q_state.shape[1], 3 * q_state.shape[0]):
        raise ValueError(
            "invalid K3 packed conv-cache shape: "
            f"cache={tuple(conv_cache.shape)} state={tuple(q_state.shape)}"
        )
    if not recurrent.is_cuda or not q_state.is_cuda:
        raise ValueError("K3 fused linear-cache store requires CUDA tensors")
    total = max(recurrent.numel(), conv_cache.numel())
    block = 256
    _linear_cache_store_kernel[(triton.cdiv(total, block),)](
        recurrent,
        q_state,
        k_state,
        v_state,
        ssm_cache,
        conv_cache,
        recurrent.stride(0),
        recurrent.stride(1),
        recurrent.stride(2),
        q_state.stride(0),
        q_state.stride(1),
        k_state.stride(0),
        k_state.stride(1),
        v_state.stride(0),
        v_state.stride(1),
        ssm_cache.stride(0),
        ssm_cache.stride(1),
        ssm_cache.stride(2),
        conv_cache.stride(0),
        conv_cache.stride(1),
        heads=recurrent.shape[0],
        state_dim=recurrent.shape[1],
        conv_channels=q_state.shape[0],
        history_size=q_state.shape[1],
        block=block,
        num_warps=4,
    )


__all__ = [
    "kimi_k3_interleave_tp_hidden",
    "kimi_k3_rms_norm_strided",
    "kimi_k3_situ",
    "kimi_k3_store_linear_cache_state",
    "kimi_k3_two_way_attn_res",
]
