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
def _pack_a2a_projection_kernel(
    projected,
    packed,
    tokens,
    elements,
    payload: tl.constexpr,
    tp_size: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < elements
    destination = offsets // (tokens * payload)
    destination_offset = offsets % (tokens * payload)
    token = destination_offset // payload
    column = destination_offset % payload
    source_offset = token * (tp_size * payload) + destination * payload + column
    value = tl.load(projected + source_offset, mask=mask)
    tl.store(packed + offsets, value, mask=mask)


@torch.compiler.disable
def kimi_k3_pack_a2a_projection(
    projected: torch.Tensor,
    tp_size: int,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pack token-major projection columns into destination-major A2A chunks."""

    if not projected.is_cuda or projected.ndim != 2 or not projected.is_contiguous():
        raise ValueError(
            "K3 KDA A2A projection pack requires a contiguous rank-2 CUDA tensor"
        )
    if tp_size <= 1 or projected.shape[1] % tp_size:
        raise ValueError(
            "K3 KDA A2A projection width must be divisible by TP: "
            f"shape={tuple(projected.shape)}, tp={tp_size}"
        )
    payload = projected.shape[1] // tp_size
    expected_shape = (tp_size, projected.shape[0], payload)
    if output is None:
        output = torch.empty(
            expected_shape,
            dtype=projected.dtype,
            device=projected.device,
        )
    elif (
        tuple(output.shape) != expected_shape
        or output.dtype != projected.dtype
        or output.device != projected.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "K3 KDA A2A projection pack output mismatch: "
            f"got={tuple(output.shape)}, expected={expected_shape}"
        )
    block = 1024
    _pack_a2a_projection_kernel[(triton.cdiv(projected.numel(), block),)](
        projected,
        output,
        projected.shape[0],
        projected.numel(),
        payload=payload,
        tp_size=tp_size,
        block=block,
        num_warps=4,
    )
    return output


@triton.jit
def _a2a_unpack_rms_norm_sigmoid_gate_kernel(
    received,
    gate,
    weight,
    output,
    token_rows,
    total_rows,
    eps,
    local_heads: tl.constexpr,
    total_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_rows: tl.constexpr,
    block_features: tl.constexpr,
):
    row_offsets = (tl.program_id(0) * block_rows + tl.arange(0, block_rows))[:, None]
    feature_offsets = tl.arange(0, block_features)[None, :]
    mask = (row_offsets < total_rows) & (feature_offsets < head_dim)

    token = row_offsets // total_heads
    global_head = row_offsets % total_heads
    source_rank = global_head // local_heads
    local_head = global_head % local_heads
    received_offsets = (
        (source_rank * token_rows + token) * local_heads + local_head
    ) * head_dim + feature_offsets
    values = tl.load(received + received_offsets, mask=mask, other=0.0).to(tl.float32)
    normalized_values = tl.where(feature_offsets < head_dim, values, 0.0)
    variance = tl.sum(normalized_values * normalized_values, axis=1) / head_dim
    inverse_rms = 1.0 / tl.sqrt(variance + eps)

    output_offsets = row_offsets * head_dim + feature_offsets
    gate_values = tl.load(gate + output_offsets, mask=mask, other=0.0).to(tl.float32)
    norm_weight = tl.load(
        weight + feature_offsets,
        mask=feature_offsets < head_dim,
        other=0.0,
    ).to(tl.float32)
    result = values * inverse_rms[:, None] * norm_weight * tl.sigmoid(gate_values)
    tl.store(output + output_offsets, result, mask=mask)


@torch.compiler.disable
def kimi_k3_a2a_unpack_rms_norm_sigmoid_gate(
    received: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fuse source-head unpack, per-head RMSNorm and KDA sigmoid output gate."""

    if not received.is_cuda or received.ndim != 4 or not received.is_contiguous():
        raise ValueError(
            "K3 KDA A2A receive must be contiguous [source,tokens,heads,dim]"
        )
    tp_size, token_rows, local_heads, head_dim = received.shape
    total_heads = tp_size * local_heads
    expected_gate_shape = (1, token_rows, total_heads, head_dim)
    if tuple(gate.shape) != expected_gate_shape or not gate.is_contiguous():
        raise ValueError(
            "K3 KDA A2A gate shape mismatch: "
            f"got={tuple(gate.shape)}, expected={expected_gate_shape}"
        )
    if weight.shape != (head_dim,):
        raise ValueError(
            f"K3 KDA A2A norm weight must have shape {(head_dim,)}, got "
            f"{tuple(weight.shape)}"
        )
    if (
        received.dtype != gate.dtype
        or received.device != gate.device
        or received.device != weight.device
    ):
        raise ValueError("K3 KDA A2A receive/gate/norm tensors must match")
    if output is None:
        output = torch.empty_like(gate)
    elif (
        tuple(output.shape) != expected_gate_shape
        or output.dtype != gate.dtype
        or output.device != gate.device
        or not output.is_contiguous()
    ):
        raise ValueError("K3 KDA A2A fused output buffer does not match gate")

    total_rows = token_rows * total_heads
    block_features = triton.next_power_of_2(head_dim)
    if triton.cdiv(total_rows, 2048 * 32) == 1:
        block_rows = 16
        num_warps = 16
    else:
        block_rows = 32
        num_warps = 4
    _a2a_unpack_rms_norm_sigmoid_gate_kernel[(triton.cdiv(total_rows, block_rows),)](
        received,
        gate,
        weight,
        output,
        token_rows,
        total_rows,
        float(eps),
        local_heads=local_heads,
        total_heads=total_heads,
        head_dim=head_dim,
        block_rows=block_rows,
        block_features=block_features,
        num_warps=num_warps,
        num_stages=3,
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
    # K3's source RMSNorm rounds the normalized activation to BF16 before
    # applying the BF16 affine weight.  Keeping both operations in FP32 and
    # rounding only the final product changes one ULP at every layer, which
    # can cross a near-tied MoE routing boundary after recurrent Decode.
    normalized = (values * inverse_rms).to(tl.bfloat16)
    gamma = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output + row * stride_o_m + offsets * stride_o_n,
        normalized.to(tl.float32) * gamma,
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
    """Store one backend-native KDA state without eager cat/transpose/copy ops."""

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
    "kimi_k3_a2a_unpack_rms_norm_sigmoid_gate",
    "kimi_k3_interleave_tp_hidden",
    "kimi_k3_pack_a2a_projection",
    "kimi_k3_rms_norm_strided",
    "kimi_k3_situ",
    "kimi_k3_store_linear_cache_state",
    "kimi_k3_two_way_attn_res",
]
