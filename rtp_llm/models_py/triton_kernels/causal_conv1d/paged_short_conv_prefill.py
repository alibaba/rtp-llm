"""Paged BF16 short convolution with three output planes and optional auxiliary packing.

Adapted from the fixed feat/k3_dev implementation. The operator is independent
of model, attention backend, and transport.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl

_PREFILL_BLOCK_T = 64


@dataclass(frozen=True)
class PagedShortConvMetadata:
    """Sequence-to-program mapping reusable across convolution layers."""

    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor
    total_chunks: int


def prepare_paged_short_conv_metadata(
    cu_seqlens_host: torch.Tensor,
    device: torch.device,
) -> PagedShortConvMetadata:
    if (
        cu_seqlens_host.ndim != 1
        or cu_seqlens_host.numel() < 2
        or cu_seqlens_host.device.type != "cpu"
    ):
        raise ValueError("paged short conv metadata requires CPU cu_seqlens=[N+1]")
    values = cu_seqlens_host.to(dtype=torch.int64).numpy()
    lengths = np.diff(values)
    if values[0] != 0 or np.any(lengths < 0):
        raise ValueError(
            f"paged short conv cu_seqlens must start at zero and be nondecreasing: {values.tolist()}"
        )
    chunk_counts = (lengths + _PREFILL_BLOCK_T - 1) // _PREFILL_BLOCK_T
    batch = np.repeat(np.arange(len(lengths), dtype=np.int32), chunk_counts)
    offsets = np.concatenate(
        [np.arange(count, dtype=np.int32) for count in chunk_counts]
    )
    return PagedShortConvMetadata(
        batch_ptr=torch.from_numpy(batch).to(device=device),
        token_chunk_offset_ptr=torch.from_numpy(offsets).to(device=device),
        total_chunks=int(batch.size),
    )


@triton.jit(do_not_specialize=["max_block_count", "physical_block_count"])
def _paged_short_conv_prefill_kernel(
    x,
    weight,
    conv_state,
    block_map,
    prefix_lengths,
    query_start_loc,
    batch_ptr,
    token_chunk_offset_ptr,
    output,
    aux,
    packed_aux,
    current_conv_state,
    continuation_mask,
    final_conv_state,
    stride_x_t,
    stride_x_d,
    stride_w_d,
    stride_w_w,
    stride_s_block,
    stride_s_w,
    stride_s_d,
    stride_bm_b,
    stride_bm_page,
    stride_o_p,
    stride_o_t,
    stride_o_d,
    stride_aux_t,
    stride_aux_h,
    stride_packed_aux_t,
    stride_packed_aux_h,
    stride_cs_b,
    stride_cs_w,
    stride_cs_d,
    stride_fs_b,
    stride_fs_w,
    stride_fs_d,
    max_block_count,
    physical_block_count,
    PROJECTION_SIZE: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BW: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HAS_CURRENT_STATE: tl.constexpr,
    RETURN_FINAL_STATE: tl.constexpr,
    HAS_RAW_AUX: tl.constexpr,
    AUX_HEADS: tl.constexpr,
    AUX_BLOCK: tl.constexpr,
):
    """FLA-compatible fused Q/K/V Prefill with direct paged state writes."""

    program = tl.program_id(0)
    i_d = tl.program_id(1)
    i_b = tl.load(batch_ptr + program).to(tl.int32)
    i_t = tl.load(token_chunk_offset_ptr + program).to(tl.int32)

    sequence_start = tl.load(query_start_loc + i_b).to(tl.int64)
    sequence_end = tl.load(query_start_loc + i_b + 1).to(tl.int64)
    sequence_length = (sequence_end - sequence_start).to(tl.int32)
    prefix = tl.load(prefix_lengths + i_b).to(tl.int64)
    token_offset = i_t * BT

    o_d = i_d * BD + tl.arange(0, BD)
    o_d_i64 = o_d.to(tl.int64)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    initial_page = (prefix - 1) // PAGE_SIZE
    initial_page_valid = (
        (prefix > 0) & (initial_page >= 0) & (initial_page < max_block_count)
    )
    initial_page_address = tl.where(initial_page_valid, initial_page, 0)
    initial_block = tl.load(
        block_map + i_b * stride_bm_b + initial_page_address * stride_bm_page,
        mask=initial_page_valid,
        other=0,
    ).to(tl.int64)
    has_initial = (
        initial_page_valid
        & (initial_block > 0)
        & (initial_block < physical_block_count)
    )
    if HAS_CURRENT_STATE:
        use_current_state = tl.load(continuation_mask + i_b) != 0
    else:
        use_current_state = False

    b_w = tl.load(
        weight + o_d[:, None] * stride_w_d + o_w * stride_w_w,
        mask=m_d[:, None] & m_w,
        other=0,
    ).to(tl.float32)
    b_y = tl.zeros((BT, BD), dtype=tl.float32)

    if i_t > 0:
        for i_w in tl.static_range(-W + 1, 1):
            p_x = tl.make_block_ptr(
                x + sequence_start * stride_x_t,
                (sequence_length, D),
                (stride_x_t, stride_x_d),
                (token_offset + i_w, i_d * BD),
                (BT, BD),
                (1, 0),
            )
            b_yi = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
            b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            source_t = o_t + i_w
            source_t_i64 = source_t.to(tl.int64)
            m_x = ((source_t >= 0) & (source_t < sequence_length))[:, None] & m_d[
                None, :
            ]
            history_idx = source_t + W - 1
            m_h = (has_initial & (source_t >= -W + 1) & (source_t < 0))[:, None] & m_d[
                None, :
            ]
            b_yi = tl.load(
                x
                + (sequence_start + source_t_i64)[:, None] * stride_x_t
                + o_d_i64[None, :] * stride_x_d,
                mask=m_x,
                other=0,
            ).to(tl.float32)
            b_yi += tl.load(
                conv_state
                + initial_block * stride_s_block
                + history_idx[:, None] * stride_s_w
                + o_d_i64[None, :] * stride_s_d,
                mask=m_h,
                other=0,
            ).to(tl.float32)
            if HAS_CURRENT_STATE:
                b_yi = tl.where(
                    (use_current_state & (source_t < 0))[:, None],
                    tl.load(
                        current_conv_state
                        + i_b * stride_cs_b
                        + history_idx[:, None] * stride_cs_w
                        + o_d_i64[None, :] * stride_cs_d,
                        mask=(source_t >= -W + 1)[:, None]
                        & (source_t < 0)[:, None]
                        & m_d[None, :],
                        other=0,
                    ).to(tl.float32),
                    b_yi,
                )
            b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    b_y = b_y * tl.sigmoid(b_y)
    output_t = token_offset + tl.arange(0, BT)
    output_plane = o_d // PROJECTION_SIZE
    output_d = o_d % PROJECTION_SIZE
    tl.store(
        output
        + output_plane[None, :] * stride_o_p
        + (sequence_start + output_t)[:, None] * stride_o_t
        + output_d[None, :] * stride_o_d,
        tl.cast(b_y, dtype=output.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=(output_t[:, None] < sequence_length) & m_d[None, :],
    )

    if HAS_RAW_AUX and i_d == 0:
        o_h = tl.arange(0, AUX_BLOCK)
        o_h_i64 = o_h.to(tl.int64)
        aux_t = (sequence_start + output_t).to(tl.int64)
        aux_mask = (output_t[:, None] < sequence_length) & (o_h[None, :] < AUX_HEADS)
        aux_values = tl.load(
            aux
            + aux_t[:, None] * stride_aux_t
            + o_h_i64[None, :] * stride_aux_h,
            mask=aux_mask,
            other=0,
        )
        tl.store(
            packed_aux
            + aux_t[:, None] * stride_packed_aux_t
            + o_h_i64[None, :] * stride_packed_aux_h,
            tl.cast(
                aux_values,
                dtype=packed_aux.dtype.element_ty,
                fp_downcast_rounding="rtne",
            ),
            mask=aux_mask,
        )

    # Page boundaries are aligned to BT on the fast path. The last partial
    # chunk also publishes a request-owned tail state for immediate Decode.
    local_end = tl.minimum(token_offset + BT, sequence_length)
    absolute_end = prefix + local_end
    should_write = (absolute_end % PAGE_SIZE == 0) | (local_end == sequence_length)
    write_page = (absolute_end - 1) // PAGE_SIZE
    write_page_valid = (write_page >= 0) & (write_page < max_block_count)
    write_page_address = tl.where(write_page_valid, write_page, 0)
    write_block = tl.load(
        block_map + i_b * stride_bm_b + write_page_address * stride_bm_page,
        mask=should_write & write_page_valid,
        other=0,
    ).to(tl.int64)

    state_w = tl.arange(0, BW)
    history_size = W - 1
    state_source_t = local_end - history_size + state_w
    state_source_t_i64 = state_source_t.to(tl.int64)
    state_history_idx = state_source_t + history_size
    state_from_x = tl.load(
        x
        + (sequence_start + state_source_t_i64)[None, :] * stride_x_t
        + o_d_i64[:, None] * stride_x_d,
        mask=m_d[:, None]
        & (state_w[None, :] < history_size)
        & (state_source_t[None, :] >= 0)
        & (state_source_t[None, :] < sequence_length),
        other=0,
    )
    state_from_history = tl.load(
        conv_state
        + initial_block * stride_s_block
        + state_history_idx[None, :] * stride_s_w
        + o_d_i64[:, None] * stride_s_d,
        mask=has_initial
        & m_d[:, None]
        & (state_w[None, :] < history_size)
        & (state_source_t[None, :] < 0),
        other=0,
    )
    if HAS_CURRENT_STATE:
        state_from_current = tl.load(
            current_conv_state
            + i_b * stride_cs_b
            + state_history_idx[None, :] * stride_cs_w
            + o_d_i64[:, None] * stride_cs_d,
            mask=use_current_state
            & m_d[:, None]
            & (state_w[None, :] < history_size)
            & (state_source_t[None, :] < 0),
            other=0,
        )
        state_from_history = tl.where(
            use_current_state & (state_source_t[None, :] < 0),
            state_from_current,
            state_from_history,
        )
    state_value = tl.where(
        state_source_t[None, :] >= 0, state_from_x, state_from_history
    )
    write_valid = (write_block > 0) & (write_block < physical_block_count)
    tl.store(
        conv_state
        + write_block * stride_s_block
        + state_w[None, :] * stride_s_w
        + o_d_i64[:, None] * stride_s_d,
        state_value,
        mask=should_write
        & write_page_valid
        & write_valid
        & m_d[:, None]
        & (state_w[None, :] < history_size),
    )
    if RETURN_FINAL_STATE:
        tl.store(
            final_conv_state
            + i_b * stride_fs_b
            + state_w[None, :] * stride_fs_w
            + o_d_i64[:, None] * stride_fs_d,
            state_value,
            mask=(local_end == sequence_length)
            & m_d[:, None]
            & (state_w[None, :] < history_size),
        )


def paged_short_conv_prefill(
    mixed_qkv: torch.Tensor,
    fused_weight: torch.Tensor,
    conv_state: torch.Tensor,
    linear_block_map: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    page_size: int,
    metadata: PagedShortConvMetadata,
    *,
    aux: torch.Tensor | None = None,
    current_conv_state: torch.Tensor | None = None,
    continuation_mask: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]
):
    """Run a three-plane convolution and optionally pack an auxiliary tensor."""

    if mixed_qkv.ndim != 2 or fused_weight.ndim != 2:
        raise ValueError("paged short conv expects input=[T,3D], weight=[3D,W]")
    tokens, channels = mixed_qkv.shape
    if channels % 3:
        raise ValueError(
            f"paged short conv channels must be divisible by 3, got {channels}"
        )
    projection_size = channels // 3
    if fused_weight.shape[0] != channels or fused_weight.shape[1] < 2:
        raise ValueError(
            "paged short conv weight shape does not match input: "
            f"input={tuple(mixed_qkv.shape)} weight={tuple(fused_weight.shape)}"
        )
    history_size = int(fused_weight.shape[1]) - 1
    if conv_state.ndim != 3 or tuple(conv_state.shape[1:]) != (
        history_size,
        channels,
    ):
        raise ValueError(
            "paged short conv cache must be [physical_blocks,history,3D], got "
            f"{tuple(conv_state.shape)}"
        )
    sequence_count = int(cu_seqlens.numel()) - 1
    if (
        linear_block_map.ndim != 2
        or linear_block_map.shape[0] != sequence_count
        or linear_block_map.shape[1] == 0
        or prefix_lengths.ndim != 1
        or prefix_lengths.numel() != sequence_count
    ):
        raise ValueError(
            "paged short conv sequence metadata disagree: "
            f"sequences={sequence_count} blocks={tuple(linear_block_map.shape)} "
            f"prefixes={tuple(prefix_lengths.shape)}"
        )
    if page_size <= 0 or page_size % _PREFILL_BLOCK_T:
        raise ValueError(
            "paged short conv page size must be a positive multiple of "
            f"{_PREFILL_BLOCK_T}, got {page_size}"
        )
    tensors = (
        mixed_qkv,
        fused_weight,
        conv_state,
        linear_block_map,
        prefix_lengths,
        cu_seqlens,
        metadata.batch_ptr,
        metadata.token_chunk_offset_ptr,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("paged short conv requires CUDA tensors")
    if mixed_qkv.stride(1) != 1 or fused_weight.stride(1) != 1:
        raise ValueError("paged short conv requires channel-last input and weights")
    if linear_block_map.dtype not in (torch.int32, torch.int64):
        raise ValueError("paged short conv block map must be int32/int64")
    if prefix_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("paged short conv prefix lengths must be int32/int64")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("paged short conv cu_seqlens must be int32/int64")
    if metadata.total_chunks <= 0 and tokens > 0:
        raise ValueError("paged short conv metadata contains no token chunks")
    has_aux = aux is not None
    if has_aux:
        assert aux is not None
        if aux.ndim != 2 or aux.shape[0] != tokens:
            raise ValueError(
                "paged short conv auxiliary input must be [tokens,features], got "
                f"{tuple(aux.shape)}"
            )
        if aux.device != mixed_qkv.device:
            raise ValueError(
                "paged short conv auxiliary device must match input: "
                f"aux={aux.device} input={mixed_qkv.device}"
            )
        if aux.shape[1] <= 0:
            raise ValueError("paged short conv auxiliary input must have features")
    has_current_state = current_conv_state is not None
    if has_current_state != (continuation_mask is not None):
        raise ValueError(
            "current paged short conv state and continuation mask must be provided together"
        )
    if has_current_state:
        assert current_conv_state is not None and continuation_mask is not None
        if tuple(current_conv_state.shape) != (
            sequence_count,
            history_size,
            channels,
        ):
            raise ValueError(
                "current paged short conv state must be [N,history,3D], got "
                f"{tuple(current_conv_state.shape)}"
            )
        if continuation_mask.ndim != 1 or continuation_mask.numel() != sequence_count:
            raise ValueError("paged short conv continuation mask must be [N]")
        if not current_conv_state.is_cuda or not continuation_mask.is_cuda:
            raise ValueError("current paged short conv state requires CUDA tensors")
        if current_conv_state.dtype != mixed_qkv.dtype:
            raise ValueError("current paged short conv state dtype must match projected QKV")

    output = torch.empty(
        (3, tokens, projection_size),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    pack_aux = aux is not None and (
        aux.dtype != mixed_qkv.dtype or not aux.is_contiguous()
    )
    packed_aux = (
        torch.empty(
            (tokens, aux.shape[1]),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        if pack_aux
        else None
    )
    final_state = (
        torch.empty(
            (sequence_count, history_size, channels),
            dtype=mixed_qkv.dtype,
            device=mixed_qkv.device,
        )
        if return_final_state
        else None
    )
    current_arg = current_conv_state if current_conv_state is not None else conv_state
    mask_arg = continuation_mask if continuation_mask is not None else prefix_lengths
    final_arg = final_state if final_state is not None else conv_state
    aux_arg = aux if pack_aux else mixed_qkv
    packed_aux_arg = packed_aux if packed_aux is not None else output
    block_d = 64
    grid = (metadata.total_chunks, triton.cdiv(channels, block_d))
    _paged_short_conv_prefill_kernel[grid](
        mixed_qkv,
        fused_weight,
        conv_state,
        linear_block_map,
        prefix_lengths,
        cu_seqlens,
        metadata.batch_ptr,
        metadata.token_chunk_offset_ptr,
        output,
        aux_arg,
        packed_aux_arg,
        current_arg,
        mask_arg,
        final_arg,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        fused_weight.stride(0),
        fused_weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        linear_block_map.stride(0),
        linear_block_map.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        aux_arg.stride(0),
        aux_arg.stride(1),
        packed_aux_arg.stride(0),
        packed_aux_arg.stride(1),
        current_arg.stride(0),
        current_arg.stride(1),
        current_arg.stride(2),
        final_arg.stride(0),
        final_arg.stride(1),
        final_arg.stride(2),
        linear_block_map.shape[1],
        conv_state.shape[0],
        PROJECTION_SIZE=projection_size,
        D=channels,
        W=fused_weight.shape[1],
        BW=triton.next_power_of_2(fused_weight.shape[1]),
        BT=_PREFILL_BLOCK_T,
        BD=block_d,
        PAGE_SIZE=page_size,
        HAS_CURRENT_STATE=has_current_state,
        RETURN_FINAL_STATE=return_final_state,
        HAS_RAW_AUX=pack_aux,
        AUX_HEADS=(aux.shape[1] if pack_aux else 1),
        AUX_BLOCK=(triton.next_power_of_2(aux.shape[1]) if pack_aux else 1),
        num_warps=4,
    )
    result = (output[0], output[1], output[2], final_state)
    if aux is None:
        return result
    return (*result, packed_aux if packed_aux is not None else aux)
