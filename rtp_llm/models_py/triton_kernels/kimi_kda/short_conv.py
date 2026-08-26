"""Kimi KDA short convolution with FLA-compatible arithmetic.

The K3 correctness model uses FLA's Triton ``ShortConvolution``.  In
particular, prefill accumulates the four taps in chronological order in FP32,
while decode shifts the convolution cache and reduces all four taps.  Keeping
those two paths distinct is important: a composition of eager Torch kernels
can differ by one BF16 ULP and move a token across a later MoE routing
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl

_PREFILL_BLOCK_T = 64


@dataclass(frozen=True)
class KimiKDAShortConvMetadata:
    """Sequence-to-program mapping reused by every KDA layer in one round."""

    batch_ptr: torch.Tensor
    token_chunk_offset_ptr: torch.Tensor
    total_chunks: int


def prepare_kimi_kda_short_conv_metadata(
    cu_seqlens_host: torch.Tensor,
    device: torch.device,
) -> KimiKDAShortConvMetadata:
    if (
        cu_seqlens_host.ndim != 1
        or cu_seqlens_host.numel() < 2
        or cu_seqlens_host.device.type != "cpu"
    ):
        raise ValueError("KDA conv metadata requires CPU cu_seqlens=[N+1]")
    values = cu_seqlens_host.to(dtype=torch.int64).numpy()
    lengths = np.diff(values)
    if values[0] != 0 or np.any(lengths <= 0):
        raise ValueError(
            f"KDA conv cu_seqlens must start at zero and increase: {values.tolist()}"
        )
    chunk_counts = (lengths + _PREFILL_BLOCK_T - 1) // _PREFILL_BLOCK_T
    batch = np.repeat(np.arange(len(lengths), dtype=np.int32), chunk_counts)
    offsets = np.concatenate(
        [np.arange(count, dtype=np.int32) for count in chunk_counts]
    )
    return KimiKDAShortConvMetadata(
        batch_ptr=torch.from_numpy(batch).to(device=device),
        token_chunk_offset_ptr=torch.from_numpy(offsets).to(device=device),
        total_chunks=int(batch.size),
    )


@triton.jit(do_not_specialize=["max_block_count", "physical_block_count"])
def _kimi_kda_short_conv_paged_prefill_kernel(
    x,
    weight,
    conv_state,
    block_map,
    prefix_lengths,
    query_start_loc,
    batch_ptr,
    token_chunk_offset_ptr,
    output,
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


@triton.jit
def _kimi_kda_short_conv_decode_kernel(
    x,
    history,
    weight,
    output,
    stride_x_d,
    stride_h_d,
    stride_h_w,
    stride_w_d,
    stride_w_w,
    stride_o_d,
    D: tl.constexpr,
    W: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
):
    """Decode kernel matching FLA causal_conv1d_update_kernel."""

    i_d = tl.program_id(0)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW)
    m_d = o_d < D
    m_w = o_w < W

    b_x = tl.load(x + o_d * stride_x_d, mask=m_d, other=0).to(tl.float32)
    b_cache = tl.zeros((BD, BW), dtype=tl.float32)
    b_history = tl.load(
        history + o_d[:, None] * stride_h_d + o_w[None, :] * stride_h_w,
        mask=m_d[:, None] & (o_w[None, :] < W - 1),
        other=0,
    ).to(tl.float32)
    b_cache = tl.where(
        (o_w < W - 1)[None, :],
        b_history,
        b_cache,
    )
    b_cache = tl.where((o_w == W - 1)[None, :], b_x[:, None], b_cache)

    # Deliberately do not cast weight separately: this is the same expression
    # used by FLA's update kernel, where the FP32 cache promotes the product.
    b_w = tl.load(
        weight + o_d[:, None] * stride_w_d + o_w[None, :] * stride_w_w,
        mask=m_d[:, None] & m_w[None, :],
        other=0,
    )
    b_y = tl.sum(b_cache * b_w, 1)
    b_y = b_y * tl.sigmoid(b_y)
    tl.store(
        output + o_d * stride_o_d,
        tl.cast(
            b_y,
            dtype=output.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
        mask=m_d,
    )


@triton.jit
def _kimi_kda_short_conv_paged_decode_kernel(
    q,
    k,
    v,
    weight,
    conv_state,
    block_map,
    sequence_lengths_plus_one,
    output,
    stride_q_b,
    stride_q_d,
    stride_k_b,
    stride_k_d,
    stride_v_b,
    stride_v_d,
    stride_w_d,
    stride_w_w,
    stride_s_block,
    stride_s_w,
    stride_s_d,
    stride_bm_b,
    stride_bm_page,
    stride_o_p,
    stride_o_b,
    stride_o_d,
    max_block_count,
    physical_block_count,
    seq_size_per_block: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
):
    """Update packed Q/K/V convolution state for a decode batch.

    One program handles one request, one Q/K/V projection and ``BD`` channels.
    The physical state remains packed as ``[block, history, 3 * D]``; block
    table indirection is resolved inside the kernel so CUDA Graph replay sees
    the live request-to-state mapping without Python-side gather/scatter ops.
    """

    i_b = tl.program_id(0)
    i_p = tl.program_id(1)
    i_d = tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW)
    m_d = o_d < D
    m_w = o_w < W

    sequence_length_plus_one = tl.load(sequence_lengths_plus_one + i_b).to(tl.int64)
    # ``sequence_lengths_plus_one`` is past_length + one decode token.  The
    # old state therefore lives at past_length - 1 and the new state at
    # past_length. Invalid pages use a masked block-0 address; they are never
    # aliased to the last valid table entry.
    read_page_raw = (sequence_length_plus_one - 2) // seq_size_per_block
    write_page_raw = (sequence_length_plus_one - 1) // seq_size_per_block
    read_position_valid = sequence_length_plus_one > 1
    write_position_valid = sequence_length_plus_one > 0
    read_page_valid = (
        read_position_valid & (read_page_raw >= 0) & (read_page_raw < max_block_count)
    )
    write_page_valid = (
        write_position_valid
        & (write_page_raw >= 0)
        & (write_page_raw < max_block_count)
    )
    read_page = tl.where(read_page_valid, read_page_raw, 0)
    write_page = tl.where(write_page_valid, write_page_raw, 0)
    read_block_id = tl.load(
        block_map + i_b * stride_bm_b + read_page * stride_bm_page,
        mask=read_page_valid,
        other=0,
    ).to(tl.int64)
    write_block_id = tl.load(
        block_map + i_b * stride_bm_b + write_page * stride_bm_page,
        mask=write_page_valid,
        other=0,
    ).to(tl.int64)
    read_valid = (
        read_page_valid & (read_block_id > 0) & (read_block_id < physical_block_count)
    )
    write_valid = (
        write_page_valid
        & (write_block_id > 0)
        & (write_block_id < physical_block_count)
    )

    b_q = tl.load(
        q + i_b * stride_q_b + o_d * stride_q_d,
        mask=m_d & (i_p == 0),
        other=0,
    ).to(tl.float32)
    b_k = tl.load(
        k + i_b * stride_k_b + o_d * stride_k_d,
        mask=m_d & (i_p == 1),
        other=0,
    ).to(tl.float32)
    b_v = tl.load(
        v + i_b * stride_v_b + o_d * stride_v_d,
        mask=m_d & (i_p == 2),
        other=0,
    ).to(tl.float32)
    b_x = b_q + b_k + b_v
    packed_d = i_p * D + o_d

    b_history = tl.load(
        conv_state
        + read_block_id * stride_s_block
        + o_w[None, :] * stride_s_w
        + packed_d[:, None] * stride_s_d,
        mask=read_valid & m_d[:, None] & (o_w[None, :] < W - 1),
        other=0,
    ).to(tl.float32)
    b_cache = tl.where((o_w < W - 1)[None, :], b_history, 0.0)
    b_cache = tl.where((o_w == W - 1)[None, :], b_x[:, None], b_cache)
    # Keep the expression identical to the one-request FLA-compatible path:
    # the FP32 cache promotes the product without an extra weight cast.
    b_weight = tl.load(
        weight + packed_d[:, None] * stride_w_d + o_w[None, :] * stride_w_w,
        mask=m_d[:, None] & m_w[None, :],
        other=0,
    )
    b_y = tl.sum(b_cache * b_weight, 1)
    b_y = b_y * tl.sigmoid(b_y)
    tl.store(
        output + i_p * stride_o_p + i_b * stride_o_b + o_d * stride_o_d,
        tl.cast(
            b_y,
            dtype=output.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
        mask=m_d,
    )

    # Shift the compact W-1 history and publish it directly to the physical
    # destination page.  At a page boundary read_block_id and write_block_id
    # intentionally differ.
    b_shifted_history = tl.load(
        conv_state
        + read_block_id * stride_s_block
        + (o_w[None, :] + 1) * stride_s_w
        + packed_d[:, None] * stride_s_d,
        mask=read_valid & m_d[:, None] & (o_w[None, :] < W - 2),
        other=0,
    )
    b_new_history = tl.where((o_w == W - 2)[None, :], b_x[:, None], b_shifted_history)
    tl.store(
        conv_state
        + write_block_id * stride_s_block
        + o_w[None, :] * stride_s_w
        + packed_d[:, None] * stride_s_d,
        b_new_history,
        mask=write_valid & m_d[:, None] & (o_w[None, :] < W - 1),
    )


@triton.jit
def _kimi_kda_short_conv_paged_target_verify_kernel(
    q,
    k,
    v,
    weight,
    conv_state,
    block_map,
    sequence_lengths_plus_one,
    output,
    stride_q_b,
    stride_q_t,
    stride_q_d,
    stride_k_b,
    stride_k_t,
    stride_k_d,
    stride_v_b,
    stride_v_t,
    stride_v_d,
    stride_w_d,
    stride_w_w,
    stride_s_block,
    stride_s_w,
    stride_s_d,
    stride_bm_b,
    stride_bm_page,
    stride_o_p,
    stride_o_b,
    stride_o_t,
    stride_o_d,
    max_block_count,
    physical_block_count,
    seq_size_per_block: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
):
    """Replay a target-verify sequence inside one Triton program.

    This is numerically equivalent to invoking the paged one-token decode
    kernel T times. It resolves the patched read/write mapping and replays the
    checkpoint-copy semantics in one program, removing Python dispatch and
    block-map clones. Each speculative position still publishes its physical
    checkpoint.
    """

    i_b = tl.program_id(0)
    i_p = tl.program_id(1)
    i_d = tl.program_id(2)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW)
    m_d = o_d < D
    m_w = o_w < W
    packed_d = i_p * D + o_d

    sequence_length_plus_one = tl.load(sequence_lengths_plus_one + i_b).to(tl.int64)
    write_page_base_raw = (sequence_length_plus_one - 1) // seq_size_per_block
    b_weight = tl.load(
        weight + packed_d[:, None] * stride_w_d + o_w[None, :] * stride_w_w,
        mask=m_d[:, None] & m_w[None, :],
        other=0,
    )
    previous_checkpoint = tl.zeros((BD, BW), dtype=tl.float32)

    for i_t in tl.static_range(0, T):
        read_page_raw = (sequence_length_plus_one + i_t - 2) // seq_size_per_block
        logical_write_page_raw = (
            sequence_length_plus_one + i_t - 1
        ) // seq_size_per_block
        checkpoint_page_raw = write_page_base_raw + i_t
        read_page_valid = (
            (sequence_length_plus_one + i_t > 1)
            & (read_page_raw >= 0)
            & (read_page_raw < max_block_count)
        )
        checkpoint_page_valid = (checkpoint_page_raw >= 0) & (
            checkpoint_page_raw < max_block_count
        )
        read_page = tl.where(read_page_valid, read_page_raw, 0)
        checkpoint_page = tl.where(checkpoint_page_valid, checkpoint_page_raw, 0)
        original_read_block_id = tl.load(
            block_map + i_b * stride_bm_b + read_page * stride_bm_page,
            mask=read_page_valid,
            other=0,
        ).to(tl.int64)
        checkpoint_block_id = tl.load(
            block_map + i_b * stride_bm_b + checkpoint_page * stride_bm_page,
            mask=checkpoint_page_valid,
            other=0,
        ).to(tl.int64)
        # The old loop patched only the logical write column to the reserved
        # checkpoint block. In-page steps read that copied checkpoint; page
        # transitions read the original block-table entry instead.
        read_from_checkpoint = read_page_raw == logical_write_page_raw
        read_block_id = tl.where(
            read_from_checkpoint, checkpoint_block_id, original_read_block_id
        )
        read_valid = (
            read_page_valid
            & (read_block_id > 0)
            & (read_block_id < physical_block_count)
        )
        checkpoint_valid = (
            checkpoint_page_valid
            & (checkpoint_block_id > 0)
            & (checkpoint_block_id < physical_block_count)
        )

        # Publish the old loop's destination copy before any shifted-history
        # reload. This is the only inter-warp read-after-write dependency in a
        # non-initial step, so one barrier here is sufficient; the final
        # checkpoint store is ordered by the next iteration's barrier.
        if i_t > 0:
            tl.store(
                conv_state
                + checkpoint_block_id * stride_s_block
                + o_w[None, :] * stride_s_w
                + packed_d[:, None] * stride_s_d,
                previous_checkpoint,
                mask=checkpoint_valid & m_d[:, None] & (o_w[None, :] < W - 1),
            )
            tl.debug_barrier()

        b_history = tl.load(
            conv_state
            + read_block_id * stride_s_block
            + o_w[None, :] * stride_s_w
            + packed_d[:, None] * stride_s_d,
            mask=read_valid & m_d[:, None] & (o_w[None, :] < W - 1),
            other=0,
        ).to(tl.float32)
        b_q = tl.load(
            q + i_b * stride_q_b + i_t * stride_q_t + o_d * stride_q_d,
            mask=m_d & (i_p == 0),
            other=0,
        ).to(tl.float32)
        b_k = tl.load(
            k + i_b * stride_k_b + i_t * stride_k_t + o_d * stride_k_d,
            mask=m_d & (i_p == 1),
            other=0,
        ).to(tl.float32)
        b_v = tl.load(
            v + i_b * stride_v_b + i_t * stride_v_t + o_d * stride_v_d,
            mask=m_d & (i_p == 2),
            other=0,
        ).to(tl.float32)
        b_x = b_q + b_k + b_v
        b_cache = tl.where((o_w < W - 1)[None, :], b_history, 0.0)
        b_cache = tl.where((o_w == W - 1)[None, :], b_x[:, None], b_cache)
        b_y = tl.sum(b_cache * b_weight, 1)
        b_y = b_y * tl.sigmoid(b_y)
        tl.store(
            output
            + i_p * stride_o_p
            + i_b * stride_o_b
            + i_t * stride_o_t
            + o_d * stride_o_d,
            tl.cast(b_y, dtype=output.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=m_d,
        )

        b_shifted_history = tl.load(
            conv_state
            + read_block_id * stride_s_block
            + (o_w[None, :] + 1) * stride_s_w
            + packed_d[:, None] * stride_s_d,
            mask=read_valid & m_d[:, None] & (o_w[None, :] < W - 2),
            other=0,
        )
        previous_checkpoint = tl.where(
            (o_w == W - 2)[None, :], b_x[:, None], b_shifted_history
        )
        tl.store(
            conv_state
            + checkpoint_block_id * stride_s_block
            + o_w[None, :] * stride_s_w
            + packed_d[:, None] * stride_s_d,
            previous_checkpoint,
            mask=checkpoint_valid & m_d[:, None] & (o_w[None, :] < W - 1),
        )


@torch.compiler.disable
def kimi_kda_short_conv_paged_prefill(
    mixed_qkv: torch.Tensor,
    fused_weight: torch.Tensor,
    conv_state: torch.Tensor,
    linear_block_map: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    page_size: int,
    metadata: KimiKDAShortConvMetadata,
    *,
    current_conv_state: torch.Tensor | None = None,
    continuation_mask: torch.Tensor | None = None,
    return_final_state: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Run fused packed Q/K/V convolution and return contiguous Q/K/V planes."""

    if mixed_qkv.ndim != 2 or fused_weight.ndim != 2:
        raise ValueError("paged KDA conv expects mixed_qkv=[T,3D], weight=[3D,W]")
    tokens, channels = mixed_qkv.shape
    if channels % 3:
        raise ValueError(
            f"paged KDA conv channels must be divisible by 3, got {channels}"
        )
    projection_size = channels // 3
    if fused_weight.shape[0] != channels or fused_weight.shape[1] < 2:
        raise ValueError(
            "paged KDA conv weight shape does not match fused input: "
            f"input={tuple(mixed_qkv.shape)} weight={tuple(fused_weight.shape)}"
        )
    history_size = int(fused_weight.shape[1]) - 1
    if conv_state.ndim != 3 or tuple(conv_state.shape[1:]) != (
        history_size,
        channels,
    ):
        raise ValueError(
            "paged KDA conv cache must be [physical_blocks,history,3D], got "
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
            "paged KDA conv sequence metadata disagree: "
            f"sequences={sequence_count} blocks={tuple(linear_block_map.shape)} "
            f"prefixes={tuple(prefix_lengths.shape)}"
        )
    if page_size <= 0 or page_size % _PREFILL_BLOCK_T:
        raise ValueError(
            "paged KDA conv page size must be a positive multiple of "
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
        raise ValueError("paged KDA conv requires CUDA tensors")
    if mixed_qkv.stride(1) != 1 or fused_weight.stride(1) != 1:
        raise ValueError("paged KDA conv requires channel-last input and weights")
    if linear_block_map.dtype not in (torch.int32, torch.int64):
        raise ValueError("paged KDA conv LINEAR block map must be int32/int64")
    if prefix_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("paged KDA conv prefix lengths must be int32/int64")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("paged KDA conv cu_seqlens must be int32/int64")
    if metadata.total_chunks <= 0 and tokens > 0:
        raise ValueError("paged KDA conv metadata contains no token chunks")
    has_current_state = current_conv_state is not None
    if has_current_state != (continuation_mask is not None):
        raise ValueError(
            "current KDA conv state and continuation mask must be provided together"
        )
    if has_current_state:
        assert current_conv_state is not None and continuation_mask is not None
        if tuple(current_conv_state.shape) != (
            sequence_count,
            history_size,
            channels,
        ):
            raise ValueError(
                "current KDA conv state must be [N,history,3D], got "
                f"{tuple(current_conv_state.shape)}"
            )
        if continuation_mask.ndim != 1 or continuation_mask.numel() != sequence_count:
            raise ValueError("KDA continuation mask must be [N]")
        if not current_conv_state.is_cuda or not continuation_mask.is_cuda:
            raise ValueError("current KDA conv state requires CUDA tensors")
        if current_conv_state.dtype != mixed_qkv.dtype:
            raise ValueError("current KDA conv state dtype must match projected QKV")

    output = torch.empty(
        (3, tokens, projection_size),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
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
    block_d = 64
    grid = (metadata.total_chunks, triton.cdiv(channels, block_d))
    _kimi_kda_short_conv_paged_prefill_kernel[grid](
        mixed_qkv,
        fused_weight,
        conv_state,
        linear_block_map,
        prefix_lengths,
        cu_seqlens,
        metadata.batch_ptr,
        metadata.token_chunk_offset_ptr,
        output,
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
        num_warps=4,
    )
    return output[0], output[1], output[2], final_state


@torch.compiler.disable
def kimi_kda_short_conv_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    """Run FLA-compatible one-token cache update without mutating history."""

    if x.ndim != 1 or weight.ndim != 2 or history.ndim != 2:
        raise ValueError(
            "KDA decode short conv expects x=[channels], "
            "weight=[channels,kernel], history=[channels,kernel-1]"
        )
    channels = x.shape[0]
    if weight.shape[0] != channels:
        raise ValueError("KDA short conv weight channel count does not match input")
    kernel_size = int(weight.shape[1])
    if tuple(history.shape) != (channels, kernel_size - 1):
        raise ValueError(
            "KDA short conv history must have shape "
            f"{(channels, kernel_size - 1)}, got {tuple(history.shape)}"
        )
    if not x.is_cuda:
        raise ValueError("KDA Triton short conv requires CUDA input")

    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    block_d = 64
    grid = (triton.cdiv(channels, block_d),)
    _kimi_kda_short_conv_decode_kernel[grid](
        x,
        history,
        weight,
        output,
        x.stride(0),
        history.stride(0),
        history.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        D=channels,
        W=kernel_size,
        BW=triton.next_power_of_2(kernel_size),
        BD=block_d,
        num_warps=4,
    )
    return output


def is_kimi_kda_short_conv_paged_decode_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    block_map: torch.Tensor,
    sequence_lengths_plus_one: torch.Tensor,
    seq_size_per_block: int,
) -> bool:
    """Return whether the graph-safe packed decode kernel can run."""

    if not q.is_cuda:
        return False
    if q.ndim != 2 or k.shape != q.shape or v.shape != q.shape:
        return False
    if q.dtype != k.dtype or q.dtype != v.dtype:
        return False
    if q.device != k.device or q.device != v.device:
        return False
    batch, projection_size = q.shape
    if weight.ndim != 2 or weight.shape[0] != 3 * projection_size:
        return False
    kernel_size = int(weight.shape[1])
    if kernel_size < 2:
        return False
    if conv_state.ndim != 3 or tuple(conv_state.shape[1:]) != (
        kernel_size - 1,
        3 * projection_size,
    ):
        return False
    if block_map.ndim != 2 or block_map.shape[0] != batch or block_map.shape[1] == 0:
        return False
    if (
        sequence_lengths_plus_one.ndim != 1
        or sequence_lengths_plus_one.numel() != batch
    ):
        return False
    if block_map.dtype not in (torch.int32, torch.int64):
        return False
    if sequence_lengths_plus_one.dtype not in (torch.int32, torch.int64):
        return False
    tensors = (weight, conv_state, block_map, sequence_lengths_plus_one)
    return seq_size_per_block > 0 and all(
        tensor.is_cuda and tensor.device == q.device for tensor in tensors
    )


@torch.compiler.disable
def kimi_kda_short_conv_paged_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    block_map: torch.Tensor,
    sequence_lengths_plus_one: torch.Tensor,
    seq_size_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one packed, indexed short-conv update for the whole decode batch.

    ``conv_state`` is RTP's physical ``[blocks, history, packed QKV]`` cache and
    is updated in place.  The returned Q/K/V tensors are separate contiguous
    planes so the recurrent kernel can consume them without materialization.
    """

    if not is_kimi_kda_short_conv_paged_decode_supported(
        q,
        k,
        v,
        weight,
        conv_state,
        block_map,
        sequence_lengths_plus_one,
        seq_size_per_block,
    ):
        raise ValueError("unsupported KDA paged short-conv decode inputs")

    batch, projection_size = q.shape
    kernel_size = int(weight.shape[1])
    output = torch.empty(
        (3, batch, projection_size),
        dtype=q.dtype,
        device=q.device,
    )
    block_d = 64
    grid = (batch, 3, triton.cdiv(projection_size, block_d))
    _kimi_kda_short_conv_paged_decode_kernel[grid](
        q,
        k,
        v,
        weight,
        conv_state,
        block_map,
        sequence_lengths_plus_one,
        output,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        block_map.stride(0),
        block_map.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_map.shape[1],
        conv_state.shape[0],
        seq_size_per_block=seq_size_per_block,
        D=projection_size,
        W=kernel_size,
        BW=triton.next_power_of_2(kernel_size),
        BD=block_d,
        num_warps=4,
    )
    return output[0], output[1], output[2]


@torch.compiler.disable
def kimi_kda_short_conv_paged_target_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    block_map: torch.Tensor,
    sequence_lengths_plus_one: torch.Tensor,
    seq_size_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a complete multi-token target-verify short convolution.

    Inputs use ``[batch, verify_length, projection]``.  The implementation
    produces and checkpoints every speculative position in one Triton launch.
    """

    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("KDA target verify expects matching Q/K/V [batch,seq,dim]")
    batch, sequence_length, projection_size = q.shape
    if sequence_length <= 0:
        raise ValueError("KDA target verify sequence must not be empty")
    if not is_kimi_kda_short_conv_paged_decode_supported(
        q[:, 0, :],
        k[:, 0, :],
        v[:, 0, :],
        weight,
        conv_state,
        block_map,
        sequence_lengths_plus_one,
        seq_size_per_block,
    ):
        raise ValueError("unsupported KDA paged target-verify short-conv inputs")

    kernel_size = int(weight.shape[1])
    output = torch.empty(
        (3, batch, sequence_length, projection_size),
        dtype=q.dtype,
        device=q.device,
    )
    block_d = 64
    grid = (batch, 3, triton.cdiv(projection_size, block_d))
    _kimi_kda_short_conv_paged_target_verify_kernel[grid](
        q,
        k,
        v,
        weight,
        conv_state,
        block_map,
        sequence_lengths_plus_one,
        output,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        block_map.stride(0),
        block_map.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        block_map.shape[1],
        conv_state.shape[0],
        seq_size_per_block=seq_size_per_block,
        T=sequence_length,
        D=projection_size,
        W=kernel_size,
        BW=triton.next_power_of_2(kernel_size),
        BD=block_d,
        num_warps=4,
    )
    return output[0], output[1], output[2]
