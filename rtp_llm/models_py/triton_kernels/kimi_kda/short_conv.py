"""Kimi KDA short convolution with FLA-compatible arithmetic.

The K3 correctness model uses FLA's Triton ``ShortConvolution``.  In
particular, prefill accumulates the four taps in chronological order in FP32,
while decode shifts the convolution cache and reduces all four taps.  Keeping
those two paths distinct is important: a composition of eager Torch kernels
can differ by one BF16 ULP and move a token across a later MoE routing
boundary.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kimi_kda_short_conv_prefill_kernel(
    x,
    history,
    weight,
    output,
    final_state,
    T,
    stride_x_t,
    stride_x_d,
    stride_h_d,
    stride_h_w,
    stride_w_d,
    stride_w_w,
    stride_o_t,
    stride_o_d,
    stride_f_d,
    stride_f_w,
    D: tl.constexpr,
    W: tl.constexpr,
    BW: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    LAST_T_BLOCK: tl.constexpr,
    USE_HISTORY: tl.constexpr,
):
    """Forward kernel matching FLA causal_conv1d_fwd_kernel."""

    i_d, i_t = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    o_d_i64 = o_d.to(tl.int64)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    b_w = tl.load(
        weight + o_d[:, None] * stride_w_d + o_w * stride_w_w,
        mask=m_d[:, None] & m_w,
        other=0,
    ).to(tl.float32)
    b_y = tl.zeros((BT, BD), dtype=tl.float32)

    if not USE_HISTORY or i_t * BT >= W:
        for i_w in tl.static_range(-W + 1, 1):
            p_x = tl.make_block_ptr(
                x,
                (T, D),
                (stride_x_t, stride_x_d),
                (i_t * BT + i_w, i_d * BD),
                (BT, BD),
                (1, 0),
            )
            b_yi = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
            b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            o_x_i64 = o_x.to(tl.int64)
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :]
            # RTP stores only the W-1 values consumed by the next call.  FLA
            # stores W values and indexes its initial state at o_x + W, so the
            # equivalent compact-history index is o_x + W - 1.
            m_h = ((o_x >= -W + 1) & (o_x < 0))[:, None] & m_d[None, :]
            b_yi = tl.load(
                x + o_x_i64[:, None] * stride_x_t + o_d_i64[None, :] * stride_x_d,
                mask=m_x,
                other=0,
            ).to(tl.float32)
            b_yi += tl.load(
                history
                + o_d[None, :] * stride_h_d
                + (o_x + W - 1)[:, None] * stride_h_w,
                mask=m_h,
                other=0,
            ).to(tl.float32)
            b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    b_y = b_y * tl.sigmoid(b_y)
    p_output = tl.make_block_ptr(
        output,
        (T, D),
        (stride_o_t, stride_o_d),
        (i_t * BT, i_d * BD),
        (BT, BD),
        (1, 0),
    )
    tl.store(
        p_output,
        tl.cast(
            b_y,
            dtype=p_output.dtype.element_ty,
            fp_downcast_rounding="rtne",
        ),
        boundary_check=(0, 1),
    )

    # Export only the W-1 values needed by the next physical page. This avoids
    # materializing [history, x.T] and slicing it again in Python.
    o_h = tl.arange(0, BW)
    history_size = W - 1
    combined_idx = T + o_h
    from_history = combined_idx < history_size
    history_idx = combined_idx
    x_idx = combined_idx - history_size
    x_idx_i64 = x_idx.to(tl.int64)
    b_final_history = tl.load(
        history + o_d[:, None] * stride_h_d + history_idx[None, :] * stride_h_w,
        mask=m_d[:, None] & (o_h[None, :] < history_size) & from_history[None, :],
        other=0,
    )
    b_final_x = tl.load(
        x + x_idx_i64[None, :] * stride_x_t + o_d_i64[:, None] * stride_x_d,
        mask=m_d[:, None]
        & (o_h[None, :] < history_size)
        & (~from_history[None, :])
        & (x_idx[None, :] >= 0)
        & (x_idx[None, :] < T),
        other=0,
    )
    tl.store(
        final_state + o_d[:, None] * stride_f_d + o_h[None, :] * stride_f_w,
        b_final_history + b_final_x,
        mask=m_d[:, None] & (o_h[None, :] < history_size) & (i_t == LAST_T_BLOCK),
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
    # past_length.  Clamp only the table address; validity still masks empty
    # and synthetic DP streams.
    read_page_unclamped = (sequence_length_plus_one - 2) // seq_size_per_block
    write_page_unclamped = (sequence_length_plus_one - 1) // seq_size_per_block
    last_page = max_block_count - 1
    read_page = tl.minimum(tl.maximum(read_page_unclamped, 0), last_page)
    write_page = tl.minimum(tl.maximum(write_page_unclamped, 0), last_page)
    read_position_valid = sequence_length_plus_one > 1
    write_position_valid = sequence_length_plus_one > 0
    read_block_id = tl.load(
        block_map + i_b * stride_bm_b + read_page * stride_bm_page,
        mask=read_position_valid,
        other=0,
    ).to(tl.int64)
    write_block_id = tl.load(
        block_map + i_b * stride_bm_b + write_page * stride_bm_page,
        mask=write_position_valid,
        other=0,
    ).to(tl.int64)
    read_valid = read_position_valid & (read_block_id > 0)
    write_valid = write_position_valid & (write_block_id > 0)

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
    seq_size_per_block: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
):
    """Replay a target-verify sequence inside one Triton program.

    This is numerically equivalent to invoking the paged one-token decode
    kernel T times, but keeps the W-1 convolution history in registers and
    removes Python dispatch, block-map clones and inter-step cache copies.
    Each speculative position still publishes its physical checkpoint.
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
    reserve_page_unclamped = (sequence_length_plus_one - 2) // seq_size_per_block
    last_page = max_block_count - 1
    reserve_page = tl.minimum(tl.maximum(reserve_page_unclamped, 0), last_page)
    read_block_id = tl.load(
        block_map + i_b * stride_bm_b + reserve_page * stride_bm_page,
        mask=sequence_length_plus_one > 1,
        other=0,
    ).to(tl.int64)
    read_valid = (sequence_length_plus_one > 1) & (read_block_id > 0)
    b_weight = tl.load(
        weight + packed_d[:, None] * stride_w_d + o_w[None, :] * stride_w_w,
        mask=m_d[:, None] & m_w[None, :],
        other=0,
    )

    for i_t in tl.range(0, T):
        # Build this token's W-wide convolution window directly from the
        # original checkpoint and the preceding verify inputs.  This avoids
        # cross-program synchronization while preserving the one-token
        # kernel's reduction order exactly.
        source_t = i_t + o_w - (W - 1)
        history_w = source_t + (W - 1)
        b_history = tl.load(
            conv_state
            + read_block_id * stride_s_block
            + history_w[None, :] * stride_s_w
            + packed_d[:, None] * stride_s_d,
            mask=read_valid
            & m_d[:, None]
            & (source_t[None, :] < 0)
            & (history_w[None, :] < W - 1),
            other=0,
        ).to(tl.float32)
        b_q = tl.load(
            q
            + i_b * stride_q_b
            + source_t[None, :] * stride_q_t
            + o_d[:, None] * stride_q_d,
            mask=m_d[:, None] & (source_t[None, :] >= 0) & (i_p == 0),
            other=0,
        ).to(tl.float32)
        b_k = tl.load(
            k
            + i_b * stride_k_b
            + source_t[None, :] * stride_k_t
            + o_d[:, None] * stride_k_d,
            mask=m_d[:, None] & (source_t[None, :] >= 0) & (i_p == 1),
            other=0,
        ).to(tl.float32)
        b_v = tl.load(
            v
            + i_b * stride_v_b
            + source_t[None, :] * stride_v_t
            + o_d[:, None] * stride_v_d,
            mask=m_d[:, None] & (source_t[None, :] >= 0) & (i_p == 2),
            other=0,
        ).to(tl.float32)
        b_cache = tl.where(source_t[None, :] < 0, b_history, b_q + b_k + b_v)
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

    # Publish checkpoints newest-to-oldest so the final write to the source
    # page cannot change the original history used by earlier checkpoints.
    for reverse_offset in tl.range(0, T):
        i_t = T - 1 - reverse_offset
        state_source_t = i_t + o_w - (W - 2)
        state_history_w = state_source_t + (W - 1)
        state_history = tl.load(
            conv_state
            + read_block_id * stride_s_block
            + state_history_w[None, :] * stride_s_w
            + packed_d[:, None] * stride_s_d,
            mask=read_valid
            & m_d[:, None]
            & (o_w[None, :] < W - 1)
            & (state_source_t[None, :] < 0)
            & (state_history_w[None, :] < W - 1),
            other=0,
        ).to(tl.float32)
        state_q = tl.load(
            q
            + i_b * stride_q_b
            + state_source_t[None, :] * stride_q_t
            + o_d[:, None] * stride_q_d,
            mask=m_d[:, None]
            & (o_w[None, :] < W - 1)
            & (state_source_t[None, :] >= 0)
            & (i_p == 0),
            other=0,
        ).to(tl.float32)
        state_k = tl.load(
            k
            + i_b * stride_k_b
            + state_source_t[None, :] * stride_k_t
            + o_d[:, None] * stride_k_d,
            mask=m_d[:, None]
            & (o_w[None, :] < W - 1)
            & (state_source_t[None, :] >= 0)
            & (i_p == 1),
            other=0,
        ).to(tl.float32)
        state_v = tl.load(
            v
            + i_b * stride_v_b
            + state_source_t[None, :] * stride_v_t
            + o_d[:, None] * stride_v_d,
            mask=m_d[:, None]
            & (o_w[None, :] < W - 1)
            & (state_source_t[None, :] >= 0)
            & (i_p == 2),
            other=0,
        ).to(tl.float32)
        history = tl.where(
            state_source_t[None, :] < 0,
            state_history,
            state_q + state_k + state_v,
        )

        write_page = tl.minimum(reserve_page + i_t, last_page)
        write_block_id = tl.load(
            block_map + i_b * stride_bm_b + write_page * stride_bm_page
        ).to(tl.int64)
        tl.store(
            conv_state
            + write_block_id * stride_s_block
            + o_w[None, :] * stride_s_w
            + packed_d[:, None] * stride_s_d,
            history,
            mask=(write_block_id > 0)
            & m_d[:, None]
            & (o_w[None, :] < W - 1),
        )


@torch.compiler.disable
def kimi_kda_short_conv_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    history: torch.Tensor,
    *,
    use_history: bool,
    output: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FLA-compatible causal short convolution for one packed sequence."""

    if x.ndim != 2 or weight.ndim != 2 or history.ndim != 2:
        raise ValueError("KDA short conv expects x, weight and history to be 2D")
    token_count, channels = x.shape
    if weight.shape[0] != channels:
        raise ValueError("KDA short conv weight channel count does not match input")
    kernel_size = int(weight.shape[1])
    if tuple(history.shape) != (channels, kernel_size - 1):
        raise ValueError(
            "KDA short conv history must have shape "
            f"{(channels, kernel_size - 1)}, got {tuple(history.shape)}"
        )
    if token_count == 0:
        if output is None:
            output = torch.empty_like(x)
        elif tuple(output.shape) != tuple(x.shape):
            raise ValueError(
                "KDA short conv output must have shape "
                f"{tuple(x.shape)}, got {tuple(output.shape)}"
            )
        elif (
            output.dtype != x.dtype
            or output.device != x.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "KDA short conv output must be contiguous and match input "
                f"dtype/device: input={x.dtype}/{x.device}, "
                f"output={output.dtype}/{output.device}"
            )
        if final_state is None:
            final_state = history.clone()
        else:
            final_state.copy_(history)
        return output, final_state
    if not x.is_cuda:
        raise ValueError("KDA Triton short conv requires CUDA input")

    if output is None:
        output = torch.empty_like(x, memory_format=torch.contiguous_format)
    elif tuple(output.shape) != tuple(x.shape):
        raise ValueError(
            "KDA short conv output must have shape "
            f"{tuple(x.shape)}, got {tuple(output.shape)}"
        )
    elif (
        output.dtype != x.dtype
        or output.device != x.device
        or not output.is_contiguous()
    ):
        raise ValueError(
            "KDA short conv output must be contiguous and match input "
            f"dtype/device: input={x.dtype}/{x.device}, "
            f"output={output.dtype}/{output.device}"
        )
    if final_state is None:
        final_state = torch.empty_like(history, memory_format=torch.contiguous_format)
    elif tuple(final_state.shape) != tuple(history.shape):
        raise ValueError(
            "KDA short conv final_state must have shape "
            f"{tuple(history.shape)}, got {tuple(final_state.shape)}"
        )
    elif final_state.dtype != x.dtype or final_state.device != x.device:
        raise ValueError(
            "KDA short conv final_state must match input dtype/device: "
            f"input={x.dtype}/{x.device}, "
            f"final={final_state.dtype}/{final_state.device}"
        )
    block_t = 64
    block_d = 64
    grid = (
        triton.cdiv(channels, block_d),
        triton.cdiv(token_count, block_t),
    )
    _kimi_kda_short_conv_prefill_kernel[grid](
        x,
        history,
        weight,
        output,
        final_state,
        token_count,
        x.stride(0),
        x.stride(1),
        history.stride(0),
        history.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        final_state.stride(0),
        final_state.stride(1),
        D=channels,
        W=kernel_size,
        BW=triton.next_power_of_2(kernel_size),
        BT=block_t,
        BD=block_d,
        LAST_T_BLOCK=triton.cdiv(token_count, block_t) - 1,
        USE_HISTORY=use_history,
        num_warps=4,
    )
    return output, final_state


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
        seq_size_per_block=seq_size_per_block,
        T=sequence_length,
        D=projection_size,
        W=kernel_size,
        BW=triton.next_power_of_2(kernel_size),
        BD=block_d,
        num_warps=4,
    )
    return output[0], output[1], output[2]
