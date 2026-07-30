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
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d[None, :]
            # RTP stores only the W-1 values consumed by the next call.  FLA
            # stores W values and indexes its initial state at o_x + W, so the
            # equivalent compact-history index is o_x + W - 1.
            m_h = ((o_x >= -W + 1) & (o_x < 0))[:, None] & m_d[None, :]
            b_yi = tl.load(
                x + o_x[:, None] * stride_x_t + o_d[None, :] * stride_x_d,
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
    b_final_history = tl.load(
        history + o_d[:, None] * stride_h_d + history_idx[None, :] * stride_h_w,
        mask=m_d[:, None] & (o_h[None, :] < history_size) & from_history[None, :],
        other=0,
    )
    b_final_x = tl.load(
        x + x_idx[None, :] * stride_x_t + o_d[:, None] * stride_x_d,
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
