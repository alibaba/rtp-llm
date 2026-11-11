"""SM90 single-token paged GDN Conv1D and independent scalar gate producer."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qwen4_gdn_decode_conv1d_gate_kernel(
    X,
    W,
    STATE,
    BLOCK_MAP,
    LENGTHS,
    A,
    B,
    A_LOG,
    DT_BIAS,
    OUT,
    G,
    BETA,
    STRIDE_X: tl.constexpr,
    STRIDE_AB: tl.constexpr,
    STRIDE_STATE: tl.constexpr,
    STRIDE_MAP: tl.int64,
    PAGE: tl.constexpr,
    DIM: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    # BA and QKV are independent after projection. One existing conv CTA per
    # request publishes the same scalar gates as fused_gdn_gating.
    if tile == 0:
        h = tl.arange(0, HEAD_BLOCK)
        valid_h = h < HEADS
        raw_a = tl.load(A + row * STRIDE_AB + h, valid_h, other=0).to(tl.float32)
        raw_b = tl.load(B + row * STRIDE_AB + h, valid_h, other=0).to(tl.float32)
        dt = tl.load(DT_BIAS + h, valid_h, other=0).to(tl.float32)
        log_a = tl.load(A_LOG + h, valid_h, other=0).to(tl.float32)
        decay = raw_a + dt
        softplus = tl.where(decay <= 20.0, tl.log(1.0 + tl.exp(decay)), decay)
        tl.store(G + row * HEADS + h, -tl.exp(log_a) * softplus, valid_h)
        # Preserve the intermediate BF16 beta rounding before recurrence.
        tl.store(BETA + row * HEADS + h, tl.sigmoid(raw_b), valid_h)

    length = tl.load(LENGTHS + row).to(tl.int32)
    if length <= 0:
        return  # CUDA Graph padding owns no state page.
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    row_map = BLOCK_MAP + row * STRIDE_MAP
    src_page = tl.load(row_map + (length - 2) // PAGE).to(tl.int64)
    dst_page = tl.load(row_map + (length - 1) // PAGE).to(tl.int64)
    src = STATE + src_page * STRIDE_STATE + channel
    h0, h1, h2 = tl.load(src), tl.load(src + DIM), tl.load(src + 2 * DIM)
    x = tl.load(X + row * STRIDE_X + channel)
    w = W + channel * 4
    w0, w1, w2, w3 = tl.load(w), tl.load(w + 1), tl.load(w + 2), tl.load(w + 3)
    # Same BF16 operands and ordered accumulation as the generic causal conv.
    # History values stay in registers for the in-place/page-crossing write.
    acc = tl.zeros((BLOCK,), tl.float32)
    acc += h0 * w0
    acc += h1 * w1
    acc += h2 * w2
    acc += x * w3
    tl.store(OUT + row * DIM + channel, acc / (1.0 + tl.exp(-acc)))
    if dst_page != -1:
        dst = STATE + dst_page * STRIDE_STATE + channel
        tl.store(dst, h1)
        tl.store(dst + DIM, h2)
        tl.store(dst + 2 * DIM, x)


def try_decode_conv_gate(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    block_map: torch.Tensor | None,
    lengths: torch.Tensor | None,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    page: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return Conv1D output, FP32 g and BF16 beta, or leave the native path.

    This accepts only S=1 inputs from the caller's ordinary decode path. Its
    page table follows the same valid-live-row contract as causal_conv1d_update.
    """
    if x.ndim != 2 or a.ndim != 2:
        return None
    dim, heads, block = x.shape[1], a.shape[1], 256
    # Qwen4 has global Hk=16, Hv=48 and D=128. TP1/2/4/8 preserves
    # Hv/Hk=3 and gives QKV widths 10240/5120/2560/1280. Each width is
    # divisible by BLOCK=256, so every channel CTA is full and independent.
    if not (
        block_map is not None
        and lengths is not None
        and (dim, heads) in ((10240, 48), (5120, 24), (2560, 12), (1280, 6))
        and 0 < x.shape[0] <= 128
        and x.is_cuda
        and torch.version.hip is None
        and page == 64
    ):
        return None
    batch = x.shape[0]
    if not (
        x.dtype == weight.dtype == state.dtype == a.dtype == b.dtype == torch.bfloat16
        and weight.shape == (dim, 4)
        and weight.is_contiguous()
        and state.ndim == 3
        and state.shape[1:] == (3, dim)
        and state.stride()[1:] == (dim, 1)
        and state.stride(0) >= 3 * dim
        and x.stride(1) == 1
        and x.stride(0) >= dim
        and a.shape == b.shape == (batch, heads)
        and a.stride() == b.stride()
        and a.stride(1) == 1
        and a.stride(0) >= heads
        and a_log.shape == dt_bias.shape == (heads,)
        and a_log.is_contiguous()
        and dt_bias.is_contiguous()
        and a_log.dtype in (torch.bfloat16, torch.float32)
        and dt_bias.dtype in (torch.bfloat16, torch.float32)
        and block_map.ndim == 2
        and block_map.shape[0] == batch
        and block_map.stride(1) == 1
        and block_map.dtype in (torch.int32, torch.int64)
        and lengths.shape == (batch,)
        and lengths.dtype == torch.int32
        and lengths.is_contiguous()
        and all(
            tensor.device == x.device
            for tensor in (weight, state, block_map, lengths, a, b, a_log, dt_bias)
        )
        and torch.cuda.get_device_capability(x.device) == (9, 0)
    ):
        return None
    output = x.new_empty((batch, dim))
    g = x.new_empty((1, batch, heads), dtype=torch.float32)
    beta = x.new_empty((1, batch, heads))
    _qwen4_gdn_decode_conv1d_gate_kernel[(batch, dim // block)](
        x,
        weight,
        state,
        block_map,
        lengths,
        a,
        b,
        a_log,
        dt_bias,
        output,
        g,
        beta,
        x.stride(0),
        a.stride(0),
        state.stride(0),
        block_map.stride(0),
        page,
        dim,
        heads,
        triton.next_power_of_2(heads),
        block,
        num_warps=4,
    )
    return output, g, beta
