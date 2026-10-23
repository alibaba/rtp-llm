"""Exact integer lowering of the audited SM120 combined-table split."""

import torch
import triton
import triton.language as tl


@triton.jit
def _divmod_slot(value, divisor, multiplier, shift):
    # For nonnegative int32 slots, ceil(2**(32+shift)/divisor) gives a
    # quotient at most one too large. Correct it without runtime division.
    value = value.to(tl.uint32)
    quotient = (tl.umulhi(value, multiplier.to(tl.uint32)) + value) >> shift
    quotient -= (quotient * divisor > value).to(tl.uint32)
    return quotient, value - quotient * divisor


@triton.jit(
    do_not_specialize=[
        "WIDTH",
        "M",
        "N",
        "WINDOW",
        "EXTRA_WIDTH",
        "EXTRA_PAD",
        "M_MULTIPLIER",
        "M_SHIFT",
    ]
)
def _split_combined_indices_kernel(
    combined,
    combined_lens,
    swa_out,
    swa_lens,
    extra_out,
    extra_lens,
    WIDTH,
    M,
    N,
    WINDOW,
    EXTRA_WIDTH,
    EXTRA_PAD,
    M_MULTIPLIER,
    M_SHIFT,
    BLOCK_COMBINED: tl.constexpr,
    BLOCK_EXTRA: tl.constexpr,
    BLOCK_SWA: tl.constexpr,
    ALIGNED_WINDOW: tl.constexpr,
    FULL_WINDOW: tl.constexpr,
    FULL_EXTRA: tl.constexpr,
    TIGHT_WIDTH: tl.constexpr,
):
    if FULL_WINDOW:
        WINDOW = BLOCK_SWA
    if FULL_EXTRA:
        EXTRA_WIDTH = BLOCK_EXTRA
        EXTRA_PAD = BLOCK_EXTRA
    if TIGHT_WIDTH:
        WIDTH = BLOCK_EXTRA + BLOCK_SWA
    WIDTH = tl.multiple_of(WIDTH, 128)
    EXTRA_PAD = tl.multiple_of(EXTRA_PAD, 64)
    if ALIGNED_WINDOW:
        WINDOW = tl.multiple_of(WINDOW, 16)
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_COMBINED)
    value = tl.load(combined + row * WIDTH + cols, cols < WIDTH, other=-1)
    safe_value = tl.maximum(value, 0)
    _, local_slot = _divmod_slot(safe_value, M, M_MULTIPLIER, M_SHIFT)
    extra_count = tl.sum(
        ((cols < WIDTH) & (value >= 0) & (local_slot < N)).to(tl.int32), 0
    )
    swa_count = tl.load(combined_lens + row) - extra_count
    tl.store(extra_lens + row, extra_count)
    tl.store(swa_lens + row, swa_count)

    extra_cols = tl.arange(0, BLOCK_EXTRA)
    extra_value = tl.load(
        combined + row * WIDTH + extra_cols, extra_cols < EXTRA_WIDTH, other=-1
    )
    safe_extra = tl.maximum(extra_value, 0)
    extra_page, extra_slot = _divmod_slot(safe_extra, M, M_MULTIPLIER, M_SHIFT)
    extra_index = extra_page * N + extra_slot
    extra_index = tl.where(
        (extra_cols < extra_count) & (extra_value >= 0) & (extra_cols < EXTRA_WIDTH),
        extra_index,
        0,
    )
    tl.store(
        extra_out + row * EXTRA_PAD + extra_cols,
        extra_index.to(tl.int32),
        extra_cols < EXTRA_PAD,
    )

    swa_cols = tl.arange(0, BLOCK_SWA)
    source_cols = tl.minimum(extra_count + swa_cols, WIDTH - 1)
    swa_value = tl.load(
        combined + row * WIDTH + source_cols, swa_cols < WINDOW, other=-1
    )
    safe_swa = tl.maximum(swa_value, 0)
    swa_page, swa_slot = _divmod_slot(safe_swa, M, M_MULTIPLIER, M_SHIFT)
    swa_index = swa_page * (M - N) + swa_slot - N
    swa_index = tl.where((swa_cols < swa_count) & (swa_value >= 0), swa_index, 0)
    tl.store(
        swa_out + row * WINDOW + swa_cols, swa_index.to(tl.int32), swa_cols < WINDOW
    )


def fused_split(
    combined_indices, combined_lens, *, M, N, window_size, extra_width, ratio, device
):
    """Execute the exact split after metadata admission at the production seam."""
    data = (
        combined_indices.squeeze(1) if combined_indices.ndim == 3 else combined_indices
    )
    assert (
        data.ndim == 2
        and data.dtype == torch.int32
        and data.is_cuda
        and data.is_contiguous()
    )
    assert combined_lens.dtype == torch.int32 and combined_lens.is_contiguous()
    assert combined_lens.ndim == 1 and combined_lens.shape[0] == data.shape[0]
    assert combined_lens.device == data.device
    assert (
        2**31 - 1 >= M > N >= 0 and window_size > 0 and 0 < extra_width <= data.shape[1]
    )
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and requested_device.index is None:
        requested_device = torch.device("cuda", torch.cuda.current_device())
    assert requested_device == data.device
    rows, width = data.shape
    assert width % 128 == 0
    padded = triton.cdiv(extra_width, 64) * 64
    swa = torch.empty((rows, window_size), dtype=torch.int32, device=data.device)
    swa_lens = torch.empty((rows,), dtype=torch.int32, device=data.device)
    extra = torch.empty((rows, padded), dtype=torch.int32, device=data.device)
    extra_lens = torch.empty((rows,), dtype=torch.int32, device=data.device)
    if rows:
        shift = (M - 1).bit_length()
        multiplier = ((1 << (32 + shift)) + M - 1) // M - (1 << 32)
        # Keep the scalar ABI int32 even when the unsigned multiplier is large.
        if multiplier >= 1 << 31:
            multiplier -= 1 << 32
        _split_combined_indices_kernel[(rows,)](
            data,
            combined_lens,
            swa,
            swa_lens,
            extra,
            extra_lens,
            WIDTH=width,
            M=M,
            N=N,
            WINDOW=window_size,
            EXTRA_WIDTH=extra_width,
            EXTRA_PAD=padded,
            M_MULTIPLIER=multiplier,
            M_SHIFT=shift,
            BLOCK_COMBINED=triton.next_power_of_2(width),
            BLOCK_EXTRA=triton.next_power_of_2(padded),
            BLOCK_SWA=triton.next_power_of_2(window_size),
            ALIGNED_WINDOW=window_size % 16 == 0,
            FULL_WINDOW=window_size == triton.next_power_of_2(window_size),
            FULL_EXTRA=extra_width == triton.next_power_of_2(padded),
            TIGHT_WIDTH=width
            == triton.next_power_of_2(padded) + triton.next_power_of_2(window_size),
            num_warps=4,
        )
    return swa, swa_lens, extra, extra_lens, 64 if ratio == 4 else 2


def can_fuse_split(
    combined_indices: torch.Tensor,
    combined_lens: torch.Tensor,
    *,
    M: int,
    N: int,
    window_size: int,
    extra_width: int,
    ratio: int,
    device: torch.device,
) -> bool:
    """Metadata-only admission for qualified SM120 producer geometries.

    Unsupported inputs keep the incumbent split. No content scan, device-to-host
    read or length-clamp change is hidden in this predicate.
    """
    from rtp_llm.models_py.utils.arch import is_sm120

    data = combined_indices
    if data.ndim == 3 and data.shape[1] == 1:
        data = data.squeeze(1)
    if (
        data.ndim != 2
        or data.dtype != torch.int32
        or not data.is_cuda
        or not data.is_contiguous()
        or combined_lens.ndim != 1
        or combined_lens.dtype != torch.int32
        or not combined_lens.is_contiguous()
        or combined_lens.device != data.device
        or combined_lens.shape[0] != data.shape[0]
    ):
        return False
    requested = torch.device(device)
    if requested.type == "cuda" and requested.index is None:
        requested = torch.device("cuda", torch.cuda.current_device())
    if requested != data.device or not is_sm120(data.device):
        return False
    return (
        0 <= data.shape[0] <= 8192
        and 128 <= data.shape[1] <= 4096
        and data.shape[1] % 128 == 0
        and 0 <= N < M <= 2**31 - 1
        and 1 <= window_size <= 128
        and 1 <= extra_width <= min(2048, data.shape[1])
        and ratio in (1, 4, 128)
    )
