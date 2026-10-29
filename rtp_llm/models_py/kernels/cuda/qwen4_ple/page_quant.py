"""Page-local FP8 domains with two batched, independently scaled projections."""

import torch
import triton
import triton.language as tl


def _valid_page_size(page_size) -> bool:
    """Pages are positive int multiples of 64; bool is not a legal page."""
    return type(page_size) is int and page_size > 0 and page_size % 64 == 0


@triton.jit(do_not_specialize=["ROWS"])
def _page_amax(
    X,
    Partial,
    ROWS,
    K: tl.constexpr,
    PAGE: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page, tile = tl.program_id(0), tl.program_id(1)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    absolute = page * PAGE * K + offset
    x = tl.load(X + absolute, (offset < PAGE * K) & (absolute < ROWS * K), other=0).to(
        tl.float32
    )
    x = tl.where(x == x, tl.abs(x), 0.0)
    tl.store(Partial + page * TILES + tile, tl.max(x, 0))


@triton.jit(do_not_specialize=["ROWS", "PADDED"])
def _page_quant(
    X,
    Partial,
    Quantized,
    RawScale,
    KeyScale,
    ValueScale,
    RowKeyScale,
    RowValueScale,
    WK,
    WV,
    ROWS,
    PADDED,
    K: tl.constexpr,
    PAGE: tl.constexpr,
    TILES: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_SCALES: tl.constexpr,
    SM90_NATIVE_SCALE: tl.constexpr,
):
    page, tile = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, REDUCE)
    maximum = tl.max(tl.load(Partial + page * TILES + i, i < TILES, other=0.0), 0)
    if SM90_NATIVE_SCALE:
        # Match the native Hopper CUDA quantizer: rounded division with FTZ.
        # Reciprocal multiplication can move BF16 inputs across FP8 ties.
        # Keep the previously validated SM120 arithmetic unchanged.
        normal_min = 1.1754943508222875e-38
        maximum = tl.where(maximum < normal_min, 0.0, maximum)
        scale = tl.div_rn(maximum, 448.0)
        scale = tl.where(scale < normal_min, 0.0, scale)
    else:
        scale = maximum * (1.0 / 448.0)
    if tile == 0:
        tl.store(RawScale + page, scale)
        # cublasLt per-tensor epilogue multiplies the two scales first.
        tl.store(KeyScale + page, scale * tl.load(WK))
        tl.store(ValueScale + page, scale * tl.load(WV))
    if ROW_SCALES:
        # Distribute bounded row-scale stores across quantization tiles.
        # The grid covers ceil(PAGE / 16) even for narrow test inputs, so
        # non-power-of-two pages need neither a large vector nor a serial loop.
        lane = tile * 16 + tl.arange(0, 16)
        row = page * PAGE + lane
        inside = (lane < PAGE) & (row < PADDED)
        tl.store(RowKeyScale + row, scale * tl.load(WK), inside)
        tl.store(RowValueScale + row, scale * tl.load(WV), inside)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    absolute = page * PAGE * K + offset
    valid = (offset < PAGE * K) & (absolute < ROWS * K)
    x = tl.load(X + absolute, valid, other=0.0).to(tl.float32)
    value = x * tl.div_rn(1.0, tl.maximum(scale, 1e-9))
    value = tl.where(value == value, value, 448.0)
    value = tl.minimum(tl.maximum(value, -448.0), 448.0)
    value = tl.where(valid, value, 0.0)
    tl.store(Quantized + absolute, value, (offset < PAGE * K) & (absolute < PADDED * K))


def quantize_pages(x, key_scale, value_scale, page_size=128, *, row_scales=False):
    rows, width = x.shape
    if not _valid_page_size(page_size):
        raise ValueError(
            f"page size must be a positive multiple of 64, got {page_size!r}"
        )
    pages = triton.cdiv(rows, page_size)
    # Keep the prior page-padded GEMM geometry and deterministic zero padding.
    padded = pages * page_size
    tiles = max(triton.cdiv(page_size * width, 4096), triton.cdiv(page_size, 16))
    partial = torch.empty((pages, tiles), device=x.device, dtype=torch.float32)
    scales = [
        torch.empty(pages, device=x.device, dtype=torch.float32) for _ in range(3)
    ]
    quantized = torch.empty((padded, width), device=x.device, dtype=torch.float8_e4m3fn)
    row_sf = (
        [
            torch.empty((padded, 1), device=x.device, dtype=torch.float32)
            for _ in range(2)
        ]
        if row_scales
        else (None, None)
    )
    _page_amax[(pages, tiles)](
        x, partial, ROWS=rows, K=width, PAGE=page_size, TILES=tiles, BLOCK=4096
    )
    _page_quant[(pages, tiles)](
        x,
        partial,
        quantized,
        *scales,
        *row_sf,
        key_scale,
        value_scale,
        ROWS=rows,
        PADDED=padded,
        K=width,
        PAGE=page_size,
        TILES=tiles,
        REDUCE=triton.next_power_of_2(tiles),
        BLOCK=4096,
        ROW_SCALES=row_scales,
        SM90_NATIVE_SCALE=torch.cuda.get_device_capability(x.device)[0] == 9,
    )
    return quantized, (scales[0], *row_sf) if row_scales else scales
