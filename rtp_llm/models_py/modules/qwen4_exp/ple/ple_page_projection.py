"""Page-local FP8 domains with two batched, independently scaled projections."""

import torch
import triton
import triton.language as tl


@triton.jit
def _page_amax(
    X,
    Partial,
    ROWS: tl.constexpr,
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


@triton.jit
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
    ROWS: tl.constexpr,
    PADDED: tl.constexpr,
    K: tl.constexpr,
    PAGE: tl.constexpr,
    TILES: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
    ROW_SCALES: tl.constexpr,
):
    page, tile = tl.program_id(0), tl.program_id(1)
    i = tl.arange(0, REDUCE)
    maximum = tl.max(tl.load(Partial + page * TILES + i, i < TILES, other=0.0), 0)
    scale = maximum * (1.0 / 448.0)
    if tile == 0:
        tl.store(RawScale + page, scale)
        # cublasLt per-tensor epilogue multiplies the two scales first.
        tl.store(KeyScale + page, scale * tl.load(WK))
        tl.store(ValueScale + page, scale * tl.load(WV))
        if ROW_SCALES:
            row = page * PAGE + tl.arange(0, PAGE)
            tl.store(RowKeyScale + row, scale * tl.load(WK))
            tl.store(RowValueScale + row, scale * tl.load(WV))
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
    padded = triton.cdiv(rows, page_size) * page_size
    pages, tiles = triton.cdiv(rows, page_size), triton.cdiv(page_size * width, 4096)
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
    )
    return quantized, (scales[0], *row_sf) if row_scales else scales


def project_pages_pair(x, first, second, page_size=128):
    from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear

    if not (
        isinstance(first, Fp8Linear)
        and isinstance(second, Fp8Linear)
        and first.use_scaled_mm
        and second.use_scaled_mm
        and first.bias is None
        and second.bias is None
        and x.ndim == 3
        and x.shape[0] == 1
        and x.shape[2] == 2560
        and 128 <= x.shape[1] <= 32768
        and page_size == 128
        and x.is_cuda
        and x.is_contiguous()
        and x.dtype == torch.bfloat16
        and torch.version.hip is None
        and torch.cuda.get_device_capability(x.device) == (12, 0)
        and tuple(first.weight.shape) == (10240, 2560)
        and tuple(second.weight.shape) == (2560, 2560)
        and first.weight.device == second.weight.device == x.device
    ):
        return None
    quantized, scales = quantize_pages(
        x.squeeze(0), first.scale, second.scale, row_scales=True
    )
    outputs = []
    for index, projection in enumerate((first, second)):
        columns = getattr(projection, "_qwen4_ple_unit_columns", None)
        if columns is None:
            # Legacy manually assembled component owners may not prepare this
            # immutable layout until their first eager call.
            if torch.cuda.is_current_stream_capturing():
                return None
            columns = projection.unit_scale.expand(
                1, projection.weight.shape[0]
            ).contiguous()
            projection._qwen4_ple_unit_columns = columns
        # Compute alpha=A_page*W first, exactly as the original per-tensor
        # epilogue. Passing separate row A and column W scales changes rounding.
        value = torch._scaled_mm(
            quantized,
            projection.weight.t(),
            scale_a=scales[index + 1],
            scale_b=columns,
            out_dtype=torch.bfloat16,
        )
        value = value[0] if isinstance(value, tuple) else value
        outputs.append(value[: x.shape[1]].unsqueeze(0))
    return (*outputs, None)
