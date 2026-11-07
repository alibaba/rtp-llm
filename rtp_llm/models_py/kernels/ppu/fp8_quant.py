"""PPU per-128-group FP8 activation quantization with FP32 arithmetic."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_quant_128(x_ptr, q_ptr, scale_ptr, HIDDEN: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    cols = group * 128 + tl.arange(0, 128)
    x = tl.load(x_ptr + row * HIDDEN + cols).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 0), 1e-10)
    scale = amax / 448.0
    # Form the multiplier directly from amax. Rounding it, or the product,
    # through BF16 changes FP8 rounding decisions relative to the reference.
    value = x * (448.0 / amax)
    q = tl.minimum(tl.maximum(value, -448.0), 448.0).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + row * HIDDEN + cols, q)
    tl.store(scale_ptr + row * (HIDDEN // 128) + group, scale)


@triton.jit(do_not_specialize=["TOTAL_GROUPS"])
def _fp8_quant_128_tiled(
    x_ptr, q_ptr, scale_ptr, TOTAL_GROUPS, GROUPS: tl.constexpr,
):
    group = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    offset = group[:, None] * 128 + tl.arange(0, 128)[None, :]
    x = tl.load(x_ptr + offset, group[:, None] < TOTAL_GROUPS, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-10)
    scale = amax / 448.0
    q = tl.clamp(x * (448.0 / amax[:, None]), -448.0, 448.0).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + offset, q, group[:, None] < TOTAL_GROUPS)
    tl.store(scale_ptr + group, scale, group < TOTAL_GROUPS)


def fp8_quantize_128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous BF16 [M, K] with row-major FP32 group scales.

    The FP32 scale and quantization order match SGLang's per-token-group
    quantizer; no BF16 reciprocal or intermediate product is introduced.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("FP8 quantization expects contiguous BF16 [M, K]")
    rows, hidden = x.shape
    if hidden == 0 or hidden % 128:
        raise ValueError("hidden size must be a positive multiple of 128")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, hidden // 128), device=x.device, dtype=torch.float32)
    if rows >= 16:
        groups = rows * (hidden // 128)
        _fp8_quant_128_tiled[(triton.cdiv(groups, 4),)](
            x, q, scales, groups, 4, num_warps=4,
        )
    elif rows:
        _fp8_quant_128[(rows, hidden // 128)](x, q, scales, hidden, num_warps=4)
    return q, scales


@triton.jit(do_not_specialize=["ROWS"])
def _silu_mul_fp8_quant_128(
    x_ptr, q_ptr, scale_ptr, ROWS, HIDDEN: tl.constexpr,
    GATE_FIRST: tl.constexpr, GROUPS: tl.constexpr,
):
    groups = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    row = groups // (HIDDEN // 128)
    cols = (groups % (HIDDEN // 128))[:, None] * 128 + tl.arange(0, 128)[None, :]
    base = row[:, None] * (2 * HIDDEN) + cols
    gate = tl.load(x_ptr + base + (0 if GATE_FIRST else HIDDEN), row[:, None] < ROWS, other=0).to(tl.float32)
    up = tl.load(x_ptr + base + (HIDDEN if GATE_FIRST else 0), row[:, None] < ROWS, other=0).to(tl.float32)
    value = (gate * tl.sigmoid(gate) * up).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), 1), 1e-10)
    scale = amax / 448.0
    quantized = tl.clamp(value * (448.0 / amax[:, None]), -448.0, 448.0).to(q_ptr.dtype.element_ty)
    tl.store(q_ptr + row[:, None] * HIDDEN + cols, quantized, row[:, None] < ROWS)
    tl.store(scale_ptr + groups, scale, row < ROWS)


def silu_mul_fp8_quantize_128(
    x: torch.Tensor, *, gate_first: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SiLU/multiply and FP8 quantization, retaining the BF16 boundary.

    Dense MLP uses [gate, up]; routed DeepGEMM experts use [up, gate].
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("SiLU FP8 quantization expects contiguous BF16 [M, 2K]")
    rows, width = x.shape
    if width == 0 or width % 256:
        raise ValueError("SiLU activation width must be a positive multiple of 256")
    hidden = width // 2
    q = torch.empty((rows, hidden), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((rows, hidden // 128), device=x.device, dtype=torch.float32)
    if rows:
        _silu_mul_fp8_quant_128[(triton.cdiv(rows * (hidden // 128), 4),)](
            x, q, scales, rows, hidden, gate_first, 4,
            num_warps=4, enable_fp_fusion=False,
        )
    return q, scales


@triton.jit
def _masked_fp8_quant_128(
    x_ptr,
    q_ptr,
    scale_ptr,
    counts_ptr,
    CAPACITY: tl.constexpr,
    HIDDEN: tl.constexpr,
    SCALE_CAPACITY: tl.constexpr,
):
    expert = tl.program_id(2)
    group = tl.program_id(0)
    row_start = tl.program_id(1)
    row_stride = tl.num_programs(1)
    count = tl.load(counts_ptr + expert)
    cols = group * 128 + tl.arange(0, 128)
    for row in range(row_start, count, row_stride):
        offset = (expert * CAPACITY + row) * HIDDEN + cols
        x = tl.load(x_ptr + offset).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), 0), 1e-10)
        scale = amax / 448.0
        scaled = x * (448.0 / amax)
        q = tl.minimum(tl.maximum(scaled, -448.0), 448.0).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + offset, q)
        scale_offset = (expert * (HIDDEN // 128) + group) * SCALE_CAPACITY + row
        tl.store(scale_ptr + scale_offset, scale)


def masked_fp8_quantize_128(
    x: torch.Tensor, masked_m: torch.Tensor, expected_m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize valid rows only, with column-major scales for masked DeepGEMM.

    Padding is deliberately left unwritten: masked GEMM/DeepEP consume only
    rows below each GPU-resident count. Counts may change during graph replay.
    ``expected_m`` only sizes the launch; the loop covers every valid row.
    """
    if x.ndim != 3 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("masked FP8 quantization expects contiguous BF16 [E, M, K]")
    experts, capacity, hidden = x.shape
    if hidden == 0 or hidden % 128 or masked_m.shape != (experts,):
        raise ValueError("hidden size must divide by 128 and counts must match experts")
    scale_capacity = triton.cdiv(capacity, 4) * 4
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (experts, hidden // 128, scale_capacity), device=x.device, dtype=torch.float32
    ).transpose(1, 2)[:, :capacity, :]
    if x.numel():
        rows = max(1, min(expected_m, capacity, 8))
        _masked_fp8_quant_128[(hidden // 128, rows, experts)](
            x, q, scales, masked_m, capacity, hidden, scale_capacity,
            num_warps=4, num_stages=1,
        )
    return q, scales


@triton.jit
def _masked_silu_mul_fp8_quant_128(
    x_ptr, q_ptr, scale_ptr, counts_ptr,
    CAPACITY: tl.constexpr, HIDDEN: tl.constexpr, SCALE_CAPACITY: tl.constexpr,
):
    expert = tl.program_id(2)
    group = tl.program_id(0)
    row_start = tl.program_id(1)
    row_stride = tl.num_programs(1)
    count = tl.load(counts_ptr + expert)
    cols = group * 128 + tl.arange(0, 128)
    for row in range(row_start, count, row_stride):
        offset = (expert * CAPACITY + row) * HIDDEN * 2 + cols
        up = tl.load(x_ptr + offset).to(tl.float32)
        gate = tl.load(x_ptr + offset + HIDDEN).to(tl.float32)
        # Match a BF16 SiLU-and-mul output followed by group quantization.
        value = ((gate * tl.sigmoid(gate)) * up).to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(value), 0), 1e-10)
        scale = amax / 448.0
        scaled = value * (448.0 / amax)
        q = tl.minimum(tl.maximum(scaled, -448.0), 448.0).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + (expert * CAPACITY + row) * HIDDEN + cols, q)
        tl.store(scale_ptr + (expert * (HIDDEN // 128) + group) * SCALE_CAPACITY + row, scale)


def masked_silu_mul_fp8_quantize_128(
    x: torch.Tensor, masked_m: torch.Tensor, expected_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused [up, gate] SiLU/FP8 with a BF16 activation rounding boundary."""
    if x.ndim != 3 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("masked SiLU FP8 quantization expects contiguous BF16 [E, M, 2K]")
    experts, capacity, width = x.shape
    if width == 0 or width % 256 or masked_m.shape != (experts,):
        raise ValueError("activation width must divide by 256 and counts must match experts")
    hidden = width // 2
    scale_capacity = triton.cdiv(capacity, 4) * 4
    q = torch.empty((experts, capacity, hidden), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((experts, hidden // 128, scale_capacity), device=x.device,
                         dtype=torch.float32).transpose(1, 2)[:, :capacity, :]
    if x.numel():
        rows = max(1, min(expected_m, capacity, 8))
        _masked_silu_mul_fp8_quant_128[(hidden // 128, rows, experts)](
            x, q, scales, masked_m, capacity, hidden, scale_capacity,
            num_warps=4, num_stages=1,
        )
    return q, scales
