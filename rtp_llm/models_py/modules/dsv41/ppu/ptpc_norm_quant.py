"""Experimental single-kernel RMSNorm/PTPC producer with an ATen-shaped sum.

This candidate is not wired into production. Its reduction follows the
installed Torch 2.11 Reduce.cuh FP32 contiguous MeanOps specialization:
four lane-local accumulators, their left-to-right combination, block-x folds,
then descending warp shuffle folds. A plain sum over the feature tile would
change that order. Native byte/Graph validation remains a promotion condition.
"""

from functools import lru_cache
import math
import struct

import torch

from rtp_llm.models_py.modules.dsv41.ppu.quant_conversion import FP8_INV_MAX


WIDTHS = (512, 1280, 5120)
MAX_ROWS = 65536


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def aten_mean_launch_config(rows, width):
    """The traced CUDA-like Torch 2.11 config, not reduce_kernel's bound512.

    The canonical square output is a new aligned FP32 tensor. For these
    shapes input_vec_size=4, output_vec_size=1, warp_size=32, max_threads=512.
    Block-y processes independent output rows; no block-y/global sum occurs.
    """
    if rows <= 0 or rows > MAX_ROWS or width not in WIDTHS:
        raise ValueError("ATen-shaped norm sum requires 1..65536 rows and a supported width")
    dim0 = min(1 << ((width // 4).bit_length() - 1), 512)
    dim1 = min(1 << (rows.bit_length() - 1), 512)
    block_width = min(dim0, 32)
    block_height = min(dim1, 512 // block_width)
    block_width = min(dim0, 512 // block_height)
    values_per_thread = (width + block_width - 1) // block_width
    if values_per_thread >= min(block_height * 16, 256):
        raise ValueError("ATen block-y reduction is outside this candidate's contract")
    # MeanOps project uses a host FP32 M/(M*K), after the sum.
    factor = _f32(_f32(rows) / _f32(rows * width))
    return block_width, block_height, factor


def is_supported(hidden, gamma, eps):
    if (
        not torch.__version__.startswith("2.11.")
        or not hidden.is_cuda
        or hidden.dtype != torch.bfloat16
        or hidden.layout != torch.strided
        or hidden.ndim not in (2, 3)
        or not hidden.is_contiguous()
        or hidden.shape[-1] not in WIDTHS
        or gamma.dtype != torch.bfloat16
        or gamma.layout != torch.strided
        or gamma.ndim != 1
        or gamma.numel() != hidden.shape[-1]
        or not gamma.is_contiguous()
        or gamma.device != hidden.device
        or hidden.requires_grad
        or gamma.requires_grad
        or torch.is_grad_enabled()
        or not isinstance(eps, (int, float))
        or not math.isfinite(eps)
        or eps < torch.finfo(torch.float32).tiny
        or eps > torch.finfo(torch.float32).max
    ):
        return False
    return hidden.numel() // hidden.shape[-1] <= MAX_ROWS


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def widen_bf16(value):
        return (value.to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(
            tl.float32, bitcast=True
        )

    @triton.jit
    def add_rn(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
            dtype=tl.float32, is_pure=True, pack=1,
        )

    @triton.jit
    def mul_rn(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b],
            dtype=tl.float32, is_pure=True, pack=1,
        )

    @triton.jit
    def norm_quant(
        X, Gamma, Norm, Quant, Scale,
        WIDTH: tl.constexpr,
        LANES: tl.constexpr,
        LOG_LANES: tl.constexpr,
        FACTOR: tl.constexpr,
        EPS: tl.constexpr,
        INV_MAX: tl.constexpr,
        BLOCK: tl.constexpr,
        MAKE_QUANT: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        lane = tl.arange(0, LANES)
        vector_col = lane[:, None] * 4 + tl.arange(0, 4)[None, :]
        accum = tl.full((LANES, 4), 0.0, tl.float32)
        # CUDA input_vectorized_thread_reduce_impl advances vector indices
        # by block_width. Each of the four components has its own accumulator.
        for chunk in tl.static_range(0, triton.cdiv(WIDTH, LANES * 4)):
            columns = vector_col + chunk * LANES * 4
            values = widen_bf16(tl.load(X + row * WIDTH + columns, columns < WIDTH, 0))
            square = mul_rn(values, values)
            accum = tl.where(columns < WIDTH, add_rn(accum, square), accum)
        even, odd = tl.split(tl.reshape(accum, (LANES, 2, 2)))
        value0, value2 = tl.split(even)
        value1, value3 = tl.split(odd)
        partial = add_rn(add_rn(add_rn(value0, value1), value2), value3)

        # Match the shared-memory folds above a hardware warp first.
        for step in tl.static_range(0, LOG_LANES - 5):
            offset = LANES >> (step + 1)
            source = tl.minimum(lane + offset, LANES - 1)
            other = tl.gather(partial, source, axis=0)
            partial = tl.where(lane < offset, add_rn(partial, other), partial)
        # Match CUDA shfl_down16,8,4,2,1. Out-of-warp lanes retain their own
        # value just as __shfl_down_sync does; only lane0 is finally projected.
        for step in tl.static_range(0, 5):
            offset = 16 >> step
            source = tl.where((lane % 32) + offset < 32, lane + offset, lane)
            partial = add_rn(partial, tl.gather(partial, source, axis=0))
        total = tl.reshape(tl.gather(partial, tl.full((1,), 0, tl.int32), axis=0), ())
        variance = mul_rn(total, FACTOR)
        inverse = tl.rsqrt(add_rn(variance, EPS))

        columns = tl.arange(0, BLOCK)
        mask = columns < WIDTH
        values = widen_bf16(tl.load(X + row * WIDTH + columns, mask, 0))
        gamma = widen_bf16(tl.load(Gamma + columns, mask, 0))
        normalized = mul_rn(mul_rn(values, inverse), gamma).to(tl.bfloat16)
        tl.store(Norm + row * WIDTH + columns, normalized, mask)
        if MAKE_QUANT:
            # Materialize the same BF16 landing before the frozen whole-row
            # PTPC arithmetic. Padding must not participate in the row maximum.
            quant_input = widen_bf16(normalized)
            maximum = tl.maximum(tl.max(tl.where(mask, tl.abs(quant_input), 0.0), 0), 1.0e-10)
            scale = mul_rn(maximum, INV_MAX)
            inverse_scale = mul_rn(tl.div_rn(1.0, maximum), 448.0)
            payload = mul_rn(quant_input, inverse_scale)
            payload = tl.minimum(tl.maximum(payload, -448.0), 448.0)
            tl.store(Quant + row * WIDTH + columns, payload, mask)
            tl.store(Scale + row, scale)

    return triton, norm_quant


def _candidate(hidden, gamma, eps, make_quant):
    if not is_supported(hidden, gamma, eps):
        return None
    width = hidden.shape[-1]
    rows = hidden.numel() // width
    normalized = torch.empty_like(hidden)
    quantized = torch.empty_like(hidden, dtype=torch.float8_e4m3fn) if make_quant else normalized
    scales = torch.empty((*hidden.shape[:-1], 1), dtype=torch.float32, device=hidden.device) if make_quant else normalized
    if rows:
        lanes, _, factor = aten_mean_launch_config(rows, width)
        triton, kernel = _kernels()
        kernel[(rows,)](
            hidden, gamma, normalized, quantized, scales,
            WIDTH=width, LANES=lanes, LOG_LANES=lanes.bit_length() - 1,
            FACTOR=factor, EPS=_f32(eps), INV_MAX=FP8_INV_MAX,
            BLOCK=triton.next_power_of_2(width), MAKE_QUANT=make_quant,
            num_warps=4 if width <= 1280 else 8,
            num_stages=1, enable_fp_fusion=False,
        )
    return (normalized, quantized, scales) if make_quant else normalized


def ptpc_rms_norm_candidate(hidden, gamma, eps=1e-6):
    """Return one-kernel norm candidate, or None for unsupported metadata.

    Inputs are finite BF16 inference tensors on the Torch2.11 PPU runtime.
    The kernel has not been promoted until exact native component/Graph tests
    establish that its rsqrt and BF16 landing match the installed provider.
    """
    return _candidate(hidden, gamma, eps, False)


def ptpc_rms_norm_quant_candidate(hidden, gamma, eps=1e-6):
    """Return (BF16 norm, E4M3 payload, FP32 [...,1] scale), or None.

    This is an explicit producer result. A consumer must use the returned
    payload/scale directly to avoid running a second activation quantizer.
    There is no global tensor cache or stale Graph-replay reuse.
    """
    return _candidate(hidden, gamma, eps, True)
