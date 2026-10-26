"""RTP G32 quantization with direct FP32 scales for SM90 DeepGEMM.

No GEMM or scale-policy change: mirror the local legacy/v2 CUDA quantizers,
including their different small-value floors. WO_A also preserves the
existing SM90 inverse-RoPE BF16 boundary and group-major physical layout.
Prepared packed-scale consumers may use the one-launch unpack helper, but
newly quantized inputs should avoid allocating packed scales entirely.

Arithmetic sources: bindings/cuda/kernels/per_token_group_quant_8bit{,_v2}.cu
and _fused_inv_rope_fp8_quant_triton.py. Upstream quantization references:
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/fp8_utils.py
The local CUDA kernels remain the compatibility oracle, not upstream main.
"""

import os

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


def use_v2(rows, width):
    policy = os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
    if policy not in ("auto", "legacy", "v2"):
        raise ValueError("DSV4_FP8_QUANT_KERNEL must be one of auto, legacy, v2")
    return policy == "v2" or (policy == "auto" and rows * width >= 4 * 1024 * 1024)


@triton.jit
def _decode_exponent(exponent):
    bits = exponent.to(tl.uint32) << 23
    bits = tl.where(exponent == 0, 0x00400000, bits)
    # Match Torch Float8_e8m0fnu::operator float exactly, including its
    # signaling-NaN payload. Substituting a canonical qNaN changes the ABI.
    bits = tl.where(exponent == 255, 0x7F800001, bits)
    return bits.to(tl.uint32).to(tl.float32, bitcast=True)


@triton.jit
def _quantize(values, V2: tl.constexpr, EPS: tl.constexpr = 1.1754943508222875e-38):
    # CUDA fmaxf ignores NaNs during absmax and the final saturation.
    magnitude = tl.where(values == values, tl.abs(values), 0.0)
    maximum = tl.maximum(tl.max(magnitude, 1), EPS)
    if V2:
        raw = tl.inline_asm_elementwise(
            "mul.rn.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            [maximum, 1.0 / 448.0],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        raw_bits = raw.to(tl.uint32, bitcast=True)
        exponent = ((raw_bits >> 23) & 255) + ((raw_bits & 0x7FFFFF) != 0)
        # Matches fast_pow2(-fast_log2_ceil(raw)), including raw == 0.
        inverse = ((254 - exponent) << 23).to(tl.uint32).to(tl.float32, bitcast=True)
        scaled = tl.inline_asm_elementwise(
            "mul.rn.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            [values, inverse[:, None]],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        raw = tl.maximum(maximum / 448.0, 1.0e-10)
        scale = tl.exp2(tl.ceil(tl.log2(raw)))
        exponent = (scale.to(tl.uint32, bitcast=True) >> 23) & 255
        # Mirror the legacy float-to-int conversion for an infinite scale;
        # extracting its exponent byte would not reproduce that conversion.
        infinity_code = (libdevice.float2int_rz(scale) + 127) & 255
        exponent = tl.where(scale == float("inf"), infinity_code, exponent)
        scaled = values / scale[:, None]
    scaled = tl.where(scaled == scaled, scaled, -448.0)
    payload = tl.minimum(tl.maximum(scaled, -448.0), 448.0).to(tl.float8e4nv)
    # Return the decoded *stored byte*, not the internal v2 scale: byte zero
    # denotes 2**-127 in UE8M0, whereas v2's internal fast_pow2(-127) is +0.
    return payload, _decode_exponent(exponent & 255)


@triton.jit(do_not_specialize=["rows", "input_stride"])
def _dense_quant_kernel(
    x, q, scales, rows, input_stride, scale_row_stride, scale_column_stride,
    K: tl.constexpr, V2: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1) * 4 + tl.arange(0, 4)
    columns = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    values = tl.load(x + row * input_stride + columns, columns < K, other=0).to(
        tl.float32
    )
    payload, scale = _quantize(values, V2)
    tl.store(q + row * K + columns, payload, columns < K)
    tl.store(
        scales + row * scale_row_stride + group * scale_column_stride,
        scale, group < K // 32,
    )


@triton.jit(
    do_not_specialize=["rows", "input_stride", "head_stride", "frequency_stride"]
)
def _woa_quant_kernel(
    x,
    frequencies,
    q,
    scales,
    rows,
    input_stride,
    head_stride,
    frequency_stride,
    frequency_column_stride,
    scale_group_stride,
    scale_row_stride,
    scale_column_stride,
    G: tl.constexpr,
    HEADS: tl.constexpr,
    TOKENS_PER_FREQ: tl.constexpr,
    V2: tl.constexpr,
    IMAGINARY_FMA_FIRST: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    group, local_head = head // HEADS, head % HEADS
    columns = tl.arange(0, 512)
    base = x + row * input_stride + head * head_stride
    values = tl.load(base + columns).to(tl.float32)
    rotary = columns >= 448
    partner = tl.load(base + (columns ^ 1), rotary, other=0).to(tl.float32)
    angle = tl.maximum((columns - 448) // 2, 0)
    frequency = (
        frequencies
        + (row // TOKENS_PER_FREQ) * frequency_stride
        + angle * frequency_column_stride
    )
    cosine = tl.load(frequency, rotary, other=1)
    sine = tl.load(frequency + 1, rotary, other=0)
    real = tl.fma(values, cosine, partner * sine)
    if IMAGINARY_FMA_FIRST:
        imag = tl.fma(partner, -sine, values * cosine)
    else:
        imag = tl.fma(values, cosine, partner * (-sine))
    rotated = tl.where(columns % 2 == 0, real, imag)
    rounded = tl.where(rotary, rotated, values).to(tl.bfloat16).to(tl.float32)
    payload, scale = _quantize(rounded.reshape((16, 32)), V2)
    destination = (group * rows + row) * (HEADS * 512) + local_head * 512
    tl.store(q + destination + columns, payload.reshape((512,)))
    tl.store(
        scales + group * scale_group_stride + row * scale_row_stride
        + (local_head * 16 + tl.arange(0, 16)) * scale_column_stride,
        scale,
    )


@triton.jit(do_not_specialize=["rows", "row_stride", "column_stride"])
def _unpack_kernel(
    packed, out, rows, row_stride, column_stride,
    rows_per_group, out_group_stride, out_row_stride,
    out_column_stride, COLUMNS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    column = tl.program_id(1) * 128 + tl.arange(0, 128)
    word = tl.load(
        packed + row * row_stride + (column // 4) * column_stride,
        column < COLUMNS,
        other=0,
    ).to(tl.uint32)
    exponent = (word >> ((column % 4) * 8)) & 255
    tl.store(
        out + (row // rows_per_group) * out_group_stride
        + (row % rows_per_group) * out_row_stride + column * out_column_stride,
        _decode_exponent(exponent), column < COLUMNS,
    )


def _input(x):
    if (
        not isinstance(x, torch.Tensor)
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or x.shape[1] <= 0
        or x.shape[1] % 32
        or x.stride(1) != 1
        or x.stride(0) < x.shape[1]
        or x.requires_grad
    ):
        raise ValueError("Expected CUDA inference BF16 [M,K], K%32==0, contiguous rows")


def _output(out, shape, dtype, device, operands):
    if out is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if (
        out.shape != shape
        or out.dtype != dtype
        or out.device != device
        or not out.is_contiguous()
        or out.requires_grad
        or any(torch._C._overlaps(out, value) for value in operands)
    ):
        raise ValueError(
            "Output must be contiguous, match shape/dtype/device and not alias operands"
        )
    return out


def _scale_output(rows, columns, device, *, groups=1, mn_major=False):
    if mn_major:
        padded_rows = max(4, triton.cdiv(rows, 4) * 4)
        return torch.empty_strided(
            (groups, rows, columns),
            (padded_rows * columns, 1, padded_rows),
            dtype=torch.float32,
            device=device,
        )
    return torch.empty((groups, rows, columns), dtype=torch.float32, device=device)


def quantize_g32(x, *, out_q=None, out_scale=None, mn_major=False):
    """BF16 [M,K] -> E4M3 [M,K], FP32 scales; optional TMA MN-major."""
    _input(x)
    rows, width = x.shape
    v2 = use_v2(rows, width)
    q = _output(out_q, x.shape, torch.float8_e4m3fn, x.device, (x,))
    if mn_major:
        expected = (1, max(4, triton.cdiv(rows, 4) * 4))
        if out_scale is None:
            scale = _scale_output(rows, width // 32, x.device, mn_major=True)[0]
        elif (
            out_scale.shape != (rows, width // 32)
            or out_scale.dtype != torch.float32
            or out_scale.device != x.device
            or out_scale.stride() != expected
            or out_scale.requires_grad
            or any(torch._C._overlaps(out_scale, value) for value in (x, q))
        ):
            raise ValueError("Expected nonaliasing TMA MN-major FP32 output scales")
        else:
            scale = out_scale
    else:
        scale = _output(out_scale, (rows, width // 32), torch.float32, x.device, (x, q))
    if rows:
        _dense_quant_kernel[(rows, triton.cdiv(width, 128))](
            x,
            q,
            scale,
            rows,
            x.stride(0),
            scale.stride(0),
            scale.stride(1),
            width,
            v2,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, scale


def quantize_wo_a_g32(
    o, freqs, *, n_groups, heads_per_group, imaginary_fma_first=True, mn_major=False
):
    """Return logical [M,G,K] and [M,G,K/32], both physically group-major.

    Dispatch policy is per original linear's M*K, never G*M*K. BF16 inverse
    RoPE is materialized in registers before group absmax/quantization.
    """
    if (
        o.ndim not in (3, 4)
        or o.dtype != torch.bfloat16
        or not o.is_cuda
        or o.shape[-1] != 512
        or o.shape[-2] != n_groups * heads_per_group
        or n_groups < 1
        or heads_per_group < 1
        or o.stride(-1) != 1
        or o.requires_grad
        or freqs.dtype != torch.complex64
        or freqs.device != o.device
        or freqs.ndim != 2
        or freqs.shape[1] != 32
        or freqs.is_conj()
        or freqs.requires_grad
    ):
        raise ValueError(
            "Expected BF16 attention rows and complex64 [B or M,32] frequencies"
        )
    if (
        o.ndim == 4
        and o.shape[0] > 1
        and o.shape[1] > 1
        and o.stride(0) != o.shape[1] * o.stride(1)
    ):
        raise ValueError("Attention batch/sequence dimensions must be view-flattenable")
    rows = o.numel() // (n_groups * heads_per_group * 512)
    if rows and (freqs.shape[0] == 0 or rows % freqs.shape[0]):
        raise ValueError("Frequency rows must divide attention token count")
    if rows and o.ndim == 4 and freqs.shape[0] not in (o.shape[0], rows):
        raise ValueError("4D attention frequencies must be per-request or per-token")
    width = heads_per_group * 512
    v2 = use_v2(rows, width)
    q = torch.empty((n_groups, rows, width), dtype=torch.float8_e4m3fn, device=o.device)
    scale = _scale_output(
        rows, width // 32, o.device, groups=n_groups, mn_major=mn_major
    )
    if rows:
        flat = o.view(rows, n_groups * heads_per_group, 512)
        freq = torch.view_as_real(freqs)
        _woa_quant_kernel[(rows, n_groups * heads_per_group)](
            flat,
            freq,
            q,
            scale,
            rows,
            flat.stride(0),
            flat.stride(1),
            freq.stride(0),
            freq.stride(1),
            scale.stride(0),
            scale.stride(1),
            scale.stride(2),
            n_groups,
            heads_per_group,
            rows // freqs.shape[0],
            v2,
            imaginary_fma_first,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q.transpose(0, 1), scale.transpose(0, 1)


def unpack_ue8m0_to_fp32(packed, *, mn_major=False, n_groups=1):
    """Decode packed [G*M,K/128] scales, preserving all UE8M0 bit patterns.

    G=1 returns [M,K/32]; G>1 returns [G,M,K/32]. Optional MN-major
    storage independently pads every group to four rows for TMA alignment.
    """
    if (
        not packed.is_cuda
        or packed.dtype != torch.int32
        or packed.ndim != 2
        or packed.stride(0) < 0
        or packed.stride(1) < 0
    ):
        raise ValueError("Expected CUDA int32 [M,packed_K] scales")
    rows, words = packed.shape
    if not isinstance(n_groups, int) or n_groups < 1 or rows % n_groups:
        raise ValueError("Scale rows must be divisible by positive n_groups")
    rows_per_group = rows // n_groups
    grouped = _scale_output(
        rows_per_group, words * 4, packed.device,
        groups=n_groups, mn_major=mn_major,
    )
    out = grouped[0] if n_groups == 1 else grouped
    if rows and words:
        _unpack_kernel[(rows, triton.cdiv(words * 4, 128))](
            packed,
            out,
            rows,
            packed.stride(0),
            packed.stride(1),
            rows_per_group,
            grouped.stride(0),
            grouped.stride(1),
            grouped.stride(2),
            words * 4,
            num_warps=4,
        )
    return out
