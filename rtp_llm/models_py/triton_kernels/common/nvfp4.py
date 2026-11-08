"""NVFP4 quantization with one FP32 scale per activation shard."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import Fp4Activation


@triton.jit
def _reciprocal_approx(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[x],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _load_nvfp4_group(address, valid):
    # One lane reads all 16 BF16 values with a 32-byte aligned load.
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; .reg .b32 a<8>; .reg .b16 lo,hi; .reg .f32 x,y;"
        "setp.ne.u32 p, $10, 0; @!p bra zero;"
        "ld.global.v8.b32 {$1,$2,$3,$4,$5,$6,$7,$8}, [$9];"
        "abs.bf16x2 a0, $1;"
        "abs.bf16x2 a1, $2;"
        "abs.bf16x2 a2, $3;"
        "abs.bf16x2 a3, $4;"
        "abs.bf16x2 a4, $5;"
        "abs.bf16x2 a5, $6;"
        "abs.bf16x2 a6, $7;"
        "abs.bf16x2 a7, $8;"
        "max.bf16x2 a0, a0, a1;"
        "max.bf16x2 a2, a2, a3;"
        "max.bf16x2 a4, a4, a5;"
        "max.bf16x2 a6, a6, a7;"
        "max.bf16x2 a0, a0, a2;"
        "max.bf16x2 a4, a4, a6;"
        "max.bf16x2 a0, a0, a4;"
        "mov.b32 {lo,hi}, a0; cvt.f32.bf16 x, lo; cvt.f32.bf16 y, hi;"
        "max.f32 $0, x, y; bra done;"
        "zero: mov.f32 $0, 0f00000000;"
        "mov.b32 $1,0; mov.b32 $2,0; mov.b32 $3,0; mov.b32 $4,0;"
        "mov.b32 $5,0; mov.b32 $6,0; mov.b32 $7,0; mov.b32 $8,0; done: }",
        constraints="=f,=r,=r,=r,=r,=r,=r,=r,=r,l,r",
        args=[address, valid.to(tl.int32)],
        dtype=(
            tl.float32, tl.uint32, tl.uint32, tl.uint32, tl.uint32,
            tl.uint32, tl.uint32, tl.uint32, tl.uint32,
        ),
        is_pure=True, pack=1,
    )


@triton.jit
def _pack_nvfp4_eight(multiplier, v0, v1, v2, v3):
    return tl.inline_asm_elementwise(
        "{ .reg .b8 b<4>; .reg .b16 lo,hi; .reg .f32 x,y;"
        "mov.b32 {lo,hi}, $2; cvt.f32.bf16 x, lo; cvt.f32.bf16 y, hi;"
        "mul.f32 x, x, $1; mul.f32 y, y, $1; cvt.rn.satfinite.e2m1x2.f32 b0, y, x;"
        "mov.b32 {lo,hi}, $3; cvt.f32.bf16 x, lo; cvt.f32.bf16 y, hi;"
        "mul.f32 x, x, $1; mul.f32 y, y, $1; cvt.rn.satfinite.e2m1x2.f32 b1, y, x;"
        "mov.b32 {lo,hi}, $4; cvt.f32.bf16 x, lo; cvt.f32.bf16 y, hi;"
        "mul.f32 x, x, $1; mul.f32 y, y, $1; cvt.rn.satfinite.e2m1x2.f32 b2, y, x;"
        "mov.b32 {lo,hi}, $5; cvt.f32.bf16 x, lo; cvt.f32.bf16 y, hi;"
        "mul.f32 x, x, $1; mul.f32 y, y, $1; cvt.rn.satfinite.e2m1x2.f32 b3, y, x;"
        "mov.b32 $0, {b0,b1,b2,b3}; }",
        constraints="=r,f,r,r,r,r",
        args=[multiplier, v0, v1, v2, v3],
        dtype=tl.uint32, is_pure=True, pack=1,
    )


@triton.jit
def _nvfp4_quantize_from_partials(
    X, Partials, Values, Scales, Tail, M, PaddedGroups,
    COUNT: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr,
):
    # Reduce the AttnRes CTA maxima once, then reuse the rank scale for every tile.
    indices = tl.arange(0, triton.next_power_of_2(COUNT))
    amax = tl.max(tl.load(Partials + indices, indices < COUNT, other=0), 0)
    scale = tl.maximum(amax, 1e-8) / (6.0 * 448.0)
    inverse = tl.div_rn(1.0, scale)
    outer_approx = _reciprocal_approx(inverse)
    if tl.program_id(0) == 0:
        tail = tl.arange(0, 4)
        tl.store(Tail + tail, tl.where(tail == 0, scale, 0.0))
    for start in range(tl.program_id(0) * BLOCK, PaddedGroups, tl.num_programs(0) * BLOCK):
        index = start + tl.arange(0, BLOCK)
        row = index // GROUPS
        group = index % GROUPS
        valid = row < M
        maximum, v0, v1, v2, v3, v4, v5, v6, v7 = _load_nvfp4_group(
            X + index.to(tl.int64) * 16, valid,
        )
        group_scale = tl.minimum(inverse * (maximum * _reciprocal_approx(6.0)), 448.0).to(tl.float8e4nv)
        multiplier = tl.where(maximum != 0, _reciprocal_approx(group_scale.to(tl.float32) * outer_approx), 0.0)
        low = _pack_nvfp4_eight(multiplier, v0, v1, v2, v3).to(tl.uint64)
        high = _pack_nvfp4_eight(multiplier, v4, v5, v6, v7).to(tl.uint64)
        packed = low | (high << 32)
        tl.store(Values + index, packed, valid)
        offset = (row // 128 * (128 * GROUPS) + group // 4 * 512
                  + row % 32 * 16 + row % 128 // 32 * 4 + group % 4)
        tl.store(Scales + offset, group_scale.to(tl.uint8, bitcast=True))


def _quantize_from_partials(input, amax_partials, values, block_scales, tail):
    rows, width = input.shape
    groups = width // 16
    padded_groups = triton.cdiv(rows, 128) * 128 * groups
    # Large shards exceed L2 and need more independent memory requests in flight.
    block, waves = (256, 48) if rows >= 32768 else (128, 16)
    # Limit the grid so a CTA can reuse its rank scale across several tiles.
    grid = min(triton.cdiv(padded_groups, block), amax_partials.numel() * waves)
    _nvfp4_quantize_from_partials[(grid,)](
        input, amax_partials, values.view(torch.uint64), block_scales, tail,
        rows, padded_groups, amax_partials.numel(), groups, block,
        num_warps=block // 32, enable_fp_fusion=False,
    )


@triton.jit
def _pull_nvfp4_rows(Values, Source, GlobalScale, Out, Target, RowScale,
                     start, destination, rows, padded_rows, GROUPS: tl.constexpr):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    group = tl.program_id(1) * 32 + tl.arange(0, 32)
    src = row + start
    dst = row + destination
    src_offset = (
        src[:, None] // 128 * (128 * GROUPS) + group[None, :] // 4 * 512
        + src[:, None] % 32 * 16 + src[:, None] % 128 // 32 * 4 + group[None, :] % 4
    )
    dst_offset = (
        dst[:, None] // 128 * (128 * GROUPS) + group[None, :] // 4 * 512
        + dst[:, None] % 32 * 16 + dst[:, None] % 128 // 32 * 4 + group[None, :] % 4
    )
    valid = row[:, None] < rows
    packed = tl.load(Values + src[:, None].to(tl.int64) * GROUPS + group[None, :], valid, other=0)
    tl.store(Out + dst[:, None].to(tl.int64) * GROUPS + group[None, :], packed, valid)
    value = tl.load(Source + src_offset, valid, other=0)
    tl.store(Target + dst_offset, value, dst[:, None] < padded_rows)
    if tl.program_id(1) == 0:
        tl.store(RowScale + dst, tl.load(GlobalScale), row < rows)


def pull_nvfp4_rows(source, target, segment):
    """Write an interval into the final GEMM row and block-scale layouts."""
    source_activation = source.unpack(source.wire)
    rows = target.values.shape[0]
    groups = source.shape[1] // 16
    padded_rows = segment.destination_offset + segment.rows
    if padded_rows == rows:
        padded_rows = triton.cdiv(rows, 128) * 128
    grid_rows = triton.cdiv(padded_rows - segment.destination_offset, 16)
    _pull_nvfp4_rows[(grid_rows, groups // 32)](
        source_activation.values.view(torch.uint64),
        source_activation.block_scales, source_activation.scale,
        target.values.view(torch.uint64), target.block_scales, target.scale,
        segment.source_offset, segment.destination_offset, segment.rows, padded_rows, groups,
    )


@triton.jit
def _unpack_decode_nvfp4(
    Wire, Wire64, Wire32, WireFloat, Codes, Blocks, Global, Fp8Codes, Fp8Scales, HeadGate,
    ROWS: tl.constexpr, LOCAL_ROWS: tl.constexpr, GROUPS: tl.constexpr,
    PACKET_BYTES: tl.constexpr, FP4_BYTES: tl.constexpr, GATE_COLS: tl.constexpr,
    ROWWISE: tl.constexpr,
):
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    group = tl.program_id(1) * 32 + tl.arange(0, 32)
    local = row % LOCAL_ROWS
    base = (row // LOCAL_ROWS) * PACKET_BYTES
    valid = row[:, None] < ROWS

    src = base[:, None] // 8 + local[:, None] * GROUPS + group[None, :]
    dst = row[:, None] * GROUPS + group[None, :]
    packed = tl.load(Wire64 + src, valid, other=0)
    tl.store(Codes + dst, packed, valid)
    local_block = (
        local[:, None] // 128 * (128 * GROUPS) + group[None, :] // 4 * 512
        + local[:, None] % 32 * 16 + local[:, None] % 128 // 32 * 4 + group[None, :] % 4
    )
    global_block = (
        row[:, None] // 128 * (128 * GROUPS) + group[None, :] // 4 * 512
        + row[:, None] % 32 * 16 + row[:, None] % 128 // 32 * 4 + group[None, :] % 4
    )
    value = tl.load(Wire + base[:, None] + LOCAL_ROWS * GROUPS * 8 + local_block, valid, other=0)
    tl.store(Blocks + global_block, value)
    if tl.program_id(1) == 0:
        if ROWWISE:
            scale_idx = (base + FP4_BYTES) // 4 - tl.cdiv(LOCAL_ROWS, 4) * 4 + local
        else:
            scale_idx = (base + FP4_BYTES - 16) // 4
        scale = tl.load(WireFloat + scale_idx, row < ROWS, other=0)
        tl.store(Global + row, scale, row < ROWS)

    if GATE_COLS:
        fp8_src = (base[:, None] + FP4_BYTES) // 8 + (local[:, None] * GROUPS + group[None, :]) * 2
        for i in tl.static_range(2):
            bits = tl.load(Wire64 + fp8_src + i, valid, other=0)
            tl.store(Fp8Codes + dst * 2 + i, bits, valid)
        scale_start = FP4_BYTES + LOCAL_ROWS * GROUPS * 16
        local_scale_rows = (LOCAL_ROWS + 3) // 4 * 4
        scale_rows = (ROWS + 3) // 4 * 4
        scale_src = (base[:, None] + scale_start) // 4 + group[None, :] * local_scale_rows + local[:, None]
        scale_mask = group[None, :] < GROUPS // 32
        packed_scales = tl.load(Wire32 + scale_src, valid & scale_mask, other=0)
        tl.store(
            Fp8Scales + group[None, :] * scale_rows + row[:, None],
            packed_scales, scale_mask & (row[:, None] < scale_rows),
        )
        gate_start = scale_start + GROUPS // 32 * local_scale_rows * 4
        gate_src = (base[:, None] + gate_start) // 4 + local[:, None] * GATE_COLS + group[None, :]
        gate_mask = valid & (group[None, :] < GATE_COLS)
        gate = tl.load(WireFloat + gate_src, gate_mask, other=0)
        tl.store(HeadGate + row[:, None] * GATE_COLS + group[None, :], gate, gate_mask)


def unpack_decode_nvfp4(wire, packet, tp_size):
    """Restore GEMM layouts in one launch, retaining the source scales."""
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
        DsaDecodeActivation, nvfp4_packet_capacity,
    )

    if isinstance(packet, DsaDecodeActivation):
        local_rows, width = packet.local_hidden.shape
        gate_cols = packet.head_gate_columns
    else:
        local_rows, width = packet.shape
        gate_cols = 0
    rows = tp_size * local_rows
    groups = width // 16
    padded_rows = triton.cdiv(rows, 128) * 128
    fp4 = Fp4Activation(
        wire.new_empty((rows, width // 2)),
        wire.new_empty((padded_rows, groups)),
        wire.new_empty(rows, dtype=torch.float32),
    )
    if gate_cols:
        fp8 = wire.new_empty((rows, width), dtype=torch.float8_e4m3fn)
        fp8_scales = wire.new_empty((width // 512, triton.cdiv(rows, 4) * 4), dtype=torch.int32)
        head_gate = wire.new_empty((rows, gate_cols), dtype=torch.float32)
    else:
        fp8, fp8_scales, head_gate = fp4.values, fp4.block_scales, fp4.scale
    _unpack_decode_nvfp4[(padded_rows // 16, groups // 32)](
        wire, wire.view(torch.uint64), wire.view(torch.int32), wire.view(torch.float32),
        fp4.values.view(torch.uint64), fp4.block_scales, fp4.scale,
        fp8.view(torch.uint64), fp8_scales, head_gate,
        rows, local_rows, groups, packet.wire.numel(),
        nvfp4_packet_capacity(local_rows, width, rowwise=packet.rowwise),
        gate_cols, packet.rowwise,
    )
    return fp4, (fp8, fp8_scales.T[:rows]), head_gate


def quantize_nvfp4_from_partials(input, amax_partials, *, packet=None):
    """Pack normalized BF16 rows using producer maxima, optionally in-place."""
    rows, width = input.shape
    if not rows or width % 64 or input.dtype != torch.bfloat16 or not input.is_contiguous():
        raise ValueError("NVFP4 producer requires dense nonempty BF16 rows and K aligned to 64")
    if packet is not None:
        if packet.shape != input.shape or packet.device != input.device:
            raise ValueError("NVFP4 packet must match its BF16 producer")
        activation = packet.unpack(packet.wire)
        _quantize_from_partials(
            input, amax_partials, activation.values, activation.block_scales,
            packet.wire[-16:].view(torch.float32),
        )
        return activation
    values = input.new_empty((rows, width // 2), dtype=torch.uint8)
    scales = input.new_empty((triton.cdiv(rows, 128) * 128, width // 16), dtype=torch.uint8)
    tail = input.new_empty(4, dtype=torch.float32)
    _quantize_from_partials(input, amax_partials, values, scales, tail)
    return Fp4Activation(values, scales, tail[:1])


def quantize_strided_nvfp4(input: torch.Tensor) -> Fp4Activation:
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    rows, width = input.shape
    if input.dtype != torch.bfloat16 or input.stride(1) != 1 or width % 64 or not width or not rows:
        raise ValueError("Strided NVFP4 quantization requires nonempty BF16 rows and K aligned to 64")
    if not hasattr(rtp_llm_ops, "nvfp4_quantize_strided_from_amax"):
        raise RuntimeError("Strided NVFP4 quantization requires the CUDA 13 RTP build")
    count = torch.cuda.get_device_properties(input.device).multi_processor_count * 4
    partials = torch.empty(count, device=input.device, dtype=torch.float32)
    scale = torch.empty(1, device=input.device, dtype=torch.float32)
    values = torch.empty((rows, width // 2), device=input.device, dtype=torch.uint8)
    padded_rows = triton.cdiv(rows, 128) * 128
    groups = width // 16
    block_scales = torch.empty((padded_rows, groups), device=input.device, dtype=torch.uint8)
    rtp_llm_ops.nvfp4_quantize_strided_from_amax(
        input, partials, values, block_scales, scale, compute_amax=True,
    )
    return Fp4Activation(values, block_scales, scale)
