# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small per-token FP8 quantization helpers implemented in FlyDSL."""

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import Stream as _FlyStream
from flydsl.expr.typing import T
from flydsl.expr.vector import ReductionOp
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

_quant_cf_cache = {}


def _launch_cached(cache, key, launch_fn, args, stream):
    cf = cache.get(key)
    stream_arg = _FlyStream(stream)
    if cf is not None:
        cf(*args, stream_arg)
    else:
        launch_fn(*args, stream=stream)
        cf = flyc.compile(launch_fn, *args, stream_arg)
        cache[key] = cf


_FP8_E4M3FNUZ_MAX = 240.0
_FP32_MIN_NORMAL = 1.1754943508222875e-38
_WARP_SIZE = 64


def _tile_config(n: int) -> tuple[int, int]:
    if n == 128:
        return 64, 2
    if n == 256:
        return 64, 4
    if n % 2048 == 0:
        return 256, 8
    if n % 1024 == 0:
        return 128, 8
    if n % 256 == 0:
        return 64, 4
    return 256, 8


@functools.lru_cache(maxsize=128)
def compile_fp8_quant_per_token(
    *,
    n: int,
    in_dtype: str,
    use_row_weights: bool = False,
    input_cache_modifier: int = 0,
    output_cache_modifier: int = 0,
):
    if in_dtype not in ("bf16", "fp16"):
        raise ValueError(f"in_dtype must be 'bf16' or 'fp16', got {in_dtype!r}")
    block_threads, vec_width = _tile_config(int(n))
    tile_cols = block_threads * vec_width
    if int(n) % tile_cols != 0:
        raise ValueError(
            f"fp8 quant fast path requires n divisible by block_threads*vec_width, "
            f"got n={n}, block_threads={block_threads}, vec_width={vec_width}"
        )
    num_tiles = int(n) // tile_cols
    red_slots = max(1, block_threads // _WARP_SIZE)

    allocator = SmemAllocator(None, arch=get_hip_arch())
    red_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_offset + red_slots * 4

    @flyc.kernel
    def fp8_quant_kernel(
        inp: fx.Tensor,
        out: fx.Tensor,
        scales: fx.Tensor,
        row_weights: fx.Tensor,
    ):
        row = gpu.block_id("x")
        tid = gpu.thread_id("x")
        tid_i32 = arith.index_cast(T.i32, tid)

        base_ptr = allocator.get_base()
        s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(red_slots,))
        s_red.get()

        inp_rsrc = buffer_ops.create_buffer_resource(
            inp,
            max_size=False,
            num_records_bytes=fx.Index(n * 2) * row + fx.Index(n * 2),
        )
        out_rsrc = buffer_ops.create_buffer_resource(
            out,
            max_size=False,
            num_records_bytes=fx.Index(n) * row + fx.Index(n),
        )
        scale_rsrc = buffer_ops.create_buffer_resource(
            scales, max_size=False, num_records_bytes=(row + fx.Index(1)) * fx.Index(4)
        )
        row_weight_rsrc = -1
        if const_expr(use_row_weights):
            row_weight_rsrc = buffer_ops.create_buffer_resource(
                row_weights,
                max_size=False,
                num_records_bytes=(row + fx.Index(1)) * fx.Index(4),
            )
        in_elem = T.bf16 if const_expr(in_dtype == "bf16") else T.f16

        def wave_reduce_max(val):
            width = arith.constant(_WARP_SIZE, type=T.i32)
            w = val
            for shift in [32, 16, 8, 4, 2, 1]:
                peer = w.shuffle_xor(arith.constant(shift, type=T.i32), width)
                w = w.maximumf(peer)
            return w

        def block_reduce_max(val):
            if const_expr(red_slots == 1):
                return wave_reduce_max(val)

            lane = tid_i32 % arith.constant(_WARP_SIZE, type=T.i32)
            wave = tid_i32 // arith.constant(_WARP_SIZE, type=T.i32)
            w = wave_reduce_max(val)
            if lane == fx.Int32(0):
                wave_idx = arith.index_cast(T.index, wave)
                SmemPtr.store(s_red, w, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < arith.constant(red_slots, type=T.i32)
                lane_safe = arith.select(in_range, lane, fx.Int32(0))
                lane_safe_idx = arith.index_cast(T.index, lane_safe)
                v = SmemPtr.load(s_red, [lane_safe_idx])
                ww = arith.select(in_range, v, arith.constant(0.0, type=T.f32))
                ww = wave_reduce_max(ww)
                if lane == fx.Int32(0):
                    SmemPtr.store(s_red, ww, [fx.Index(0)])
            gpu.barrier()

            return SmemPtr.load(s_red, [fx.Index(0)])

        abs_mask = fx.Vector.filled(vec_width, 0x7FFFFFFF, fx.Int32)
        local_max = arith.constant(0.0, type=T.f32)
        cached_vecs = []
        row_base = row * fx.Index(n)
        thread_base = tid * fx.Index(vec_width)

        for tile_i in range_constexpr(num_tiles):
            elem_off = row_base + fx.Index(tile_i * tile_cols) + thread_base
            raw = buffer_ops.buffer_load(
                inp_rsrc,
                elem_off,
                vec_width=vec_width,
                dtype=in_elem,
                **({"cache_modifier": int(input_cache_modifier)} if int(input_cache_modifier) != 0 else {}),
            )
            vals = fx.Vector(raw).to(fx.Float32)
            cached_vecs.append(vals)
            vals_abs = (vals.bitcast(fx.Int32) & abs_mask).bitcast(fx.Float32)
            chunk_max = vals_abs.reduce(ReductionOp.MAX)
            local_max = local_max.maximumf(chunk_max.ir_value())

        row_max = block_reduce_max(local_max)
        scale = row_max / arith.constant(_FP8_E4M3FNUZ_MAX, type=T.f32)
        safe_nonzero_scale = scale.maximumf(arith.constant(_FP32_MIN_NORMAL, type=T.f32))
        final_scale = arith.select(
            scale == arith.constant(0.0, type=T.f32),
            arith.constant(1.0, type=T.f32),
            safe_nonzero_scale,
        )
        scale_out = final_scale
        if const_expr(use_row_weights):
            scale_out = scale_out * buffer_ops.buffer_load(row_weight_rsrc, row, vec_width=1, dtype=T.f32)
        if tid_i32 == fx.Int32(0):
            buffer_ops.buffer_store(scale_out, scale_rsrc, row)

        inv_scale = arith.constant(1.0, type=T.f32) / final_scale

        for tile_i in range_constexpr(num_tiles):
            vals = cached_vecs[tile_i] * inv_scale
            elem_off = row_base + fx.Index(tile_i * tile_cols) + thread_base
            if const_expr(vec_width == 2):
                lo = rocdl.cvt_pk_fp8_f32(T.i32, vals[0], vals[1], fx.Int32(0), False)
                byte0 = arith.trunci(T.i8, lo)
                byte1 = arith.trunci(T.i8, lo >> arith.constant(8, type=T.i32))
                buffer_ops.buffer_store(
                    byte0,
                    out_rsrc,
                    elem_off,
                    **({"cache_modifier": int(output_cache_modifier)} if int(output_cache_modifier) != 0 else {}),
                )
                buffer_ops.buffer_store(
                    byte1,
                    out_rsrc,
                    elem_off + fx.Index(1),
                    **({"cache_modifier": int(output_cache_modifier)} if int(output_cache_modifier) != 0 else {}),
                )
            else:
                for wi in range_constexpr(vec_width // 4):
                    base = wi * 4
                    lo = rocdl.cvt_pk_fp8_f32(T.i32, vals[base], vals[base + 1], fx.Int32(0), False)
                    packed = rocdl.cvt_pk_fp8_f32(T.i32, vals[base + 2], vals[base + 3], lo, True)
                    word_off = (elem_off // fx.Index(4)) + fx.Index(wi)
                    buffer_ops.buffer_store(
                        packed,
                        out_rsrc,
                        word_off,
                        **({"cache_modifier": int(output_cache_modifier)} if int(output_cache_modifier) != 0 else {}),
                    )

    @flyc.jit
    def launch_fp8_quant(
        inp: fx.Tensor,
        out: fx.Tensor,
        scales: fx.Tensor,
        row_weights: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        rows_idx = arith.index_cast(T.index, rows)
        fp8_quant_kernel(inp, out, scales, row_weights).launch(
            grid=(rows_idx, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_fp8_quant


@functools.lru_cache(maxsize=128)
def compile_fp8_quant_with_zero(
    *,
    n: int,
    in_dtype: str,
    zero_dtype: str,
    zero_vec_width: int = 8,
    use_row_weights: bool = False,
    input_cache_modifier: int = 0,
    output_cache_modifier: int = 0,
):
    """Compile fused FP8 quant + zero_fill: two kernels in one @flyc.jit dispatch."""
    quant_launcher = compile_fp8_quant_per_token(
        n=n,
        in_dtype=in_dtype,
        use_row_weights=use_row_weights,
        input_cache_modifier=input_cache_modifier,
        output_cache_modifier=output_cache_modifier,
    )

    from .zero_fill import compile_zero_fill

    zero_launcher_inner = compile_zero_fill(dtype=zero_dtype, vec_width=zero_vec_width)

    # Extract the kernel objects from the existing launchers by calling them
    # in a new @flyc.jit that launches both sequentially on the same stream.
    quant_block_threads, _ = _tile_config(int(n))

    is_bf16_zero = zero_dtype == "bf16"

    @flyc.kernel(known_block_size=[256, 1, 1])
    def zero_fill_kernel_fused(out: fx.Tensor, i32_numel: fx.Int32):
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        vec_id = bid * fx.Index(256) + tid
        elem_base = vec_id * fx.Index(zero_vec_width)
        numel_idx = arith.index_cast(T.index, i32_numel)
        vec_count = numel_idx // fx.Index(zero_vec_width)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=False, num_records_bytes=numel_idx * fx.Index(2))
        from flydsl.expr import vector

        in_range = arith.cmpi(arith.CmpIPredicate.ult, vec_id, vec_count)
        elem_type = T.bf16 if is_bf16_zero else T.f16
        zero_scalar = arith.constant(0.0, type=elem_type)
        zero_vec = vector.broadcast(T.vec(zero_vec_width, elem_type), zero_scalar)
        buffer_ops.buffer_store(zero_vec, out_rsrc, elem_base, mask=in_range)

    quant_allocator = SmemAllocator(None, arch=get_hip_arch())
    quant_red_slots = max(1, quant_block_threads // _WARP_SIZE)
    quant_red_offset = quant_allocator._align(quant_allocator.ptr, 16)
    quant_allocator.ptr = quant_red_offset + quant_red_slots * 4

    quant_compiled = compile_fp8_quant_per_token(
        n=n,
        in_dtype=in_dtype,
        use_row_weights=use_row_weights,
        input_cache_modifier=input_cache_modifier,
        output_cache_modifier=output_cache_modifier,
    )

    @flyc.jit
    def launch_fp8_quant_with_zero(
        inp: fx.Tensor,
        out: fx.Tensor,
        scales: fx.Tensor,
        row_weights: fx.Tensor,
        rows: fx.Int32,
        zero_out: fx.Tensor,
        zero_numel: fx.Int32,
        stream: fx.Stream,
    ):
        quant_compiled(inp, out, scales, row_weights, rows, stream=stream)
        zero_numel_idx = arith.index_cast(T.index, zero_numel)
        zero_vec_count = zero_numel_idx // fx.Index(zero_vec_width)
        zero_blocks = (zero_vec_count + fx.Index(255)) // fx.Index(256)
        zero_fill_kernel_fused(zero_out, zero_numel).launch(
            grid=(zero_blocks, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_fp8_quant_with_zero


_quant_zero_cf_cache = {}


def fp8_quantize_per_token_with_zero(
    x: torch.Tensor,
    scale_shape,
    zero_tensor: torch.Tensor,
    *,
    stream,
    row_weights: torch.Tensor | None = None,
    input_cache_modifier: int | None = None,
    output_cache_modifier: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 per-token quantization + zero_fill in one dispatch (2 kernels, 1 Python call)."""
    if x.dtype == torch.bfloat16:
        in_dtype = "bf16"
    elif x.dtype == torch.float16:
        in_dtype = "fp16"
    else:
        raise ValueError(f"fp8 quant supports fp16/bf16 input, got {x.dtype}")
    if not x.is_contiguous():
        x = x.contiguous()

    zero_dtype = "bf16" if zero_tensor.dtype == torch.bfloat16 else "f16"
    zero_numel = int(zero_tensor.numel())
    zero_vec_width = 8
    if zero_numel == 0 or zero_numel % zero_vec_width != 0:
        raise ValueError(f"zero_tensor numel must be positive and divisible by {zero_vec_width}, got {zero_numel}")

    n = int(x.shape[-1])
    rows = int(x.numel() // n)
    x_2d = x.reshape(rows, n)
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fnuz, device=x.device)
    scales = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
    use_row_weights = row_weights is not None
    if row_weights is None:
        row_weights_arg = torch.empty((0,), dtype=torch.float32, device=x.device)
    else:
        row_weights_arg = row_weights.reshape(rows).contiguous()
    if input_cache_modifier is None:
        input_cache_modifier = 0
    if output_cache_modifier is None:
        output_cache_modifier = 0

    launcher = compile_fp8_quant_with_zero(
        n=n,
        in_dtype=in_dtype,
        zero_dtype=zero_dtype,
        zero_vec_width=zero_vec_width,
        use_row_weights=use_row_weights,
        input_cache_modifier=input_cache_modifier,
        output_cache_modifier=output_cache_modifier,
    )
    _qz_args = (
        x_2d,
        out.reshape(rows, n),
        scales.reshape(rows),
        row_weights_arg,
        rows,
        zero_tensor,
        zero_numel,
    )
    _launch_cached(_quant_zero_cf_cache, id(launcher), launcher, _qz_args, stream)
    return out, scales


def fp8_quantize_per_token(
    x: torch.Tensor,
    scale_shape,
    *,
    stream,
    row_weights: torch.Tensor | None = None,
    input_cache_modifier: int | None = None,
    output_cache_modifier: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dtype == torch.bfloat16:
        in_dtype = "bf16"
    elif x.dtype == torch.float16:
        in_dtype = "fp16"
    else:
        raise ValueError(f"fp8 quant supports fp16/bf16 input, got {x.dtype}")
    if not x.is_contiguous():
        x = x.contiguous()

    n = int(x.shape[-1])
    rows = int(x.numel() // n)
    x_2d = x.reshape(rows, n)
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fnuz, device=x.device)
    scales = torch.empty(scale_shape, dtype=torch.float32, device=x.device)
    use_row_weights = row_weights is not None
    if row_weights is None:
        row_weights_arg = torch.empty((0,), dtype=torch.float32, device=x.device)
    else:
        row_weights_arg = row_weights.reshape(rows).contiguous()
    if input_cache_modifier is None:
        input_cache_modifier = 0
    if output_cache_modifier is None:
        output_cache_modifier = 0
    launcher = compile_fp8_quant_per_token(
        n=n,
        in_dtype=in_dtype,
        use_row_weights=use_row_weights,
        input_cache_modifier=input_cache_modifier,
        output_cache_modifier=output_cache_modifier,
    )
    _q_args = (x_2d, out.reshape(rows, n), scales.reshape(rows), row_weights_arg, rows)
    _launch_cached(_quant_cf_cache, id(launcher), launcher, _q_args, stream)
    return out, scales
