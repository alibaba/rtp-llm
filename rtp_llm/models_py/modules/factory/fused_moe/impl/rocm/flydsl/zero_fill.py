# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Small tensor fill helpers implemented in FlyDSL."""

import functools

import torch

import flydsl.compiler as flyc
from flydsl.expr import Stream as _FlyStream

_zero_cf_cache = {}


def _launch_cached(cache, key, launch_fn, args, stream):
    cf = cache.get(key)
    stream_arg = _FlyStream(stream)
    if cf is not None:
        cf(*args, stream_arg)
    else:
        launch_fn(*args, stream=stream)
        cf = flyc.compile(launch_fn, *args, stream_arg)
        cache[key] = cf
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu, vector
from flydsl.expr.typing import T


@functools.lru_cache(maxsize=16)
def compile_zero_fill(*, dtype: str, vec_width: int = 8, block_threads: int = 256):
    if dtype not in ("bf16", "f16"):
        raise ValueError(f"zero fill supports bf16/f16 outputs, got {dtype!r}")
    if int(vec_width) <= 0 or int(block_threads) <= 0:
        raise ValueError("vec_width and block_threads must be positive")

    is_bf16 = dtype == "bf16"

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def zero_fill_kernel(out: fx.Tensor, i32_numel: fx.Int32):
        bid = gpu.block_id("x")
        tid = gpu.thread_id("x")
        vec_id = bid * fx.Index(block_threads) + tid
        elem_base = vec_id * fx.Index(vec_width)
        numel_idx = arith.index_cast(T.index, i32_numel)
        vec_count = numel_idx // fx.Index(vec_width)

        out_rsrc = buffer_ops.create_buffer_resource(
            out, max_size=False, num_records_bytes=numel_idx * fx.Index(2)
        )
        in_range = arith.cmpi(arith.CmpIPredicate.ult, vec_id, vec_count)
        elem_type = T.bf16 if is_bf16 else T.f16
        zero_scalar = arith.constant(0.0, type=elem_type)
        zero_vec = vector.broadcast(T.vec(vec_width, elem_type), zero_scalar)
        buffer_ops.buffer_store(zero_vec, out_rsrc, elem_base, mask=in_range)

    @flyc.jit
    def launch_zero_fill(out: fx.Tensor, i32_numel: fx.Int32, stream: fx.Stream):
        numel_idx = fx.Index(i32_numel)
        vec_count = numel_idx // fx.Index(vec_width)
        blocks = (vec_count + fx.Index(block_threads - 1)) // fx.Index(block_threads)
        zero_fill_kernel(out, i32_numel).launch(
            grid=(blocks, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_zero_fill


def zero_fill_tensor(x: torch.Tensor, *, stream) -> bool:
    """Zero a contiguous fp16/bf16 CUDA tensor.

    Returns True when the FlyDSL fast path launched. Unsupported shapes return
    False so callers can keep their existing fallback.
    """
    if x.dtype == torch.bfloat16:
        dtype = "bf16"
    elif x.dtype == torch.float16:
        dtype = "f16"
    else:
        return False
    if (not x.is_cuda) or (not x.is_contiguous()):
        return False

    numel = int(x.numel())
    vec_width = 8
    if numel == 0 or numel % vec_width != 0:
        return False

    launcher = compile_zero_fill(dtype=dtype, vec_width=vec_width, block_threads=256)
    _launch_cached(_zero_cf_cache, id(launcher), launcher, (x, numel), stream)
    return True
