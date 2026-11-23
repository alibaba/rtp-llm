# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Extracted from vLLM d07b7db9785564549d163937b5d5d9c04c76d626.
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/common/primitives.py
import math
from functools import wraps

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


def ptx(assembly, constraints, operands=(), result=None, *, effects, loc=None, ip=None):
    """One verbatim asm block; a tuple `result` (even a 1-tuple) returns a tuple."""
    if isinstance(result, tuple) and len(result) == 1:
        args = (assembly, constraints, operands, result[0])
        return (ptx(*args, effects=effects, loc=loc, ip=ip),)
    if result is None:
        ir_type = None
    elif isinstance(result, tuple):
        ir_type = llvm.StructType.get_literal([dtype.mlir_type for dtype in result])
    else:
        ir_type = result.mlir_type
    value = llvm.inline_asm(
        ir_type,
        [operand.ir_value(loc=loc, ip=ip) for operand in operands],
        assembly,
        constraints,
        has_side_effects=effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(
            dtype(llvm.extractvalue(dtype.mlir_type, value, [i], loc=loc, ip=ip))
            for i, dtype in enumerate(result)
        )
    return result(value)


def asm(result=None, *, pure=False):
    """`f(*args) -> (code, constraints, operands[, result, pure])` as one inline-PTX
    op. Pointer operands pass their address: 32-bit for "r" (shared), 64-bit for "l"."""

    def wrap(fn):
        @dsl_user_op
        @wraps(fn)
        def op(*args, loc=None, ip=None, **kwargs):
            code, constraints, operands, *rest = fn(*args, **kwargs)
            res, fx = (*rest, pure)[:2] if rest else (result, pure)
            kinds = [c for c in constraints.split(",") if c[:1] not in ("=", "~", "")]
            values = []
            for kind, v in zip(kinds, operands):
                if hasattr(v, "toint"):
                    v = v.toint(loc=loc, ip=ip)
                    v = Int32(v) if kind == "r" else v
                values.append(v)
            return ptx(code, constraints, values, res, effects=not fx, loc=loc, ip=ip)

        return op

    return wrap


@asm(Float32, pure=True)
def fma(a, b, c):
    return "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, c)


@asm((Float32, Float32), pure=True)
def fma2(a, b, c):
    """Lanewise a * b + c of FP32 pairs in one packed instruction (IEEE per lane)."""
    code = "{ .reg .b64 a,b,c,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
    code += "mov.b64 c,{$6,$7}; fma.rn.f32x2 d,a,b,c; mov.b64 {$0,$1},d; }"
    return code, "=f,=f,f,f,f,f,f,f", (*a, *b, *c)


@asm((Float32,) * 4, pure=True)
def mma(a, b, accumulator):
    code = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    code += "{$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};"
    return code, "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", (*a, *b, *accumulator)


def mma_into(acc, a, b):
    """`acc += a @ b` with one BF16 MMA; `acc` is a four-element register view."""
    result = mma(a, b, tuple(acc[k] for k in range(4)))
    for k in range(4):
        acc[k] = result[k]


def at(tensor, *coord):
    """Pointer to `tensor[coord]`."""
    return tensor.iterator + cute.crd2idx(coord, tensor.layout)


def zeros(shape, dtype):
    tensor = cute.make_rmem_tensor(shape, dtype)
    tensor.fill(0)
    return tensor


def alloc(allocator, dtype, shape, align=16):
    """Row-major `shape` tensor from a `SmemAllocator`."""
    stride = None
    if isinstance(shape, tuple):
        stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    layout = cute.make_layout(shape, stride=stride)
    return allocator.allocate_tensor(dtype, layout, align)


@asm((Float32,) * 4)
def load_shared_f32x4(pointer):
    return "{ ld.shared.v4.f32 {$0,$1,$2,$3}, [$4]; }", "=f,=f,=f,=f,r", (pointer,)


@asm()
def copy_g2s(dst, src):
    return "{ cp.async.cg.shared.global [$0], [$1], 16; }", "r,l", (dst, src)


@asm()
def load_bulk(dst, src, size, barrier):
    code = (
        "{ cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        f" [$0], [$1], {size}, [$2]; }}"
    )
    return code, "r,l,r", (dst, src, barrier)


@asm()
def acquire_tensor_map(tensor_map):
    """Acquire a descriptor initialized in global memory before TMA use."""
    return (
        "fence.proxy.tensormap::generic.acquire.sys [$0], 128;",
        "l,~{memory}",
        (tensor_map,),
    )


@asm()
def load_tma_tile(dst, tensor_map, coords, barrier):
    """Issue a rank-five tensor load on the caller's transaction barrier."""
    code = (
        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.tile "
        "[$0], [$1, {$2, $3, $4, $5, $6}], [$7];"
    )
    return (
        code,
        "r,l,r,r,r,r,r,r,~{memory}",
        (
            dst,
            tensor_map,
            *(Int32(coord) for coord in coords),
            barrier,
        ),
    )


@asm()
def store_tma_tile(src, tensor_map, coords):
    """Issue a rank-five tensor store for a later bulk group commit."""
    code = (
        "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group.tile "
        "[$0, {$1, $2, $3, $4, $5}], [$6];"
    )
    return (
        code,
        "l,r,r,r,r,r,r,~{memory}",
        (
            tensor_map,
            *(Int32(coord) for coord in coords),
            src,
        ),
    )


@asm()
def commit_bulk_group():
    return "cp.async.bulk.commit_group;", "~{memory}", ()


@asm(Int32)
def try_wait(barrier, phase):
    code = "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
    code += "p,[$1],$2; selp.u32 $0,1,0,p; }"
    return code, "=r,r,r,~{memory}", (barrier, Int32(phase))


@cute.jit
def wait(barrier, phase):
    done = try_wait(barrier, phase)
    while done == 0:
        done = try_wait(barrier, phase)


@cute.jit
def scatter_reduce(values, lane, count: cutlass.Constexpr[int]):
    """Warp-reduce `count` values; lane groups end up owning one value each."""
    selected = Int32(0)
    for stage in cutlass.range_constexpr(5):
        offset = 16 >> stage
        remaining = max(count >> stage, 1)
        if cutlass.const_expr(remaining > 1):
            half = remaining // 2
            upper = (lane & offset) != 0
            for i in cutlass.range_constexpr(half):
                send = values[i] if upper else values[i + half]
                keep = values[i + half] if upper else values[i]
                values[i] = keep + cute.arch.shuffle_sync_bfly(send, offset)
            if upper:
                selected += half
        else:
            values[0] += cute.arch.shuffle_sync_bfly(values[0], offset)
    return values[0], selected


# Extracted from vLLM d07b7db9785564549d163937b5d5d9c04c76d626.
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/kda/kernel.py


@asm(Float32, pure=True)
def triton_exp(value):
    code = "{ .reg .f32 t; mul.f32 t,$1,0f3FB8AA3B; ex2.approx.ftz.f32 $0,t; }"
    return code, "=f,f", (value,)


@asm(Float32, pure=True)
def div_full(numerator, denominator):
    """The quotient by a positive denominator as div.full.f32 computes it."""
    code = "{ .reg .f32 a,b,r; .reg .pred big; mov.f32 a,$1; mov.f32 b,$2; "
    code += "setp.gt.f32 big,b,0f7E800000; @big mul.f32 a,a,0f3E800000; "
    code += "@big mul.f32 b,b,0f3E800000; rcp.approx.ftz.f32 r,b; mul.f32 $0,a,r; }"
    return code, "=f,f,f", (numerator, denominator)


@asm(Float32, pure=True)
def reciprocal(value):
    """1 / x as div.full.f32 computes it for 2^-126 <= x <= 2^126."""
    return "rcp.approx.ftz.f32 $0,$1;", "=f,f", (value,)


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.f32 $0,$1,$2;", "=f,f,f", (a, b)


@asm(Float32, pure=True)
def sqrt_approx(value):
    return "sqrt.approx.ftz.f32 $0,$1;", "=f,f", (value,)


@cute.jit
def triton_sigmoid(value):
    """tl.sigmoid: 1 / (1 + exp(-x))."""
    return div_full(Float32(1), Float32(1) + triton_exp(-value))


# Extracted from vLLM d07b7db9785564549d163937b5d5d9c04c76d626.
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/kda/specialized.py


@asm((Uint32,) * 2)
def load_shared_u32x2(pointer):
    return "{ ld.shared.v2.u32 {$0,$1}, [$2]; }", "=r,=r,r", (pointer,)


@asm()
def sync_threads(barrier_id, count):
    """bar.sync with a runtime barrier id."""
    return "bar.sync $0, $1;", "r,r,~{memory}", (Int32(barrier_id), Int32(count))


@asm()
def fence_async_shared():
    """Order this thread's shared-memory writes before later bulk copies."""
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def store_bulk(dst, src, size):
    """Shared -> global bulk copy (one bulk group)."""
    code = "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2; "
    code += "cp.async.bulk.commit_group;"
    return code, "l,r,r,~{memory}", (dst, src, Int32(size))


@asm()
def wait_bulk_reads():
    """The issued bulk stores have read their shared-memory sources."""
    return "cp.async.bulk.wait_group.read 0;", "~{memory}", ()


@asm()
def wait_bulk_writes():
    """The issued bulk stores have completed their global writes."""
    return "cp.async.bulk.wait_group 0;", "~{memory}", ()


@asm()
def set_max_registers(count, *, increase):
    op = "inc" if increase else "dec"
    return f"setmaxnreg.{op}.sync.aligned.u32 {count};", "", ()


# Extracted from vLLM d07b7db9785564549d163937b5d5d9c04c76d626.
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/kda/wide.py


@asm((Uint32,) * 2, pure=True)
def load_weight_bf16x4(pointer):
    return "{ ld.global.nc.v2.u32 {$0,$1}, [$2]; }", "=r,=r,l", (pointer,)
