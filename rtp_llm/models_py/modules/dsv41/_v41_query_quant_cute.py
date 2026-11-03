# Copyright 2026 Co-developed by KDA (Kernel Design Agents)

from __future__ import annotations

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute.arch as cute_arch
import torch
from cutlass import Float32, Float64, Int32, Int64, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

D = 1280
VEC = 8
THREADS = D // VEC
WARPS = THREADS // 32
PACKS = D // 128


@dsl_user_op
def _ldg128(addr, *, loc=None, ip=None):
    struct = llvm.StructType.get_literal([T.i32()] * 4)
    res = llvm.inline_asm(
        struct,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _stg128(addr, w0, w1, w2, w3, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            Uint32(w2).ir_value(loc=loc, ip=ip),
            Uint32(w3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _stg64(addr, w0, w1, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _unpack2p(wd, *, loc=None, ip=None) -> Float64:
    res = llvm.inline_asm(
        T.f64(),
        [Uint32(wd).ir_value(loc=loc, ip=ip)],
        """{
            .reg .b16 h<2>;
            .reg .f32 lo, hi;
            mov.b32 {h0, h1}, $1;
            cvt.f32.bf16 lo, h0;
            cvt.f32.bf16 hi, h1;
            mov.b64 $0, {lo, hi};
        }""",
        "=d,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float64(res)


@dsl_user_op
def _dup2(v, *, loc=None, ip=None) -> Float64:
    res = llvm.inline_asm(
        T.f64(),
        [Float32(v).ir_value(loc=loc, ip=ip)],
        "mov.b64 $0, {$1, $1};",
        "=d,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float64(res)


@dsl_user_op
def _dup2b(v, *, loc=None, ip=None) -> Float64:
    res = llvm.inline_asm(
        T.f64(),
        [Uint32(v).ir_value(loc=loc, ip=ip)],
        "mov.b64 $0, {$1, $1};",
        "=d,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float64(res)


@dsl_user_op
def _fold8_p(p0, p1, p2, p3, *, loc=None, ip=None) -> Float32:
    ins = [Float64(v).ir_value(loc=loc, ip=ip) for v in (p0, p1, p2, p3)]
    res = llvm.inline_asm(
        T.f32(),
        ins,
        """{
            .reg .b64 e<4>;
            .reg .f32 s<8>, a<7>;
            mul.rn.f32x2 e0, $1, $1;
            mul.rn.f32x2 e1, $2, $2;
            mul.rn.f32x2 e2, $3, $3;
            mul.rn.f32x2 e3, $4, $4;
            mov.b64 {s0, s1}, e0;
            mov.b64 {s2, s3}, e1;
            mov.b64 {s4, s5}, e2;
            mov.b64 {s6, s7}, e3;
            add.rn.f32 a0, s0, s1;
            add.rn.f32 a1, a0, s2;
            add.rn.f32 a2, a1, s3;
            add.rn.f32 a3, a2, s4;
            add.rn.f32 a4, a3, s5;
            add.rn.f32 a5, a4, s6;
            add.rn.f32 a6, a5, s7;
            mov.f32 $0, a6;
        }""",
        "=f,d,d,d,d",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(res)


@dsl_user_op
def _inv_rsqrt(s, eps, *, loc=None, ip=None) -> Float32:
    res = llvm.inline_asm(
        T.f32(),
        [Float32(s).ir_value(loc=loc, ip=ip), Float32(eps).ir_value(loc=loc, ip=ip)],
        """{
            .reg .f32 d, e;
            div.full.f32 d, $1, 0f44A00000;
            add.rn.f32 e, d, $2;
            rsqrt.approx.ftz.f32 $0, e;
        }""",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(res)


@dsl_user_op
def _norm_pair2(xp, gp, inv2, *, loc=None, ip=None):
    struct = llvm.StructType.get_literal([T.i32(), T.f32(), T.f64()])
    ins = [
        Float64(xp).ir_value(loc=loc, ip=ip),
        Float64(gp).ir_value(loc=loc, ip=ip),
        Float64(inv2).ir_value(loc=loc, ip=ip),
    ]
    res = llvm.inline_asm(
        struct,
        ins,
        """{
            .reg .b64 tp, up;
            .reg .f32 u<2>, v<2>;
            .reg .b16 b<2>;
            mul.rn.f32x2 tp, $3, $5;
            mul.rn.f32x2 up, tp, $4;
            mov.b64 {u0, u1}, up;
            abs.f32 v0, u0;
            abs.f32 v1, u1;
            max.f32 $1, v0, v1;
            cvt.rn.bf16.f32 b0, u0;
            cvt.rn.bf16.f32 b1, u1;
            mov.b32 $0, {b0, b1};
            cvt.f32.bf16 u0, b0;
            cvt.f32.bf16 u1, b1;
            mov.b64 $2, {u0, u1};
        }""",
        "=r,=f,=d,d,d,d",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
        Float64(llvm.extractvalue(T.f64(), res, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def _max4_f32(m0, m1, m2, m3, *, loc=None, ip=None) -> Float32:
    ins = [Float32(v).ir_value(loc=loc, ip=ip) for v in (m0, m1, m2, m3)]
    res = llvm.inline_asm(
        T.f32(),
        ins,
        """{
            .reg .f32 a, b;
            max.f32 a, $1, $2;
            max.f32 b, $3, $4;
            max.f32 a, a, b;
            mov.f32 $0, a;
        }""",
        "=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(res)


@dsl_user_op
def _bf16_rt(a, *, loc=None, ip=None) -> Float32:
    res = llvm.inline_asm(
        T.f32(),
        [Float32(a).ir_value(loc=loc, ip=ip)],
        """{
            .reg .b16 h;
            cvt.rn.bf16.f32 h, $1;
            cvt.f32.bf16 $0, h;
        }""",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Float32(res)


@dsl_user_op
def _sf_pack4(e0, e1, e2, e3, *, loc=None, ip=None) -> Uint32:
    ins = [Uint32(v).ir_value(loc=loc, ip=ip) for v in (e0, e1, e2, e3)]
    res = llvm.inline_asm(
        T.i32(),
        ins,
        """{
            .reg .b32 lo, hi;
            prmt.b32 lo, $1, $2, 0x0040;   // [e0.b0, e1.b0, -, -]
            prmt.b32 hi, $3, $4, 0x0040;   // [e2.b0, e3.b0, -, -]
            prmt.b32 $0, lo, hi, 0x5410;   // [e0, e1, e2, e3] bytes
        }""",
        "=r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Uint32(res)


@dsl_user_op
def _exp_rec(a, *, loc=None, ip=None):
    struct = llvm.StructType.get_literal([T.i32(), T.i32()])
    res = llvm.inline_asm(
        struct,
        [Float32(a).ir_value(loc=loc, ip=ip)],
        """{
            .reg .b32 b, t<3>;
            .reg .pred p<2>;
            mov.b32 b, $2;
            add.u32 t0, b, 2064384;            // 0x1F8000
            shr.u32 t0, t0, 23;                // ea + [ma8 > 192]
            sub.s32 t1, t0, 8;                 // E'
            max.s32 t1, t1, 94;                // 1e-10 clamp: E >= 94
            setp.eq.u32 p0, b, 2139095040;     // +inf -> E=255
            setp.gt.u32 p1, b, 2139095040;     // NaN  -> E=94
            selp.b32 t2, 255, t1, p0;
            selp.b32 t1, 94, t2, p1;           // E
            mov.b32 $1, t1;
            sub.u32 t2, 254, t1;
            shl.b32 t2, t2, 23;                // bits of 2^(127-E)
            selp.b32 $0, 0, t2, p0;            // +0.0 reciprocal when amax=+inf
        }""",
        "=r,=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _quant4_f(y01, y23, rec2, *, loc=None, ip=None) -> Uint32:
    ins = [
        Float64(y01).ir_value(loc=loc, ip=ip),
        Float64(y23).ir_value(loc=loc, ip=ip),
        Float64(rec2).ir_value(loc=loc, ip=ip),
    ]
    res = llvm.inline_asm(
        T.i32(),
        ins,
        """{
            .reg .b64 qa, qb;
            .reg .b16 p0, p1;
            .reg .f32 q<4>;
            mul.rn.f32x2 qa, $1, $3;
            mul.rn.f32x2 qb, $2, $3;
            mov.b64 {q0, q1}, qa;
            mov.b64 {q2, q3}, qb;
            max.f32 q0, q0, 0fC3E00000;
            max.f32 q1, q1, 0fC3E00000;
            max.f32 q2, q2, 0fC3E00000;
            max.f32 q3, q3, 0fC3E00000;
            min.f32 q0, q0, 0f43E00000;
            min.f32 q1, q1, 0f43E00000;
            min.f32 q2, q2, 0f43E00000;
            min.f32 q3, q3, 0f43E00000;
            cvt.rn.satfinite.e4m3x2.f32 p0, q1, q0;
            cvt.rn.satfinite.e4m3x2.f32 p1, q3, q2;
            mov.b32 $0, {p0, p1};
        }""",
        "=r,d,d,d",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Uint32(res)


class _QueryNormQuant1280:
    def __init__(self, stride_x: int, pdl: bool):
        self.stride_x = int(stride_x)
        self.pdl = pdl

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        rows: Int32,
        stride_sf: Int32,
        eps: Float32,
        stream: cuda_drv.CUstream = None,
    ):
        self.kernel(x_ptr, w_ptr, y_ptr, q_ptr, sf_ptr, stride_sf, eps).launch(
            grid=[rows, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        q_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        stride_sf: Int32,
        eps: Float32,
    ):
        t, _, _ = cute_arch.thread_idx()
        row, _, _ = cute_arch.block_idx()
        warp = t >> 5
        lane = t & 31

        row64 = Int64(row)
        t64 = Int64(t)
        t16 = t64 * 16
        x_addr = x_ptr.toint() + row64 * (2 * self.stride_x) + t16
        w_addr = w_ptr.toint() + t16

        ww = _ldg128(w_addr)
        if cutlass.const_expr(self.pdl):
            cute_arch.griddepcontrol_wait()
        xw = _ldg128(x_addr)
        y_addr = y_ptr.toint() + row64 * (2 * D) + t16
        q_addr = q_ptr.toint() + row64 * D + t64 * 8

        xs = [_unpack2p(xw[i]) for i in range(4)]
        gs = [_unpack2p(ww[i]) for i in range(4)]

        # Preserve the native sum-of-squares reduction order.
        p = _fold8_p(*xs)
        p = p + cute_arch.shuffle_sync_bfly(p, offset=16)
        p = p + cute_arch.shuffle_sync_bfly(p, offset=8)
        p = p + cute_arch.shuffle_sync_bfly(p, offset=4)
        p = p + cute_arch.shuffle_sync_bfly(p, offset=2)
        p = p + cute_arch.shuffle_sync_bfly(p, offset=1)

        swp = cute_arch.alloc_smem(Float32, 16)
        sw_t = cute.make_tensor(swp, cute.make_layout((16,)))
        if lane == 0:
            sw_t[warp] = p
        cute_arch.barrier()
        w0, w1, w2, w3, w4 = (sw_t[i] for i in range(WARPS))
        total = ((w0 + w4) + w2) + (w1 + w3)
        inv = _inv_rsqrt(total, eps)
        inv2 = _dup2(inv)

        nw = [_norm_pair2(xs[i], gs[i], inv2) for i in range(4)]
        yw = [nw[i][0] for i in range(4)]
        _stg128(y_addr, yw[0], yw[1], yw[2], yw[3])

        m8 = _max4_f32(nw[0][1], nw[1][1], nw[2][1], nw[3][1])
        base = lane & 28
        p1 = (lane + 1) & 3
        p2 = (lane + 2) & 3
        p3 = (lane + 3) & 3
        f1 = cute_arch.shuffle_sync(m8, base + p1)
        f2 = cute_arch.shuffle_sync(m8, base + p2)
        f3 = cute_arch.shuffle_sync(m8, base + p3)
        g = _max4_f32(m8, f1, f2, f3)
        # Scale selection must retain the materialized BF16 boundary.
        m = _bf16_rt(g)
        rec, e = _exp_rec(m)
        rec2 = _dup2b(rec)

        _stg64(
            q_addr,
            _quant4_f(nw[0][2], nw[1][2], rec2),
            _quant4_f(nw[2][2], nw[3][2], rec2),
        )
        if cutlass.const_expr(self.pdl):
            cute_arch.griddepcontrol_launch_dependents()

        gsf = cute.make_tensor(
            sf_ptr, cute.make_layout((PACKS, stride_sf), stride=(stride_sf, 1))
        )
        hs = lane & 16
        e1 = cute_arch.shuffle_sync(e, hs + 4)
        e2 = cute_arch.shuffle_sync(e, hs + 8)
        e3 = cute_arch.shuffle_sync(e, hs + 12)
        u = _sf_pack4(e, e1, e2, e3)
        if (lane & 15) == 0:
            pack = 2 * warp + (lane >> 4)
            gsf[(pack, row)] = Int32(u)


_COMPILED: dict = {}


def _supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.shape[-1] != D
        or weight.dtype != torch.bfloat16
        or weight.shape != (D,)
        or not weight.is_contiguous()
        or weight.device != x.device
        or torch.cuda.get_device_capability(x.device) != (10, 0)
    ):
        return False
    flat = x.view(-1, D)
    return (
        flat.stride(1) == 1
        and (flat.stride(0) * 2) % 16 == 0
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    )


def try_query_norm_quant(
    x: torch.Tensor, weight: torch.Tensor, eps: float, *, pdl: bool
):
    if not _supported(x, weight):
        return None
    flat = x.view(-1, D)
    rows = flat.shape[0]
    with torch.cuda.device(x.device):
        normalized = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
        quantized = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
        scales = torch.empty(
            (PACKS, (rows + 3) // 4 * 4), device=x.device, dtype=torch.int32
        ).T[:rows]
        result = normalized, (quantized, scales)
        if not rows:
            return result
        gmem = cute.AddressSpace.gmem
        bf16 = cutlass.BFloat16
        args = (
            make_ptr(bf16, x.data_ptr(), gmem, assumed_align=16),
            make_ptr(bf16, weight.data_ptr(), gmem, assumed_align=16),
            make_ptr(bf16, normalized.data_ptr(), gmem, assumed_align=16),
            make_ptr(bf16, quantized.data_ptr(), gmem, assumed_align=16),
            make_ptr(Int32, scales.data_ptr(), gmem, assumed_align=4),
            Int32(rows),
            Int32(scales.stride(1)),
            Float32(eps),
        )
        stride_x = flat.stride(0)
        key = (x.device.index, stride_x, pdl)
        executor = _COMPILED.get(key)
        stream = cuda_drv.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        if executor is None:
            executor = cute.compile(_QueryNormQuant1280(stride_x, pdl), *args, stream)
            _COMPILED[key] = executor
        executor(*args, stream)
    return result
