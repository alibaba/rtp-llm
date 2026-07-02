#!/usr/bin/env python3
"""FlyDSL megakernel V2a for MI300X/MI308X (CDNA3, gfx942).

Starting from the MI308 V1 baseline:
  - V43 keeps BLOCK_DV=64 for large T to halve DV-block duplication
  - V45 trims the h/W/Vn LDS stride from 140 to 132 without changing counters
  - V46 delays producer K^T LDS stores until after the first loop barrier
  - V47 moves direct O stores after Phase-E barrier to overlap next prefetch
  - lds_k_row: REMOVED (GEMM1 K loaded from GMEM, saves 17KB)
  - LDS: 88KB → 62.8KB (fits MI300X 64KB/CU)
  - B-side LDS: transposed layout with 140/76 stride padding for lower bank conflict
  - h_acc: 8 v4f32; U/O/Vd_acc: 4
  - 4 waves → 8 waves: waves 0-3 compute, waves 4-7 producer-load Q/K^T/A

Architecture: 8 waves (512 threads), single-buffer LDS under the CDNA3 64KB limit.
Target: MI300X (gfx942), DK=DV=128, BT=64, block_DV=64, LDS ≈ 62.8KB
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import triton
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import math as math_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# See flydsl_chunk_gdn_mi308x.py for rationale: the global RocmBackend
# pipeline_fragments monkey-patch is removed because the O=3 upgrade was a
# no-op (or microscopic regression) on the measured shapes, and the patch
# leaked into any other FlyDSL kernel that happened to be imported next.

BT = 64
K_SIZE = 128
V_SIZE = 128
H_SIZE = 32
Hg_SIZE = 8
BLOCK_DV = 64  # V43: reduce DV-block duplication for large T
BLOCK_SIZE = 512  # V2a: 4 compute waves + 4 producer waves

# LDS strides (element count per row, including padding)
STRIDE_KT = 140  # [BT, K]: K+12; GEMM3 B reads K-contiguous vectors
STRIDE_Q = 140  # [BT, K]: K+12
STRIDE_A = 76  # [BT, BT]: BT+12   (also for Pg, Vd storage)
STRIDE_H = 132  # V45: [BDV, K_or_BT], K contiguous; saves 1KB LDS vs 140
STRIDE_VN = 76  # B-operand layout: [BDV, BT], BT contiguous

# N_tiles for BLOCK_DV dimension
NT_BDV = BLOCK_DV // 16  # 2 (was 4 on MI355X)

# V24: producer waves stage current chunk Q/K^T/A while compute waves preload
# the non-Q/K/A operands. Empty producer phase barriers are anchored below so
# LLVM cannot collapse them and let producer overwrite LDS before compute is done.
COMPUTE_STAGE_Q = False
COMPUTE_STAGE_KT = False
COMPUTE_STAGE_A = False
PRODUCER_STAGE_Q = True
PRODUCER_STAGE_KT = True
PRODUCER_STAGE_A = True


def build_megakernel(
    is_varlen=False,
    use_initial_state=False,
    store_ssm_state=False,
    ssm_state_dtype=None,
    h_size=H_SIZE,
    hg_size=Hg_SIZE,
    output_final_state=True,
):
    K = K_SIZE
    V = V_SIZE
    H = h_size
    Hg = hg_size

    LDS_KT_ELEMS = BT * STRIDE_KT  # 8960
    LDS_Q_ELEMS = BT * STRIDE_Q  # 8960
    LDS_A_ELEMS = BT * STRIDE_A  # 4864
    LDS_H_ELEMS = BLOCK_DV * STRIDE_H  # 8448, holds h/W, later V' after W is consumed
    LDS_VN_ELEMS = BLOCK_DV * STRIDE_VN  # 4864, aliases lds_h after Phase D GEMM4
    LDS_G_ELEMS = BT
    LDS_BETA_ELEMS = BT

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="megakernel_mi300x_v2_smem"
    )

    # No lds_k_row — K for GEMM1 loaded from GMEM directly

    lds_kT_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_kT_off + LDS_KT_ELEMS * 2

    lds_q_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_q_off + LDS_Q_ELEMS * 2

    lds_a_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_a_off + LDS_A_ELEMS * 2

    lds_h_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_h_off + LDS_H_ELEMS * 2

    lds_g_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_g_off + LDS_G_ELEMS * 4

    lds_beta_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_beta_off + LDS_BETA_ELEMS * 2

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def megakernel_fn(
        q_ptr: fx.Tensor,
        k_ptr: fx.Tensor,
        v_ptr: fx.Tensor,
        a_ptr: fx.Tensor,
        g_ptr: fx.Tensor,
        beta_ptr: fx.Tensor,
        o_ptr: fx.Tensor,
        h0_ptr: fx.Tensor,
        ht_ptr: fx.Tensor,
        cu_seqlens_ptr: fx.Tensor,
        chunk_offsets_ptr: fx.Tensor,
        prefix_lengths_ptr: fx.Tensor,
        block_map_ptr: fx.Tensor,
        ssm_states_ptr: fx.Tensor,
        max_block_size: fx.Int32,
        seq_size_per_block: fx.Int32,
        ssm_state_stride: fx.Int32,
        token_offset: fx.Int32,
        total_T_for_store: fx.Int32,
        scale: fx.Float32,
        T_val: fx.Int32,
    ):
        v4f32 = T.vec(4, T.f32)
        v4bf16 = T.vec(4, T.bf16)
        v1bf16 = T.vec(1, T.bf16)
        v1f32 = T.vec(1, T.f32)
        v4i16 = T.vec(4, T.i16)
        _z = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def mfma(a, b, c):
            r = rocdl.mfma_f32_16x16x16bf16_1k(
                v4f32,
                [vector.bitcast(v4i16, a), vector.bitcast(v4i16, b), c, _z, _z, _z],
            )
            return r.result if hasattr(r, "result") else r

        def to_idx(v):
            return arith.index_cast(T.index, v)

        def v4e(vec, i):
            return vector.extract(vec, static_position=[i], dynamic_position=[])

        def v4m(vals):
            return vector.from_elements(v4f32, vals)

        def v4_scale(v, s):
            elems = [arith.mulf(v4e(v, i), s) for i in range_constexpr(4)]
            return v4m(elems)

        def bf16_trunc(f32_val):
            return arith.truncf(T.bf16, f32_val)

        def load_b_k16_scalar(lds_ref, stride, k_base, n_col):
            vals = []
            for ii in range_constexpr(4):
                val_vec = vector.load_op(
                    v1bf16, lds_ref, [to_idx((k_base + ii) * stride + n_col)]
                )
                vals.append(
                    vector.extract(val_vec, static_position=[0], dynamic_position=[])
                )
            return vector.from_elements(v4bf16, vals)

        def load_b_k16_vec(lds_ref, stride, k_base, n_col):
            return vector.load_op(v4bf16, lds_ref, [to_idx(n_col * stride + k_base)])

        def load_a_bt_scalar(lds_ref, stride, bt_base, dk_col):
            vals = []
            for ii in range_constexpr(4):
                val_vec = vector.load_op(
                    v1bf16, lds_ref, [to_idx((bt_base + ii) * stride + dk_col)]
                )
                vals.append(
                    vector.extract(val_vec, static_position=[0], dynamic_position=[])
                )
            return vector.from_elements(v4bf16, vals)

        # Thread/Block indices
        bbhv = gpu.block_idx.x
        i_bh = bbhv // (V // BLOCK_DV)
        i_v = bbhv % (V // BLOCK_DV)
        i_b = i_bh // H
        i_h = i_bh % H
        k_head = i_h // (H // Hg)
        tid = gpu.thread_idx.x
        wave_id = tid // 64
        lane = tid % 64

        bos = i_b * T_val
        T_actual = T_val
        NT = (T_val + BT - 1) // BT

        if is_varlen:
            rsrc_cu = buffer_ops.create_buffer_resource(cu_seqlens_ptr, max_size=True)
            rsrc_co = buffer_ops.create_buffer_resource(
                chunk_offsets_ptr, max_size=True
            )
            bos = arith.trunci(
                T.i32, buffer_ops.buffer_load(rsrc_cu, i_b, vec_width=1, dtype=T.i64)
            )
            eos = arith.trunci(
                T.i32,
                buffer_ops.buffer_load(rsrc_cu, i_b + 1, vec_width=1, dtype=T.i64),
            )
            T_actual = eos - bos
            NT = (T_actual + BT - 1) // BT

        # Strides
        stride_q = Hg * K
        stride_k = Hg * K
        stride_v = H * V
        stride_a = H * BT
        stride_g = H
        stride_o = H * V

        # Buffer resources
        rsrc_q = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        rsrc_k = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
        rsrc_v = buffer_ops.create_buffer_resource(v_ptr, max_size=True)
        rsrc_a = buffer_ops.create_buffer_resource(a_ptr, max_size=True)
        rsrc_g = buffer_ops.create_buffer_resource(g_ptr, max_size=True)
        rsrc_beta = buffer_ops.create_buffer_resource(beta_ptr, max_size=True)
        rsrc_o = buffer_ops.create_buffer_resource(o_ptr, max_size=True)
        rsrc_ht = buffer_ops.create_buffer_resource(ht_ptr, max_size=True)
        rsrc_prefix = buffer_ops.create_buffer_resource(
            prefix_lengths_ptr, max_size=True
        )
        rsrc_block_map = buffer_ops.create_buffer_resource(block_map_ptr, max_size=True)
        rsrc_ssm = buffer_ops.create_buffer_resource(ssm_states_ptr, max_size=True)

        # LDS
        lds_base = allocator.get_base()
        lds_kT = SmemPtr(lds_base, lds_kT_off, T.bf16, shape=(LDS_KT_ELEMS,)).get()
        lds_q = SmemPtr(lds_base, lds_q_off, T.bf16, shape=(LDS_Q_ELEMS,)).get()
        lds_a = SmemPtr(lds_base, lds_a_off, T.bf16, shape=(LDS_A_ELEMS,)).get()
        lds_h = SmemPtr(lds_base, lds_h_off, T.bf16, shape=(LDS_H_ELEMS,)).get()
        lds_g = SmemPtr(lds_base, lds_g_off, T.f32, shape=(LDS_G_ELEMS,)).get()
        lds_beta = SmemPtr(
            lds_base, lds_beta_off, T.bf16, shape=(LDS_BETA_ELEMS,)
        ).get()

        # Aliases
        lds_w = lds_h  # W[BT, BDV] reuses lds_h after h reads done
        lds_pg = lds_q  # Pg[BT, BT] reuses lds_q after Q reads done
        lds_vd = lds_a  # Vd[BT, BDV] reuses lds_a after Ag consumed
        lds_vn = lds_h  # V'[BT, BDV] reuses lds_h after W consumed in Phase D

        zero_f32 = arith.constant(0.0, type=T.f32)
        zero_v4 = v4m([zero_f32] * 4)

        def store_h_to_ssm_block(h_vals, dest_block_pos):
            zero_i32 = arith.constant(0, type=T.i32)
            block_map_off = i_b * max_block_size + dest_block_pos
            block_idx = buffer_ops.buffer_load(
                rsrc_block_map, block_map_off, vec_width=1, dtype=T.i32
            )
            valid_block = arith.cmpi(arith.CmpIPredicate.sgt, block_idx, zero_i32)
            block_if = scf.IfOp(valid_block, [], has_else=True)
            with ir.InsertionPoint(block_if.then_block):
                ssm_base = block_idx * ssm_state_stride + i_h * K * V
                for mc in range_constexpr(2):
                    for nt in range_constexpr(NT_BDV):
                        for ii in range_constexpr(4):
                            dk = (wave_id * 2 + mc) * 16 + (lane // 16) * 4 + ii
                            dv = i_v * BLOCK_DV + nt * 16 + lane % 16
                            ssm_off = ssm_base + dv * K + dk
                            val = v4e(h_vals[mc * NT_BDV + nt], ii)
                            if ssm_state_dtype == torch.bfloat16:
                                val = bf16_trunc(val)
                            buffer_ops.buffer_store(val, rsrc_ssm, ssm_off)
                scf.YieldOp([])
            with ir.InsertionPoint(block_if.else_block):
                scf.YieldOp([])

        def maybe_store_h_to_ssm(h_vals, chunk_idx_i32):
            zero_i32 = arith.constant(0, type=T.i32)
            one_i32 = arith.constant(1, type=T.i32)
            bt_i32 = arith.constant(BT, type=T.i32)
            has_total_len = arith.cmpi(
                arith.CmpIPredicate.sgt, total_T_for_store, zero_i32
            )
            local_total_len = token_offset + T_actual
            global_input_len = arith.select(
                has_total_len, total_T_for_store, local_total_len
            )
            global_chunk_base = token_offset + chunk_idx_i32 * bt_i32
            global_chunk_end = global_chunk_base + bt_i32
            is_last = arith.cmpi(
                arith.CmpIPredicate.sge, global_chunk_end, global_input_len
            )
            global_chunk_idx = token_offset // bt_i32 + chunk_idx_i32
            next_global_chunk = global_chunk_idx + one_i32
            chunk_gt_zero = arith.cmpi(
                arith.CmpIPredicate.sgt, global_chunk_idx, zero_i32
            )
            boundary_rem = (next_global_chunk * bt_i32) % seq_size_per_block
            on_block_boundary = arith.cmpi(
                arith.CmpIPredicate.eq, boundary_rem, zero_i32
            )
            is_middle_store = arith.andi(chunk_gt_zero, on_block_boundary)
            prefix = buffer_ops.buffer_load(rsrc_prefix, i_b, vec_width=1, dtype=T.i32)
            last_dest = (prefix + global_input_len - one_i32) // seq_size_per_block
            middle_dest = (
                prefix + global_chunk_idx * bt_i32 + bt_i32 - one_i32
            ) // seq_size_per_block

            last_if = scf.IfOp(is_last, [], has_else=True)
            with ir.InsertionPoint(last_if.then_block):
                store_h_to_ssm_block(h_vals, last_dest)
                scf.YieldOp([])
            with ir.InsertionPoint(last_if.else_block):
                middle_if = scf.IfOp(is_middle_store, [], has_else=True)
                with ir.InsertionPoint(middle_if.then_block):
                    store_h_to_ssm_block(h_vals, middle_dest)
                    scf.YieldOp([])
                with ir.InsertionPoint(middle_if.else_block):
                    scf.YieldOp([])
                scf.YieldOp([])

        # Wave role classification. Only waves 0-3 own the h state and stores.
        is_compute = arith.cmpi(
            arith.CmpIPredicate.slt, wave_id, arith.constant(4, type=T.i32)
        )
        prod_wave = wave_id - 4
        h_result_types = [ir.Type.parse("vector<4xf32>")] * (2 * NT_BDV)

        # Initialize h state: mc=2, nt=NT_BDV=2 → 4 v4f32 (was 8)
        init_h = [zero_v4] * (2 * NT_BDV)  # 4

        if use_initial_state:
            rsrc_h0 = buffer_ops.create_buffer_resource(h0_ptr, max_size=True)
            h0_base = i_bh * K * V
            init_if = scf.IfOp(is_compute, h_result_types, has_else=True)
            with ir.InsertionPoint(init_if.then_block):
                loaded_h = [zero_v4] * (2 * NT_BDV)
                for mc in range_constexpr(2):
                    for nt in range_constexpr(NT_BDV):
                        h0_vals = []
                        for ii in range_constexpr(4):
                            dk = (wave_id * 2 + mc) * 16 + (lane // 16) * 4 + ii
                            dv = i_v * BLOCK_DV + nt * 16 + lane % 16
                            h0_off = h0_base + dv * K + dk
                            h0_vals.append(
                                buffer_ops.buffer_load(
                                    rsrc_h0, h0_off, vec_width=1, dtype=T.f32
                                )
                            )
                        loaded_h[mc * NT_BDV + nt] = v4m(h0_vals)
                scf.YieldOp([loaded_h[i] for i in range(2 * NT_BDV)])
            with ir.InsertionPoint(init_if.else_block):
                scf.YieldOp([init_h[i] for i in range(2 * NT_BDV)])
            init_h = [init_if.results[i] for i in range(2 * NT_BDV)]

        scale_v = arith.ArithValue(scale)

        # Producer waves cooperatively stage the current chunk's Q/K^T/A.
        # The LDS layout remains single-buffered to stay below CDNA3's 64KB limit.
        p_row0 = prod_wave * 8 + lane // 8
        p_col0 = (lane % 8) * 16
        p_a_col0 = (lane % 8) * 8

        def producer_load_q_a_gb(chunk_base_val):
            # Q[BT, K] → lds_q
            if PRODUCER_STAGE_Q:
                for ei in range_constexpr(2):
                    q_row = p_row0 + 32 * ei
                    q_off = (chunk_base_val + q_row) * stride_q + k_head * K + p_col0
                    q_vec = buffer_ops.buffer_load(
                        rsrc_q, q_off, vec_width=8, dtype=T.bf16
                    )
                    vector.store(q_vec, lds_q, [to_idx(q_row * STRIDE_Q + p_col0)])
                    q_vec2 = buffer_ops.buffer_load(
                        rsrc_q, q_off + 8, vec_width=8, dtype=T.bf16
                    )
                    vector.store(q_vec2, lds_q, [to_idx(q_row * STRIDE_Q + p_col0 + 8)])

            # A[BT, BT] → lds_a
            if PRODUCER_STAGE_A:
                for ei in range_constexpr(2):
                    a_row = p_row0 + 32 * ei
                    a_off = (chunk_base_val + a_row) * stride_a + i_h * BT + p_a_col0
                    a_vec = buffer_ops.buffer_load(
                        rsrc_a, a_off, vec_width=8, dtype=T.bf16
                    )
                    vector.store(a_vec, lds_a, [to_idx(a_row * STRIDE_A + p_a_col0)])

            # FlashQLA keeps gamma/beta in shared memory so consumer groups do
            # not issue repeated scalar global loads. Only lane 0..15 of each
            # producer wave write one contiguous quarter of the chunk.
            gb_active = arith.cmpi(
                arith.CmpIPredicate.slt, lane, arith.constant(16, type=T.i32)
            )
            gb_if = scf.IfOp(gb_active, [], has_else=True)
            with ir.InsertionPoint(gb_if.then_block):
                gb_idx = prod_wave * 16 + lane
                g_off = (chunk_base_val + gb_idx) * stride_g + i_h
                g_val = buffer_ops.buffer_load(rsrc_g, g_off, vec_width=1, dtype=T.f32)
                vector.store(
                    vector.from_elements(v1f32, [g_val]),
                    lds_g,
                    [to_idx(gb_idx)],
                    alignment=4,
                )

                beta_off = (chunk_base_val + gb_idx) * stride_g + i_h
                beta_val = buffer_ops.buffer_load(
                    rsrc_beta, beta_off, vec_width=1, dtype=T.bf16
                )
                vector.store(
                    vector.from_elements(v1bf16, [beta_val]), lds_beta, [to_idx(gb_idx)]
                )
                scf.YieldOp([])
            with ir.InsertionPoint(gb_if.else_block):
                scf.YieldOp([])

        def producer_load_kt(chunk_base_val):
            # K[BT, K] → lds_kT row-major. GEMM1 still reads K directly from GMEM.
            if PRODUCER_STAGE_KT:
                for ei in range_constexpr(2):
                    k_row = p_row0 + 32 * ei
                    k_off = (chunk_base_val + k_row) * stride_k + k_head * K + p_col0
                    k_vec = buffer_ops.buffer_load(
                        rsrc_k, k_off, vec_width=8, dtype=T.bf16
                    )
                    k_vec2 = buffer_ops.buffer_load(
                        rsrc_k, k_off + 8, vec_width=8, dtype=T.bf16
                    )
                    vector.store(k_vec, lds_kT, [to_idx(k_row * STRIDE_KT + p_col0)])
                    vector.store(
                        k_vec2, lds_kT, [to_idx(k_row * STRIDE_KT + p_col0 + 8)]
                    )

        def producer_load_kt_regs(chunk_base_val):
            # V46: issue K^T global loads before the first barrier, but keep
            # producer LDS stores after the barrier so LLVM does not need an
            # entry `lgkmcnt(0)` for those stores.
            kt_vecs = [None] * 2
            kt_vecs2 = [None] * 2
            if PRODUCER_STAGE_KT:
                for ei in range_constexpr(2):
                    k_row = p_row0 + 32 * ei
                    k_off = (chunk_base_val + k_row) * stride_k + k_head * K + p_col0
                    kt_vecs[ei] = buffer_ops.buffer_load(
                        rsrc_k, k_off, vec_width=8, dtype=T.bf16
                    )
                    kt_vecs2[ei] = buffer_ops.buffer_load(
                        rsrc_k, k_off + 8, vec_width=8, dtype=T.bf16
                    )
            return kt_vecs, kt_vecs2

        def producer_store_kt_regs(kt_vecs, kt_vecs2):
            if PRODUCER_STAGE_KT:
                for ei in range_constexpr(2):
                    k_row = p_row0 + 32 * ei
                    vector.store(
                        kt_vecs[ei], lds_kT, [to_idx(k_row * STRIDE_KT + p_col0)]
                    )
                    vector.store(
                        kt_vecs2[ei], lds_kT, [to_idx(k_row * STRIDE_KT + p_col0 + 8)]
                    )

        def producer_prefetch_next_q_a_gb(next_chunk_base_val, has_next_iter):
            next_if = scf.IfOp(has_next_iter, [], has_else=True)
            with ir.InsertionPoint(next_if.then_block):
                producer_load_q_a_gb(next_chunk_base_val)
                rocdl.s_waitcnt(0)
                scf.YieldOp([])
            with ir.InsertionPoint(next_if.else_block):
                producer_barrier_anchor()
                scf.YieldOp([])

        def compute_load_q_kt_a(chunk_base_val):
            q_row0 = tid // 8
            q_col0 = (tid % 8) * 16
            if COMPUTE_STAGE_Q:
                for ei in range_constexpr(2):
                    q_row = q_row0 + 32 * ei
                    q_off = (chunk_base_val + q_row) * stride_q + k_head * K + q_col0
                    q_vec = buffer_ops.buffer_load(
                        rsrc_q, q_off, vec_width=8, dtype=T.bf16
                    )
                    vector.store(q_vec, lds_q, [to_idx(q_row * STRIDE_Q + q_col0)])
                    q_vec2 = buffer_ops.buffer_load(
                        rsrc_q, q_off + 8, vec_width=8, dtype=T.bf16
                    )
                    vector.store(q_vec2, lds_q, [to_idx(q_row * STRIDE_Q + q_col0 + 8)])

            if COMPUTE_STAGE_KT:
                for ei in range_constexpr(2):
                    k_row = q_row0 + 32 * ei
                    k_off = (chunk_base_val + k_row) * stride_k + k_head * K + q_col0
                    k_vec = buffer_ops.buffer_load(
                        rsrc_k, k_off, vec_width=8, dtype=T.bf16
                    )
                    k_vec2 = buffer_ops.buffer_load(
                        rsrc_k, k_off + 8, vec_width=8, dtype=T.bf16
                    )
                    vector.store(k_vec, lds_kT, [to_idx(k_row * STRIDE_KT + q_col0)])
                    vector.store(
                        k_vec2, lds_kT, [to_idx(k_row * STRIDE_KT + q_col0 + 8)]
                    )

            a_row0 = tid // 8
            a_col0 = (tid % 8) * 8
            if COMPUTE_STAGE_A:
                for ei in range_constexpr(2):
                    a_row = a_row0 + 32 * ei
                    a_off = (chunk_base_val + a_row) * stride_a + i_h * BT + a_col0
                    a_vec = buffer_ops.buffer_load(
                        rsrc_a, a_off, vec_width=8, dtype=T.bf16
                    )
                    vector.store(a_vec, lds_a, [to_idx(a_row * STRIDE_A + a_col0)])

        def producer_barrier_anchor():
            llvm_dialect.inline_asm(
                None,
                [],
                "s_nop 0",
                "",
                has_side_effects=True,
            )

        # Producer prologue: stage chunk0 Q/A/g/beta before the loop. The loop
        # entry only needs to stage K^T, matching later iterations.
        prologue_if = scf.IfOp(is_compute, [], has_else=True)
        with ir.InsertionPoint(prologue_if.then_block):
            scf.YieldOp([])
        with ir.InsertionPoint(prologue_if.else_block):
            producer_load_q_a_gb(bos)
            rocdl.s_waitcnt(0)
            scf.YieldOp([])

        # ═══════════════════════════════════════════════
        # MAIN LOOP over chunks
        # ═══════════════════════════════════════════════
        for i_c, inner_iter_args, loop_results in scf.for_(
            arith.index(0),
            to_idx(NT),
            arith.index(1),
            iter_args=init_h,
        ):
            i_c_i32 = arith.index_cast(T.i32, i_c)
            h_acc_incoming = list(inner_iter_args)
            chunk_base = bos + i_c_i32 * BT

            role_if = scf.IfOp(is_compute, h_result_types, has_else=True)

            # ────────────────────────────────────────────
            # COMPUTE PATH (waves 0-3): owns h, MFMA, O/ht stores
            # ────────────────────────────────────────────
            with ir.InsertionPoint(role_if.then_block):
                h_acc = list(h_acc_incoming)

                # ════════════════════════════════════════
                # PHASE A: load Q, K^T, A and h; preload V, g, beta
                # ════════════════════════════════════════

                compute_load_q_kt_a(chunk_base)

                for mc in range_constexpr(2):
                    for nt in range_constexpr(NT_BDV):
                        for ii in range_constexpr(4):
                            dk = (wave_id * 2 + mc) * 16 + (lane // 16) * 4 + ii
                            dv = nt * 16 + lane % 16
                            val_bf16 = bf16_trunc(v4e(h_acc[mc * NT_BDV + nt], ii))
                            vector.store(
                                vector.from_elements(v1bf16, [val_bf16]),
                                lds_h,
                                [to_idx(dv * STRIDE_H + dk)],
                            )

                v_preloads = [[None] * 4 for _ in range(NT_BDV)]
                for nt in range_constexpr(NT_BDV):
                    for ii in range_constexpr(4):
                        v_row = wave_id * 16 + (lane // 16) * 4 + ii
                        v_col = i_v * BLOCK_DV + nt * 16 + lane % 16
                        v_off = (chunk_base + v_row) * stride_v + i_h * V + v_col
                        v_preloads[nt][ii] = buffer_ops.buffer_load(
                            rsrc_v, v_off, vec_width=1, dtype=T.bf16
                        )

                last_idx_candidate = arith.ArithValue(
                    arith.muli((i_c_i32 + 1), arith.constant(BT, type=T.i32))
                )
                T_act_v = arith.ArithValue(T_actual)
                cmp_lt = arith.cmpi(
                    arith.CmpIPredicate.slt, last_idx_candidate, T_act_v
                )
                last_idx_clamped = arith.select(cmp_lt, last_idx_candidate, T_act_v)
                last_idx = arith.subi(last_idx_clamped, arith.constant(1, type=T.i32))
                g_last_off = (bos + last_idx) * stride_g + i_h
                g_last = buffer_ops.buffer_load(
                    rsrc_g, g_last_off, vec_width=1, dtype=T.f32
                )

                gpu.barrier()

                g_row_vals = [None] * 4
                for ii in range_constexpr(4):
                    g_abs_row = wave_id * 16 + (lane // 16) * 4 + ii
                    g_vec = vector.load_op(v1f32, lds_g, [to_idx(g_abs_row)])
                    g_row_vals[ii] = vector.extract(
                        g_vec, static_position=[0], dynamic_position=[]
                    )

                g_col_vals = [None] * 4
                for nt in range_constexpr(4):
                    g_abs_col = nt * 16 + lane % 16
                    g_vec = vector.load_op(v1f32, lds_g, [to_idx(g_abs_col)])
                    g_col_vals[nt] = vector.extract(
                        g_vec, static_position=[0], dynamic_position=[]
                    )

                beta_col_vals = [None] * 4
                for nt in range_constexpr(4):
                    beta_abs_col = nt * 16 + lane % 16
                    beta_vec = vector.load_op(v1bf16, lds_beta, [to_idx(beta_abs_col)])
                    beta_col_vals[nt] = vector.extract(
                        beta_vec, static_position=[0], dynamic_position=[]
                    )

                # ════════════════════════════════════════
                # PHASE B: GEMM1 U=K@h, GEMM2 O_inter=Q@h, W, O scaling
                # GEMM1: K loaded from GMEM (no lds_k_row on MI300X)
                # ════════════════════════════════════════

                U_acc = [zero_v4] * NT_BDV
                O_acc = [zero_v4] * NT_BDV

                for kt in range_constexpr(8):  # K/16 = 8 reduction tiles on gfx942
                    b_h = [None] * NT_BDV
                    for nt in range_constexpr(NT_BDV):
                        k_base = kt * 16 + (lane // 16) * 4
                        n_col = nt * 16 + lane % 16
                        b_h[nt] = load_b_k16_vec(lds_h, STRIDE_H, k_base, n_col)

                    k_row_idx = wave_id * 16 + lane % 16
                    k_col_idx = kt * 16 + (lane // 16) * 4
                    k_gmem_off = (
                        (chunk_base + k_row_idx) * stride_k + k_head * K + k_col_idx
                    )
                    a_K = buffer_ops.buffer_load(
                        rsrc_k, k_gmem_off, vec_width=4, dtype=T.bf16
                    )

                    a_q_off = (
                        (wave_id * 16 + lane % 16) * STRIDE_Q
                        + kt * 16
                        + (lane // 16) * 4
                    )
                    a_Q = vector.load_op(v4bf16, lds_q, [to_idx(a_q_off)])

                    for nt in range_constexpr(NT_BDV):
                        U_acc[nt] = mfma(a_K, b_h[nt], U_acc[nt])
                        O_acc[nt] = mfma(a_Q, b_h[nt], O_acc[nt])

                g_exp_row = [
                    math_dialect.exp2(g_row_vals[ii]) for ii in range_constexpr(4)
                ]
                # g_col_vals used directly below via exp2(g_row - g_col)
                # to avoid fp32 overflow from separate exp2(-g_col)

                for nt in range_constexpr(NT_BDV):
                    for ii in range_constexpr(4):
                        v_f32 = arith.extf(T.f32, v_preloads[nt][ii])
                        u_val = v4e(U_acc[nt], ii)
                        w_val = arith.subf(v_f32, arith.mulf(g_exp_row[ii], u_val))
                        U_acc[nt] = vector.insert(
                            w_val, U_acc[nt], static_position=[ii], dynamic_position=[]
                        )

                        o_val = v4e(O_acc[nt], ii)
                        o_scaled = arith.mulf(o_val, arith.mulf(scale_v, g_exp_row[ii]))
                        O_acc[nt] = vector.insert(
                            o_scaled,
                            O_acc[nt],
                            static_position=[ii],
                            dynamic_position=[],
                        )

                gpu.barrier()

                # ════════════════════════════════════════
                # PHASE C: W→lds_w, GEMM3 P=Q@K^T, Ag, Pg
                # ════════════════════════════════════════

                for nt in range_constexpr(NT_BDV):
                    for ii in range_constexpr(4):
                        w_row = wave_id * 16 + (lane // 16) * 4 + ii
                        w_col = nt * 16 + lane % 16
                        w_bf16 = bf16_trunc(v4e(U_acc[nt], ii))
                        vector.store(
                            vector.from_elements(v1bf16, [w_bf16]),
                            lds_w,
                            [to_idx(w_col * STRIDE_H + w_row)],
                        )

                P_acc = [zero_v4] * 4

                for kt in range_constexpr(8):
                    b_kT = [None] * 4
                    for nt in range_constexpr(4):
                        k_base = kt * 16 + (lane // 16) * 4
                        n_col = nt * 16 + lane % 16
                        b_kT[nt] = load_b_k16_vec(lds_kT, STRIDE_KT, k_base, n_col)

                    a_q_off = (
                        (wave_id * 16 + lane % 16) * STRIDE_Q
                        + kt * 16
                        + (lane // 16) * 4
                    )
                    a_Q = vector.load_op(v4bf16, lds_q, [to_idx(a_q_off)])

                    for nt in range_constexpr(4):
                        P_acc[nt] = mfma(a_Q, b_kT[nt], P_acc[nt])

                for nt in range_constexpr(4):
                    for ii in range_constexpr(4):
                        abs_row = wave_id * 16 + (lane // 16) * 4 + ii
                        abs_col = nt * 16 + lane % 16

                        g_gate = math_dialect.exp2(
                            arith.subf(g_row_vals[ii], g_col_vals[nt])
                        )
                        causal = arith.cmpi(arith.CmpIPredicate.sge, abs_row, abs_col)
                        g_masked = arith.select(causal, g_gate, zero_f32)

                        a_lds_off = abs_row * STRIDE_A + abs_col
                        a_elem_vec = vector.load_op(v1bf16, lds_a, [to_idx(a_lds_off)])
                        a_elem = arith.extf(
                            T.f32,
                            vector.extract(
                                a_elem_vec, static_position=[0], dynamic_position=[]
                            ),
                        )
                        beta_f32 = arith.extf(T.f32, beta_col_vals[nt])
                        a_masked = arith.select(causal, a_elem, zero_f32)
                        ag_val = arith.mulf(a_masked, beta_f32)
                        ag_bf16 = bf16_trunc(ag_val)
                        vector.store(
                            vector.from_elements(v1bf16, [ag_bf16]),
                            lds_a,
                            [to_idx(a_lds_off)],
                        )

                        p_val = v4e(P_acc[nt], ii)
                        pg_val = arith.mulf(scale_v, arith.mulf(g_masked, p_val))
                        pg_bf16 = bf16_trunc(pg_val)
                        pg_lds_off = abs_row * STRIDE_A + abs_col
                        vector.store(
                            vector.from_elements(v1bf16, [pg_bf16]),
                            lds_pg,
                            [to_idx(pg_lds_off)],
                        )

                gpu.barrier()

                # ════════════════════════════════════════
                # PHASE D: GEMM4 Vd=Ag@W, V', Vd→lds, V'→lds_vn
                # ════════════════════════════════════════

                Vd_acc = [zero_v4] * NT_BDV

                for kt in range_constexpr(4):  # BT/16 = 4 reduction tiles on gfx942
                    b_w = [None] * NT_BDV
                    for nt in range_constexpr(NT_BDV):
                        k_base = kt * 16 + (lane // 16) * 4
                        n_col = nt * 16 + lane % 16
                        b_w[nt] = load_b_k16_vec(lds_w, STRIDE_H, k_base, n_col)

                    a_ag_off = (
                        (wave_id * 16 + lane % 16) * STRIDE_A
                        + kt * 16
                        + (lane // 16) * 4
                    )
                    a_Ag = vector.load_op(v4bf16, lds_a, [to_idx(a_ag_off)])

                    for nt in range_constexpr(NT_BDV):
                        Vd_acc[nt] = mfma(a_Ag, b_w[nt], Vd_acc[nt])

                for nt in range_constexpr(NT_BDV):
                    for ii in range_constexpr(4):
                        abs_row = wave_id * 16 + (lane // 16) * 4 + ii
                        vd_val = v4e(Vd_acc[nt], ii)

                        vd_bf16 = bf16_trunc(vd_val)
                        vd_col = nt * 16 + lane % 16
                        vd_lds_off = vd_col * STRIDE_A + abs_row
                        vector.store(
                            vector.from_elements(v1bf16, [vd_bf16]),
                            lds_vd,
                            [to_idx(vd_lds_off)],
                        )

                        g_rev = arith.subf(g_last, g_row_vals[ii])
                        g_rev_exp = math_dialect.exp2(g_rev)
                        vn_val = arith.mulf(g_rev_exp, vd_val)
                        vn_bf16 = bf16_trunc(vn_val)
                        vn_col = nt * 16 + lane % 16
                        vn_lds_off = vn_col * STRIDE_VN + abs_row
                        vector.store(
                            vector.from_elements(v1bf16, [vn_bf16]),
                            lds_vn,
                            [to_idx(vn_lds_off)],
                        )

                gpu.barrier()

                # ════════════════════════════════════════
                # PHASE E: GEMM5 O+=Pg@Vd, Store O → GMEM
                # ════════════════════════════════════════

                for kt in range_constexpr(4):
                    b_vd = [None] * NT_BDV
                    for nt in range_constexpr(NT_BDV):
                        k_base = kt * 16 + (lane // 16) * 4
                        n_col = nt * 16 + lane % 16
                        b_vd[nt] = load_b_k16_vec(lds_vd, STRIDE_A, k_base, n_col)

                    a_pg_off = (
                        (wave_id * 16 + lane % 16) * STRIDE_A
                        + kt * 16
                        + (lane // 16) * 4
                    )
                    a_Pg = vector.load_op(v4bf16, lds_pg, [to_idx(a_pg_off)])

                    for nt in range_constexpr(NT_BDV):
                        O_acc[nt] = mfma(a_Pg, b_vd[nt], O_acc[nt])

                gpu.barrier()

                # V47: O stores are direct GMEM stores, but delaying them past
                # this barrier lets producer waves prefetch next Q/A/g/beta
                # while compute waves drain O stores.
                for nt in range_constexpr(NT_BDV):
                    for ii in range_constexpr(4):
                        o_row = wave_id * 16 + (lane // 16) * 4 + ii
                        o_col = i_v * BLOCK_DV + nt * 16 + lane % 16
                        o_off = (chunk_base + o_row) * stride_o + i_h * V + o_col
                        o_bf16 = bf16_trunc(v4e(O_acc[nt], ii))
                        buffer_ops.buffer_store(o_bf16, rsrc_o, o_off)

                # ════════════════════════════════════════
                # PHASE F: h *= exp2(g_last); GEMM6 h += K^T @ V'
                # ════════════════════════════════════════

                g_last_exp = math_dialect.exp2(g_last)
                for idx in range_constexpr(2 * NT_BDV):
                    h_acc[idx] = v4_scale(h_acc[idx], g_last_exp)

                for mc in range_constexpr(2):
                    for kt in range_constexpr(4):  # BT/16 = 4 reduction tiles on gfx942
                        b_vn = [None] * NT_BDV
                        for nt in range_constexpr(NT_BDV):
                            k_base = kt * 16 + (lane // 16) * 4
                            n_col = nt * 16 + lane % 16
                            b_vn[nt] = load_b_k16_vec(lds_vn, STRIDE_VN, k_base, n_col)

                        a_dk = (wave_id * 2 + mc) * 16 + lane % 16
                        a_bt = kt * 16 + (lane // 16) * 4
                        a_kT = load_a_bt_scalar(lds_kT, STRIDE_KT, a_bt, a_dk)

                        for nt in range_constexpr(NT_BDV):
                            h_acc[mc * NT_BDV + nt] = mfma(
                                a_kT, b_vn[nt], h_acc[mc * NT_BDV + nt]
                            )

                if store_ssm_state:
                    maybe_store_h_to_ssm(h_acc, i_c_i32)

                gpu.barrier()
                scf.YieldOp([h_acc[i] for i in range(2 * NT_BDV)])

            # ────────────────────────────────────────────
            # PRODUCER PATH (waves 4-7): stage current chunk Q/K^T/A
            # ────────────────────────────────────────────
            with ir.InsertionPoint(role_if.else_block):
                kt_vecs, kt_vecs2 = producer_load_kt_regs(chunk_base)
                gpu.barrier()
                producer_store_kt_regs(kt_vecs, kt_vecs2)
                # K^T is first consumed in Phase C; keep the explicit full wait
                # on the Phase-C gate. LLVM may still insert an LDS wait before
                # the first barrier to order the producer's ds stores.
                rocdl.s_waitcnt(0)
                gpu.barrier()
                producer_barrier_anchor()
                gpu.barrier()
                producer_barrier_anchor()
                gpu.barrier()
                producer_barrier_anchor()
                gpu.barrier()
                next_i_c_i32 = arith.addi(i_c_i32, arith.constant(1, type=T.i32))
                has_next_iter = arith.cmpi(arith.CmpIPredicate.slt, next_i_c_i32, NT)
                next_chunk_base = chunk_base + BT
                producer_prefetch_next_q_a_gb(next_chunk_base, has_next_iter)
                gpu.barrier()
                scf.YieldOp([h_acc_incoming[i] for i in range(2 * NT_BDV)])

            h_result = [role_if.results[i] for i in range(2 * NT_BDV)]
            yield h_result

        # ═══════════ Store final state h → ht ═══════════
        # Compile-time guard: omit the entire ht store IR when the caller does
        # not request final state, so an empty/dummy ht_ptr is never written.
        if output_final_state:
            final_h = list(loop_results)
            ht_base = i_bh * K * V
            store_if = scf.IfOp(is_compute, [], has_else=True)
            with ir.InsertionPoint(store_if.then_block):
                for mc in range_constexpr(2):
                    for nt in range_constexpr(NT_BDV):
                        for ii in range_constexpr(4):
                            dk = (wave_id * 2 + mc) * 16 + (lane // 16) * 4 + ii
                            dv = i_v * BLOCK_DV + nt * 16 + lane % 16
                            ht_off = ht_base + dv * K + dk
                            buffer_ops.buffer_store(
                                v4e(final_h[mc * NT_BDV + nt], ii), rsrc_ht, ht_off
                            )
                scf.YieldOp([])
            with ir.InsertionPoint(store_if.else_block):
                scf.YieldOp([])

    @flyc.jit
    def launch(
        q_ptr: fx.Tensor,
        k_ptr: fx.Tensor,
        v_ptr: fx.Tensor,
        a_ptr: fx.Tensor,
        g_ptr: fx.Tensor,
        beta_ptr: fx.Tensor,
        o_ptr: fx.Tensor,
        h0_ptr: fx.Tensor,
        ht_ptr: fx.Tensor,
        cu_seqlens_ptr: fx.Tensor,
        chunk_offsets_ptr: fx.Tensor,
        prefix_lengths_ptr: fx.Tensor,
        block_map_ptr: fx.Tensor,
        ssm_states_ptr: fx.Tensor,
        max_block_size: fx.Int32,
        seq_size_per_block: fx.Int32,
        ssm_state_stride: fx.Int32,
        token_offset: fx.Int32,
        total_T_for_store: fx.Int32,
        scale: fx.Float32,
        T_val: fx.Int32,
        N_val: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = (V_SIZE // BLOCK_DV) * N_val * H
        megakernel_fn(
            q_ptr,
            k_ptr,
            v_ptr,
            a_ptr,
            g_ptr,
            beta_ptr,
            o_ptr,
            h0_ptr,
            ht_ptr,
            cu_seqlens_ptr,
            chunk_offsets_ptr,
            prefix_lengths_ptr,
            block_map_ptr,
            ssm_states_ptr,
            max_block_size,
            seq_size_per_block,
            ssm_state_stride,
            token_offset,
            total_T_for_store,
            scale,
            T_val,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch


_megakernel_cache = {}


def megakernel_fwd(
    q,
    k,
    v,
    a,
    g,
    beta,
    scale,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_offsets=None,
    prefix_lengths=None,
    block_map=None,
    ssm_states=None,
    seq_size_per_block=None,
):
    """Fused recompute_wu + fwd_h + fwd_o megakernel (MI300X V1).

    Args:
        q: [B, T, Hg, K] bf16
        k: [B, T, Hg, K] bf16
        v: [B, T, H, V] bf16
        a: [B, T, H, BT] bf16 -- A_inv from kkt_solve
        g: [B, T, H] fp32 -- cumsum output (log2 domain)
        beta: [B, T, H] bf16
        scale: float
        initial_state: [B, H, K, V] fp32 or None
        output_final_state: bool
        cu_seqlens: [N+1] int64 or None
        chunk_offsets: [N+1] int64 or None

    Returns:
        o: [B, T, H, V] bf16
        final_state: [N, H, K, V] fp32 or None
    """
    B, T_total, Hg, K = q.shape
    _, _, H, V = v.shape
    if ssm_states is not None:
        if prefix_lengths is None or block_map is None or seq_size_per_block is None:
            raise ValueError(
                "prefix_lengths, block_map and seq_size_per_block are required "
                "when FlyDSL writes ssm_states directly"
            )

    use_h0 = initial_state is not None
    use_vl = cu_seqlens is not None
    store_ssm_state = ssm_states is not None

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        if chunk_offsets is None:
            lens = cu_seqlens[1:] - cu_seqlens[:-1]
            chunk_offsets = torch.cat(
                [cu_seqlens.new_tensor([0]), triton.cdiv(lens, BT)]
            ).cumsum(-1)
    else:
        N = B

    cache_key = (
        H,
        Hg,
        K,
        V,
        use_h0,
        use_vl,
        store_ssm_state,
        ssm_states.dtype if store_ssm_state else None,
        output_final_state,
    )
    if cache_key not in _megakernel_cache:
        _megakernel_cache[cache_key] = build_megakernel(
            is_varlen=use_vl,
            use_initial_state=use_h0,
            store_ssm_state=store_ssm_state,
            ssm_state_dtype=ssm_states.dtype if store_ssm_state else None,
            h_size=H,
            hg_size=Hg,
            output_final_state=output_final_state,
        )
    fn = _megakernel_cache[cache_key]

    o = torch.empty(B, T_total, H, V, device=q.device, dtype=q.dtype)
    # Kernel writes ht with V-first byte formula: offset = dv*K + dk,
    # so the allocation must be [N, H, V, K] (contiguous K innermost).
    ht = (
        torch.empty(N, H, V, K, device=q.device, dtype=torch.float32)
        if output_final_state
        else None
    )

    dummy = torch.empty(0, device=q.device, dtype=torch.float32)
    dummy_i32 = torch.empty(0, device=q.device, dtype=torch.int32)
    h0_arg = initial_state if initial_state is not None else dummy
    ht_arg = ht if ht is not None else dummy
    cu_arg = cu_seqlens if cu_seqlens is not None else dummy.to(torch.int64)
    co_arg = chunk_offsets if chunk_offsets is not None else dummy.to(torch.int64)
    prefix_arg = prefix_lengths if prefix_lengths is not None else dummy_i32
    block_map_arg = block_map if block_map is not None else dummy_i32
    ssm_arg = ssm_states if ssm_states is not None else dummy
    max_block_size = block_map.shape[1] if block_map is not None else 0
    ssm_state_stride = ssm_states.stride(0) if ssm_states is not None else 0
    seq_block = seq_size_per_block if seq_size_per_block is not None else 1
    total_t_for_store = T_total if cu_seqlens is None else 0

    fn(
        q,
        k,
        v,
        a,
        g,
        beta,
        o,
        h0_arg,
        ht_arg,
        cu_arg,
        co_arg,
        prefix_arg,
        block_map_arg,
        ssm_arg,
        max_block_size,
        seq_block,
        ssm_state_stride,
        0,
        total_t_for_store,
        scale,
        T_total,
        N,
    )

    return o, ht
