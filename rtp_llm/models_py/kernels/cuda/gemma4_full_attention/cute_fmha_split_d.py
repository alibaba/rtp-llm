# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Adapted for RTP-LLM (Gemma4 split-D full-attention prototype) from the
# FlashInfer CuTe-DSL Blackwell FMHA reference snapshots:
#   - build_logs/.../references/flashinfer_attention/prefill.py
#       (kernel __call__ / @cute.kernel structure, NHD layouts with GQA
#        broadcast, SharedStorage, fragment creation, launch parameters)
#   - build_logs/.../references/flashinfer_attention/roles/mma.py
#       (gemm_qk kphase unrolling with a fresh local TiledMma per helper,
#        alloc_tmem/dealloc_tmem lifecycle)
#   - build_logs/.../references/flashinfer_attention/roles/softmax.py
#       (online-softmax step: TMEM load, row_max reduce, exp2_scale,
#        packed-f32x2 row_sum reduction; single-warpgroup variant here)
#   - build_logs/.../references/flashinfer_attention/roles/loader_tma.py
#       (TMA partitioning + copy patterns)
#   - build_logs/.../references/flashinfer_attention/pipeline_topology.py
#       (pipeline type/thread-count rules, create() kwargs)
#   - build_logs/.../references/flashinfer_attention/warp_schedule.py
#       (warp role IDs, register budgets, barrier IDs)
# The upstream scheduler/ (persistent.py), fusion/ (mask.py, variant.py) and
# softmax_math.py modules are not present in the local snapshot; their logic
# (causal masking, trip counts, exp2_scale) is inlined below.

"""Gemma4 split-D full attention kernels (SM100/SM103).

Gemma4's full (non-sliding-window) attention uses head_dim=512 for Q/K/V.
A single tcgen05 MMA cannot execute P@V at N=512 (instruction N is capped at
256), and staging whole [128, 512] K rows next to a resident Q tile overflows
the 232,448-byte SM100 SMEM budget.  This module implements the two-phase
split-D plan from docs/c6b-implementation-plan.md:

* Phase 1 splits the D=512 QK reduction into four 128-wide chunks and computes
  one causal online-softmax trajectory.  It writes the final per-row ``(m,l)``
  statistics to a ``[T, Hq, 2]`` FP32 tensor.
* Phase 2 consumes those fixed statistics, recomputes the same full-D score
  tiles, forms BF16 ``P = exp((S-m)*scale)``, and computes both 256-wide P@V
  output halves.  The two output-half CTAs read the same Phase-1 ``(m,l)``;
  neither computes or updates a private softmax trajectory.  Each FP32 output
  accumulator is divided by the shared ``l`` before a contiguous BF16
  ``[T,Hq,512]`` result is stored.

Both phases use four resident [128,128] Q chunks.  K streams in [128,128]
chunks.  Phase 2 separately stages one [128,256] V-half tile and aliases the
finished Q storage for its output epilogue, keeping its SMEM allocation below
the SM100 limit.

Prototype constraints (single sequence, no varlen/batch):
* one causal sequence per call: s_q == s_k == s_v == T
* Hq % Hkv == 0 (GQA), head_dim == 512, BF16 full output, FP32 stats
* non-persistent grids; every CTA owns one Q tile and one query head.  Phase 2
  adds a two-element grid dimension for the two 256-wide output halves.
"""

from typing import Tuple

from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

setup_cutlass_import_path()

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.pipeline import Agent, CooperativeGroup, PipelineConsumer, PipelineProducer

# The pinned nvidia-cutlass-dsl exposes OperandMajorMode under tcgen05 (see
# references/cutlass_cute/blackwell_helpers.py imports); newer trees re-export
# it from cutlass.cute.nvgpu.  Import both ways for compatibility.
try:
    from cutlass.cute.nvgpu.tcgen05 import OperandMajorMode
except ImportError:  # pragma: no cover - depends on DSL version
    from cutlass.cute.nvgpu import OperandMajorMode


LOG2_E = 1.4426950408889634


@cute.jit
def _exp2_scale(frg: cute.Tensor, scale: Float32, row_max: Float32):
    """In-place exp2((x - m) * scale) over a register fragment.

    Inline replacement for the (not snapshotted) flashinfer
    ``softmax_math.exp2_scale``.
    """
    for i in cutlass.range_constexpr(cute.size(frg)):
        frg[i] = cute.arch.exp2(frg[i] * scale - row_max * scale)


class Gemma4SplitDStatsKernel:
    """Phase-1 statistics kernel: causal GQA softmax (m, l) at head_dim=512.

    Warp layout (8 warps, 256 threads — trimmed from the vendor 16-warp
    PREFILL_SCHEDULE; the correction/epilogue/second-softmax roles are unused
    because stats never leave the softmax warpgroup)::

        warps 0-3  softmax   (one warpgroup; owns one 128-row Q tile)
        warp  4    mma       (tcgen05 UMMA issue)
        warp  5    load      (TMA)
        warp  6    idle      (register deallocation only)
        warp  7    empty     (barrier init, register deallocation)

    Like the vendor schedule, every warpgroup is complete (setmaxnreg moves
    registers between full 4-warp groups): warpgroup 0 allocates up to the
    softmax budget, warpgroup 1 deallocates the four single-warp roles.

    Pipelines::

        load --[load_q, TMA_UMMA, 4 stages]--> mma     (Q chunk residency)
        load --[load_k, TMA_UMMA, 2 stages]--> mma     (K chunk streaming)
        mma  --[mma_s,  UMMA_ASYNC, 1 stage]--> softmax (S tile handshake)
    """

    def __init__(self):
        # Tile geometry (compile-time constants).
        # GEMMA4_SPLIT_D_TILE_M is a RATE-PROBE knob only (20261008): at 64
        # the softmax Ld32x32b partition maps 2 threads per row, so the
        # stored (m, l) stats are WRONG — the kernel still completes and
        # its wall time measures the M64 tcgen05 MMA aggregate rate vs the
        # M128 default. Never use 64 for production numerics.
        import os as _os_tm

        self.tile_m = int(_os_tm.environ.get("GEMMA4_SPLIT_D_TILE_M", "128"))
        if self.tile_m not in (64, 128):
            raise ValueError(
                f"GEMMA4_SPLIT_D_TILE_M={self.tile_m} unsupported (64 or 128)"
            )
        self.tile_n = 128
        self.d_chunk = 128
        self.num_d_chunks = 4  # head_dim 512 / d_chunk 128
        self.head_dim = self.d_chunk * self.num_d_chunks
        # Full-D tiler describes the logical GEMM; the chunk tiler drives the
        # SMEM staging of both Q (A operand) and K (B operand).
        self.qk_mma_tiler = (self.tile_m, self.tile_n, self.head_dim)
        self.qk_chunk_tiler = (self.tile_m, self.tile_n, self.d_chunk)

        self.qk_acc_dtype = Float32
        self.q_stages = self.num_d_chunks  # Q chunk residency: stage i <- chunk i
        self.kv_stages = 2  # double-buffered K chunks

        # Warp schedule (subset of the vendor PREFILL_SCHEDULE; both
        # warpgroups complete so setmaxnreg reallocation stays well-formed).
        self.num_softmax_warps = 4
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.idle_warp_id = 6
        self.empty_warp_id = 7
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * 8
        self.num_regs_softmax = 192
        self.num_regs_other = 32
        self.num_regs_empty = 24
        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.tmem_dealloc_arrive_count = self.threads_per_warp * self.num_softmax_warps

        # TMEM: S tile [128, 128] fp32 occupies 128 columns; allocate the full
        # 512-column capacity exactly like the vendor TmemLayout.
        self.tmem_s_offset = 0
        self.tmem_alloc_cols = 512

        # Populated at __call__ time (vendor pattern).
        self.q_dtype = None
        self.k_dtype = None

    # ------------------------------------------------------------------
    # MMA primitives (vendor MmaRole.gemm_qk, extended with an `accumulate`
    # flag so the four D-chunk GEMMs of one KV tile chain in TMEM).
    # ------------------------------------------------------------------

    @cute.jit
    def _make_local_qk_mma(self) -> cute.TiledMma:
        return sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.qk_chunk_tiler[:2],
        )

    @cute.jit
    def _gemm_qk(
        self,
        tStS: cute.Tensor,
        tSrQ_slice: cute.Tensor,
        tSrK_slice: cute.Tensor,
        accumulate: bool,
    ):
        """S (+)= Q_chunk * K_chunk^T over the chunk's kphases (16 each)."""
        local_mma = self._make_local_qk_mma()
        num_kphases = cute.size(tSrQ_slice, mode=[2])
        for kphase_idx in cutlass.range_constexpr(num_kphases):
            coord = (None, None, kphase_idx)
            local_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0 or accumulate)
            cute.gemm(local_mma, tStS, tSrQ_slice[coord], tSrK_slice[coord], tStS)

    @cute.jit
    def _alloc_tmem(self, storage: cute.Tensor):
        tmem_alloc_cols = Int32(self.tmem_alloc_cols)
        cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
        cute.arch.barrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            number_of_threads=self.threads_per_warp,
        )

    @cute.jit
    def _dealloc_tmem(self, storage: cute.Tensor, tmem_dealloc_mbar_ptr: Int32):
        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
        tmem_alloc_cols = Int32(self.tmem_alloc_cols)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32,
            alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

    # ------------------------------------------------------------------
    # Roles
    # ------------------------------------------------------------------

    @cute.jit
    def _loader_run(
        self,
        qk_thr_mma: cute.core.ThrMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        load_q_producer: PipelineProducer,
        load_k_producer: PipelineProducer,
        q_tile_coord: Int32,
        head_coord: Int32,
        trip_count: Int32,
    ):
        """TMA loads: four resident Q chunks, then K chunks per KV tile."""
        # Q: flat-divide (s, d) by the chunk tile (128, 128).  The TMA atom
        # covers one [128, 128] chunk (box dims <= 256 — a whole [128, 512]
        # row would be an illegal TMA box).
        gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_chunk_tiler, mode=[0, 2]))
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )
        # Keep (vals, q_tiles); select the D chunk and (head, batch=0).
        for dk in cutlass.range_constexpr(self.num_d_chunks):
            tQgQ = tQgQ_qdl[None, None, dk, (head_coord, 0)]
            q_handle = load_q_producer.acquire_and_advance()
            cute.copy(
                tma_atom_q,
                tQgQ[None, q_tile_coord],
                tQsQ[None, q_handle.index],
                tma_bar_ptr=q_handle.barrier,
            )

        # K: same chunked partition on the B side.
        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_chunk_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK_kdl, 0, 3),
        )
        kv_coord = Int32(0)
        for _i in cutlass.range(0, trip_count, 1, unroll=1):
            for dk in cutlass.range_constexpr(self.num_d_chunks):
                tKgK = tKgK_kdl[None, None, dk, (head_coord, 0)]
                k_handle = load_k_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_k,
                    tKgK[None, kv_coord],
                    tKsK[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )
            kv_coord += 1

    @cute.jit
    def _mma_run(
        self,
        tStS0: cute.Tensor,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        load_q_consumer: PipelineConsumer,
        load_k_consumer: PipelineConsumer,
        mma_s_producer: PipelineProducer,
        trip_count: Int32,
        storage: cute.Tensor,
        tmem_dealloc_mbar_ptr: Int32,
    ):
        self._alloc_tmem(storage)

        # Wait for the four resident Q chunk stages (loaded once; stage i
        # holds chunk i, matching the loader's in-order fill).
        for _dk in cutlass.range_constexpr(self.num_d_chunks):
            load_q_consumer.wait_and_advance()

        for _i in cutlass.range(0, trip_count, 1, unroll=1):
            s_handle = mma_s_producer.acquire_and_advance()
            # S = sum over the four D chunks of Q_chunk @ K_chunk^T.  The
            # first chunk overwrites S (fresh tile); the rest accumulate, so
            # TMEM holds the full-D=512 reduction before softmax sees it.
            for dk in cutlass.range_constexpr(self.num_d_chunks):
                k_handle = load_k_consumer.wait_and_advance()
                tSrK_c = tSrK[None, None, None, k_handle.index]
                self._gemm_qk(tStS0, tSrQ[None, None, None, dk], tSrK_c, dk != 0)
                k_handle.release()
            s_handle.commit()

        self._dealloc_tmem(storage, tmem_dealloc_mbar_ptr)

    @cute.jit
    def _softmax_step(
        self,
        need_apply_mask: bool,
        cS: cute.Tensor,
        row_max: Float32,
        row_sum: Float32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        scale_softmax_log2: Float32,
        qk_thr_mma: cute.core.ThrMma,
        tiled_tmem_load,
        thr_tmem_load,
        tTMEM_LOADtS: cute.Tensor,
        mma_s_consumer: PipelineConsumer,
    ) -> Tuple[Float32, Float32, PipelineConsumer]:
        """One online-softmax step over one 128x128 S tile (vendor pattern,
        minus the P/vec TMEM write-back: stats stay in registers)."""
        tScS = qk_thr_mma.partition_C(cS)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)

        si_handle = mma_s_consumer.wait_and_advance()
        tTMEM_LOADrS = cute.make_fragment(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)

        frg_cnt = 4
        frg_tile = cute.size(tTMEM_LOADrS) // frg_cnt
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        tTMEM_LOADcS_frg = cute.logical_divide(tTMEM_LOADcS, cute.make_layout(frg_tile))

        # Causal + boundary mask (inlined fusion/mask.apply_mask): cS carries
        # global coordinates, so key j is visible to query row i iff j <= i
        # and j < seqlen_k.
        if need_apply_mask:
            for j in cutlass.range_constexpr(frg_cnt):
                for k in cutlass.range_constexpr(cute.size(tTMEM_LOADrS_frg, mode=[0])):
                    qo_idx, kv_idx = tTMEM_LOADcS_frg[k, j]
                    if kv_idx > qo_idx or kv_idx >= seqlen_k:
                        tTMEM_LOADrS_frg[k, j] = -cutlass.Float32.inf

        for j in cutlass.range_constexpr(frg_cnt):
            score_vec = tTMEM_LOADrS_frg[None, j].load()
            tTMEM_LOADrS_frg[None, j].store(
                score_vec.to(self.q_dtype).to(self.qk_acc_dtype)
            )

        old_row_max = row_max
        row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        row_max_safe = row_max
        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0

        # P = exp2((S - m) * scale), in registers only.
        for j in cutlass.range_constexpr(frg_cnt):
            _exp2_scale(tTMEM_LOADrS_frg[None, j], scale_softmax_log2, row_max_safe)

        # S is free for the next tile's MMA once every thread's TMEM load has
        # landed in registers.
        si_handle.release()

        # di = di-1 * exp2((mi-1 - mi) * scale) + sum_j exp2((xi - mi) * scale)
        acc_scale_ = scale_softmax_log2 * (old_row_max - row_max_safe)
        # * 0.5 compensates for seeding both packed elements with row_sum below
        acc_scale = cute.arch.exp2(acc_scale_) * 0.5
        row_sum *= acc_scale
        # 4-way unrolled packed reduction (vendor SoftmaxRole.step verbatim).
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)

        reduction_unroll = 4
        frg_tile_r = cute.size(tTMEM_LOADrS) // reduction_unroll
        tTMEM_LOADrS_frg_r = cute.logical_divide(
            tTMEM_LOADrS, cute.make_layout(frg_tile_r)
        )

        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg_r, mode=[0]), 2):
            local_row_sum_0 = cute.arch.add_packed_f32x2(
                local_row_sum_0,
                (tTMEM_LOADrS_frg_r[j, 0], tTMEM_LOADrS_frg_r[j + 1, 0]),
            )
            local_row_sum_1 = cute.arch.add_packed_f32x2(
                local_row_sum_1,
                (tTMEM_LOADrS_frg_r[j, 1], tTMEM_LOADrS_frg_r[j + 1, 1]),
            )
            local_row_sum_2 = cute.arch.add_packed_f32x2(
                local_row_sum_2,
                (tTMEM_LOADrS_frg_r[j, 2], tTMEM_LOADrS_frg_r[j + 1, 2]),
            )
            local_row_sum_3 = cute.arch.add_packed_f32x2(
                local_row_sum_3,
                (tTMEM_LOADrS_frg_r[j, 3], tTMEM_LOADrS_frg_r[j + 1, 3]),
            )

        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]

        return (row_max, row_sum, mma_s_consumer)

    @cute.jit
    def _softmax_run(
        self,
        seqlen_q: Int32,
        seqlen_k: Int32,
        scale_softmax_log2: Float32,
        mStats: cute.Tensor,
        qk_thr_mma: cute.core.ThrMma,
        tStS0: cute.Tensor,
        mma_s_consumer: PipelineConsumer,
        q_tile_coord: Int32,
        head_coord: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * self.num_softmax_warps)

        cS_base = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS0)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_tmem_load.partition_S(tStS0)

        # Causal single-sequence trip counts (inlined fusion/mask helpers):
        # KV tiles strictly below the diagonal tile are fully unmasked; the
        # diagonal tile j == q_tile needs masking.
        unmask_count = q_tile_coord
        trip_count = q_tile_coord + 1

        row_max = -Float32.inf
        row_sum = 0.0

        cS_unmasked = cute.domain_offset((q_tile_coord * self.tile_m, 0), cS_base)
        for i in cutlass.range(0, unmask_count, 1, unroll=1):
            cS_iter = cute.domain_offset((0, i * self.tile_n), cS_unmasked)
            (
                row_max,
                row_sum,
                mma_s_consumer,
            ) = self._softmax_step(
                False,
                cS_iter,
                row_max,
                row_sum,
                seqlen_q,
                seqlen_k,
                scale_softmax_log2,
                qk_thr_mma,
                tiled_tmem_load,
                thr_tmem_load,
                tTMEM_LOADtS,
                mma_s_consumer,
            )

        cS_diag = cute.domain_offset(
            (q_tile_coord * self.tile_m, unmask_count * self.tile_n), cS_base
        )
        (
            row_max,
            row_sum,
            mma_s_consumer,
        ) = self._softmax_step(
            True,
            cS_diag,
            row_max,
            row_sum,
            seqlen_q,
            seqlen_k,
            scale_softmax_log2,
            qk_thr_mma,
            tiled_tmem_load,
            thr_tmem_load,
            tTMEM_LOADtS,
            mma_s_consumer,
        )

        # Each softmax thread owns exactly one row of the 128-row tile (the
        # Ld32x32b tmem copy maps 4 warps x 32 lanes onto 128 TMEM rows).
        # Recover the thread's global row from the identity-coordinate
        # fragment and store (m, l).
        cS_row = cute.domain_offset((q_tile_coord * self.tile_m, 0), cS_base)
        tScS_row = qk_thr_mma.partition_C(cS_row)
        tTMEM_LOADcS_row = thr_tmem_load.partition_D(tScS_row)
        qo_idx, _ = tTMEM_LOADcS_row[0]
        if qo_idx < seqlen_q:
            mStats[qo_idx, head_coord, 0] = row_max
            mStats[qo_idx, head_coord, 1] = row_sum

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        stats_in: cute.Tensor,
        problem_size: Tuple[Int32, Int32, Int32, Int32],
        scale_softmax_log2: Float32,
        stream,
    ):
        """Compute causal GQA softmax statistics.

        :param q_in: queries, ``[T, Hq, 512]`` row-major (NHD)
        :param k_in: keys, ``[T, Hkv, 512]`` row-major (NHD)
        :param stats_in: output ``[T, Hq, 2]`` FP32; ``[..., 0]`` = row max,
            ``[..., 1]`` = sum(exp((S - m) * sm_scale)) over unmasked keys
        :param problem_size: ``(T, Hq, Hkv, D)``
        :param scale_softmax_log2: ``log2(e) * sm_scale``
        :param stream: CUDA stream
        """
        s_q, h_q, h_k, d = problem_size
        h_r = h_q // h_k

        # NOTE: head_dim must be 512 (four 128-wide D chunks are hardcoded in
        # the loader); the wrapper validates this before compile/call.  Like
        # the vendor kernel, all of problem_size stays dynamic Int32 so
        # sequence length and head counts are runtime parameters.

        # (s, d, ((h_r, h_k), b=1)) — K-major (d contiguous), GQA broadcast
        # on the K side via a 0-stride h_r mode (vendor prefill layouts).
        q_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), 1)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), 0)),
        )
        q = cute.make_tensor(q_in.iterator, q_layout)
        k_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), 1)),
            stride=(d * h_k, 1, ((0, d), 0)),
        )
        k = cute.make_tensor(k_in.iterator, k_layout)
        stats_layout = cute.make_layout((s_q, h_q, 2), stride=(h_q * 2, 2, 1))
        stats = cute.make_tensor(stats_in.iterator, stats_layout)

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: q={self.q_dtype} != k={self.k_dtype}")

        grid = (cute.ceil_div(s_q, self.tile_m), h_q, 1)

        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.qk_chunk_tiler[:2],
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (qk_tiled_mma.thr_id.shape,)
        )

        # SMEM: Q resident as 4 chunk stages (131,072 B) + K double-buffered
        # (65,536 B) = 196,608 B of the 232,448 B SM100 budget.
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma, self.qk_chunk_tiler, self.q_dtype, self.q_stages
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, self.qk_chunk_tiler, self.k_dtype, self.kv_stages
        )

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_chunk_tiler,
            qk_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_chunk_tiler,
            qk_tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_copy_chunk_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)

        @cute.struct
        class SharedStorage:
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stages * 2]
            load_k_mbar_ptr: cute.struct.MemRange[Int64, self.kv_stages * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, 2]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
            tmem_holding_buf: cutlass.Int32
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                1024,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                1024,
            ]

            @classmethod
            def size_in_bytes(cls) -> int: ...  # noqa: F811

        smem_bytes = SharedStorage.size_in_bytes()
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        if cutlass.const_expr(smem_bytes > smem_capacity):
            raise ValueError(
                f"SharedStorage requires {smem_bytes} bytes but SM100 provides "
                f"{smem_capacity} bytes."
            )

        # Kernel-time state, referenced from self.kernel (prefill.py:232,244 pattern)
        self.shared_storage = SharedStorage
        self.tma_copy_chunk_bytes = tma_copy_chunk_bytes

        self.kernel(
            qk_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            stats,
            scale_softmax_log2,
            q_smem_layout_staged,
            k_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            smem=smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        mStats: cute.Tensor,
        scale_softmax_log2: Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
    ):
        """Device kernel: warp-specialized split-D attention statistics."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        q_tile_coord = bidx
        head_coord = bidy
        trip_count = q_tile_coord + 1

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            num_stages=self.q_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_chunk_bytes,
            defer_sync=True,
        )
        load_q_producer, load_q_consumer = load_q_pipe.make_participants()
        load_k_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            num_stages=self.kv_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_chunk_bytes,
            defer_sync=True,
        )
        load_k_producer, load_k_consumer = load_k_pipe.make_participants()
        mma_s_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(
                Agent.Thread, self.threads_per_warp * self.num_softmax_warps
            ),
            defer_sync=True,
        )
        mma_s_producer, mma_s_consumer = mma_s_pipe.make_participants()

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        if warp_idx == self.empty_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                self.tmem_dealloc_arrive_count,
            )
        cute.arch.mbarrier_init_fence()

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )

        qk_thr_mma = qk_tiled_mma.get_slice(0)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tStS = qk_thr_mma.make_fragment_C(
            qk_thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        )
        tStS0 = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)

        cute.arch.barrier(
            barrier_id=self.cta_sync_bar_id,
            number_of_threads=self.threads_per_cta,
        )

        # ///////////////////////////////////////////////////////////////////
        #  EMPTY + IDLE
        # ///////////////////////////////////////////////////////////////////
        if warp_idx == self.empty_warp_id or warp_idx == self.idle_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        # ///////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self._loader_run(
                qk_thr_mma,
                tma_atom_q,
                mQ_qdl,
                tma_atom_k,
                mK_kdl,
                sQ,
                sK,
                load_q_producer,
                load_k_producer,
                q_tile_coord,
                head_coord,
                trip_count,
            )

        # ///////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self._mma_run(
                tStS0,
                tSrQ,
                tSrK,
                load_q_consumer,
                load_k_consumer,
                mma_s_producer,
                trip_count,
                storage,
                tmem_dealloc_mbar_ptr,
            )

        # ///////////////////////////////////////////////////////////////////
        #  SOFTMAX (warps 0-3, one warpgroup)
        # ///////////////////////////////////////////////////////////////////
        if warp_idx < self.num_softmax_warps:
            cute.arch.warpgroup_reg_alloc(self.num_regs_softmax)
            self._softmax_run(
                mQ_qdl.shape[0],
                mK_kdl.shape[0],
                scale_softmax_log2,
                mStats,
                qk_thr_mma,
                tStS0,
                mma_s_consumer,
                q_tile_coord,
                head_coord,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
        return


class Gemma4SplitDApplyKernel:
    """Phase-2 fixed-statistics apply kernel for BF16 D512 attention.

    The grid's Z dimension selects one of two 256-wide V/O halves.  Both halves
    recompute QK but consume the same Phase-1 ``(m, l)`` tensor, so there is no
    per-half max/sum reduction and no per-half softmax trajectory.

    Warp layout (8 warps, 256 threads)::

        warps 0-3  fixed-stat P formation + normalized output epilogue
        warp  4    QK and PV tcgen05 MMA issue
        warp  5    Q/K/V TMA loads
        warp  6    output TMA store
        warp  7    empty / barrier initialization

    SMEM keeps four Q chunks resident (128 KiB), streams one K chunk (32 KiB),
    and stages one V half (64 KiB).  Once all QK work is complete, the 64 KiB
    output tile aliases the no-longer-needed Q storage.
    """

    def __init__(self):
        self.tile_m = 128
        self.tile_n = 128
        self.d_chunk = 128
        self.num_d_chunks = 4
        self.head_dim = self.d_chunk * self.num_d_chunks
        self.output_half_dim = 256
        self.num_output_halves = 2

        self.qk_chunk_tiler = (self.tile_m, self.tile_n, self.d_chunk)
        self.pv_mma_tiler = (self.tile_m, self.output_half_dim, self.tile_n)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32

        self.q_stages = self.num_d_chunks
        self.k_stages = 1
        self.v_stages = 1

        self.num_apply_warps = 4
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.epilogue_warp_id = 6
        self.empty_warp_id = 7
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * 8
        self.num_regs_apply = 192
        self.num_regs_other = 32
        self.num_regs_empty = 24
        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.tmem_dealloc_arrive_count = self.threads_per_warp * self.num_apply_warps

        # S occupies FP32 TMEM columns [0, 128).  Its BF16 P alias starts at
        # offset 32 (64 columns after pointer recasting), while the 256-column
        # FP32 O accumulator occupies [128, 384).
        self.tmem_s_offset = 0
        self.tmem_p_offset = 32
        self.tmem_o_offset = 128
        self.tmem_alloc_cols = 512

        self.q_dtype = None
        self.k_dtype = None
        self.v_dtype = None
        self.o_dtype = None
        self.o_layout = None
        self.epi_tile = self.pv_mma_tiler[:2]

    @cute.jit
    def _make_local_qk_mma(self) -> cute.TiledMma:
        return sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.qk_chunk_tiler[:2],
        )

    @cute.jit
    def _make_local_pv_mma(self) -> cute.TiledMma:
        return sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.pv_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.pv_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

    @cute.jit
    def _gemm_qk(
        self,
        tStS: cute.Tensor,
        tSrQ_slice: cute.Tensor,
        tSrK_slice: cute.Tensor,
        accumulate: bool,
    ):
        local_mma = self._make_local_qk_mma()
        num_kphases = cute.size(tSrQ_slice, mode=[2])
        for kphase_idx in cutlass.range_constexpr(num_kphases):
            coord = (None, None, kphase_idx)
            local_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0 or accumulate)
            cute.gemm(local_mma, tStS, tSrQ_slice[coord], tSrK_slice[coord], tStS)

    @cute.jit
    def _gemm_pv(
        self,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        tOrV_slice: cute.Tensor,
        accumulate: bool,
    ):
        local_mma = self._make_local_pv_mma()
        num_kphases = cute.size(tOrP, mode=[2])
        for kphase_idx in cutlass.range_constexpr(num_kphases):
            coord = (None, None, kphase_idx)
            local_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0 or accumulate)
            cute.gemm(local_mma, tOtO, tOrP[coord], tOrV_slice[coord], tOtO)

    @cute.jit
    def _alloc_tmem(self, storage: cute.Tensor):
        tmem_alloc_cols = Int32(self.tmem_alloc_cols)
        cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
        cute.arch.barrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            number_of_threads=self.threads_per_warp,
        )

    @cute.jit
    def _dealloc_tmem(self, storage: cute.Tensor, tmem_dealloc_mbar_ptr: Int32):
        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
        tmem_alloc_cols = Int32(self.tmem_alloc_cols)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32,
            alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf,
        )
        cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

    @cute.jit
    def _loader_run(
        self,
        qk_thr_mma: cute.core.ThrMma,
        pv_thr_mma: cute.core.ThrMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dklh: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        load_q_producer: PipelineProducer,
        load_k_producer: PipelineProducer,
        load_v_producer: PipelineProducer,
        q_tile_coord: Int32,
        head_coord: Int32,
        half_coord: Int32,
        trip_count: Int32,
    ):
        gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_chunk_tiler, mode=[0, 2]))
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )
        for dk in cutlass.range_constexpr(self.num_d_chunks):
            tQgQ = tQgQ_qdl[None, None, dk, (head_coord, 0)]
            q_handle = load_q_producer.acquire_and_advance()
            cute.copy(
                tma_atom_q,
                tQgQ[None, q_tile_coord],
                tQsQ[None, q_handle.index],
                tma_bar_ptr=q_handle.barrier,
            )

        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_chunk_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK_kdl, 0, 3),
        )

        gV_dklh = cute.flat_divide(mV_dklh, cute.select(self.pv_mma_tiler, mode=[1, 2]))
        tSgV_dklh = pv_thr_mma.partition_B(gV_dklh)
        tVsV, tVgV_dklh = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tSgV_dklh, 0, 3),
        )
        tVgV = tVgV_dklh[None, 0, None, (head_coord, half_coord)]

        kv_coord = Int32(0)
        for _i in cutlass.range(0, trip_count, 1, unroll=1):
            for dk in cutlass.range_constexpr(self.num_d_chunks):
                tKgK = tKgK_kdl[None, None, dk, (head_coord, 0)]
                k_handle = load_k_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_k,
                    tKgK[None, kv_coord],
                    tKsK[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )

            v_handle = load_v_producer.acquire_and_advance()
            cute.copy(
                tma_atom_v,
                tVgV[None, kv_coord],
                tVsV[None, v_handle.index],
                tma_bar_ptr=v_handle.barrier,
            )
            kv_coord += 1

    @cute.jit
    def _mma_run(
        self,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tOrP: cute.Tensor,
        tOrV: cute.Tensor,
        load_q_consumer: PipelineConsumer,
        load_k_consumer: PipelineConsumer,
        load_v_consumer: PipelineConsumer,
        mma_s_producer: PipelineProducer,
        p_mma_consumer: PipelineConsumer,
        mma_o_producer: PipelineProducer,
        trip_count: Int32,
        storage: cute.Tensor,
        tmem_dealloc_mbar_ptr: Int32,
    ):
        self._alloc_tmem(storage)

        for _dk in cutlass.range_constexpr(self.num_d_chunks):
            load_q_consumer.wait_and_advance()

        # First KV tile initializes O.  Keep it outside the runtime loop so the
        # tcgen05 ACCUMULATE field remains a compile-time constant.
        s_handle = mma_s_producer.acquire_and_advance()
        for dk in cutlass.range_constexpr(self.num_d_chunks):
            k_handle = load_k_consumer.wait_and_advance()
            self._gemm_qk(
                tStS,
                tSrQ[None, None, None, dk],
                tSrK[None, None, None, k_handle.index],
                dk != 0,
            )
            k_handle.release()
        s_handle.commit()
        p_handle = p_mma_consumer.wait_and_advance()
        v_handle = load_v_consumer.wait_and_advance()
        self._gemm_pv(
            tOtO,
            tOrP,
            tOrV[None, None, None, v_handle.index],
            False,
        )
        p_handle.release()
        v_handle.release()

        for _i in cutlass.range(1, trip_count, 1, unroll=1):
            s_handle = mma_s_producer.acquire_and_advance()
            for dk in cutlass.range_constexpr(self.num_d_chunks):
                k_handle = load_k_consumer.wait_and_advance()
                self._gemm_qk(
                    tStS,
                    tSrQ[None, None, None, dk],
                    tSrK[None, None, None, k_handle.index],
                    dk != 0,
                )
                k_handle.release()
            s_handle.commit()

            p_handle = p_mma_consumer.wait_and_advance()
            v_handle = load_v_consumer.wait_and_advance()
            self._gemm_pv(
                tOtO,
                tOrP,
                tOrV[None, None, None, v_handle.index],
                True,
            )
            p_handle.release()
            v_handle.release()

        o_handle = mma_o_producer.acquire_and_advance()
        o_handle.commit()
        self._dealloc_tmem(storage, tmem_dealloc_mbar_ptr)

    @cute.jit
    def _fixed_stats_step(
        self,
        need_apply_mask: bool,
        cS: cute.Tensor,
        row_max: Float32,
        row_sum: Float32,
        seqlen_k: Int32,
        scale_softmax_log2: Float32,
        qk_thr_mma: cute.core.ThrMma,
        tiled_tmem_load,
        thr_tmem_load,
        tTMEM_LOADtS: cute.Tensor,
        tiled_tmem_store,
        tTMEM_STOREtP: cute.Tensor,
        tTMEM_STOREcP: cute.Tensor,
        mma_s_consumer: PipelineConsumer,
        p_mma_producer: PipelineProducer,
    ) -> Tuple[PipelineConsumer, PipelineProducer]:
        tScS = qk_thr_mma.partition_C(cS)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)

        si_handle = mma_s_consumer.wait_and_advance()
        tTMEM_LOADrS = cute.make_fragment(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)

        frg_cnt = 4
        frg_tile = cute.size(tTMEM_LOADrS) // frg_cnt
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        tTMEM_LOADcS_frg = cute.logical_divide(tTMEM_LOADcS, cute.make_layout(frg_tile))
        if need_apply_mask:
            for j in cutlass.range_constexpr(frg_cnt):
                for k in cutlass.range_constexpr(cute.size(tTMEM_LOADrS_frg, mode=[0])):
                    qo_idx, kv_idx = tTMEM_LOADcS_frg[k, j]
                    if kv_idx > qo_idx or kv_idx >= seqlen_k:
                        tTMEM_LOADrS_frg[k, j] = -cutlass.Float32.inf

        for j in cutlass.range_constexpr(frg_cnt):
            score_vec = tTMEM_LOADrS_frg[None, j].load()
            tTMEM_LOADrS_frg[None, j].store(
                score_vec.to(self.q_dtype).to(self.qk_acc_dtype)
            )

        # Match eager softmax's quantization boundary: normalize in FP32, then
        # cast the normalized probabilities to BF16 before PV.
        tTMEM_STORErP = cute.make_fragment(tTMEM_STOREcP.shape, self.qk_acc_dtype)
        tTMEM_STORErP_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErP.iterator, dtype=self.q_dtype),
            tTMEM_LOADrS.layout,
        )
        tTMEM_STORErP_e_frg = cute.logical_divide(
            tTMEM_STORErP_e, cute.make_layout(frg_tile)
        )
        inv_l = 1.0 / row_sum
        for j in cutlass.range_constexpr(frg_cnt):
            _exp2_scale(tTMEM_LOADrS_frg[None, j], scale_softmax_log2, row_max)
            for k in cutlass.range_constexpr(cute.size(tTMEM_LOADrS_frg, mode=[0])):
                tTMEM_LOADrS_frg[k, j] *= inv_l
            p_vec = tTMEM_LOADrS_frg[None, j].load()
            tTMEM_STORErP_e_frg[None, j].store(p_vec.to(self.q_dtype))

        p_handle = p_mma_producer.acquire_and_advance()
        cute.copy(tiled_tmem_store, tTMEM_STORErP, tTMEM_STOREtP)
        cute.arch.fence_view_async_tmem_store()
        si_handle.release()
        p_handle.commit()
        return (mma_s_consumer, p_mma_producer)

    @cute.jit
    def _output_to_smem(
        self,
        pv_thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        sO: cute.Tensor,
    ):
        cO = cute.make_identity_tensor(self.pv_mma_tiler[:2])
        tOsO = pv_thr_mma.partition_C(sO)
        tOcO = pv_thr_mma.partition_C(cO)

        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOtO_i = cute.logical_divide(
            tOtO, cute.make_layout((self.tile_m, corr_tile_size))
        )
        tOcO_i = cute.logical_divide(
            tOcO, cute.make_layout((self.tile_m, corr_tile_size))
        )
        tOsO_i = cute.logical_divide(
            tOsO, cute.make_layout((self.tile_m, corr_tile_size))
        )

        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * self.num_apply_warps)
        epi_subtile = (self.tile_m, corr_tile_size)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            self.pv_mma_tiler,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO_i[(None, None), 0]
        )
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i[(None, None), None])

        for i in cutlass.range_constexpr(self.output_half_dim // corr_tile_size):
            tTMrO = cute.make_fragment(
                tTMEM_LOADcO[None, 0, 0, i].shape, self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO[None, 0, 0, i], tTMrO)
            tSMrO = cute.make_fragment(tTMrO.shape, self.o_dtype)
            tSMrO.store(tTMrO.load().to(self.o_dtype))
            cute.copy(
                tiled_smem_store,
                tSMrO,
                tTMEM_LOADsO[None, 0, 0, i],
            )
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def _apply_run(
        self,
        seqlen_q: Int32,
        seqlen_k: Int32,
        scale_softmax_log2: Float32,
        mStats: cute.Tensor,
        qk_thr_mma: cute.core.ThrMma,
        pv_thr_mma: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sO: cute.Tensor,
        mma_s_consumer: PipelineConsumer,
        p_mma_producer: PipelineProducer,
        mma_o_consumer: PipelineConsumer,
        output_producer: PipelineProducer,
        q_tile_coord: Int32,
        head_coord: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * self.num_apply_warps)

        cS_base = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS = qk_thr_mma.partition_C(cS_base)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_tmem_load.partition_S(tStS)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)

        tile_p_like_fp32 = self.tile_n // Float32.width * self.q_dtype.width
        tStS_P_layout = cute.composition(
            tStS.layout, cute.make_layout((self.tile_m, tile_p_like_fp32))
        )
        tStS_P = cute.make_tensor(tStS.iterator + self.tmem_p_offset, tStS_P_layout)
        tScS_P_layout = cute.composition(
            tScS.layout, cute.make_layout((self.tile_m, tile_p_like_fp32))
        )
        tScS_P = cute.make_tensor(tScS.iterator, tScS_P_layout)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        tTMEM_STOREtP = thr_tmem_store.partition_D(tStS_P)
        tTMEM_STOREcP = thr_tmem_store.partition_S(tScS_P)

        cS_row = cute.domain_offset((q_tile_coord * self.tile_m, 0), cS_base)
        tScS_row = qk_thr_mma.partition_C(cS_row)
        tTMEM_LOADcS_row = thr_tmem_load.partition_D(tScS_row)
        qo_idx, _ = tTMEM_LOADcS_row[0]
        row_max = 0.0
        row_sum = 1.0
        if qo_idx < seqlen_q:
            row_max = mStats[qo_idx, head_coord, 0]
            row_sum = mStats[qo_idx, head_coord, 1]

        unmask_count = q_tile_coord
        cS_unmasked = cute.domain_offset((q_tile_coord * self.tile_m, 0), cS_base)
        for i in cutlass.range(0, unmask_count, 1, unroll=1):
            cS_iter = cute.domain_offset((0, i * self.tile_n), cS_unmasked)
            mma_s_consumer, p_mma_producer = self._fixed_stats_step(
                False,
                cS_iter,
                row_max,
                row_sum,
                seqlen_k,
                scale_softmax_log2,
                qk_thr_mma,
                tiled_tmem_load,
                thr_tmem_load,
                tTMEM_LOADtS,
                tiled_tmem_store,
                tTMEM_STOREtP,
                tTMEM_STOREcP,
                mma_s_consumer,
                p_mma_producer,
            )

        cS_diag = cute.domain_offset(
            (q_tile_coord * self.tile_m, unmask_count * self.tile_n), cS_base
        )
        mma_s_consumer, p_mma_producer = self._fixed_stats_step(
            True,
            cS_diag,
            row_max,
            row_sum,
            seqlen_k,
            scale_softmax_log2,
            qk_thr_mma,
            tiled_tmem_load,
            thr_tmem_load,
            tTMEM_LOADtS,
            tiled_tmem_store,
            tTMEM_STOREtP,
            tTMEM_STOREcP,
            mma_s_consumer,
            p_mma_producer,
        )

        o_handle = mma_o_consumer.wait_and_advance()
        output_handle = output_producer.acquire_and_advance()
        self._output_to_smem(pv_thr_mma, tOtO, sO)
        o_handle.release()
        output_handle.commit()

    @cute.jit
    def _output_store_run(
        self,
        tma_atom_o: cute.CopyAtom,
        mO_qdlh: cute.Tensor,
        sO: cute.Tensor,
        output_consumer: PipelineConsumer,
        q_tile_coord: Int32,
        head_coord: Int32,
        half_coord: Int32,
    ):
        gO_qdlh = cute.flat_divide(mO_qdlh, cute.select(self.pv_mma_tiler, mode=[0, 1]))
        gO = gO_qdlh[None, None, None, 0, (head_coord, half_coord)]
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_o,
            0,
            cute.make_layout(1),
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )

        output_handle = output_consumer.wait_and_advance()
        cute.copy(tma_atom_o, tOsO[None, 0], tOgO[None, q_tile_coord])
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        output_handle.release()

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        v_in: cute.Tensor,
        stats_in: cute.Tensor,
        o_in: cute.Tensor,
        problem_size: Tuple[Int32, Int32, Int32, Int32],
        scale_softmax_log2: Float32,
        stream,
    ):
        """Apply fixed Phase-1 statistics and write BF16 ``[T,Hq,512]``."""
        s_q, h_q, h_k, d = problem_size
        h_r = h_q // h_k

        q_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), 1)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), 0)),
        )
        q = cute.make_tensor(q_in.iterator, q_layout)
        k_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), 1)),
            stride=(d * h_k, 1, ((0, d), 0)),
        )
        k = cute.make_tensor(k_in.iterator, k_layout)
        v_layout = cute.make_layout(
            (self.output_half_dim, s_q, ((h_r, h_k), self.num_output_halves)),
            stride=(
                1,
                d * h_k,
                ((0, d), self.output_half_dim),
            ),
        )
        v = cute.make_tensor(v_in.iterator, v_layout)
        stats_layout = cute.make_layout((s_q, h_q, 2), stride=(h_q * 2, 2, 1))
        stats = cute.make_tensor(stats_in.iterator, stats_layout)
        o_layout = cute.make_layout(
            (s_q, self.output_half_dim, ((h_r, h_k), self.num_output_halves)),
            stride=(
                d * h_q,
                1,
                ((d, d * h_r), self.output_half_dim),
            ),
        )
        o = cute.make_tensor(o_in.iterator, o_layout)

        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type
        self.o_layout = utils.LayoutEnum.from_tensor(o)
        if cutlass.const_expr(
            self.q_dtype != self.k_dtype
            or self.q_dtype != self.v_dtype
            or self.q_dtype != self.o_dtype
            or self.q_dtype != cutlass.BFloat16
        ):
            raise TypeError("Phase-2 split-D apply requires matching BF16 Q/K/V/O")

        grid = (
            cute.ceil_div(s_q, self.tile_m),
            h_q,
            self.num_output_halves,
        )
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.qk_chunk_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            self.pv_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.pv_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )
        qk_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (qk_tiled_mma.thr_id.shape,)
        )
        pv_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (pv_tiled_mma.thr_id.shape,)
        )

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma, self.qk_chunk_tiler, self.q_dtype, self.q_stages
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma, self.qk_chunk_tiler, self.k_dtype, self.k_stages
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma, self.pv_mma_tiler, self.v_dtype, self.v_stages
        )
        p_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma, self.pv_mma_tiler, self.q_dtype, 1
        )
        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, 1
        )

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_chunk_tiler,
            qk_tiled_mma,
            qk_cluster_layout_vmnk.shape,
        )
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_chunk_tiler,
            qk_tiled_mma,
            qk_cluster_layout_vmnk.shape,
        )
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.pv_mma_tiler,
            pv_tiled_mma,
            pv_cluster_layout_vmnk.shape,
        )
        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            o,
            o_smem_layout,
            self.epi_tile,
        )

        tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        tma_copy_k_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        tma_copy_v_bytes = cute.size_in_bytes(self.v_dtype, v_smem_layout)

        @cute.struct
        class SharedStorage:
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stages * 2]
            load_k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stages * 2]
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stages * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, 2]
            p_mma_mbar_ptr: cute.struct.MemRange[Int64, 2]
            mma_o_mbar_ptr: cute.struct.MemRange[Int64, 2]
            output_mbar_ptr: cute.struct.MemRange[Int64, 2]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[Int64, 1]
            tmem_holding_buf: cutlass.Int32
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                1024,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                1024,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(v_smem_layout_staged)],
                1024,
            ]

            @classmethod
            def size_in_bytes(cls) -> int: ...  # noqa: F811

        smem_bytes = SharedStorage.size_in_bytes()
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        if cutlass.const_expr(smem_bytes > smem_capacity):
            raise ValueError(
                f"Phase-2 SharedStorage requires {smem_bytes} bytes but SM100 "
                f"provides {smem_capacity} bytes."
            )

        self.shared_storage = SharedStorage
        self.tma_copy_q_bytes = tma_copy_q_bytes
        self.tma_copy_k_bytes = tma_copy_k_bytes
        self.tma_copy_v_bytes = tma_copy_v_bytes

        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            stats,
            scale_softmax_log2,
            q_smem_layout_staged,
            k_smem_layout_staged,
            v_smem_layout_staged,
            p_tmem_layout_staged,
            o_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            smem=smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dklh: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_qdlh: cute.Tensor,
        mStats: cute.Tensor,
        scale_softmax_log2: Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        q_tile_coord = bidx
        head_coord = bidy
        half_coord = bidz
        trip_count = q_tile_coord + 1

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
        if warp_idx == self.epilogue_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            num_stages=self.q_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_q_bytes,
            defer_sync=True,
        )
        load_q_producer, load_q_consumer = load_q_pipe.make_participants()
        load_k_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            num_stages=self.k_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_k_bytes,
            defer_sync=True,
        )
        load_k_producer, load_k_consumer = load_k_pipe.make_participants()
        load_v_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            num_stages=self.v_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_v_bytes,
            defer_sync=True,
        )
        load_v_producer, load_v_consumer = load_v_pipe.make_participants()
        mma_s_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(
                Agent.Thread, self.threads_per_warp * self.num_apply_warps
            ),
            defer_sync=True,
        )
        mma_s_producer, mma_s_consumer = mma_s_pipe.make_participants()
        p_mma_pipe = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.p_mma_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(
                Agent.Thread, self.threads_per_warp * self.num_apply_warps
            ),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            defer_sync=True,
        )
        p_mma_producer, p_mma_consumer = p_mma_pipe.make_participants()
        mma_o_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mma_o_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(
                Agent.Thread, self.threads_per_warp * self.num_apply_warps
            ),
            defer_sync=True,
        )
        mma_o_producer, mma_o_consumer = mma_o_pipe.make_participants()
        output_pipe = pipeline.PipelineAsync.create(
            barrier_storage=storage.output_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(
                Agent.Thread, self.threads_per_warp * self.num_apply_warps
            ),
            consumer_group=CooperativeGroup(Agent.Thread, self.threads_per_warp),
            defer_sync=True,
        )
        output_producer, output_consumer = output_pipe.make_participants()

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        if warp_idx == self.empty_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                self.tmem_dealloc_arrive_count,
            )
        cute.arch.mbarrier_init_fence()

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        sO = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, o_smem_layout_staged.inner),
            o_smem_layout_staged.outer,
        )

        qk_thr_mma = qk_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)
        tStS_base = qk_thr_mma.make_fragment_C(
            qk_thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        )
        tStS = cute.make_tensor(
            tStS_base.iterator + self.tmem_s_offset, tStS_base.layout
        )
        tOtO_base = pv_thr_mma.make_fragment_C(
            pv_thr_mma.partition_shape_C(self.pv_mma_tiler[:2])
        )
        tOtO = cute.make_tensor(
            tOtO_base.iterator + self.tmem_o_offset, tOtO_base.layout
        )
        tP = cute.make_tensor(tStS_base.iterator, p_tmem_layout_staged.outer)
        tOrP_base = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
        p_scale = self.qk_acc_dtype.width // self.q_dtype.width
        tOrP = cute.make_tensor(
            tOrP_base.iterator + p_scale * self.tmem_p_offset,
            tOrP_base.layout,
        )

        cute.arch.barrier(
            barrier_id=self.cta_sync_bar_id,
            number_of_threads=self.threads_per_cta,
        )

        if warp_idx == self.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self._loader_run(
                qk_thr_mma,
                pv_thr_mma,
                tma_atom_q,
                mQ_qdl,
                tma_atom_k,
                mK_kdl,
                tma_atom_v,
                mV_dklh,
                sQ,
                sK,
                sV,
                load_q_producer,
                load_k_producer,
                load_v_producer,
                q_tile_coord,
                head_coord,
                half_coord,
                trip_count,
            )

        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self._mma_run(
                tStS,
                tOtO,
                tSrQ,
                tSrK,
                tOrP,
                tOrV,
                load_q_consumer,
                load_k_consumer,
                load_v_consumer,
                mma_s_producer,
                p_mma_consumer,
                mma_o_producer,
                trip_count,
                storage,
                tmem_dealloc_mbar_ptr,
            )

        if warp_idx == self.epilogue_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self._output_store_run(
                tma_atom_o,
                mO_qdlh,
                sO,
                output_consumer,
                q_tile_coord,
                head_coord,
                half_coord,
            )

        if warp_idx < self.num_apply_warps:
            cute.arch.warpgroup_reg_alloc(self.num_regs_apply)
            self._apply_run(
                mQ_qdl.shape[0],
                mK_kdl.shape[0],
                scale_softmax_log2,
                mStats,
                qk_thr_mma,
                pv_thr_mma,
                tStS,
                tOtO,
                sO[None, None, 0],
                mma_s_consumer,
                p_mma_producer,
                mma_o_consumer,
                output_producer,
                q_tile_coord,
                head_coord,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
        return
