# Copyright 2025-2026 Ant Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Head-window experiment derived from cuLA dev88 dcce4719 pre_scan.py.
Read a head window directly from compact full-head tensors, avoiding packing.

Single fused CuTeDSL kernel with grid-level dispatch:
  blockIdx.x < num_v_tiles → he mode: computes he [K, V] = exit h-state
  blockIdx.x >= num_v_tiles → m mode: computes m [K, K] = transition matrix
  (8-warp SM100 MMA pipeline, identical MMA shapes for both modes)

Output tensor: hm [S_split, H, K, V+K] FP32 or BF16; register state remains FP32
  columns [0:V]   = he  (exit h-state)
  columns [V:V+K] = m   (transition matrix)
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Float32, Int32, Int64

from cula.utils import USE_FAST_MATH, assert_blackwell

PRINT_DEBUG = False

LN2 = 0.6931471805599453
INV_LN2 = 1.4426950408889634


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


# =====================================================================
# Fused CuTeDSL Kernel: he + m with grid-level dispatch
# =====================================================================


class ChunkDeltaRulePreScanFused:
    """
    Fused pre-scan kernel: computes both he (exit h-state) and m (transition matrix).

    Grid-level dispatch: blockIdx.x < num_v_tiles → he mode, else → m mode.
    Both modes share identical MMA structure (BS=BV=64, BT=64, BK=128).
    MMA warp code is unchanged; only CUDA warps have mode-specific branches.

    Grid: (num_v_tiles + num_k_tiles, S_split * H, 1) — non-persistent.
    Each CTA processes one (tile, sub-sequence, head) work unit.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim_k: int = 128,
        head_dim_v: int = 128,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        use_fast_math: bool = True,
        source_heads: int = 0,
        head_offset: int = 0,
        identity_residual: bool = False,
    ):
        assert head_dim_k == 128 and head_dim_v == 128
        assert_blackwell()

        self.use_fast_math = use_fast_math
        self.source_heads = source_heads
        self.head_offset = head_offset
        self.identity_residual = identity_residual
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype

        self.BT = chunk_size  # 64
        self.BK = head_dim_k  # 128
        self.BV = 64  # V tiling fixed at 64
        self.BS = 64  # K tiling for m mode (= BV)

        # Warp assignment (same as fwd_h)
        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.load_warp_id = 5
        self.store_warp_id = 6
        self.empty_warp_id = 7
        self.min_occupancy = 1
        self.num_regs_cuda = 232
        self.num_regs_others = 40
        self.threads_per_cta = self.threads_per_warp * 8

        # MMA tiling (same as fwd_h)
        # WH MMA: state(BV,BK) @ W(BT,BK) → acc(BV,BT)
        self.wh_mma_tiler = (self.BV, self.BT, self.BK)
        # KV MMA: vnew(BV,BT) @ K^T(BK,BT) → update(BV,BK)
        self.kv_mma_tiler = (self.BV, self.BK, self.BT)

        # Pipeline stages (simplified: no h_out, no vnew_store)
        self.k_stage = 3
        self.w_stage = 3
        self.u_stage = 2
        self.gk_stage = 2
        self.acc_stage = 1
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE

        self.buffer_align_bytes = 1024

        # Barrier for TMEM dealloc sync
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )
        # Barrier for CUDA warp-group sync during gk_scale precomputation
        self.gk_precompute_bar = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp * len(self.cuda_warp_ids),  # 128
        )

    @staticmethod
    def _plan_tmem_offsets(tiled_mma_wh, tile_wh, tiled_mma_kv, tile_kv, state_tmem_layout, vnew_tmem_layout, acc_stages):
        """Plan TMEM column allocation. Same as fwd_h."""
        SM100_TMEM_CAPACITY_COLS = 512
        wh_shape = tiled_mma_wh.partition_shape_C(tile_wh[:2])
        wh_fake = tiled_mma_wh.make_fragment_C(cute.append(wh_shape, acc_stages))
        num_wh = tcgen05.find_tmem_tensor_col_offset(wh_fake)

        tCrState_fake = tiled_mma_wh.make_fragment_A(state_tmem_layout.outer.shape)
        num_state = tcgen05.find_tmem_tensor_col_offset(tCrState_fake)

        tCrVnew_fake = tiled_mma_kv.make_fragment_A(vnew_tmem_layout.outer.shape)
        num_vnew = tcgen05.find_tmem_tensor_col_offset(tCrVnew_fake)

        kv_shape = tiled_mma_kv.partition_shape_C(tile_kv[:2])
        kv_fake = tiled_mma_kv.make_fragment_C(cute.append(kv_shape, 1))
        num_kv = tcgen05.find_tmem_tensor_col_offset(kv_fake)

        wh_off = 0
        state_off = wh_off + num_wh
        vnew_off = state_off + num_state
        kv_off = vnew_off + num_vnew
        total_tmp = kv_off + num_kv
        total = 1
        while total < total_tmp:
            total *= 2
        assert total <= SM100_TMEM_CAPACITY_COLS
        return wh_off, state_off, vnew_off, kv_off, total

    def _compute_grid(self, S_split, H, K, V):
        """Grid: (num_v_tiles + num_k_tiles, S_split * H, 1). Non-persistent."""
        num_v_tiles = (V + self.BV - 1) // self.BV
        num_k_tiles = (K + self.BS - 1) // self.BS
        return (num_v_tiles + num_k_tiles, S_split * H, 1)

    def _tma_partition_B(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, hidx):
        """Partition B operand tensors for TMA copy."""
        coord = (0, None, None)
        gX = cute.local_tile(tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, batch_idx)))
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _epilog_partition(self, atom, gC_mnl, epi_tile, sC):
        """Partition for epilogue-style TMA load."""
        gC_epi = cute.flat_divide(gC_mnl, epi_tile)
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            sC_g,
            gC_g,
        )
        return atom, bSG_sC, bSG_gC

    @cute.jit
    def __call__(
        self,
        # ── Input tensors (varlen packed, B=1) ──
        k_in: cute.Tensor,  # [T_total, H, K]  bf16
        w_in: cute.Tensor,  # [T_total, H, K]  bf16
        u_in: cute.Tensor,  # [T_total, H, V]  bf16
        gk_in: cute.Tensor,  # [T_total, H, K]  fp32
        # ── Output tensor ──
        hm_in: cute.Tensor,  # [S_split, H, K, V+K]  fp32  (packed he+m)
        # ── Sequence metadata ──
        cu_seqlens_in: cute.Tensor,  # [S_split+1]  int32
        # ── Scalar parameters ──
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],  # (S_split, T_total, H, K, V)
        use_gk: Int32,  # 1 if gk is provided, 0 otherwise
        num_v_tiles: Int32,  # cdiv(V, BV) — dispatch threshold
        stream,
    ):
        """
        Launch the pre-scan kernel.

        Args:
            k_in:  key tensor, varlen packed [T_total, H, K] bf16
            w_in:  decay weight tensor [T_total, H, K] bf16
            u_in:  value tensor [T_total, H, V] bf16
            gk_in: key gate [T_total, H, K] fp32 (zeros if unused)
            hm_in: output tensor [S_split, H, K, V+K] fp32
                   he written to columns [0:V], m written to columns [V:V+K]
            cu_seqlens_in: cumulative sequence lengths [S_split+1] int32
            problem_size: (S_split, T_total, H, K, V)
            use_gk: flag for gk gating
            num_v_tiles: number of V tiles (dispatch threshold for he vs m)
        """
        k_ptr = k_in.iterator + self.head_offset * self.head_dim_k
        w_ptr = w_in.iterator + self.head_offset * self.head_dim_k
        u_ptr = u_in.iterator + self.head_offset * self.head_dim_v
        gk_ptr = gk_in.iterator + self.head_offset * self.head_dim_k
        hm_ptr = hm_in.iterator
        cu_seqlens_ptr = cu_seqlens_in.iterator

        S_split, T_total, H, K, V = problem_size

        # ===================== GMEM layouts =====================
        # All data tensors are varlen packed [T_total, H, dim]
        # K^T view: (K, T, (H, 1)) with K contiguous — for KV MMA B operand
        source_heads = self.source_heads
        packed_k_batch_stride = Int64(T_total) * source_heads * K
        packed_v_batch_stride = Int64(T_total) * source_heads * V
        kt_layout = cute.make_layout(
            (K, T_total, (H, Int32(1))),
            stride=(1, source_heads * K, (K, packed_k_batch_stride)),
        )
        kt = cute.make_tensor(k_ptr, kt_layout)

        # W view: (T, K, (H, 1)) with K contiguous — for WH MMA B operand
        w_layout = cute.make_layout(
            (T_total, K, (H, Int32(1))),
            stride=(source_heads * K, 1, (K, packed_k_batch_stride)),
        )
        w = cute.make_tensor(w_ptr, w_layout)

        # U transposed view: (V, T, (H, 1)) with V contiguous — for TMA load
        u_T_layout = cute.make_layout(
            (V, T_total, (H, Int32(1))),
            stride=(1, source_heads * V, (V, packed_v_batch_stride)),
        )
        u_T = cute.make_tensor(u_ptr, u_T_layout)

        # U row-major view: (T, V, H) — for address computation in CUDA warps
        u_layout = cute.make_layout(
            (T_total, V, H),
            stride=(Int64(source_heads) * V, 1, V),
        )
        u = cute.make_tensor(u_ptr, u_layout)

        # gk K-first view: (K, T_gk, (H, 1)) with K contiguous — for TMA load
        # T_gk = 1 when gk is unused (dummy 1-row tensor), T_total otherwise
        T_gk = gk_in.shape[0]
        gk_K_layout = cute.make_layout(
            (K, T_gk, (H, Int32(1))),
            stride=(1, source_heads * K, (K, Int64(T_gk) * source_heads * K)),
        )
        gk_K = cute.make_tensor(gk_ptr, gk_K_layout)

        # he output: writes columns [0:V] of packed [S_split, H, K, V+K]
        he_layout = cute.make_layout(
            (K, V, (H, S_split)),
            stride=(V + K, 1, (K * (V + K), H * K * (V + K))),
        )
        he = cute.make_tensor(hm_ptr, he_layout)

        # m output: writes columns [V:V+K] of packed [S_split, H, K, V+K]
        m_layout = cute.make_layout(
            (K, K, (H, S_split)),
            stride=(V + K, 1, (K * (V + K), H * K * (V + K))),
        )
        m = cute.make_tensor(hm_ptr + V, m_layout)

        # cu_seqlens: [S_split+1]
        cu_seqlens = cute.make_tensor(cu_seqlens_ptr, cute.make_layout((S_split + 1,)))

        self.k_dtype = kt.element_type
        self.w_dtype = w.element_type
        self.u_dtype = u.element_type

        # ===================== MMA setup (same as fwd_h) =====================
        wh_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.wh_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )
        kv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.kv_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

        vnew_tmem_layout = sm100_utils.make_smem_layout_a(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.io_dtype,
            1,
        )
        state_tmem_layout = sm100_utils.make_smem_layout_a(
            wh_tiled_mma,
            self.wh_mma_tiler,
            self.io_dtype,
            1,
        )

        # ===================== TMEM offsets =====================
        (self.tmem_wh_off, self.tmem_state_off, self.tmem_vnew_off, self.tmem_kv_off, self.tmem_total) = (
            self._plan_tmem_offsets(
                wh_tiled_mma,
                self.wh_mma_tiler,
                kv_tiled_mma,
                self.kv_mma_tiler,
                state_tmem_layout,
                vnew_tmem_layout,
                self.acc_stage,
            )
        )

        # ===================== SMEM layouts =====================
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        w_smem_staged = sm100_utils.make_smem_layout_b(
            wh_tiled_mma,
            self.wh_mma_tiler,
            self.io_dtype,
            self.w_stage,
        )
        kt_smem_staged = sm100_utils.make_smem_layout_b(
            kv_tiled_mma,
            self.kv_mma_tiler,
            self.io_dtype,
            self.k_stage,
        )
        u_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),
            self.u_stage,
        )

        # ===================== TMA descriptors =====================
        cluster_layout = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (wh_tiled_mma.thr_id.shape,),
        )

        w_smem = cute.select(w_smem_staged, mode=[0, 1, 2])
        tma_atom_w, tma_tensor_w = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            w,
            w_smem,
            self.wh_mma_tiler,
            wh_tiled_mma,
            cluster_layout.shape,
        )
        kt_smem = cute.select(kt_smem_staged, mode=[0, 1, 2])
        tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            kt,
            kt_smem,
            self.kv_mma_tiler,
            kv_tiled_mma,
            cluster_layout.shape,
        )
        u_smem = cute.select(u_epi_staged, mode=[0, 1])
        tma_atom_u, tma_tensor_u = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            u_T,
            u_smem,
            (self.BV, self.BT),
        )
        gk_smem_2d = cute.make_layout((self.BK, 1))
        tma_atom_gk, tma_tensor_gk = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            gk_K,
            gk_smem_2d,
            (self.BK, 1),
        )

        self.tma_w_bytes = cute.size_in_bytes(self.io_dtype, w_smem)
        self.tma_kt_bytes = cute.size_in_bytes(self.io_dtype, kt_smem)
        self.tma_u_bytes = cute.size_in_bytes(self.io_dtype, u_smem)
        self.tma_gk_bytes = self.BK * 4

        # ===================== SharedStorage =====================
        @cute.struct
        class SharedStorage:
            # -- Pipelines: Load → MMA --
            load_w_mbar: cute.struct.MemRange[Int64, self.w_stage * 2]
            load_kt_mbar: cute.struct.MemRange[Int64, self.k_stage * 2]
            load_u_mbar: cute.struct.MemRange[Int64, self.u_stage * 2]
            load_gk_mbar: cute.struct.MemRange[Int64, self.gk_stage * 2]
            # -- Pipelines: CUDA ↔ MMA --
            state_tmem_mbar: cute.struct.MemRange[Int64, 1 * 2]
            wh_done_mbar: cute.struct.MemRange[Int64, self.acc_stage * 2]
            vnew_smem_mbar: cute.struct.MemRange[Int64, 1 * 2]
            kv_done_mbar: cute.struct.MemRange[Int64, 1 * 2]

            # -- TMEM holding --
            tmem_holding_buf: Int32
            # -- Data buffers --
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(w_smem_staged)],
                self.buffer_align_bytes,
            ]
            sKt: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(kt_smem_staged)],
                self.buffer_align_bytes,
            ]
            sU: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(u_epi_staged)],
                self.buffer_align_bytes,
            ]
            sGK: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK * self.gk_stage],
                128,
            ]

        self.shared_storage = SharedStorage
        self.grid = self._compute_grid(S_split, H, K, V)

        self.kernel(
            wh_tiled_mma,
            kv_tiled_mma,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_kt,
            tma_tensor_kt,
            tma_atom_u,
            tma_tensor_u,
            tma_atom_gk,
            tma_tensor_gk,
            u,
            u_T,
            he,
            m,
            w_smem_staged,
            kt_smem_staged,
            state_tmem_layout,
            vnew_tmem_layout,
            u_epi_staged,
            cu_seqlens,
            problem_size,
            use_gk,
            num_v_tiles,
        ).launch(
            grid=self.grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=self.min_occupancy,
        )

    @cute.kernel
    def kernel(
        self,
        wh_tiled_mma: cute.TiledMma,
        kv_tiled_mma: cute.TiledMma,
        # TMA atoms + descriptors
        tma_atom_w: cute.CopyAtom,
        tma_tensor_w: cute.Tensor,
        tma_atom_kt: cute.CopyAtom,
        tma_tensor_kt: cute.Tensor,
        tma_atom_u: cute.CopyAtom,
        tma_tensor_u: cute.Tensor,
        tma_atom_gk: cute.CopyAtom,
        tma_tensor_gk: cute.Tensor,
        # GMEM tensors for address computation
        u_tensor: cute.Tensor,  # (T, V, H)
        u_T_tensor: cute.Tensor,  # (V, T, H)
        he_tensor: cute.Tensor,  # (K, V, (H, S_split)) — he columns of packed hm
        m_tensor: cute.Tensor,  # (K, K, (H, S_split)) — m columns of packed hm
        # SMEM layouts
        w_smem_staged: cute.ComposedLayout,
        kt_smem_staged: cute.ComposedLayout,
        state_tmem_layout: cute.ComposedLayout,
        vnew_tmem_layout: cute.ComposedLayout,
        u_epi_staged: cute.ComposedLayout,
        # Sequence metadata
        cu_seqlens: cute.Tensor,  # (S_split+1,)
        # Scalars
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32],
        use_gk: Int32,
        num_v_tiles: Int32,  # dispatch: tile_idx < num_v_tiles → he mode
    ):
        """
        Device kernel. Each CTA processes one (tile, sub-seq, head) triple.

        Grid-level dispatch:
          tile_idx < num_v_tiles → he mode (exit h-state)
          tile_idx >= num_v_tiles → m mode (transition matrix)

        Both modes share identical MMA structure. Only CUDA warps and
        Load warp (U TMA) differ between modes.

        Warp roles:
          Load warp (5):  TMA G2S for W, K^T, gk, U(he mode only).
          MMA warp (4):   WH/WM + KV/KM MMA (code unchanged).
          CUDA warps (0-3):
            he mode: h recursion (same as fwd_h minus outputs)
            m mode: M^T recursion via associativity reformulation
          Store warp (6): idle.
          Empty warp (7): idle.
        """
        S_split, T_total, H, K, V = problem_size
        BT = self.BT

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # Prefetch TMA descriptors (Load warp)
        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_w)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kt)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_u)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_gk)

        # ===================== SMEM allocation =====================
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sGK_smem = storage.sGK.get_tensor(cute.make_layout((self.BK, self.gk_stage)))
        sGK_3d = storage.sGK.get_tensor(cute.make_layout((self.BK, 1, self.gk_stage), stride=(1, self.BK, self.BK)))

        # ===================== Pipelines =====================
        # Load → MMA: W, K^T (TmaUmma)
        load_w_P, load_w_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.w_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_w_bytes,
            barrier_storage=storage.load_w_mbar.data_ptr(),
        ).make_participants()

        load_kt_P, load_kt_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(1),
            tx_count=self.tma_kt_bytes,
            barrier_storage=storage.load_kt_mbar.data_ptr(),
        ).make_participants()

        # CUDA → MMA: state TMEM (AsyncUmma)
        state_smem_P, state_smem_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.state_tmem_mbar.data_ptr(),
        ).make_participants()

        # MMA → CUDA: WH done (UmmaAsync)
        wh_done_P, wh_done_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stage,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.wh_done_mbar.data_ptr(),
        ).make_participants()

        # CUDA → MMA: vnew TMEM (AsyncUmma)
        vnew_smem_P, vnew_smem_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.vnew_smem_mbar.data_ptr(),
        ).make_participants()

        # MMA → CUDA: KV done (UmmaAsync)
        kv_done_P, kv_done_C = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=make_thread_cooperative_group(1),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * len(self.cuda_warp_ids)),
            barrier_storage=storage.kv_done_mbar.data_ptr(),
        ).make_participants()

        # Load → CUDA: U (TmaAsync)
        load_u_P, load_u_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.u_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len(self.cuda_warp_ids)),
            tx_count=self.tma_u_bytes,
            barrier_storage=storage.load_u_mbar.data_ptr(),
        ).make_participants()

        # Load → CUDA: gk (TmaAsync)
        load_gk_P, load_gk_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.gk_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len(self.cuda_warp_ids)),
            tx_count=self.tma_gk_bytes,
            barrier_storage=storage.load_gk_mbar.data_ptr(),
        ).make_participants()

        # ===================== TMEM allocation =====================
        tmem_alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_bar,
            allocator_warp_id=self.load_warp_id,
        )
        tmem.allocate(self.tmem_total)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # ===================== SMEM views =====================
        sW = storage.sW.get_tensor(w_smem_staged.outer, swizzle=w_smem_staged.inner)
        sKt = storage.sKt.get_tensor(kt_smem_staged.outer, swizzle=kt_smem_staged.inner)
        sU_epi = storage.sU.get_tensor(u_epi_staged.outer, swizzle=u_epi_staged.inner)

        # ===================== MMA fragments =====================
        # WH MMA: A=state(TMEM), B=sW, acc=WH TMEM
        tCrState_fake = wh_tiled_mma.make_fragment_A(state_tmem_layout.outer.shape)
        tCrState = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_off, dtype=tCrState_fake.element_type),
            tCrState_fake.layout,
        )
        tCrW = wh_tiled_mma.make_fragment_B(sW)
        wh_shape = wh_tiled_mma.partition_shape_C(self.wh_mma_tiler[:2])
        tCtAccWH_fake = wh_tiled_mma.make_fragment_C(cute.append(wh_shape, self.acc_stage))
        tCtAccWH = cute.make_tensor(tmem_ptr + self.tmem_wh_off, tCtAccWH_fake.layout)

        # KV MMA: A=v_new(TMEM), B=sKt, acc=KV TMEM
        tCrVnew_fake = kv_tiled_mma.make_fragment_A(vnew_tmem_layout.outer.shape)
        tCrVnew = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_vnew_off, dtype=tCrVnew_fake.element_type),
            tCrVnew_fake.layout,
        )
        tCrKt = kv_tiled_mma.make_fragment_B(sKt)
        kv_shape = kv_tiled_mma.partition_shape_C(self.kv_mma_tiler[:2])
        tCtAccKV_fake = kv_tiled_mma.make_fragment_C(cute.append(kv_shape, 1))
        tCtAccKV = cute.make_tensor(tmem_ptr + self.tmem_kv_off, tCtAccKV_fake.layout)

        # ===================== Work unit decode (non-persistent) =====================
        # Release references to non-serializable Python objects before runtime if-blocks
        del storage, smem
        tile_idx = cute.arch.block_idx()[0]
        combined = cute.arch.block_idx()[1]
        i_subseq = combined // H
        i_h = combined % H
        bos = cu_seqlens[i_subseq]
        eos = cu_seqlens[i_subseq + 1]
        seq_len = eos - bos
        NT = (seq_len + BT - 1) // BT

        # Grid-level dispatch: he mode vs m mode
        is_he_mode = tile_idx < num_v_tiles

        # =========================================================================
        # LOAD WARP
        # =========================================================================
        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

            # TMA partition: shift by bos for varlen
            bos_i64 = Int64(bos)
            tma_tensor_w_v = cute.domain_offset((bos_i64, 0, (0, 0)), tma_tensor_w)
            tma_tensor_kt_v = cute.domain_offset((0, bos_i64, (0, 0)), tma_tensor_kt)
            tma_tensor_u_v = cute.domain_offset((0, bos_i64, (0, 0)), tma_tensor_u)
            tma_tensor_gk_v = cute.domain_offset((0, bos_i64, (0, 0)), tma_tensor_gk)

            tWsW, tWgW = self._tma_partition_B(
                tma_atom_w,
                tma_tensor_w_v,
                sW,
                self.wh_mma_tiler,
                wh_tiled_mma,
                Int32(0),
                i_h,
            )
            tKsK, tKgK = self._tma_partition_B(
                tma_atom_kt,
                tma_tensor_kt_v,
                sKt,
                self.kv_mma_tiler,
                kv_tiled_mma,
                Int32(0),
                i_h,
            )

            # U TMA partition
            gU_ld = tma_tensor_u_v[None, None, (i_h, Int32(0))]
            _, bSG_sU, bSG_gU = self._epilog_partition(
                tma_atom_u,
                gU_ld,
                (self.BV, self.BT),
                sU_epi,
            )

            # gk TMA partition
            gGK_ld = tma_tensor_gk_v[None, None, (i_h, Int32(0))]
            _, bSG_sGK, bSG_gGK = self._epilog_partition(
                tma_atom_gk,
                gGK_ld,
                (self.BK, 1),
                sGK_3d,
            )

            # Chunk loop: issue TMA loads
            for chunk_idx in cutlass.range(0, NT, unroll=0):
                w_h = load_w_P.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_w,
                    src=tWgW[None, chunk_idx, 0],
                    dst=tWsW[None, w_h.index],
                    tma_bar_ptr=w_h.barrier,
                )

                kt_h = load_kt_P.acquire_and_advance()
                cute.copy(
                    atom=tma_atom_kt,
                    src=tKgK[None, 0, chunk_idx],
                    dst=tKsK[None, kt_h.index],
                    tma_bar_ptr=kt_h.barrier,
                )

                # U TMA: he mode only (m mode skips U entirely)
                if is_he_mode:
                    u_h = load_u_P.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_u,
                        src=bSG_gU[(None, tile_idx, chunk_idx)],
                        dst=bSG_sU[None, u_h.index],
                        tma_bar_ptr=u_h.barrier,
                    )

                # Load gk only when gk gating is active
                if use_gk != 0:
                    gk_t_idx = chunk_idx * self.BT + self.BT - 1
                    remaining = seq_len - chunk_idx * self.BT
                    if remaining < self.BT:
                        gk_t_idx = seq_len - 1
                    gk_h = load_gk_P.acquire_and_advance()
                    cute.copy(
                        atom=tma_atom_gk,
                        src=bSG_gGK[(None, 0, gk_t_idx)],
                        dst=bSG_sGK[None, gk_h.index],
                        tma_bar_ptr=gk_h.barrier,
                    )

        # =========================================================================
        # MMA WARP
        # =========================================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

            for chunk_idx in cutlass.range(0, NT, unroll=0):
                # WH MMA: acc = state @ W
                state_h = state_smem_C.wait_and_advance()
                w_h = load_w_C.wait_and_advance()
                wh_h = wh_done_P.acquire_and_advance()
                for kp in cutlass.range(cute.size(tCrW, mode=[2]), unroll_full=True):
                    wh_tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        wh_tiled_mma,
                        tCtAccWH[None, None, None, wh_h.index],
                        tCrState[None, None, kp, state_h.index],
                        tCrW[None, None, kp, w_h.index],
                        tCtAccWH[None, None, None, wh_h.index],
                    )
                wh_h.commit()
                w_h.release()
                state_h.release()

                # KV MMA: update = vnew @ K^T
                vnew_h = vnew_smem_C.wait_and_advance()
                kt_h = load_kt_C.wait_and_advance()
                kv_h = kv_done_P.acquire_and_advance()
                for kp in cutlass.range(cute.size(tCrKt, mode=[2]), unroll_full=True):
                    kv_tiled_mma.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        kv_tiled_mma,
                        tCtAccKV[None, None, None, 0],
                        tCrVnew[None, None, kp, vnew_h.index],
                        tCrKt[None, None, kp, kt_h.index],
                        tCtAccKV[None, None, None, 0],
                    )
                kv_h.commit()
                kt_h.release()
                vnew_h.release()

        # =========================================================================
        # CUDA CORE WARPS (0-3)
        # =========================================================================
        elif warp_idx in self.cuda_warp_ids:
            cute.arch.setmaxregister_increase(self.num_regs_cuda)
            local_tidx = tidx % (self.threads_per_warp * len(self.cuda_warp_ids))

            # ----- T2R setup for KV acc (BV, BK fp32) → h update -----
            t2r_atom_kv = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(16), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAccKV_flat = tCtAccKV[((None, None), 0, 0, None)]
            fake_sKV = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.kv_mma_tiler, (1, 1, None)),
            )
            tiled_t2r_kv = tcgen05.make_tmem_copy(t2r_atom_kv, tCtAccKV_flat[(None, None, 0)])
            thr_t2r_kv = tiled_t2r_kv.get_slice(local_tidx)
            tTR_tKV = thr_t2r_kv.partition_S(tCtAccKV_flat)
            tTR_sKV = thr_t2r_kv.partition_D(fake_sKV)
            # h state in registers (persistent across chunks)
            tTR_rKV = cute.make_rmem_tensor(tTR_sKV.shape, self.acc_dtype)

            # ----- T2R setup for WH acc (BV, BT fp32) → v_new -----
            t2r_atom_wh = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(8), tcgen05.Pack.NONE),
                self.acc_dtype,
            )
            tCtAccWH_flat = tCtAccWH[((None, None), 0, 0, None)]
            fake_sWH = cute.make_tensor(
                cute.make_ptr(self.io_dtype, 0, cute.AddressSpace.smem),
                cute.dice(self.wh_mma_tiler, (1, 1, None)),
            )
            tiled_t2r_wh = tcgen05.make_tmem_copy(t2r_atom_wh, tCtAccWH_flat[(None, None, 0)])
            thr_t2r_wh = tiled_t2r_wh.get_slice(local_tidx)
            tTR_tWH = thr_t2r_wh.partition_S(tCtAccWH_flat)
            tTR_sWH = thr_t2r_wh.partition_D(fake_sWH)

            # ----- R2T: h regs → TMEM for WH MMA A operand -----
            copy_atom_r2t_state = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(16), tcgen05.Unpack.NONE),
                self.io_dtype,
            )
            tiled_r2t_state = tcgen05.make_tmem_copy(copy_atom_r2t_state, tCrState)
            thr_r2t_state = tiled_r2t_state.get_slice(local_tidx)
            r2t_state_shape = cute.slice_(thr_r2t_state.partition_S(tCrState).shape, (None, None, None, None, 0))
            tRT_tState = thr_r2t_state.partition_D(tCrState)

            # ----- R2T: v_new regs → TMEM for KV MMA A operand -----
            copy_atom_r2t_vnew = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
                self.io_dtype,
            )
            tiled_r2t_vnew = tcgen05.make_tmem_copy(copy_atom_r2t_vnew, tCrVnew)
            thr_r2t_vnew = tiled_r2t_vnew.get_slice(local_tidx)
            r2t_vnew_shape = cute.slice_(thr_r2t_vnew.partition_S(tCrVnew).shape, (None, None, None, None, 0))
            tRT_tVnew = thr_r2t_vnew.partition_D(tCrVnew)

            # ----- Identity tensors for coordinate mapping -----
            vnew_tile = cute.dice(self.wh_mma_tiler, (1, 1, None))  # (BV, BT)
            cM_vnew = cute.make_identity_tensor(vnew_tile)
            tTR_cM = thr_t2r_wh.partition_D(cM_vnew)

            h_tile = cute.dice(self.kv_mma_tiler, (1, 1, None))  # (BV, BK)
            cM_h = cute.make_identity_tensor(h_tile)
            tTR_cM_h = thr_t2r_kv.partition_D(cM_h)

            # ----- Initialize state: h=0 (he mode) or M^T=I (m mode) -----
            if is_he_mode:
                for ei in cutlass.range(cute.size(tTR_rKV), unroll_full=True):
                    tTR_rKV[ei] = Float32(0.0)
            else:
                k_col_tile = tile_idx - num_v_tiles
                for ei in cutlass.range(cute.size(tTR_rKV), unroll_full=True):
                    v_coord, k_coord = tTR_cM_h[ei]
                    col_global = v_coord + k_col_tile * self.BS
                    if k_coord == col_global:
                        tTR_rKV[ei] = Float32(1.0)
                    else:
                        tTR_rKV[ei] = Float32(0.0)

            # ===== Main chunk loop =====
            for chunk_idx in cutlass.range(0, NT, unroll=0):
                # ========================================
                # Phase 1: Publish state for WH/WM MMA
                # ========================================
                tRT_rState = cute.make_rmem_tensor(r2t_state_shape, self.io_dtype)
                h_vec = tTR_rKV.load()
                h_vec_bf16 = h_vec.to(self.io_dtype)

                # R2T state → TMEM (triggers WH/WM MMA)
                tRT_rState.store(h_vec_bf16)
                state_h = state_smem_P.acquire_and_advance()
                cute.copy(tiled_r2t_state, tRT_rState, tRT_tState[(None, None, None, None, 0)])
                cute.arch.fence_view_async_tmem_store()
                state_h.commit()

                # Preload U from SMEM → registers (he mode only, overlapping WH MMA)
                tTR_rU = cute.make_rmem_tensor(tTR_sWH.shape, self.acc_dtype)
                if is_he_mode:
                    u_handle = load_u_C.wait_and_advance()
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        v_coord, t_coord = tTR_cM[ei]
                        tTR_rU[ei] = sU_epi[(v_coord, t_coord, u_handle.index)].to(self.acc_dtype)
                    u_handle.release()

                # ========================================
                # Phase 2: Process WH/WM result → triggers KV/KM MMA
                # ========================================
                wh_h = wh_done_C.wait_and_advance()
                tTR_rWH = cute.make_rmem_tensor(tTR_sWH.shape, self.acc_dtype)
                cute.copy(tiled_t2r_wh, tTR_tWH[(None, None, None, wh_h.index)], tTR_rWH)
                cute.arch.fence_view_async_tmem_load()
                wh_h.release()

                if is_he_mode:
                    # he mode: v_new = u - WH
                    for ei in cutlass.range_constexpr(cute.size(tTR_rWH)):
                        tTR_rWH[ei] = tTR_rU[ei] - tTR_rWH[ei]
                # else: m mode — tTR_rWH = WM result, used as-is for KM MMA

                # Varlen tail chunk zero mask (both modes)
                valid_len_chunk = seq_len - chunk_idx * self.BT
                if valid_len_chunk < self.BT:
                    for ei in cutlass.range_constexpr(cute.size(tTR_cM)):
                        v_coord, t_coord = tTR_cM[ei]
                        if t_coord >= valid_len_chunk:
                            tTR_rWH[ei] = Float32(0.0)

                # R2T vnew/temp → TMEM (triggers KV/KM MMA)
                vnew_vec_bf16 = tTR_rWH.load().to(self.io_dtype)
                tRT_rVnew = cute.make_rmem_tensor(r2t_vnew_shape, self.io_dtype)
                tRT_rVnew.store(vnew_vec_bf16)
                vnew_h = vnew_smem_P.acquire_and_advance()
                cute.copy(tiled_r2t_vnew, tRT_rVnew, tRT_tVnew[(None, None, None, None, 0)])
                cute.arch.fence_view_async_tmem_store()
                vnew_h.commit()

                # ========================================
                # Phase 3: gk decay (overlapping with KV/KM MMA)
                # ========================================
                if use_gk != 0:
                    gk_h = load_gk_C.wait_and_advance()
                    gk_raw = sGK_smem[(tidx, gk_h.index)]
                    sGK_smem[(tidx, gk_h.index)] = cute.exp2(gk_raw, fastmath=self.use_fast_math)
                    self.gk_precompute_bar.arrive_and_wait()
                    for ei in cutlass.range(cute.size(tTR_rKV), unroll_full=True):
                        v_coord, k_coord = tTR_cM_h[ei]
                        tTR_rKV[ei] = tTR_rKV[ei] * sGK_smem[(k_coord, gk_h.index)]
                    gk_h.release()

                # ========================================
                # Phase 4: KV/KM update
                # ========================================
                kv_h = kv_done_C.wait_and_advance()
                tTR_rUpdate = cute.make_rmem_tensor(tTR_sKV.shape, self.acc_dtype)
                cute.copy(tiled_t2r_kv, tTR_tKV[(None, None, None, 0)], tTR_rUpdate)
                cute.arch.fence_view_async_tmem_load()
                kv_h.release()

                h_vec = tTR_rKV.load()
                update_vec = tTR_rUpdate.load()
                if is_he_mode:
                    tTR_rKV.store(h_vec + update_vec)  # h += K^T @ v_new
                else:
                    tTR_rKV.store(h_vec - update_vec)  # M -= K^T @ (W @ M)

            # ===== After loop: write output to GMEM =====
            if is_he_mode:
                # Write he (exit h-state) → hm[:, :, :, :V]
                for ei in cutlass.range(cute.size(tTR_rKV), unroll_full=True):
                    v_coord, k_coord = tTR_cM_h[ei]
                    he_tensor[(k_coord, v_coord + tile_idx * self.BV, (i_h, i_subseq))] = tTR_rKV[ei].to(he_tensor.element_type)
            else:
                # Write M^T (transition matrix, transposed) → hm[:, :, :, V:]
                k_col_tile = tile_idx - num_v_tiles
                for ei in cutlass.range(cute.size(tTR_rKV), unroll_full=True):
                    v_coord, k_coord = tTR_cM_h[ei]
                    col_global = v_coord + k_col_tile * self.BS
                    value = tTR_rKV[ei]
                    if cutlass.const_expr(self.identity_residual):
                        if k_coord == col_global:
                            value = value - Float32(1.0)
                    m_tensor[(k_coord, col_global, (i_h, i_subseq))] = value.to(m_tensor.element_type)

        # =========================================================================
        # STORE WARP
        # =========================================================================
        elif warp_idx == self.store_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)
            # Store warp idle — CUDA warps write hm directly to GMEM
            pass

        # =========================================================================
        # EMPTY WARP
        # =========================================================================
        else:
            cute.arch.setmaxregister_decrease(self.num_regs_others)
            # Empty warp idle
            pass

        # ===================== TMEM dealloc =====================
        self.tmem_dealloc_sync_barrier.sync()
        tmem.free(tmem_ptr)


# =====================================================================
# Compile cache + Python API
# =====================================================================

_pre_scan_kernel_cache: dict = {}


def _compile_pre_scan_variant(H, K, V, chunk_size, use_fast_math, source_heads, head_offset, bf16_output, identity_residual):
    """Compile one ChunkDeltaRulePreScanFused kernel variant."""
    kernel_obj = ChunkDeltaRulePreScanFused(
        chunk_size=chunk_size,
        head_dim_k=K,
        head_dim_v=V,
        use_fast_math=use_fast_math,
        source_heads=source_heads,
        head_offset=head_offset,
        identity_residual=identity_residual,
    )

    sym_t = cute.sym_int()  # T_total
    sym_s = cute.sym_int()  # S_split
    sym_cu = cute.sym_int()  # cu_seqlens length = S_split+1

    # varlen packed: [T_total, H, dim]
    sym_gk = cute.sym_int()  # independent: 1 when gk unused, T_total when used

    k_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_t, source_heads, K), stride_order=(2, 1, 0), assumed_align=128)
    w_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_t, source_heads, K), stride_order=(2, 1, 0), assumed_align=128)
    u_fake = make_fake_compact_tensor(cutlass.BFloat16, (sym_t, source_heads, V), stride_order=(2, 1, 0), assumed_align=128)
    gk_fake = make_fake_compact_tensor(cutlass.Float32, (sym_gk, source_heads, K), stride_order=(2, 1, 0), assumed_align=128)

    # output: [S_split, H, K, V+K] fp32  (packed hm)
    hm_fake = make_fake_compact_tensor(cutlass.BFloat16 if bf16_output else cutlass.Float32, (sym_s, H, K, V + K), stride_order=(3, 2, 1, 0), assumed_align=128)

    # cu_seqlens: [S_split+1]
    cu_fake = make_fake_compact_tensor(cutlass.Int32, (sym_cu,), assumed_align=128)

    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_fn = cute.compile(
        kernel_obj,
        k_fake,
        w_fake,
        u_fake,
        gk_fake,
        hm_fake,
        cu_fake,
        (Int32(1), Int32(1), Int32(H), Int32(K), Int32(V)),  # problem_size
        Int32(0),  # use_gk
        Int32(0),  # num_v_tiles (concrete value passed at runtime)
        stream_fake,
        options="--enable-tvm-ffi",
    )
    return compiled_fn


def _get_compiled_pre_scan(H, K, V, chunk_size, source_heads, head_offset, bf16_output, identity_residual):
    """Get compiled pre-scan kernel with lazy compilation + caching."""
    key = (H, K, V, chunk_size, USE_FAST_MATH, source_heads, head_offset, bf16_output, identity_residual)
    if key not in _pre_scan_kernel_cache:
        _pre_scan_kernel_cache[key] = _compile_pre_scan_variant(H, K, V, chunk_size, USE_FAST_MATH, source_heads, head_offset, bf16_output, identity_residual)
    return _pre_scan_kernel_cache[key]


# =====================================================================
# Python API
# =====================================================================


def chunk_delta_rule_pre_scan(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor | None = None,
    cu_seqlens_split: torch.Tensor = None,
    S_split: int = 0,
    chunk_size: int = 64,
    head_start: int = 0,
    head_count: int | None = None,
    out: torch.Tensor | None = None,
    identity_residual: bool = False,
) -> torch.Tensor:
    """
    Compute packed (he, m) state for each split sub-sequence.

    Single fused CuTeDSL kernel with grid-level dispatch:
      blockIdx.x < num_v_tiles → he (exit h-state) → hm[:, :, :, :V]
      blockIdx.x >= num_v_tiles → m (transition matrix) → hm[:, :, :, V:]

    Args:
        k:  [1, T, H, K] bf16  (varlen packed, B=1)
        w:  [1, T, H, K] bf16
        u:  [1, T, H, V] bf16
        gk: [1, T, H, K] fp32 or None (key gate)
        cu_seqlens_split: [S_split+1] int32  (sub-sequence boundaries)
        S_split: number of sub-sequences
        chunk_size: chunk size (default 64)

    Returns:
        hm: [S_split, H, K, V+K] fp32
            hm[:, :, :, :V]  = he  (K×V exit h-state)
            hm[:, :, :, V:]  = m   (K×K transition matrix)
    """
    assert cu_seqlens_split is not None, "cu_seqlens_split is required"
    assert k.shape[0] == 1, "pre_scan requires varlen mode (B=1)"

    T = k.shape[1]
    source_heads = k.shape[2]
    H = source_heads if head_count is None else head_count
    assert 0 <= head_start < source_heads and head_start + H <= source_heads
    assert all(x.is_contiguous() for x in (k, w, u))
    assert gk is not None and gk.is_contiguous(), "This experiment requires a compact gate tensor"
    K = k.shape[3]
    V = u.shape[3]
    device = k.device

    # Squeeze batch dim for kernel (varlen: [T, H, dim])
    k_kern = k[0]
    w_kern = w[0]
    u_kern = u[0]

    use_gk_flag = 1 if gk is not None else 0
    gk_kern = gk[0] if gk is not None else torch.zeros(1, H, K, device=device, dtype=torch.float32)

    # Ensure cu_seqlens is int32
    cu_seqlens_i32 = cu_seqlens_split.int() if cu_seqlens_split.dtype != torch.int32 else cu_seqlens_split

    # Allocate packed output: [S_split, H, K, V+K] fp32
    hm = out if out is not None else torch.empty(S_split, H, K, V + K, device=device, dtype=torch.float32)
    assert hm.shape == (S_split, H, K, V + K) and hm.is_contiguous()
    assert hm.dtype in (torch.float32, torch.bfloat16) and hm.device == device

    # Single fused kernel: he + m via grid-level dispatch
    BV = 64
    num_v_tiles = (V + BV - 1) // BV

    compiled_fn = _get_compiled_pre_scan(H, K, V, chunk_size, source_heads, head_start, hm.dtype == torch.bfloat16, identity_residual)
    compiled_fn(
        k_kern,
        w_kern,
        u_kern,
        gk_kern,
        hm,
        cu_seqlens_i32,
        (S_split, T, H, K, V),
        use_gk_flag,
        num_v_tiles,
    )

    return hm
