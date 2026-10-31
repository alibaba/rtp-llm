# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This file is ported from the CUTLASS dense block-scaled GEMM example
# and adapted for the current Blackwell GeForce target.
#
# Ported from the b12x kernel library to FlashInfer.

# The M=16 SF layouts and 16x64x512 tiling follow SGLang
# 80afdc1280bf37c1440e6e7009741cbc94e2fba0 (BSD-3-Clause).
# Quantization follows TensorRT-LLM quantization_utils.cuh via FlashInfer.

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm120 import (
    Sm120BlockScaledDenseGemmKernel,
)

# The reciprocal, SiLU and FP4 conversion below are adapted from
# TensorRT-LLM quantization_utils.cuh, Copyright (c) 2019-2023 NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0. You may obtain a copy at
# http://www.apache.org/licenses/LICENSE-2.0 . Distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See that License for the specific permissions and limitations.


@dsl_user_op
def _rcp(v: Float32, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [v.ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _silu(v: Float32, *, loc=None, ip=None):
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [v.ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 a,b; mul.f32 a,$1,0fbfb8aa3b; ex2.approx.ftz.f32 b,a; add.f32 a,b,0f3f800000; div.rn.f32 $0,$1,a; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_fp4_64(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    v8: Float32,
    v9: Float32,
    v10: Float32,
    v11: Float32,
    v12: Float32,
    v13: Float32,
    v14: Float32,
    v15: Float32,
    *,
    loc=None,
    ip=None,
):
    return cutlass.Uint64(
        llvm.inline_asm(
            T.i64(),
            [
                v0.ir_value(loc=loc, ip=ip),
                v1.ir_value(loc=loc, ip=ip),
                v2.ir_value(loc=loc, ip=ip),
                v3.ir_value(loc=loc, ip=ip),
                v4.ir_value(loc=loc, ip=ip),
                v5.ir_value(loc=loc, ip=ip),
                v6.ir_value(loc=loc, ip=ip),
                v7.ir_value(loc=loc, ip=ip),
                v8.ir_value(loc=loc, ip=ip),
                v9.ir_value(loc=loc, ip=ip),
                v10.ir_value(loc=loc, ip=ip),
                v11.ir_value(loc=loc, ip=ip),
                v12.ir_value(loc=loc, ip=ip),
                v13.ir_value(loc=loc, ip=ip),
                v14.ir_value(loc=loc, ip=ip),
                v15.ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .b8 p0; .reg .b8 p1; .reg .b8 p2; .reg .b8 p3; .reg .b8 p4; .reg .b8 p5; .reg .b8 p6; .reg .b8 p7;cvt.rn.satfinite.e2m1x2.f32 p0,$2,$1; cvt.rn.satfinite.e2m1x2.f32 p1,$4,$3; cvt.rn.satfinite.e2m1x2.f32 p2,$6,$5; cvt.rn.satfinite.e2m1x2.f32 p3,$8,$7; cvt.rn.satfinite.e2m1x2.f32 p4,$10,$9; cvt.rn.satfinite.e2m1x2.f32 p5,$12,$11; cvt.rn.satfinite.e2m1x2.f32 p6,$14,$13; cvt.rn.satfinite.e2m1x2.f32 p7,$16,$15;.reg .b32 lo,hi; mov.b32 lo,{p0,p1,p2,p3}; mov.b32 hi,{p4,p5,p6,p7}; mov.b64 $0,{lo,hi}; }",
            "=l,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def sm120_make_smem_layout_sfa(
    tiled_mma,
    tile_shape_mnk,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
):
    del loc, ip
    assert sf_vec_size in (16, 32)
    # Warp MMA consumes 16 live rows, while the scale-factor storage layout has
    # a 64-row minimum and a physical 128-row basic block.
    tile_m = max(64, tile_shape_mnk[0])
    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size
    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)
    assert tile_m % 64 == 0
    sfa_tile_m = max(blk_mn, _ceil_div(tile_m, blk_mn) * blk_mn)
    sfa_shape_m = (mn_basic_block_shape, sfa_tile_m // blk_mn)
    sf_stride_m = (mn_basic_block_stride, blk_elems)
    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0
    assert blk_sf % mma_nsf == 0
    sfa_shape_k = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sf_stride_k = (
        k_basic_block_stride,
        mma_nsf,
        sfa_tile_m // blk_mn * blk_elems,
    )
    layout = cute.make_layout(
        (sfa_shape_m, sfa_shape_k), stride=(sf_stride_m, sf_stride_k)
    )
    return cute.append(
        layout,
        cute.make_layout(num_stages, stride=cute.cosize(cute.filter_zeros(layout))),
    )


def sm120_make_smem_layout_sfb(
    tiled_mma,
    tile_shape_mnk,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
):
    del loc, ip
    assert sf_vec_size in (16, 32)
    blk_mn = 128
    blk_sf = 4
    blk_elems = blk_mn * blk_sf
    assert tile_shape_mnk[1] % 16 == 0
    assert tile_shape_mnk[2] % sf_vec_size == 0
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size
    mn_basic_block_shape = (32, 4)
    mn_basic_block_stride = (16, 4)
    k_basic_block_shape = (sf_vec_size, mma_nsf)
    k_basic_block_stride = (0, 1)
    sfb_tile_n = max(blk_mn, _ceil_div(tile_shape_mnk[1], blk_mn) * blk_mn)
    sfb_shape_n = (mn_basic_block_shape, sfb_tile_n // blk_mn)
    sf_stride_n = (mn_basic_block_stride, blk_elems)
    assert tile_shape_mnk[2] % (blk_sf * mma_nsf) == 0
    assert tile_shape_mnk[2] % (sf_vec_size * blk_sf) == 0
    assert blk_sf % mma_nsf == 0
    sfb_shape_k = (
        k_basic_block_shape,
        blk_sf // mma_nsf,
        tile_shape_mnk[2] // sf_vec_size // blk_sf,
    )
    sf_stride_k = (
        k_basic_block_stride,
        mma_nsf,
        sfb_tile_n // blk_mn * blk_elems,
    )
    layout = cute.make_layout(
        (sfb_shape_n, sfb_shape_k), stride=(sf_stride_n, sf_stride_k)
    )
    return cute.append(
        layout,
        cute.make_layout(num_stages, stride=cute.cosize(cute.filter_zeros(layout))),
    )


class FusedGemmKernel(Sm120BlockScaledDenseGemmKernel):
    """SM120 NVFP4 up projection with BF16-rounded SwiGLU and NVFP4 output."""

    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        use_prefetch: bool = False,
        enable_pdl: bool = True,
    ):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        # K = sf_vec_size * 8 for FP4 (each FP4 element is 0.5 bytes, sf_vec_size
        # elements per scale factor, and we want 4 MMA k-tiles per stage)
        tile_k = sf_vec_size * (
            32 if mma_tiler_mn[0] == 16 else 16
        )  # K=512 for small M, K=256 for the wider tile
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], tile_k)
        self.sfa_tile_shape_mk = (max(128, mma_tiler_mn[0]), tile_k)
        self.sfa_tiles_per_block = self.sfa_tile_shape_mk[0] // mma_tiler_mn[0]
        self.sfb_tile_shape_nk = (max(128, mma_tiler_mn[1]), tile_k)
        self.sfb_tiles_per_block = self.sfb_tile_shape_nk[0] // mma_tiler_mn[1]
        self.cluster_shape_mnk = (1, 1, 1)  # Always (1,1,1) on the current target
        self.epi_tile = (mma_tiler_mn[0], mma_tiler_mn[1])
        self.use_prefetch = use_prefetch
        self.enable_pdl = enable_pdl

        self.tiled_mma = None
        self.occupancy = 1
        self.num_mma_warps = 2 if mma_tiler_mn[0] == 16 else 8
        self.tma_load_warp_id = self.num_mma_warps
        self.num_threads_per_warp = 32
        self.threads_per_cta = (
            self.num_mma_warps + 1  # 1 warp for DMA
        ) * self.num_threads_per_warp

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        self.ab_stage = None
        self.epi_stage = None
        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.num_mma_warps * self.num_threads_per_warp,
        )
        self.load_register_requirement = 40
        self.mma_register_requirement = 232

    def _setup_attributes(self):
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype,
            self.acc_dtype,
            self.sf_dtype,
        )
        atom_shape = (self.num_mma_warps // 2, 2, 1)
        atom_layout = cute.make_layout(atom_shape)
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, False
        )
        self.tiled_mma = cute.make_tiled_mma(
            mma_op,
            atom_layout,
            permutation_mnk=permutation_mnk,
        )
        # Bare atom for manual unroll workaround (avoids hasAuxTensor address space bug)
        self.mma_atom = cute.make_mma_atom(mma_op)
        # Compute atom loop bounds from tile shape and atom/layout shape
        # MMA atom: m16, n8, k64; atom_layout: (4,2,1) -> group: m64, n16, k64
        mma_m, mma_n, mma_k = 16, 8, 64
        self.num_m_tiles = self.tile_shape_mnk[0] // (mma_m * atom_shape[0])
        self.num_n_tiles = self.tile_shape_mnk[1] // (mma_n * atom_shape[1])
        self.num_k_blocks = self.tile_shape_mnk[2] // mma_k

        self.cta_layout_mnk = cute.make_layout(self.cluster_shape_mnk)

        # Compute the smem size of SFA/SFB
        sfa_smem_layout_per_stage = sm120_make_smem_layout_sfa(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )
        sfb_smem_layout_per_stage = sm120_make_smem_layout_sfb(
            self.tiled_mma,
            self.tile_shape_mnk,
            self.sf_vec_size,
            1,
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            sfa_smem_layout_per_stage,
            sfb_smem_layout_per_stage,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        assert (
            self.epi_stage > 0
        ), "epi_stage <= 0, not enough shared memory. This configuration will be skipped."

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
            self.sf_vec_size,
            self.tiled_mma,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
        q: cute.Tensor,
        sf: cute.Tensor,
        scale: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation.

        Args:
            a: Input tensor A
            b: Input tensor B
            sfa: Scale factor tensor for A
            sfb: Scale factor tensor for B
            c: Output tensor C
            alpha: Alpha scaling factor tensor, shape (1,), float32
            max_active_clusters: Max active clusters
            stream: CUDA stream
            epilogue_op: Elementwise epilogue function
        """
        # Setup static attributes
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = cutlass.BFloat16
        self.sf_dtype = sfa.element_type

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        self.sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa.iterator, self.sfa_layout)

        self.sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb.iterator, self.sfb_layout)

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )
        tma_atom_sfa, tma_tensor_sfa = self._make_tma_atoms_and_tensors(
            sfa_tensor,
            self.sfa_smem_layout_staged,
            self.sfa_tile_shape_mk,
            1,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = self._make_tma_atoms_and_tensors(
            sfb_tensor,
            self.sfb_smem_layout_staged,
            self.sfb_tile_shape_nk,
            1,
            internal_type=cutlass.Int16,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.tile_shape_mnk,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c,
            self.tiled_mma,
            self.mma_atom,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
            epilogue_op,
            alpha,
            q,
            sf,
            scale,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        mma_atom: cute.MmaAtom,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
        q: cute.Tensor,
        sf: cute.Tensor,
        scale: cute.Tensor,
    ):
        # Keep alpha in FP32 for precision
        alpha_value = alpha[0].to(cutlass.Float32)

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch TMA descriptors
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem_layout)
            + cute.size_in_bytes(self.b_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline setup
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_warps
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cta_layout_vmnk,
        )

        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_arrive_relaxed()

        # Generate smem tensors
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            self.sfa_tile_shape_mk,
            (None, None, None),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            self.sfb_tile_shape_nk,
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        # Partition for TiledMMA
        thr_mma = tiled_mma.get_slice(tidx)

        # TMA partitions for A
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )

        # TMA partitions for B
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        # TMA partitions for SFA
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sSFA, 0, 2),
            cute.group_modes(gSFA_mkl, 0, 2),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA partitions for SFB
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_nkl, 0, 2),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        # Make fragments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA_full = self._partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
        tCrSFB_full = self._partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)

        tCgC = thr_mma.partition_C(gC_mnl)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # Cluster/thread sync
        if cute.size(self.cluster_shape_mnk) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.sync_threads()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Tile scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # Pipeline states
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        # MMA warp group
        if warp_idx < self.num_mma_warps:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Copy atoms for SMEM->RMEM
            atom_copy_ldmatrix_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_layout.is_m_major_a(), 4),
                self.a_dtype,
            )
            atom_copy_ldmatrix_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_layout.is_n_major_b(), 4),
                self.b_dtype,
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
            smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)

            atom_copy_ldmatrix_SF = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.sf_dtype,
            )
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                self._get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                atom_copy_ldmatrix_SF,
                self._get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            thr_copy_ldmatrix_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            thr_copy_ldmatrix_SFB = smem_tiled_copy_SFB.get_slice(tidx)
            tCsSFA_copy_view_full = thr_copy_ldmatrix_SFA.partition_S(sSFA)
            tCrSFA_copy_view_full = thr_copy_ldmatrix_SFA.retile(tCrSFA_full)
            tCsSFB_copy_view_full = thr_copy_ldmatrix_SFB.partition_S(sSFB)
            tCrSFB_copy_view_full = thr_copy_ldmatrix_SFB.retile(tCrSFB_full)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                gC_mnl_slice = gC_mnl[(None, None, *tile_coord_mnl)]
                sfa_tile_offset = tile_coord_mnl[0] % self.sfa_tiles_per_block
                sfb_tile_offset = tile_coord_mnl[1] % self.sfb_tiles_per_block
                if cutlass.const_expr(self.sfa_tiles_per_block > 1):
                    sSFA_tile = cute.local_tile(
                        sSFA,
                        cute.slice_(self.tile_shape_mnk, (None, 0, None)),
                        (sfa_tile_offset, 0, None),
                    )
                    tCsSFA_tile_copy_view = thr_copy_ldmatrix_SFA.partition_S(sSFA_tile)
                    tCrSFA_tile = self._partition_fragment_SFA(
                        sSFA_tile[None, None, 0], thr_mma, tidx
                    )
                    tCrSFA_tile_copy_view = thr_copy_ldmatrix_SFA.retile(tCrSFA_tile)
                else:
                    tCsSFA_tile_copy_view = tCsSFA_copy_view_full
                    tCrSFA_tile = tCrSFA_full
                    tCrSFA_tile_copy_view = tCrSFA_copy_view_full
                if cutlass.const_expr(self.sfb_tiles_per_block > 1):
                    sSFB_tile = cute.local_tile(
                        sSFB,
                        cute.slice_(self.tile_shape_mnk, (0, None, None)),
                        (sfb_tile_offset, 0, None),
                    )
                    tCsSFB_tile_copy_view = thr_copy_ldmatrix_SFB.partition_S(sSFB_tile)
                    tCrSFB_tile = self._partition_fragment_SFB(
                        sSFB_tile[None, None, 0], thr_mma, tidx
                    )
                    tCrSFB_tile_copy_view = thr_copy_ldmatrix_SFB.retile(tCrSFB_tile)
                else:
                    tCsSFB_tile_copy_view = tCsSFB_copy_view_full
                    tCrSFB_tile = tCrSFB_full
                    tCrSFB_tile_copy_view = tCrSFB_copy_view_full
                accumulators.fill(0.0)

                # Pipelined MAINLOOP
                mainloop_consumer_state.reset_count()

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )

                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsB_p = tCsB_copy_view[None, None, None, mainloop_consumer_state.index]
                tCsSFA_p = tCsSFA_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                tCsSFB_p = tCsSFB_tile_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                cute.copy(
                    smem_tiled_copy_A,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_B,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )

                tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_tile_copy_view)
                tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_tile_copy_view)

                cute.copy(
                    smem_tiled_copy_SFA,
                    tCsSFA_p_filtered[None, None, 0],
                    tCrSFA_copy_view_filtered[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_SFB,
                    tCsSFB_p_filtered[None, None, 0],
                    tCrSFB_copy_view_filtered[None, None, 0],
                )

                for _k_tile in range(0, k_tile_cnt - 1, 1, unroll=2):  # type: ignore[call-overload]
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )

                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()

                            peek_ab_full_status = cutlass.Boolean(1)
                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )

                            tCsA_p = tCsA_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsB_p = tCsB_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFA_p = tCsSFA_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            tCsSFB_p = tCsSFB_tile_copy_view[
                                None, None, None, mainloop_consumer_state.index
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )

                        # Manual atom unroll: avoids hasAuxTensor address space bug
                        for _mt in range(self.num_m_tiles):
                            for _nt in range(self.num_n_tiles):
                                mma_atom.set(
                                    WarpField.SFA,
                                    tCrSFA_tile[None, _mt, k_block_idx].iterator,
                                )
                                mma_atom.set(
                                    WarpField.SFB,
                                    tCrSFB_tile[None, _nt, k_block_idx].iterator,
                                )
                                cute.gemm(
                                    mma_atom,
                                    accumulators[None, _mt, _nt],
                                    tCrA[None, _mt, k_block_idx],
                                    tCrB[None, _nt, k_block_idx],
                                    accumulators[None, _mt, _nt],
                                )
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )

                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(
                            tCrSFA_tile_copy_view
                        )
                        tCrSFB_copy_view_filtered = cute.filter_zeros(
                            tCrSFB_tile_copy_view
                        )
                        cute.copy(
                            smem_tiled_copy_SFA,
                            tCsSFA_p_filtered[None, None, k_block_next],
                            tCrSFA_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered[None, None, k_block_next],
                            tCrSFB_copy_view_filtered[None, None, k_block_next],
                        )

                # Hoist out last k_tile
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )

                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(
                            tCrSFA_tile_copy_view
                        )
                        tCrSFB_copy_view_filtered = cute.filter_zeros(
                            tCrSFB_tile_copy_view
                        )
                        cute.copy(
                            smem_tiled_copy_SFA,
                            tCsSFA_p_filtered[None, None, k_block_next],
                            tCrSFA_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_SFB,
                            tCsSFB_p_filtered[None, None, k_block_next],
                            tCrSFB_copy_view_filtered[None, None, k_block_next],
                        )
                    # Manual atom unroll: avoids hasAuxTensor address space bug
                    for _mt in range(self.num_m_tiles):
                        for _nt in range(self.num_n_tiles):
                            mma_atom.set(
                                WarpField.SFA,
                                tCrSFA_tile[None, _mt, k_block_idx].iterator,
                            )
                            mma_atom.set(
                                WarpField.SFB,
                                tCrSFB_tile[None, _nt, k_block_idx].iterator,
                            )
                            cute.gemm(
                                mma_atom,
                                accumulators[None, _mt, _nt],
                                tCrA[None, _mt, k_block_idx],
                                tCrB[None, _nt, k_block_idx],
                                accumulators[None, _mt, _nt],
                            )

                # EPILOGUE
                _is_m_major = self.c_layout.is_m_major_c()
                if cutlass.const_expr(self.c_dtype.width == 16):
                    copy_atom_r2s = cute.make_copy_atom(
                        cute.nvgpu.warp.StMatrix8x8x16bOp(_is_m_major, 2),
                        self.c_dtype,
                    )
                else:
                    copy_atom_r2s = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        self.c_dtype,
                    )

                copy_atom_C = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(
                        self.c_layout.is_m_major_c(),
                        2,
                    ),
                    self.c_dtype,
                )

                tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s,
                    tiled_copy_C_Atom,
                )

                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)

                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)

                epi_rest_m = 1
                epi_rest_n = 1
                epi_tile_m = self.epi_tile[0]
                epi_tile_n = self.epi_tile[1]
                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])
                has_multi_epi_store = cutlass.const_expr(
                    not (self.epi_stage == 1 and epi_rest_m == 1 and epi_rest_n == 1)
                )
                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    for epi_n in cutlass.range_constexpr(epi_rest_n):
                        MmaMPerEpiM = epi_tile_m // mma_tile_m
                        MmaNPerEpiN = epi_tile_n // mma_tile_n
                        for mma_n_in_epi in cutlass.range_constexpr(MmaNPerEpiN):
                            for mma_m_in_epi in cutlass.range_constexpr(MmaMPerEpiM):
                                mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi
                                mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi
                                tRS_rD_slice = tRS_rD[
                                    (None, mma_m_in_epi, mma_n_in_epi)
                                ]
                                tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    tRS_rD_slice[elem_idx] = tRS_rAcc_slice[elem_idx]

                        # Type conversion with alpha scaling
                        tRS_rD_out = cute.make_rmem_tensor(
                            tRS_rD_layout.shape, self.c_dtype
                        )
                        acc_vec = tRS_rD.load()
                        # Multiply alpha in FP32 before converting to c_dtype
                        # to avoid overflow when c_dtype is FP16
                        acc_vec = epilogue_op((alpha_value * acc_vec).to(self.c_dtype))
                        tRS_rD_out.store(acc_vec)

                        # Register to shared memory
                        epi_buffer = (epi_m * epi_rest_n + epi_n) % cute.size(
                            tRS_sD, mode=[3]
                        )
                        if has_multi_epi_store:
                            self.epilog_sync_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()

                        # Fused SwiGLU + NVFP4. Weight rows are [g0,u0,g1,u1,...].
                        groups_per_row = self.epi_tile[1] // 32
                        total_groups = self.epi_tile[0] * groups_per_row
                        for chunk in cutlass.range_constexpr(
                            (total_groups + self.num_mma_warps * 32 - 1)
                            // (self.num_mma_warps * 32)
                        ):
                            group = tidx + chunk * self.num_mma_warps * 32
                            if group < total_groups:
                                row = group // groups_per_row
                                block = group % groups_per_row
                                global_row = (
                                    tile_coord_mnl[0] * self.tile_shape_mnk[0]
                                    + epi_m * self.epi_tile[0]
                                    + row
                                )
                                global_block = (
                                    tile_coord_mnl[1] * self.tile_shape_mnk[1]
                                    + epi_n * self.epi_tile[1]
                                ) // 32 + block
                                if global_row < q.shape[0] and global_block * 16 < (
                                    q.shape[1] * 2
                                ):
                                    vals = cute.make_rmem_tensor((16,), cutlass.Float32)
                                    vmax = cutlass.Float32(0)
                                    for i in cutlass.range_constexpr(16):
                                        g = sC[
                                            (row, block * 32 + i * 2, epi_buffer)
                                        ].to(cutlass.Float32)
                                        u = sC[
                                            (row, block * 32 + i * 2 + 1, epi_buffer)
                                        ].to(cutlass.Float32)
                                        v = (
                                            (_silu(g) * u)
                                            .to(cutlass.BFloat16)
                                            .to(cutlass.Float32)
                                        )
                                        vals[i] = v
                                        vmax = cute.arch.fmax(vmax, cute.abs(v))
                                    sf_float = scale[0] * (
                                        vmax * _rcp(cutlass.Float32(6))
                                    )
                                    sf_local = cute.make_rmem_tensor(
                                        (1,), cutlass.Float8E4M3FN
                                    )
                                    sf_local[0] = sf_float.to(cutlass.Float8E4M3FN)
                                    sf_narrow = sf_local[0].to(cutlass.Float32)
                                    inv = cutlass.Float32(0)
                                    if vmax != 0:
                                        inv = _rcp(sf_narrow * _rcp(scale[0]))
                                    sf_offset = (
                                        (global_row // 128)
                                        * (((q.shape[1] * 2) + 63) // 64)
                                        * 512
                                        + (global_block // 4) * 512
                                        + (global_row % 32) * 16
                                        + ((global_row % 128) // 32) * 4
                                        + global_block % 4
                                    )
                                    sf[sf_offset] = cute.recast_tensor(
                                        sf_local, cutlass.Uint8
                                    )[0]
                                    packed_output = cute.make_tensor(
                                        cute.recast_ptr(
                                            q.iterator, dtype=cutlass.Uint64
                                        ),
                                        layout=cute.make_layout(
                                            q.shape[0] * (q.shape[1] // 8)
                                        ),
                                    )
                                    packed_output[
                                        global_row * (q.shape[1] // 8)
                                        + global_block * 1
                                        + 0
                                    ] = _pack_fp4_64(
                                        vals[0] * inv,
                                        vals[1] * inv,
                                        vals[2] * inv,
                                        vals[3] * inv,
                                        vals[4] * inv,
                                        vals[5] * inv,
                                        vals[6] * inv,
                                        vals[7] * inv,
                                        vals[8] * inv,
                                        vals[9] * inv,
                                        vals[10] * inv,
                                        vals[11] * inv,
                                        vals[12] * inv,
                                        vals[13] * inv,
                                        vals[14] * inv,
                                        vals[15] * inv,
                                    )
                        self.epilog_sync_barrier.arrive_and_wait()

                # Advance to the next work tile
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        # DMA warp group
        elif warp_idx == self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                tAgA_mkl = tAgA[(None, tile_coord_mnl[0], None, tile_coord_mnl[2])]
                tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, tile_coord_mnl[2])]
                sfa_tile_coord_m = tile_coord_mnl[0] // self.sfa_tiles_per_block
                tAgSFA_mkl = tAgSFA[(None, sfa_tile_coord_m, None, tile_coord_mnl[2])]
                sfb_tile_coord_n = tile_coord_mnl[1] // self.sfb_tiles_per_block
                tBgSFB_nkl = tBgSFB[(None, sfb_tile_coord_n, None, tile_coord_mnl[2])]

                mainloop_producer_state.reset_count()

                for _k_tile in range(0, k_tile_cnt, 1, unroll=2):  # type: ignore[call-overload]
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)

                    tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    tAgSFA_k = tAgSFA_mkl[(None, mainloop_producer_state.count)]
                    tAsSFA_pipe = tAsSFA[(None, mainloop_producer_state.index)]

                    tBgSFB_k = tBgSFB_nkl[(None, mainloop_producer_state.count)]
                    tBsSFB_pipe = tBsSFB[(None, mainloop_producer_state.index)]

                    cute.copy(
                        tma_atom_a,
                        tAgA_k,
                        tAsA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_k,
                        tAsSFA_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                            mainloop_producer_state
                        ),
                    )
                    mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)
        return

    @cute.jit
    def wrapper(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        q: cute.Tensor,
        sf: cute.Tensor,
        scale: cute.Tensor,
        sf_m: cutlass.Int64,
        sf_n: cutlass.Int64,
        sf_k: cutlass.Int64,
        l: cutlass.Constexpr,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        current_stream,
        swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Wrapper matching the SM100 compile interface."""
        m = cute.size(mA, mode=[0])
        k_raw = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])

        if cutlass.const_expr(
            mA.element_type == cutlass.Uint8 and mB.element_type == cutlass.Uint8
        ):
            k = k_raw * 2
            a_ptr = cute.recast_ptr(mA.iterator, dtype=cutlass.Float4E2M1FN)
            b_ptr = cute.recast_ptr(mB.iterator, dtype=cutlass.Float4E2M1FN)
        elif cutlass.const_expr(mA.element_type != mB.element_type):
            raise TypeError("Unsupported mixed input dtypes for block-scaled GEMM.")
        else:
            k = k_raw
            a_ptr = mA.iterator
            b_ptr = mB.iterator

        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)),
        )
        c_tensor = cute.make_identity_tensor((m, n, l))
        sfa_tensor = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_m, 4, sf_k, l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, sf_n, 4, sf_k, l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )

        self(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
            q,
            sf,
            scale,
            alpha_tensor,
            max_active_clusters,
            current_stream,
            epilogue_op,
        )

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple,
        epi_tile: tuple,
        a_dtype,
        a_layout,
        b_dtype,
        b_layout,
        ab_stage: int,
        c_dtype,
        c_layout,
        epi_stage: int,
        sf_vec_size: int,
        tiled_mma,
    ) -> tuple:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = a_layout.is_k_major_a()
        b_is_k_major = b_layout.is_k_major_b()
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]

        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        sfa_smem_layout_staged = sm120_make_smem_layout_sfa(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            ab_stage,
        )
        sfb_smem_layout_staged = sm120_make_smem_layout_sfb(
            tiled_mma,
            tile_shape_mnk,
            sf_vec_size,
            ab_stage,
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return (
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            epi_smem_layout_staged,
        )
