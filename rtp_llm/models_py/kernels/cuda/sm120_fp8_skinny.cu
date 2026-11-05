// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

// KDA provenance: this kernel was automatically optimized by the Humanize2
// workflow (https://github.com/PolyArch/humanize) and Kernel Design Agents
// (https://github.com/mit-han-lab/kernel-design-agents).

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software without
// specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Adapted from SGLang 80afdc1280bf37c1440e6e7009741cbc94e2fba0.
// Preserve Torch row/column scale ordering with device-resident vector scales.

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include <cuda_runtime.h>

namespace rtp_sm120_fp8_skinny {

using namespace cute;

using ElementInput       = cutlass::float_e4m3_t;
using ElementOutput      = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassTensorOp;

using ClusterShape            = Shape<_1, _1, _1>;
using LayoutA                 = cutlass::layout::RowMajor;
using LayoutB                 = cutlass::layout::ColumnMajor;
using LayoutC                 = cutlass::layout::ColumnMajor;
using LayoutD                 = cutlass::layout::ColumnMajor;
constexpr int AlignmentInput  = 16;
constexpr int AlignmentOutput = 8;

using Accum    = cutlass::epilogue::fusion::Sm90AccFetch;
using Multiply = cutlass::epilogue::fusion::
    Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;

// D.T = W @ A.T moves the tiny token dimension to the narrow N axis. A
// token tile reduces padded MMA work for small batches. The generic
// warp-specialized schedule needs only one MMA consumer group for this skinny
// tile; unlike ping-pong, it compiles to 256 threads without register spills.
template<int TileM, int TokenTile, int TileK, bool CacheHints = false>
struct Fp8Gemm {
    using TileShape       = Shape<Int<TileM>, Int<TokenTile>, Int<TileK>>;
    using ActivationScale = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float, float, Stride<_0, _1, _0>>;
    using WeightScale     = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float, Stride<_1, _0, _0>>;
    using ScaledActivation = cutlass::epilogue::fusion::Sm90EVT<Multiply, ActivationScale, Accum>;
    using ScaleOutput      = cutlass::epilogue::fusion::Sm90EVT<Multiply, WeightScale, ScaledActivation>;

    // Preserve the tuned mainloop's stage count by reserving the same carveout
    // used by the former TMA epilogue.  Only the executed epilogue changes.
    using StagingEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag,
                                                                  OperatorClass,
                                                                  TileShape,
                                                                  ClusterShape,
                                                                  Shape<Int<TileM>, Int<TokenTile>>,
                                                                  ElementAccumulator,
                                                                  float,
                                                                  void,
                                                                  LayoutC,
                                                                  AlignmentOutput,
                                                                  ElementOutput,
                                                                  LayoutD,
                                                                  AlignmentOutput,
                                                                  cutlass::epilogue::collective::EpilogueScheduleAuto,
                                                                  ScaleOutput>::CollectiveOp;
    using CollectiveEpilogue = StagingEpilogue;
    using StageCount         = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>;
    // The SM120 builder currently exposes ping-pong/cooperative schedules only.
    // Use it as a type/layout factory, then rebuild the collective with the
    // generic one-consumer-group dispatch policy below.
    using BuiltMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementInput,
        LayoutA,
        AlignmentInput,
        ElementInput,
        LayoutB,
        AlignmentInput,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        StageCount,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
    // CUTLASS's generic SM120 builder selects ldmatrix.x4 for both operands.
    // A 16-wide B tile contains only enough values for ldmatrix.x2, so retain
    // every generated policy/layout while narrowing just that shared-to-register
    // copy atom.
    using MainloopDispatch       = cutlass::gemm::MainloopSm120TmaWarpSpecialized<BuiltMainloop::DispatchPolicy::Stages,
                                                                                  2,
                                                                                  ClusterShape,
                                                                                  cutlass::gemm::KernelTmaWarpSpecialized>;
    using SmemLayoutAtomB        = cute::UMMA::Layout_K_SW128_Atom<uint8_t>;
    using BaseCollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
        MainloopDispatch,
        typename BuiltMainloop::TileShape,
        typename BuiltMainloop::ElementA,
        typename BuiltMainloop::StrideA,
        typename BuiltMainloop::ElementB,
        typename BuiltMainloop::StrideB,
        typename BuiltMainloop::TiledMma,
        typename BuiltMainloop::GmemTiledCopyA,
        typename BuiltMainloop::SmemLayoutAtomA,
        typename BuiltMainloop::SmemCopyAtomA,
        typename BuiltMainloop::TransformA,
        typename BuiltMainloop::GmemTiledCopyB,
        SmemLayoutAtomB,
        cute::Copy_Atom<cute::conditional_t<TokenTile == 16, cute::SM75_U32x2_LDSM_N, cute::SM75_U32x4_LDSM_N>,
                        uint8_t>,
        typename BuiltMainloop::TransformB>;
    // The generic GemmUniversal kernel calls the legacy mma overload. SM120's
    // collective also accepts a block coordinate for block-scaled variants; this
    // scalar-scale kernel does not use it, so bridge the two interfaces.
    struct CollectiveMainloop: BaseCollectiveMainloop {
        template<class TensorA, class TensorB, class KTileIterator, class BlockCoord>
        CUTLASS_DEVICE void load(typename BaseCollectiveMainloop::Params const&    mainloop_params,
                                 typename BaseCollectiveMainloop::MainloopPipeline pipeline,
                                 typename BaseCollectiveMainloop::PipelineState    smem_pipe_write,
                                 cute::tuple<TensorA, TensorB> const&              load_inputs,
                                 BlockCoord const&                                 block_coord,
                                 KTileIterator                                     k_tile_iter,
                                 int                                               k_tile_count,
                                 int                                               thread_idx,
                                 uint32_t                                          block_rank_in_cluster,
                                 typename BaseCollectiveMainloop::TensorStorage&   shared_tensors) {
            if constexpr (!CacheHints) {
                BaseCollectiveMainloop::load(mainloop_params,
                                             pipeline,
                                             smem_pipe_write,
                                             load_inputs,
                                             block_coord,
                                             k_tile_iter,
                                             k_tile_count,
                                             thread_idx,
                                             block_rank_in_cluster,
                                             shared_tensors);
                return;
            }

            if (cute::elect_one_sync()) {
                auto shared_a = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_A.data()),
                                                  typename BaseCollectiveMainloop::SmemLayoutA{});
                auto shared_b = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_B.data()),
                                                  typename BaseCollectiveMainloop::SmemLayoutB{});

                // This kernel always uses a 1x1 cluster, so the TMA slice is fixed.
                auto block_tma_a                          = mainloop_params.tma_load_a.get_slice(0);
                auto block_tma_b                          = mainloop_params.tma_load_b.get_slice(0);
                auto [m_coord, n_coord, k_coord, l_coord] = block_coord;
                auto global_a       = cute::get<0>(load_inputs)(cute::_, cute::_, m_coord, cute::_, l_coord);
                auto global_b       = cute::get<1>(load_inputs)(cute::_, cute::_, n_coord, cute::_, l_coord);
                auto tiled_global_a = block_tma_a.partition_S(global_a);
                auto tiled_shared_a = block_tma_a.partition_D(shared_a);
                auto tiled_global_b = block_tma_b.partition_S(global_b);
                auto tiled_shared_b = block_tma_b.partition_D(shared_b);

                CUTLASS_PRAGMA_NO_UNROLL
                for (; k_tile_count > 0; --k_tile_count) {
                    pipeline.producer_acquire(smem_pipe_write);
                    // PipelineTmaAsync::producer_acquire registers transaction_bytes;
                    // TMA completes the barrier, so producer_commit is a no-op here.
                    using Barrier     = typename BaseCollectiveMainloop::MainloopPipeline::ProducerBarrierType;
                    Barrier*  barrier = pipeline.producer_get_barrier(smem_pipe_write);
                    const int stage   = smem_pipe_write.index();
                    // Stream weights with evict-first priority so they do not displace
                    // the activation rows reused by every output tile. Keep the
                    // existing device-resident input and scale layouts.
                    cute::copy(mainloop_params.tma_load_a.with(*barrier, 0, cute::TMA::CacheHintSm90::EVICT_FIRST),
                               tiled_global_a(cute::_, cute::_, cute::_, *k_tile_iter),
                               tiled_shared_a(cute::_, cute::_, cute::_, stage));
                    cute::copy(mainloop_params.tma_load_b.with(*barrier, 0, cute::TMA::CacheHintSm90::EVICT_NORMAL),
                               tiled_global_b(cute::_, cute::_, cute::_, *k_tile_iter),
                               tiled_shared_b(cute::_, cute::_, cute::_, stage));
                    ++k_tile_iter;
                    ++smem_pipe_write;
                }
            }
        }

        template<class FrgTensorC>
        CUTLASS_DEVICE void mma(typename BaseCollectiveMainloop::MainloopPipeline pipeline,
                                typename BaseCollectiveMainloop::PipelineState    state,
                                FrgTensorC&                                       accumulators,
                                int                                               k_tile_count,
                                int                                               thread_idx,
                                typename BaseCollectiveMainloop::TensorStorage&   shared_tensors,
                                typename BaseCollectiveMainloop::Params const&    params) {
            BaseCollectiveMainloop::mma(
                pipeline, state, accumulators, k_tile_count, thread_idx, shared_tensors, params);
        }
    };
    using Kernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template<int TileM, int TokenTile, int TileK, bool CacheHints = false>
void run(void*        output,
         const void*  activation,
         const void*  weight,
         const void*  scale_a,
         const void*  scale_b,
         int          n,
         int          k,
         cudaStream_t stream,
         int          m = 9) {
    using GemmKernel = typename Fp8Gemm<TileM, TokenTile, TileK, CacheHints>::Kernel;
    using Gemm       = typename Fp8Gemm<TileM, TokenTile, TileK, CacheHints>::Device;
    using StrideA    = typename GemmKernel::StrideA;
    using StrideB    = typename GemmKernel::StrideB;
    using StrideD    = typename GemmKernel::StrideD;

    StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(n, k, 1));
    StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(m, k, 1));
    StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(n, m, 1));

    typename GemmKernel::MainloopArguments mainloop_args{
        static_cast<const ElementInput*>(weight), stride_a, static_cast<const ElementInput*>(activation), stride_b};
    auto*                                  result = static_cast<ElementOutput*>(output);
    typename GemmKernel::EpilogueArguments epilogue_args{
        {{static_cast<const float*>(scale_b), 0.f, {}}, {{static_cast<const float*>(scale_a), 0.f, {}}, {}, {}}, {}},
        nullptr,
        stride_d,
        result,
        stride_d};

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1}, mainloop_args, epilogue_args};

    Gemm gemm;
    TVM_FFI_ICHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess)
        << "KDA SM120 FP8 GEMM cannot implement M=" << m << ", N=" << n << ", K=" << k;
    TVM_FFI_ICHECK(gemm.initialize(arguments, nullptr, stream) == cutlass::Status::kSuccess)
        << "KDA SM120 FP8 GEMM initialization failed";
    TVM_FFI_ICHECK(gemm.run(stream, nullptr, false) == cutlass::Status::kSuccess) << "KDA SM120 FP8 GEMM launch failed";
}

// Fixed choices for the qualified projection sizes; scales stay device operands.
void launch_gemm(void*        output,
                 const void*  activation,
                 const void*  weight,
                 const void*  scale_a,
                 const void*  scale_b,
                 int          m,
                 int          n,
                 int          k,
                 cudaStream_t stream) {
    if (m <= 16) {
        if (n == 3072 || n == 8192)
            return run<32, 16, 512, true>(output, activation, weight, scale_a, scale_b, n, k, stream, m);
        return run<64, 16, 512, true>(output, activation, weight, scale_a, scale_b, n, k, stream, m);
    }
    if (m <= 32) {
        if (n == 3072 || n == 8192)
            return run<32, 32, 512, true>(output, activation, weight, scale_a, scale_b, n, k, stream, m);
        return run<64, 32, 128, true>(output, activation, weight, scale_a, scale_b, n, k, stream, m);
    }
    return run<64, 64, 128>(output, activation, weight, scale_a, scale_b, n, k, stream, m);
}
}  // namespace rtp_sm120_fp8_skinny

using tvm::ffi::TensorView;
void run_fp8(TensorView x, TensorView w, TensorView a, TensorView b, TensorView out) {
    tvm::ffi::CUDADeviceGuard guard(x.device().device_id);
    rtp_sm120_fp8_skinny::launch_gemm(
        out.data_ptr(),
        x.data_ptr(),
        w.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        x.size(0),
        w.size(0),
        w.size(1),
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(x.device().device_type, x.device().device_id)));
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run_fp8);
