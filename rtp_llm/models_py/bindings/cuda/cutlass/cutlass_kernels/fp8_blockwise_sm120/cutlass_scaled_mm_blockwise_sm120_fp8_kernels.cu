// SPDX-License-Identifier: Apache-2.0
//
// FP8 PER_BLOCK GEMM for sm_120 family (consumer Blackwell).
//
// Adapted from vLLM (Apache 2.0):
//   csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/
//     scaled_mm_blockwise_sm120_fp8.cu
//     scaled_mm_blockwise_sm120_fp8_dispatch.cuh
//   csrc/cutlass_extensions/common.hpp
// Source: https://github.com/vllm-project/vllm  commit 6f955986e (2026-05-24)
//
// Modifications for RTP-LLM:
//   - Replace torch::stable / torch::Library with torch::Tensor (eager) + a
//     C++ wrapper compatible with the existing fp4_gemm wiring style.
//   - Replace torch::stable::empty workspace with torch::empty (fp4 pattern).
//   - Inline enable_sm120_family<> instead of vendoring cutlass_extensions/.
//
// Bench (RTX PRO 5000 72GB Blackwell, sm120-upgrade/bench/vllm_pb_port,
// 2026-05-26): 442 TFLOPs at 4096^3, 1.78x BF16 baseline. See
// sm120-upgrade/deep-dives/fp8-gemm-sm120.md §11.6.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <type_traits>

#include "cutlass_scaled_mm_blockwise_sm120_fp8.h"

// clang-format off
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"
// clang-format on

#define CUTLASS_CHECK(status)                                                                                          \
    {                                                                                                                  \
        cutlass::Status error = (status);                                                                              \
        TORCH_CHECK(error == cutlass::Status::kSuccess, cutlassGetStatusString(error));                                \
    }

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

using namespace cute;

namespace {

int64_t ceil_div(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

void check_cuda_same_device(torch::Tensor const& tensor, char const* name, c10::Device const& device) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.device() == device, name, " must be on the same device as A, got ", tensor.device(), " vs ", device);
}

void check_contiguous(torch::Tensor const& tensor, char const* name) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// SM12x family CUDA_ARCH gate (verbatim from vllm cutlass_extensions/common.hpp)
template<typename Kernel>
struct enable_sm120_family: Kernel {
    template<typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__
#if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
        Kernel::operator()(std::forward<Args>(args)...);
#else
        printf("FP8 PER_BLOCK SM120 kernel only supports sm120f.\n");
        asm("trap;");
#endif
#endif
    }
};

// GEMM type factory (verbatim from vllm scaled_mm_blockwise_sm120_fp8_dispatch.cuh)
template<class OutType,
         int ScaleGranularityM,
         int ScaleGranularityN,
         int ScaleGranularityK,
         class MmaTileShape,
         class ClusterShape,
         class EpilogueScheduler,
         class MainloopScheduler,
         bool swap_ab_ = false>
struct cutlass_3x_gemm_fp8_blockwise {
    static constexpr bool swap_ab = swap_ab_;
    using ElementAB               = cutlass::float_e4m3_t;

    using ElementA                  = ElementAB;
    using LayoutA                   = cutlass::layout::RowMajor;
    using LayoutA_Transpose         = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    using ElementB                  = ElementAB;
    using LayoutB                   = cutlass::layout::ColumnMajor;
    using LayoutB_Transpose         = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    using ElementD                  = OutType;
    using LayoutD                   = cutlass::layout::RowMajor;
    using LayoutD_Transpose         = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using ElementC                  = void;
    using LayoutC                   = LayoutD;
    using LayoutC_Transpose         = LayoutD_Transpose;
    static constexpr int AlignmentC = AlignmentD;

    using ElementAccumulator = float;
    using ElementCompute     = float;
    using ElementBlockScale  = float;

    using ScaleConfig = std::conditional_t<swap_ab,
                                           cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM,
                                                                                      ScaleGranularityN,
                                                                                      ScaleGranularityK,
                                                                                      cute::UMMA::Major::K,
                                                                                      cute::UMMA::Major::MN>,
                                           cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM,
                                                                                      ScaleGranularityN,
                                                                                      ScaleGranularityK,
                                                                                      cute::UMMA::Major::MN,
                                                                                      cute::UMMA::Major::K>>;

    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    using ArchTag       = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    using ElementScalar              = float;
    using DefaultOperation =
        cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementCompute,
        ElementC,
        std::conditional_t<swap_ab, LayoutC_Transpose, LayoutC>,
        AlignmentC,
        ElementD,
        std::conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
        AlignmentD,
        EpilogueScheduler,
        DefaultOperation>::CollectiveOp;

    using StageCountType = cutlass::gemm::collective::StageCountAuto;
    using CollectiveMainloop =
        std::conditional_t<swap_ab,
                           typename cutlass::gemm::collective::CollectiveBuilder<
                               ArchTag,
                               OperatorClass,
                               ElementB,
                               cute::tuple<LayoutB_Transpose, LayoutSFA>,
                               AlignmentB,
                               ElementA,
                               cute::tuple<LayoutA_Transpose, LayoutSFB>,
                               AlignmentA,
                               ElementAccumulator,
                               MmaTileShape,
                               ClusterShape,
                               cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                                   sizeof(typename CollectiveEpilogue::SharedStorage))>,
                               MainloopScheduler>::CollectiveOp,
                           typename cutlass::gemm::collective::CollectiveBuilder<
                               ArchTag,
                               OperatorClass,
                               ElementA,
                               cute::tuple<LayoutA, LayoutSFA>,
                               AlignmentA,
                               ElementB,
                               cute::tuple<LayoutB, LayoutSFB>,
                               AlignmentB,
                               ElementAccumulator,
                               MmaTileShape,
                               ClusterShape,
                               cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                                   sizeof(typename CollectiveEpilogue::SharedStorage))>,
                               MainloopScheduler>::CollectiveOp>;

    using KernelType = enable_sm120_family<
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;

    struct GemmKernel: public KernelType {};
};

// Tile configurations (verbatim from vllm dispatch.cuh)
template<typename OutType>
struct sm120_blockwise_fp8_config_default {
    using KernelSchedule   = cutlass::gemm::collective::KernelScheduleAuto;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileShape        = Shape<_128, _128, _128>;
    using ClusterShape     = Shape<_1, _1, _1>;
    using Gemm =
        cutlass_3x_gemm_fp8_blockwise<OutType, 1, 128, 128, TileShape, ClusterShape, EpilogueSchedule, KernelSchedule>;
};

template<typename OutType>
struct sm120_blockwise_fp8_config_pingpong {
    using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileShape        = Shape<_64, _128, _128>;
    using ClusterShape     = Shape<_1, _1, _1>;
    using Gemm =
        cutlass_3x_gemm_fp8_blockwise<OutType, 1, 128, 128, TileShape, ClusterShape, EpilogueSchedule, KernelSchedule>;
};

template<typename OutType>
struct sm120_blockwise_fp8_config_swapab {
    using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileShape        = Shape<_128, _32, _128>;
    using ClusterShape     = Shape<_1, _1, _1>;
    using Gemm             = cutlass_3x_gemm_fp8_blockwise<OutType,
                                                           128,
                                                           1,
                                                           128,
                                                           TileShape,
                                                           ClusterShape,
                                                           EpilogueSchedule,
                                                           KernelSchedule,
                                                           true>;
};

// Launcher (port of vllm cutlass_gemm_caller_blockwise; uses torch::empty for
// workspace per the existing fp4_gemm pattern).
template<typename Gemm>
void launch_one(torch::Tensor&       D,
                torch::Tensor const& A,
                torch::Tensor const& B,
                torch::Tensor const& A_sf,
                torch::Tensor const& B_sf,
                int                  M,
                int                  N,
                int                  K,
                cudaStream_t         stream) {
    static constexpr bool swap_ab = Gemm::swap_ab;
    using GemmKernel              = typename Gemm::GemmKernel;
    using StrideA                 = typename GemmKernel::StrideA;
    using StrideB                 = typename GemmKernel::StrideB;
    using StrideC                 = typename GemmKernel::StrideC;
    using LayoutSFA               = typename Gemm::LayoutSFA;
    using LayoutSFB               = typename Gemm::LayoutSFB;
    using ScaleConfig             = typename Gemm::ScaleConfig;

    using ElementAB         = typename Gemm::ElementAB;
    using ElementD          = typename Gemm::ElementD;
    using ElementBlockScale = typename Gemm::ElementBlockScale;

    StrideA a_stride = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB b_stride = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC c_stride =
        cutlass::make_cute_packed_stride(StrideC{}, swap_ab ? cute::make_shape(N, M, 1) : cute::make_shape(M, N, 1));

    LayoutSFA layout_SFA = swap_ab ? ScaleConfig::tile_atom_to_shape_SFA(make_shape(N, M, K, 1)) :
                                     ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = swap_ab ? ScaleConfig::tile_atom_to_shape_SFB(make_shape(N, M, K, 1)) :
                                     ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    auto a_e = static_cast<ElementAB const*>(A.const_data_ptr());
    auto b_e = static_cast<ElementAB const*>(B.const_data_ptr());
    auto sa  = static_cast<ElementBlockScale const*>(A_sf.const_data_ptr());
    auto sb  = static_cast<ElementBlockScale const*>(B_sf.const_data_ptr());

    typename GemmKernel::MainloopArguments mainloop_args{};
    mainloop_args.layout_SFA = layout_SFA;
    mainloop_args.layout_SFB = layout_SFB;
    if (swap_ab) {
        mainloop_args.ptr_A   = b_e;
        mainloop_args.dA      = b_stride;
        mainloop_args.ptr_B   = a_e;
        mainloop_args.dB      = a_stride;
        mainloop_args.ptr_SFA = sb;
        mainloop_args.ptr_SFB = sa;
    } else {
        mainloop_args.ptr_A   = a_e;
        mainloop_args.dA      = a_stride;
        mainloop_args.ptr_B   = b_e;
        mainloop_args.dB      = b_stride;
        mainloop_args.ptr_SFA = sa;
        mainloop_args.ptr_SFB = sb;
    }
    auto prob_shape = swap_ab ? cute::make_shape(N, M, K, 1) : cute::make_shape(M, N, K, 1);

    auto                                   cd = static_cast<ElementD*>(D.data_ptr());
    typename GemmKernel::EpilogueArguments epilogue_args{{}, cd, c_stride, cd, c_stride};

    cutlass::KernelHardwareInfo    hw_info;
    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args, epilogue_args, hw_info, {}};

    using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    GemmOp gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(args));

    size_t     workspace_size    = gemm_op.get_workspace_size(args);
    auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
    auto       workspace         = torch::empty(static_cast<int64_t>(workspace_size), workspace_options);

    CUTLASS_CHECK(gemm_op.run(args, workspace.data_ptr(), stream));
}

// M-tier dispatch + M<=64 swap-AB heuristic (verbatim from vllm
// cutlass_gemm_blockwise_sm120_fp8_dispatch).
template<typename OutType>
void dispatch_blockwise_sm120(torch::Tensor&       D,
                              torch::Tensor const& A,
                              torch::Tensor const& B,
                              torch::Tensor const& A_sf,
                              torch::Tensor const& B_sf,
                              int                  M,
                              int                  N,
                              int                  K,
                              cudaStream_t         stream) {
    bool swap_ab = (M <= 64) || (M % 4 != 0);
    if (!swap_ab) {
        if (M <= 256) {
            launch_one<typename sm120_blockwise_fp8_config_pingpong<OutType>::Gemm>(
                D, A, B, A_sf, B_sf, M, N, K, stream);
        } else {
            launch_one<typename sm120_blockwise_fp8_config_default<OutType>::Gemm>(
                D, A, B, A_sf, B_sf, M, N, K, stream);
        }
    } else {
        launch_one<typename sm120_blockwise_fp8_config_swapab<OutType>::Gemm>(D, A, B, A_sf, B_sf, M, N, K, stream);
    }
}

}  // anonymous namespace

#endif  // CUTLASS_ARCH_MMA_SM120_SUPPORTED

void cutlass_scaled_mm_blockwise_sm120_fp8(torch::Tensor&       D,
                                           torch::Tensor const& A,
                                           torch::Tensor const& B,
                                           torch::Tensor const& A_sf,
                                           torch::Tensor const& B_sf) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    auto device = A.device();
    check_cuda_same_device(B, "B", device);
    check_cuda_same_device(D, "D", device);
    check_cuda_same_device(A_sf, "A_sf", device);
    check_cuda_same_device(B_sf, "B_sf", device);

    check_contiguous(A, "A");
    check_contiguous(B, "B");
    check_contiguous(D, "D");

    TORCH_CHECK(A.dtype() == torch::kFloat8_e4m3fn, "A must be float8_e4m3fn");
    TORCH_CHECK(B.dtype() == torch::kFloat8_e4m3fn, "B must be float8_e4m3fn");
    TORCH_CHECK(A_sf.dtype() == torch::kFloat32, "A_sf must be float32");
    TORCH_CHECK(B_sf.dtype() == torch::kFloat32, "B_sf must be float32");
    TORCH_CHECK(D.dtype() == at::ScalarType::BFloat16 || D.dtype() == at::ScalarType::Half,
                "D must be bfloat16 or float16, got ",
                D.dtype());
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2,
                "A/B/D must be 2D, got dims A=",
                A.dim(),
                ", B=",
                B.dim(),
                ", D=",
                D.dim());

    int M = static_cast<int>(A.size(0));
    int K = static_cast<int>(A.size(1));
    int N = static_cast<int>(B.size(1));
    TORCH_CHECK(B.size(0) == K, "B.size(0) (", B.size(0), ") must equal K (", K, ")");
    TORCH_CHECK(
        D.size(0) == M && D.size(1) == N, "D shape (", D.size(0), ",", D.size(1), ") must be (M=", M, ", N=", N, ")");

    int64_t scale_k = ceil_div(K, 128);
    int64_t scale_n = ceil_div(N, 128);
    TORCH_CHECK(A_sf.dim() == 2, "A_sf must be 2D, got ", A_sf.dim(), "D");
    TORCH_CHECK(A_sf.size(0) == M && A_sf.size(1) == scale_k,
                "A_sf shape (",
                A_sf.size(0),
                ",",
                A_sf.size(1),
                ") must be (M=",
                M,
                ", ceil_div(K, 128)=",
                scale_k,
                ")");
    TORCH_CHECK(A_sf.stride(0) == 1 && A_sf.stride(1) == M,
                "A_sf must use MN-major scale layout with stride (1, M), got (",
                A_sf.stride(0),
                ",",
                A_sf.stride(1),
                ")");
    TORCH_CHECK(B_sf.dim() == 2, "B_sf must be 2D, got ", B_sf.dim(), "D");
    TORCH_CHECK(B_sf.size(0) == scale_k && B_sf.size(1) == scale_n,
                "B_sf shape (",
                B_sf.size(0),
                ",",
                B_sf.size(1),
                ") must be (ceil_div(K, 128)=",
                scale_k,
                ", ceil_div(N, 128)=",
                scale_n,
                ")");
    TORCH_CHECK(B_sf.stride(0) == scale_n && B_sf.stride(1) == 1,
                "B_sf must use contiguous scale layout with stride (ceil_div(N, 128), 1), got (",
                B_sf.stride(0),
                ",",
                B_sf.stride(1),
                ")");

    at::cuda::CUDAGuard device_guard{(char)A.get_device()};
    cudaStream_t        stream = at::cuda::getCurrentCUDAStream(A.get_device());

    auto out_dtype = D.dtype();
    if (out_dtype == at::ScalarType::BFloat16) {
        dispatch_blockwise_sm120<cutlass::bfloat16_t>(D, A, B, A_sf, B_sf, M, N, K, stream);
    } else if (out_dtype == at::ScalarType::Half) {
        dispatch_blockwise_sm120<cutlass::half_t>(D, A, B, A_sf, B_sf, M, N, K, stream);
    } else {
        TORCH_CHECK(false, "Unsupported output dtype for fp8 blockwise sm120: ", out_dtype);
    }
#else
    TORCH_CHECK(false,
                "cutlass_scaled_mm_blockwise_sm120_fp8 was not compiled with "
                "CUTLASS_ARCH_MMA_SM120_SUPPORTED. Rebuild with --config=cuda12_9.");
#endif
}
