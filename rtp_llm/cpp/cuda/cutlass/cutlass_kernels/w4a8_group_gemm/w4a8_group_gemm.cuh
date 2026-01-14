#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/include/scalar_type.hpp"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/include/scaled_mm_epilogues_c3x.hpp"

using namespace cute;

#define CUTLASS_CHECK(status)                                                                                          \
    do {                                                                                                               \
        cutlass::Status error = (status);                                                                              \
        if (error != cutlass::Status::kSuccess) {                                                                      \
            std::stringstream msg;                                                                                     \
            msg << "Cutlass error: " << cutlassGetStatusString(error) << " at " << __FILE__ << ":" << __LINE__;        \
            throw std::runtime_error(msg.str());                                                                       \
        }                                                                                                              \
    } while (0)

template<typename ElementA, typename ElementB, typename ElementBScale, typename ElementD, typename ElementAccumulator>
__global__ void w4a8_group_gemm_starts(int32_t*                           expert_offsets,
                                       ElementA**                         a_offsets,
                                       ElementB**                         b_offsets,
                                       cutlass::Array<ElementBScale, 8>** b_scales_offsets,
                                       ElementD**                         out_offsets,
                                       ElementAccumulator**               a_out_scales_offsets,
                                       ElementAccumulator**               b_out_scales_offsets,
                                       ElementA*                          a_base_as_int,
                                       ElementB*                          b_base_as_int,
                                       cutlass::Array<ElementBScale, 8>*  b_scales_base_as_int,
                                       ElementD*                          out_base_as_int,
                                       ElementAccumulator*                a_out_scales_base_as_int,
                                       ElementAccumulator*                b_out_scales_base_as_int,
                                       int64_t                            n,
                                       int64_t                            k,
                                       const int                          num_experts,
                                       const int                          scale_k,
                                       const bool                         per_act_token,
                                       const bool                         per_out_ch) {
    int expert_id = threadIdx.x;
    if (expert_id >= num_experts) {
        return;
    }

    int64_t expert_offset = expert_offsets[expert_id];

    a_offsets[expert_id]            = a_base_as_int + expert_offset * k;
    b_offsets[expert_id]            = b_base_as_int + expert_id * n * k / 2;
    b_scales_offsets[expert_id]     = b_scales_base_as_int + expert_id * n * scale_k;
    out_offsets[expert_id]          = out_base_as_int + expert_offset * n;
    a_out_scales_offsets[expert_id] = a_out_scales_base_as_int + (per_act_token ? expert_offset : 0);
    b_out_scales_offsets[expert_id] = b_out_scales_base_as_int + (per_out_ch ? n * expert_id : expert_id);
}

#define __CALL_STARTS_KERNEL(OUTPUT_TYPE, D_TYPE)                                                                      \
    else if (output.dtype() == OUTPUT_TYPE) {                                                                          \
        w4a8_group_gemm_starts<cutlass::float_e4m3_t, cutlass::int4b_t, cutlass::float_e4m3_t, D_TYPE, float>          \
            <<<1, 1024, 0, stream>>>(                                                                                  \
                static_cast<int32_t*>(expert_offsets.data_ptr()),                                                      \
                static_cast<cutlass::float_e4m3_t**>(a_ptrs.data_ptr()),                                               \
                static_cast<cutlass::int4b_t**>(b_ptrs.data_ptr()),                                                    \
                static_cast<cutlass::Array<cutlass::float_e4m3_t, 8>**>(b_scales_ptrs.data_ptr()),                     \
                static_cast<D_TYPE**>(output_ptrs.data_ptr()),                                                         \
                static_cast<float**>(a_out_scales_ptrs.data_ptr()),                                                    \
                static_cast<float**>(b_out_scales_ptrs.data_ptr()),                                                    \
                static_cast<cutlass::float_e4m3_t*>(a.data_ptr()),                                                     \
                static_cast<cutlass::int4b_t*>(b.data_ptr()),                                                          \
                static_cast<cutlass::Array<cutlass::float_e4m3_t, 8>*>(b_scales.data_ptr()),                           \
                static_cast<D_TYPE*>(output.data_ptr()),                                                               \
                static_cast<float*>(a_out_scales.data_ptr()),                                                          \
                static_cast<float*>(b_out_scales.data_ptr()),                                                          \
                n,                                                                                                     \
                k,                                                                                                     \
                num_experts,                                                                                           \
                scale_k,                                                                                               \
                per_act_token,                                                                                         \
                per_out_ch);                                                                                           \
    }

template<typename Kernel>
struct enable_sm90_or_later: Kernel {
    template<typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900
        Kernel::operator()(std::forward<Args>(args)...);
#endif
    }
};

using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;

using ElementAccumulator = float;
using OperatorClass      = cutlass::arch::OpClassTensorOp;

using LayoutA           = cutlass::layout::RowMajor;
using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
using LayoutB           = cutlass::layout::ColumnMajor;
using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
using LayoutD           = cutlass::layout::RowMajor;
using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
using LayoutC           = LayoutD;
using LayoutC_Transpose = LayoutD_Transpose;

template<typename ElementA_,
         typename ElementB_,
         typename ElementBScale_,
         typename ElementC_,
         typename ArchTag_,
         template<typename, typename, typename> typename Epilogue_,
         typename TileShape,
         typename ClusterShape,
         typename KernelSchedule,
         typename EpilogueSchedule,
         bool SWAP_AB_ = false>
struct W4A8GroupGemm {
    static constexpr bool swap_ab = SWAP_AB_;
    using ElementA                = ElementA_;
    using ElementB                = ElementB_;
    using ElementBScale           = ElementBScale_;
    using ElementC                = ElementC_;
    using ElementD                = ElementC_;
    using ElementAccumulator      = float;
    using ArchTag                 = ArchTag_;

    using Epilogue = Epilogue_<ElementAccumulator, ElementD, TileShape>;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using EVTCompute = typename Epilogue::EVTCompute;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<ArchTag,
                                                                  OperatorClass,
                                                                  TileShape,
                                                                  ClusterShape,
                                                                  cutlass::epilogue::collective::EpilogueTileAuto,
                                                                  ElementAccumulator,
                                                                  ElementAccumulator,
                                                                  ElementC,
                                                                  conditional_t<swap_ab, LayoutC_Transpose*, LayoutC*>,
                                                                  AlignmentC,
                                                                  ElementD,
                                                                  conditional_t<swap_ab, LayoutD_Transpose*, LayoutD*>,
                                                                  AlignmentD,
                                                                  EpilogueSchedule,
                                                                  EVTCompute>::CollectiveOp;

    static constexpr size_t CEStorageSize = sizeof(typename CollectiveEpilogue::SharedStorage);
    using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(CEStorageSize)>;

    using CollectiveMainloop = conditional_t<
        swap_ab,
        typename cutlass::gemm::collective::CollectiveBuilder<ArchTag,
                                                              OperatorClass,
                                                              cute::tuple<ElementB, cutlass::Array<ElementBScale, 8>>,
                                                              LayoutB_Transpose*,
                                                              AlignmentB,
                                                              ElementA,
                                                              LayoutA_Transpose*,
                                                              AlignmentA,
                                                              ElementAccumulator,
                                                              TileShape,
                                                              ClusterShape,
                                                              Stages,
                                                              KernelSchedule>::CollectiveOp,
        typename cutlass::gemm::collective::CollectiveBuilder<ArchTag,
                                                              OperatorClass,
                                                              ElementA,
                                                              LayoutA*,
                                                              AlignmentA,
                                                              cute::tuple<ElementB, cutlass::Array<ElementBScale, 8>>,
                                                              LayoutB*,
                                                              AlignmentB,
                                                              ElementAccumulator,
                                                              TileShape,
                                                              ClusterShape,
                                                              Stages,
                                                              KernelSchedule>::CollectiveOp>;

    using StrideBScale = typename CollectiveMainloop::StrideScale;

    using KernelType = enable_sm90_or_later<
        cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>>;

    struct GemmKernel: public KernelType {};
};

void run_w4a8_group_gemm_starts(torch::Tensor const& expert_offsets,
                                torch::Tensor&       a_ptrs,
                                torch::Tensor&       b_ptrs,
                                torch::Tensor&       b_scales_ptrs,
                                torch::Tensor&       output_ptrs,
                                torch::Tensor&       a_out_scales_ptrs,
                                torch::Tensor&       b_out_scales_ptrs,
                                torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& b_scales,
                                torch::Tensor&       output,
                                torch::Tensor const& a_out_scales,
                                torch::Tensor const& b_out_scales,
                                const int            group_size,
                                const bool           per_act_token,
                                const bool           per_out_ch) {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kInt8);
    TORCH_CHECK(b_scales.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(a_out_scales.dtype() == torch::kFloat32);
    TORCH_CHECK(b_out_scales.dtype() == torch::kFloat32);

    int num_experts = static_cast<int>(expert_offsets.size(0));
    TORCH_CHECK(num_experts <= 1024, "Expert num must not be greater than 1024");
    auto stream = at::cuda::getCurrentCUDAStream(a.device().index());

    auto n       = output.size(1);
    auto k       = a.size(1);
    auto scale_k = cutlass::ceil_div(k, group_size);

    if (false) {}
    __CALL_STARTS_KERNEL(torch::kBFloat16, cutlass::bfloat16_t)
    __CALL_STARTS_KERNEL(torch::kFloat16, cutlass::half_t)
    else {
        TORCH_CHECK(false, "Invalid output type (must be float16 or bfloat16)");
    }
}
#undef __CALL_STARTS_KERNEL

template<typename Gemm>
void w4a8_group_gemm_caller(torch::Tensor&       output,
                            torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& b_scales,
                            torch::Tensor const& a_out_scales,
                            torch::Tensor const& b_out_scales,
                            torch::Tensor const& expert_offsets,
                            torch::Tensor const& problem_sizes,
                            torch::Tensor const& a_strides,
                            torch::Tensor const& b_strides,
                            torch::Tensor const& b_scales_strides,
                            torch::Tensor const& c_strides,
                            const int            group_size,
                            bool                 per_act_token,
                            bool                 per_out_ch) {
    static constexpr bool swap_ab = Gemm::swap_ab;

    using ElementA      = typename Gemm::ElementA;
    using ElementB      = typename Gemm::ElementB;
    using ElementBScale = typename Gemm::ElementBScale;
    using ElementC      = typename Gemm::ElementC;
    using ElementD      = typename Gemm::ElementD;

    int  num_experts = static_cast<int>(expert_offsets.size(0));
    auto stream      = at::cuda::getCurrentCUDAStream(a.device().index());
    auto options_int = torch::TensorOptions().dtype(torch::kInt64).device(a.device());

    torch::Tensor a_ptrs            = torch::empty(num_experts, options_int);
    torch::Tensor b_ptrs            = torch::empty(num_experts, options_int);
    torch::Tensor b_scales_ptrs     = torch::empty(num_experts, options_int);
    torch::Tensor output_ptrs       = torch::empty(num_experts, options_int);
    torch::Tensor a_out_scales_ptrs = torch::empty(num_experts, options_int);
    torch::Tensor b_out_scales_ptrs = torch::empty(num_experts, options_int);

    run_w4a8_group_gemm_starts(expert_offsets,
                               a_ptrs,
                               b_ptrs,
                               b_scales_ptrs,
                               output_ptrs,
                               a_out_scales_ptrs,
                               b_out_scales_ptrs,
                               a,
                               b,
                               b_scales,
                               output,
                               a_out_scales,
                               b_out_scales,
                               group_size,
                               per_act_token,
                               per_out_ch);

    using GemmKernel   = typename Gemm::GemmKernel;
    using StrideA      = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
    using StrideB      = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;
    using StrideBScale = typename Gemm::StrideBScale;
    using StrideC      = typename GemmKernel::InternalStrideC;
    using StrideD      = typename GemmKernel::InternalStrideD;

    ProblemShape::UnderlyingProblemShape* problem_sizes_as_shapes =
        static_cast<ProblemShape::UnderlyingProblemShape*>(problem_sizes.data_ptr());
    ProblemShape prob_shape{num_experts, problem_sizes_as_shapes, nullptr};

    typename GemmKernel::MainloopArguments mainloop_args;
    if constexpr (swap_ab) {
        mainloop_args = typename GemmKernel::MainloopArguments{
            static_cast<const ElementB**>(b_ptrs.data_ptr()),
            static_cast<StrideB*>(b_strides.data_ptr()),
            static_cast<const ElementA**>(a_ptrs.data_ptr()),
            static_cast<StrideA*>(a_strides.data_ptr()),
            static_cast<const cutlass::Array<ElementBScale, 8>**>(b_scales_ptrs.data_ptr()),
            static_cast<StrideBScale*>(b_scales_strides.data_ptr()),
            group_size};
    } else {
        mainloop_args = typename GemmKernel::MainloopArguments{
            static_cast<const ElementA**>(a_ptrs.data_ptr()),
            static_cast<StrideA*>(a_strides.data_ptr()),
            static_cast<const ElementB**>(b_ptrs.data_ptr()),
            static_cast<StrideB*>(b_strides.data_ptr()),
            static_cast<const cutlass::Array<ElementBScale, 8>**>(b_scales_ptrs.data_ptr()),
            static_cast<StrideBScale*>(b_scales_strides.data_ptr()),
            group_size};
    }

    // Currently, we are only able to do broadcast on either all or none a_out_scales
    // and on either all or none b_out_scales
    typename GemmKernel::EpilogueArguments epilogue_args{
        Gemm::Epilogue::prepare_args(swap_ab ? static_cast<const ElementAccumulator**>(b_out_scales_ptrs.data_ptr()) :
                                               static_cast<const ElementAccumulator**>(a_out_scales_ptrs.data_ptr()),
                                     swap_ab ? static_cast<const ElementAccumulator**>(a_out_scales_ptrs.data_ptr()) :
                                               static_cast<const ElementAccumulator**>(b_out_scales_ptrs.data_ptr()),
                                     swap_ab ? per_out_ch : per_act_token,
                                     swap_ab ? per_act_token : per_out_ch),
        nullptr,
        static_cast<StrideC*>(c_strides.data_ptr()),
        static_cast<ElementD**>(output_ptrs.data_ptr()),
        static_cast<StrideD*>(c_strides.data_ptr())};

    int                                      device_id = a.device().index();
    static const cutlass::KernelHardwareInfo hw_info{
        device_id, cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id)};

    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped, prob_shape, mainloop_args, epilogue_args, hw_info};

    using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    GemmOp gemm_op;
    CUTLASS_CHECK(gemm_op.can_implement(args));

    size_t     workspace_size    = gemm_op.get_workspace_size(args);
    auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto       workspace         = torch::empty(workspace_size, workspace_options);

    CUTLASS_CHECK(gemm_op.run(args, workspace.data_ptr(), stream));
}
