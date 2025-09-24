#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/device_memory.h"

#include "cutlass_extensions/gemm/kernel/group_gemm_traits.h"

#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/group_gemm/group_gemm.h"
#include "rtp_llm/cpp/cuda/cuda_host_utils.h"
namespace rtp_llm {

template <typename dtype, typename arch>
struct GroupGemmTraits;

template <typename arch>
struct GroupGemmTraits<cutlass::half_t, arch> {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using ThreadWrapShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using OpClassTensorOp = cutlass::arch::OpClassTensorOp;
    using GroupedGemmConfig = cutlass::gemm::device::DefaultGemmConfiguration<
                                        cutlass::arch::OpClassTensorOp,
                                        arch,
                                        cutlass::half_t,
                                        cutlass::half_t,
                                        cutlass::half_t,
                                        float>;
    using EpilogueOutputOp = typename GroupedGemmConfig::EpilogueOutputOp;
    static const int kStages = GroupedGemmConfig::kStages;
    static const int kAlignmentA = 8;
    static const int kAlignmentB = 8;
};

template<>
struct GroupGemmTraits<cutlass::half_t, cutlass::arch::Sm70> {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using ThreadWrapShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::kernel::GroupGemmArchTraits<cutlass::arch::Sm70>::InstructionShape;
    using OpClassTensorOp = cutlass::arch::OpClassTensorOp;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value, float, float>;
    static const int kStages = cutlass::gemm::kernel::GroupGemmArchTraits<cutlass::arch::Sm70>::Stages;
    static const int kAlignmentA = 8;
    static const int kAlignmentB = 8;
};

template <typename arch>
struct GroupGemmTraits<cutlass::bfloat16_t, arch> {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using ThreadWrapShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
    using OpClassTensorOp = cutlass::arch::OpClassTensorOp;
    using GroupedGemmConfig = cutlass::gemm::device::DefaultGemmConfiguration<
                                        cutlass::arch::OpClassTensorOp,
                                        arch,
                                        cutlass::bfloat16_t,
                                        cutlass::bfloat16_t,
                                        cutlass::bfloat16_t,
                                        float>;
    using EpilogueOutputOp = typename GroupedGemmConfig::EpilogueOutputOp;
    static const int kStages = GroupedGemmConfig::kStages;
    static const int kAlignmentA = 8;
    static const int kAlignmentB = 8;
};

template <typename arch>
struct GroupGemmTraits<float, arch> {
    using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 8>;
    using ThreadWrapShape = cutlass::gemm::GemmShape<64, 32, 8>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using OpClassTensorOp = cutlass::arch::OpClassSimt;
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
    static const int kStages = 3;
    static const int kAlignmentA = 1;
    static const int kAlignmentB = 1;
};



template <typename cutlassType, typename arch>
void groupedGemm_(cutlassType** A, cutlassType** B, cutlassType** C,
                  const int* m, const int* n, const int* k,
                  const float alpha, const float beta,
                  int count, cudaStream_t stream)
{

    using ElementA = cutlassType;
    using ElementB = cutlassType;
    using ElementOutput = cutlassType;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA,
        LayoutA,
        cutlass::ComplexTransform::kNone,
        GroupGemmTraits<cutlassType, arch>::kAlignmentA,
        ElementB,
        LayoutB,
        cutlass::ComplexTransform::kNone,
        GroupGemmTraits<cutlassType, arch>::kAlignmentB,
        ElementOutput,
        LayoutC,
        ElementAccumulator,
        typename GroupGemmTraits<cutlassType, arch>::OpClassTensorOp,
        arch,
        typename GroupGemmTraits<cutlassType, arch>::ThreadBlockShape,
        typename GroupGemmTraits<cutlassType, arch>::ThreadWrapShape,
        typename GroupGemmTraits<cutlassType, arch>::InstructionShape,
        typename GroupGemmTraits<cutlassType, arch>::EpilogueOutputOp,
        // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
        // This parameter is passed in at present to match the APIs of other kernels. The parameter
        // is unused within the kernel.
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        GroupGemmTraits<cutlassType, arch>::kStages,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

    using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    typename Gemm::EpilogueOutputOp::Params epilogue_op(alpha, beta);

    // Configure GEMM arguments

    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<int64_t> lda_host;
    std::vector<int64_t> ldb_host;
    std::vector<int64_t> ldc_host;

    cutlass::DeviceAllocation<int64_t> lda;
    cutlass::DeviceAllocation<int64_t> ldb;
    cutlass::DeviceAllocation<int64_t> ldc;
    cutlass::DeviceAllocation<cutlassType *> ptr_A;
    cutlass::DeviceAllocation<cutlassType *> ptr_B;
    cutlass::DeviceAllocation<cutlassType *> ptr_C;

    problem_sizes.resize(count);
    lda_host.resize(count);
    ldb_host.resize(count);
    ldc_host.resize(count);

    problem_sizes_device.reset(count);

    lda.reset(count);
    ldb.reset(count);
    ldc.reset(count);
    ptr_A.reset(count);
    ptr_B.reset(count);
    ptr_C.reset(count);

    for (int i = 0; i < count; ++i) {
        cutlass::gemm::GemmCoord problem(m[i], n[i], k[i]);
        problem_sizes.at(i) = problem;
        lda_host.at(i) = LayoutA::packed({problem.m(), problem.k()}).stride(0);
        ldb_host.at(i) = LayoutB::packed({problem.k(), problem.n()}).stride(0);
        ldc_host.at(i) = LayoutC::packed({problem.m(), problem.n()}).stride(0);
    }

    problem_sizes_device.reset(count);
    problem_sizes_device.copy_from_host(problem_sizes.data());
    lda.copy_from_host(lda_host.data());
    ldb.copy_from_host(ldb_host.data());
    ldc.copy_from_host(ldc_host.data());
    ptr_A.copy_from_host(A);
    ptr_B.copy_from_host(B);
    ptr_C.copy_from_host(C);

    int threadblock_count = Gemm::sufficient(problem_sizes.data(), count);

    typename Gemm::Arguments args(problem_sizes_device.get(), count, threadblock_count, epilogue_op,
                                  ptr_A.get(), ptr_B.get(), ptr_C.get(), ptr_C.get(),
                                  lda.get(), ldb.get(), ldc.get(), ldc.get(), problem_sizes.data());

    // Initialize the GEMM object
    Gemm gemm;

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Group kernel will fail for params. Error: " + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[FT Error][Group Runner] " + err_msg);
    }
    size_t workspace_size = gemm.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);
    auto init_status = gemm.initialize(args, workspace.get(), stream);

    if (init_status != cutlass::Status::kSuccess) {
        std::string err_msg = "Failed to initialize cutlass Group gemm. Error: "
                              + std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[FT Error][Group Runner] " + err_msg);
    }

    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to run cutlass group gemm. Error: " + std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[FT Error][Group Runner] " + err_msg);
    }
}


template<typename T>
CutlassGroupGemmRunner<T>::CutlassGroupGemmRunner()
{

    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    sm_ = get_sm();
    multi_processor_count_ = getMultiProcessorCount();
}

template<typename T>
struct CutlassTypeTrait {
    using type = T;
};

template<>
struct CutlassTypeTrait<half> {
    using type = cutlass::half_t;
};

template<>
struct CutlassTypeTrait<__nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};



template<typename T>
void CutlassGroupGemmRunner<T>::gemm(T** A, T** B, T** C,
                                     const int* m, const int* n, const int* k,
                                     const float alpha, const float beta,
                                     const int count, cudaStream_t stream) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    using type = typename CutlassTypeTrait<T>::type;
    if (sm_ >= 70 && sm_ < 75) {
        if constexpr(std::is_same<T, half>::value) {
            groupedGemm_<type, cutlass::arch::Sm70>(
                (type**)A,
                (type**)B,
                (type**)C, m, n, k,
                alpha, beta, count, stream);
        } else {
            throw std::runtime_error("[FT Error][GroupGemm][GEMM Dispatch] Arch[75-80] only support half");
        }
    }  else if (sm_ >= 75 && sm_ < 80) {
        if constexpr(std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) {
            groupedGemm_<type, cutlass::arch::Sm75>(
                (type**)A,
                (type**)B,
                (type**)C, m, n, k,
                alpha, beta, count, stream);
        } else {
            throw std::runtime_error("[FT Error][GroupGemm][GEMM Dispatch] Arch[75-80] only support half/bf16");
        }

    } else if (sm_ >= 80 && sm_ < 90) {
        groupedGemm_<type, cutlass::arch::Sm80>(
            (type**)A,
            (type**)B,
            (type**)C, m, n, k,
            alpha, beta, count, stream);
    } else {
        throw std::runtime_error("[FT Error][GroupGemm][GEMM Dispatch] Arch unsupported for Group GEMM");
    }

}

}  // namespace rtp_llm
