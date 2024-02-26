

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/util/device_memory.h"

#include "cutlass_extensions/gemm/kernel/group_gemm_traits.h"

#include "src/fastertransformer/cutlass/cutlass_kernels/group_gemm/group_gemm.h"
#include "src/fastertransformer/cuda/cuda_utils.h"
namespace fastertransformer {


template <int M1, int N1, int K1, int M2, int N2, int K2, typename cutlassType, typename arch>
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

    const int kAlignmentA = 8;
    const int kAlignmentB = 8;

    using GroupGemmArchTraits = cutlass::gemm::kernel::GroupGemmArchTraits<arch>;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<ElementA, LayoutA,
        cutlass::ComplexTransform::kNone, kAlignmentA, ElementB, LayoutB, cutlass::ComplexTransform::kNone, kAlignmentB,
        ElementOutput, LayoutC, ElementAccumulator, cutlass::arch::OpClassTensorOp, arch,
        cutlass::gemm::GemmShape<128, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        typename GroupGemmArchTraits::InstructionShape,
        cutlass::epilogue::thread::LinearCombination<ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
        // This parameter is passed in at present to match the APIs of other kernels. The parameter
        // is unused within the kernel.
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        GroupGemmArchTraits::Stages, // kStages
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
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    int device{-1};
    check_cuda_error(cudaGetDevice(&device));
    sm_ = getSMVersion();
    check_cuda_error(cudaDeviceGetAttribute(&multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}


template<typename T>
void CutlassGroupGemmRunner<T>::gemm(T** A, T** B, T** C,
                                     const int* m, const int* n, const int* k,
                                     const float alpha, const float beta,
                                     const int count, cudaStream_t stream) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if constexpr(std::is_same<T, half>::value) {
        if (sm_ >= 70 && sm_ < 75) {
            groupedGemm_<16, 32, 64, 16, 32, 64, cutlass::half_t, cutlass::arch::Sm70>(
                (cutlass::half_t**)A, 
                (cutlass::half_t**)B,
                (cutlass::half_t**)C, m, n, k, 
                alpha, beta, count, stream);
        }  else if (sm_ >= 75 && sm_ < 80) {
            groupedGemm_<16, 32, 64, 16, 32, 64, cutlass::half_t, cutlass::arch::Sm75>(
                (cutlass::half_t**)A, 
                (cutlass::half_t**)B,
                (cutlass::half_t**)C, m, n, k, 
                alpha, beta, count, stream);
        } else if (sm_ >= 80 && sm_ < 90) {
            groupedGemm_<16, 32, 64, 16, 32, 64, cutlass::half_t, cutlass::arch::Sm80>(
                (cutlass::half_t**)A, 
                (cutlass::half_t**)B,
                (cutlass::half_t**)C, m, n, k, 
                alpha, beta, count, stream);
        } else {
            throw std::runtime_error("[FT Error][GroupGemm][GEMM Dispatch] Arch unsupported for Group GEMM");
    }
        
    }
    
}

}  // namespace fastertransformer
