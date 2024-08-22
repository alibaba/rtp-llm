#include "src/fastertransformer/devices/arm_impl/ArmDevice.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/cpu_allocator.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include <cstring>
#include "autil/StringUtil.h"
#include "type_bf16/hie_bfloat16.hpp"
#include "gemm_opt/ArmGemmKernel.h"

namespace fastertransformer {


/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]
BufferPtr ArmCpuDevice::gemm(const GemmParams& params) {
        return gemm_opt(params);
}


/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]
BufferPtr ArmCpuDevice::gemm_opt(const GemmParams& params) {

#ifdef GEMM_DEBUG
    Timer timer;
    auto start = std::chrono::high_resolution_clock::now();
#endif

    params.check();

    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Dshape;

    size_t dim;
    size_t batch_size;
    size_t m;
    size_t k;
    size_t n;
    size_t lda;

    Ashape = params.A.shape();
    Bshape = params.B.shape();

    dim        = params.A.dim();
    batch_size = std::accumulate(Ashape.begin(), Ashape.end() - 2, (size_t)1, std::multiplies<size_t>());

    if (params.transA == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
    }

    if (params.transB == TransposeOperation::TRANSPOSE) {
        std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
    }

    m = Ashape[dim - 2];
    k = Ashape[dim - 1];
    n = Bshape[dim - 1];
    lda = k;

    auto data_type = params.compute_type == DataType::TYPE_INVALID ? params.A.type() : params.compute_type;

    Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
    Dshape.insert(Dshape.end(), {m, n});

    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((data_type == params.D->type()) && (Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              data_type,
                              autil::StringUtil::toString(Dshape).c_str(),
                              params.D->debugString().c_str());
    } else {
        output = allocateBuffer({data_type, Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }

#ifdef GEMM_DEBUG
    timer_recorder_.record(std::string("gemm_prepare_data, ") + "m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k), timer.elapsed_nano());
    timer.reset();
#endif
    // allocate a temp workspace to pack input fp32->bf16 or fp16->bf16
    size_t k_pack = std::ceil(k / 8.0) * 8;
    size_t m_aligned = m + m % 2;
    std::vector<size_t> workspace_shape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
    workspace_shape.insert(workspace_shape.end(), {m_aligned, k_pack});
    BufferPtr workspace = allocateBuffer({DataType::TYPE_BF16, workspace_shape, AllocationType::DEVICE}, {"gemm_workspace"});
    memset(workspace->data(), 0, workspace->sizeBytes());

    BufferPtr weight_workspace;
    const Buffer *weight_workspace_ptr = nullptr;

    size_t weight_k_pack = std::ceil(k / 8.0) * 8;
    size_t width = weight_k_pack * 2;
    size_t height = n / 2 + n % 2;
    if (params.B.type() == DataType::TYPE_FP32 ||
        params.B.type() == DataType::TYPE_FP16 ||
        params.B.type() == DataType::TYPE_BF16) {
        weight_workspace_ptr = &(params.B);
    } else {
        std::cerr << "Unsupported data type for B" << std::endl;
        return nullptr;
    }

#ifdef GEMM_DEBUG
    timer_recorder_.record(std::string("gemm_prepare_workspace(packing)") + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k), timer.elapsed_nano());
#endif

    for (size_t batch = 0; batch < batch_size; ++batch) {
#ifdef GEMM_DEBUG
        timer.reset();
#endif

        hie::bfloat16* B_bf16_ptr = reinterpret_cast<hie::bfloat16*>(weight_workspace_ptr->dataWithOffset(batch * height * width));
        float* C_fp32_ptr = reinterpret_cast<float*>(output->dataWithOffset(batch * m * n));
        float16_t* C_fp16_ptr = reinterpret_cast<float16_t*>(output->dataWithOffset(batch * m * n));
        if (params.A.type() == DataType::TYPE_FP32) {
            float* A_fp32_ptr = reinterpret_cast<float*>(params.A.dataWithOffset(batch * m * k));

#ifdef GEMM_DEBUG
            timer_recorder_.record(std::string("gemm_prepare_batch_data")  + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k), timer.elapsed_nano());
            timer.reset();
#endif

            if (data_type == DataType::TYPE_FP32) {
                gemm_kernel_.gemm_kernel_arm<float, float>(m, n, k, k_pack, lda, A_fp32_ptr, B_bf16_ptr, C_fp32_ptr, nullptr, 0, workspace->data());
            } else if (data_type == DataType::TYPE_FP16) {
                gemm_kernel_.gemm_kernel_arm<float, float16_t>(m, n, k, k_pack, lda, A_fp32_ptr, B_bf16_ptr, C_fp16_ptr, nullptr, 0, workspace->data());
            } else {
                std::cerr << "Unsupported data type for compute" << std::endl;
                return nullptr;
            }

#ifdef GEMM_DEBUG
            timer_recorder_.record(std::string("gemm_kernel") + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k), timer.elapsed_nano());
#endif

        } else if(params.A.type() == DataType::TYPE_FP16) {
            float16_t* A_fp16_ptr = reinterpret_cast<float16_t*>(params.A.dataWithOffset(batch * m * k));

#ifdef GEMM_DEBUG
            timer_recorder_.record(std::string("gemm_prepare_batch_data")  + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k), timer.elapsed_nano());
            timer.reset();
#endif
            // gemm_kernel_.gemm_kernel_arm(m, n, k, k_pack, lda, A_fp16_ptr, B_bf16_ptr, C_fp32_ptr, nullptr, 0, workspace->data());
            if (data_type == DataType::TYPE_FP32) {
                gemm_kernel_.gemm_kernel_arm<float16_t, float>(m, n, k, k_pack, lda, A_fp16_ptr, B_bf16_ptr, C_fp32_ptr, nullptr, 0, workspace->data());
            } else if (data_type == DataType::TYPE_FP16) {
                gemm_kernel_.gemm_kernel_arm<float16_t, float16_t>(m, n, k, k_pack, lda, A_fp16_ptr, B_bf16_ptr, C_fp16_ptr, nullptr, 0, workspace->data());
            } else {
                std::cerr << "Unsupported data type for compute" << std::endl;
                return nullptr;
            }

#ifdef GEMM_DEBUG
            timer_recorder_.record(std::string("gemm_kernel") + ", m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k), timer.elapsed_nano());
#endif
        } else {
            std::cerr << "Unsupported data type for A" << std::endl;
            return nullptr;
        }
    }
#ifdef GEMM_DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    float during_time = std::chrono::duration<float>(end - start).count();
    printf("gemm_opt b,m,n,k %ld %ld %ld %ld %.3f\n", batch_size, m, n, k, during_time * 1000);
#endif
    return output;
}

#ifdef GEMM_DEBUG
TimerRecorder ArmCpuDevice::timer_recorder_ = TimerRecorder();
void ArmCpuDevice::print_time() {
    timer_recorder_.print();
}
#endif

}  // namespace fastertransformer
