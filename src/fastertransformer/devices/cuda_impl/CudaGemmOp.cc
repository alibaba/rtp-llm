#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"

#include <numeric>

using namespace std;

namespace fastertransformer {


cublasOperation_t convert_cublas(TransposeOperation op) {
    switch (op) {
        case TransposeOperation::NONE: return cublasOperation_t::CUBLAS_OP_N;
        case TransposeOperation::TRANSPOSE: return cublasOperation_t::CUBLAS_OP_T;
        default: throw std::invalid_argument("");
    }
};

/// @brief   basic gemm ops
/// @details D = alpha * A * B + beta * C
///          A(array) : [m, k]
///          B(array) : [k, n]
///          C(array) : [m, n]
///          D(array) : [m, n]
///          alpha(scalar)
///          beta(scalar)
void CudaDevice::gemm(const GemmParams& params) {
    
    params.Check();

    const bool use_batch_gemm = (params.A.dim() > 2);
    auto a_op = convert_cublas(params.transA);
    auto b_op = convert_cublas(params.transB);

    std::cout << "a_op is " << a_op << std::endl;
    std::cout << "b_op is " << b_op << std::endl;

    // select compute
    if (use_batch_gemm) {

        // A [b, ..., m, k]
        // B [b, ..., k, n]
        // C [b, ..., m, n]

        const int batch_size = std::accumulate(params.A.shape().begin(),
                                               params.A.shape().end() - 2,
                                               1.0, std::multiplies<int>());

        const int m = params.D.shape()[params.D.shape().size() - 2];

        const int k = (a_op == cublasOperation_t::CUBLAS_OP_T) ? 
                      params.A.shape()[params.D.shape().size() - 2] : 
                      params.A.shape()[params.D.shape().size() - 1];

        const int n = params.D.shape()[params.D.shape().size() - 1];

        const int lda = k;
        const int stride_a = m * k;

        const int ldb = (b_op == cublasOperation_t::CUBLAS_OP_T) ? k : n;

        const int stride_b = k * n;
        
        const int ldc = n;
        const int stride_c = m * n;

        for (auto size : params.A.shape()) {
            std::cout << "A shape is " << size << std::endl;
        }
        for (auto size : params.B.shape()) {
            std::cout << "B shape is " << size << std::endl;
        }
        for (auto size : params.D.shape()) {
            std::cout << "D shape is " << size << std::endl;
        }
        

        std::cout << "m is " << m << "\n" \
                  << "n is " << n << "\n" \
                  << "k is " << k << "\n" \
                  << "b is " << batch_size << std::endl;
        // convert buffers to ptrs
        const void* A = params.A.data();
        const void* B = params.B.data();
        void* D = params.D.data();

        cudaDataType_t A_data_type = CUDA_R_16F;
        cudaDataType_t B_data_type = CUDA_R_16F;
        cudaDataType_t D_data_type = CUDA_R_32F;
        cudaDataType_t computeType = CUDA_R_32F;

        const float alpha = 1.0f;
        const float beta  = 0.0f;

        if (params.D.type() == DataType::TYPE_FP32) {
            std::cout << "test" << std::endl;
            cublas_mm_wrapper_->stridedBatchedGemm(b_op,
                                                   a_op,
                                                   n,
                                                   m,
                                                   k,
                                                   alpha,
                                                   B,
                                                   B_data_type,
                                                   ldb,
                                                   stride_b,
                                                   A,
                                                   A_data_type,
                                                   lda,
                                                   stride_a,
                                                   beta,
                                                   D,
                                                   D_data_type,
                                                   ldc,
                                                   stride_c,
                                                   batch_size,
                                                   computeType);
        } else {
            cublas_mm_wrapper_->stridedBatchedGemm(
            b_op,
            a_op,
            n,
            m,
            k,
            B,
            ldb,
            stride_b,
            A,
            lda,
            stride_a,
            D,
            ldc,
            stride_c,
            batch_size);
        }
        sync_check_cuda_error();
        return;
    } else {
        const int m = params.A.shape()[0];
        const int k = params.A.shape()[1];
        const int n = params.D.shape()[1];

        // convert buffers to ptrs
        const void* A = params.A.data();
        const void* B = params.B.data();
        void* D = params.D.data();

        cublas_mm_wrapper_->Gemm(a_op,
                                 b_op, 
                                 n, 
                                 m, 
                                 k, 
                                 B, 
                                 n, 
                                 A, 
                                 k, 
                                 D, 
                                 n);
        sync_check_cuda_error();
        return;
    }

    return;

}


} // namespace fastertransformer

