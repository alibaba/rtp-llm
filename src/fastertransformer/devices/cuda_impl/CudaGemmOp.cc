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
BufferPtr CudaDevice::gemm(const GemmParams& params) {

    params.check();

    const bool use_batch_gemm = (params.A.dim() > 2);
    auto a_op = convert_cublas(params.transA);
    auto b_op = convert_cublas(params.transB);

    const auto output_type = (params.compute_type == DataType::TYPE_INVALID) ?
                             params.A.type() : params.compute_type;
    BufferPtr output;

    // select compute
    if (use_batch_gemm) {

        // A [b, ..., m, k]
        // B [b, ..., k, n]
        // C [b, ..., m, n]

        const auto batch_size = std::accumulate(params.A.shape().begin(),
                                               params.A.shape().end() - 2,
                                               (size_t)1, std::multiplies<size_t>());

        const auto m = (a_op == cublasOperation_t::CUBLAS_OP_T) ?
                      params.A.shape().end()[-1]:
                      params.A.shape().end()[-2];

        const auto k = (a_op == cublasOperation_t::CUBLAS_OP_T) ?
                      params.A.shape().end()[-2]:
                      params.A.shape().end()[-1];

        const auto n = (a_op == cublasOperation_t::CUBLAS_OP_T) ?
                      params.B.shape().end()[-2]:
                      params.B.shape().end()[-1];

        auto output_shape = vector<size_t>(params.A.shape().begin(), params.A.shape().end() - 2);
        output_shape.insert(output_shape.end(), {m, n});
        output = allocateBuffer({params.A.type(), output_shape, AllocationType::DEVICE}, {});

        const auto lda = k;
        const auto stride_a = m * k;
        const auto ldb = (b_op == cublasOperation_t::CUBLAS_OP_T) ? k : n;
        const auto stride_b = k * n;
        const auto ldc = n;
        const auto stride_c = m * n;

        // convert buffers to ptrs
        const auto A = params.A.data();
        const auto B = params.B.data();
        auto D = output->data();

        cudaDataType_t A_data_type = CUDA_R_16F;
        cudaDataType_t B_data_type = CUDA_R_16F;
        cudaDataType_t D_data_type = CUDA_R_32F;
        cudaDataType_t computeType = CUDA_R_32F;

        const float alpha = 1.0f;
        const float beta  = 0.0f;

        if (output_type == DataType::TYPE_FP32) {
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
    } else {
        const auto m = params.A.shape()[0];
        const auto k = params.A.shape()[1];
        const auto n = params.B.shape()[1];
        output = allocateBuffer({output_type, {m, n}, AllocationType::DEVICE}, {});

        // convert buffers to ptrs
        const auto A = params.A.data();
        const auto B = params.B.data();
        auto D = output->data();

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
    }

    return move(output);
}


} // namespace fastertransformer

