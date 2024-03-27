#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/utils/ShapeCheck.h"

#include <numeric>
#include <utility>

using namespace std;

namespace fastertransformer {

cublasOperation_t opConvert(TransposeOperation op) {
    switch (op) {
        case TransposeOperation::NONE: return cublasOperation_t::CUBLAS_OP_N;
        case TransposeOperation::TRANSPOSE: return cublasOperation_t::CUBLAS_OP_T;
        default: throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
};

cudaDataType_t dtypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_FP16 : return cudaDataType_t::CUDA_R_16F;
        case DataType::TYPE_FP32 : return cudaDataType_t::CUDA_R_32F;
        default: throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
};

struct CudaGemmDispatch {

    enum GemmImplementType {
        cublas_basic_gemm,
        cublas_batch_gemm,
        cublas_batch_gemm_float32,
        invalid,
    };

    static GemmImplementType dispatch(const GemmParams& params) {
        size_t dim = params.A.dim();
        if (params.A.type() == DataType::TYPE_FP16 &&
            params.B.type() == DataType::TYPE_FP16 &&
            params.C == std::nullopt && dim == 2) {
            
            return GemmImplementType::cublas_basic_gemm;
        }

        else if (params.A.type() == DataType::TYPE_FP16 &&
                 params.B.type() == DataType::TYPE_FP16 &&
                 params.C == std::nullopt && dim > 2    &&
                 params.compute_type == DataType::TYPE_FP32) {
            
            return GemmImplementType::cublas_batch_gemm_float32;
        }

        else if (params.A.type() == DataType::TYPE_FP16 &&
                 params.B.type() == DataType::TYPE_FP16 &&
                 params.C == std::nullopt && dim > 2    &&
                 (params.compute_type == DataType::TYPE_FP16 ||
                 params.compute_type == DataType::TYPE_INVALID)) {
            
            return GemmImplementType::cublas_batch_gemm;
        }
        return GemmImplementType::invalid;
    }

};


struct CudaGemmArguments {
    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Cshape;
    std::vector<size_t> Dshape;

    DataType ADtype;
    DataType BDtype;
    DataType CDtype;
    DataType DDtype;

    size_t dim;
    size_t batch_size;
    size_t m;
    size_t k;
    size_t n;

    float alpha = 1.0f;
    float beta  = 0.0f;

    size_t lda;
    size_t stride_a;
    size_t ldb;
    size_t stride_b;
    size_t ldc;
    size_t stride_c;

    CudaGemmArguments(const GemmParams& params) {

        Ashape = params.A.shape();
        Bshape = params.B.shape();

        if (params.transA == TransposeOperation::TRANSPOSE) {
            std::iter_swap(Ashape.end() -1, Ashape.end() -2);
        }

        if (params.transB == TransposeOperation::TRANSPOSE) {
            std::iter_swap(Bshape.end() -1, Bshape.end() -2);
        }

        if (params.C != std::nullopt) {
            Cshape = params.C.value().get().shape();
        }

        ADtype = params.A.type();
        BDtype = params.A.type();
        if (params.C != std::nullopt) {
            CDtype = params.C.value().get().type();
        }
        DDtype = (params.compute_type == DataType::TYPE_INVALID) ?
                  params.A.type() : params.compute_type;
        
        dim =  params.A.dim();
        batch_size = std::accumulate(Ashape.begin(), Ashape.end() - 2,
                                     (size_t)1, std::multiplies<size_t>());
        
        m = Ashape[dim - 2];
        k = Ashape[dim - 1];
        n = Bshape[dim - 1];

        Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
        Dshape.insert(Dshape.end(), {m, n});

        lda = params.A.shape()[dim - 1];
        stride_a = m * k;
        ldb = params.B.shape()[dim - 1];
        stride_b = k * n;
        ldc = n;
        stride_c = m * n;
    }

    void dump() {
        std::cout << "Ashape is : " << ShapeStringView(Ashape) << "\n"
                  << "Bshape is : " << ShapeStringView(Bshape) << "\n"
                  << "Cshape is : " << ShapeStringView(Cshape) << "\n"
                  << "Dshape is : " << ShapeStringView(Dshape) << "\n"
                  << "dim is : " << dim << "\n"
                  << "batch size is : " << batch_size << "\n"
                  << "m is : " << m << "\n"
                  << "n is : " << n << "\n"
                  << "k is : " << k << "\n"
                  << "lda is : " << lda << "\n"
                  << "ldb is : " << ldb << "\n"
                  << "ldc is : " << ldc << "\n"
                  << "stride_a is : " << stride_a << "\n"
                  << "stride_b is : " << stride_b << "\n"
                  << "stride_c is : " << stride_c << "\n" << std::endl;
    }

};


/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]
BufferPtr CudaDevice::gemm(const GemmParams& params) {

    params.check();

    using GemmImplementType = CudaGemmDispatch::GemmImplementType;
    CudaGemmArguments arguments(params);

    if (CudaGemmDispatch::dispatch(params) == GemmImplementType::invalid) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }

    auto output = allocateBuffer({arguments.DDtype, arguments.Dshape, AllocationType::DEVICE}, {});

    if (CudaGemmDispatch::dispatch(params) == GemmImplementType::cublas_basic_gemm) {
        const auto A = params.A.data();
        const auto B = params.B.data();
        auto D = output->data();
        auto a_op = opConvert(params.transA);
        auto b_op = opConvert(params.transB);
        cublas_mm_wrapper_->Gemm(a_op,
                                 b_op,
                                 arguments.n,
                                 arguments.m,
                                 arguments.k,
                                 B,
                                 arguments.n,
                                 A,
                                 arguments.k,
                                 D,
                                 arguments.n);
        sync_check_cuda_error();
        return move(output);
    } else if (CudaGemmDispatch::dispatch(params) == GemmImplementType::cublas_batch_gemm) {

        const auto A = params.A.data();
        const auto B = params.B.data();
        auto D = output->data();

        auto a_op = opConvert(params.transA);
        auto b_op = opConvert(params.transB);
        cublas_mm_wrapper_->stridedBatchedGemm(b_op,
                                               a_op,
                                                arguments.n,
                                                arguments.m,
                                                arguments.k,
                                                B,
                                                arguments.ldb,
                                                arguments.stride_b,
                                                A,
                                                arguments.lda,
                                                arguments.stride_a,
                                                D,
                                                arguments.ldc,
                                                arguments.stride_c,
                                                arguments.batch_size);
        sync_check_cuda_error();
        return move(output);
    } else if (CudaGemmDispatch::dispatch(params) == GemmImplementType::cublas_batch_gemm_float32) {
        // convert buffers to ptrs
        const auto A = params.A.data();
        const auto B = params.B.data();
        auto D = output->data();

        auto a_op = opConvert(params.transA);
        auto b_op = opConvert(params.transB);

        auto A_data_type = dtypeConvert(arguments.ADtype);
        auto B_data_type = dtypeConvert(arguments.BDtype);
        auto D_data_type = dtypeConvert(arguments.DDtype);
        auto computeType = dtypeConvert(arguments.DDtype);
        cublas_mm_wrapper_->stridedBatchedGemm(b_op,
                                               a_op,
                                               arguments.n,
                                               arguments.m,
                                               arguments.k,
                                               arguments.alpha,
                                               B,
                                               B_data_type,
                                               arguments.ldb,
                                               arguments.stride_b,
                                               A,
                                               A_data_type,
                                               arguments.lda,
                                               arguments.stride_a,
                                               arguments.beta,
                                               D,
                                               D_data_type,
                                               arguments.ldc,
                                               arguments.stride_c,
                                               arguments.batch_size,
                                               computeType);
        sync_check_cuda_error();
        return move(output);
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    return std::move(output);
}


} // namespace fastertransformer

