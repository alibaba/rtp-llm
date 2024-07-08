#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmAllocator.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/utils/ShapeCheck.h"
#include "autil/StringUtil.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/BufferHelper.h"

#include <numeric>
#include <utility>

using namespace std;

namespace fastertransformer {
using namespace rocm;

hipblasOperation_t opConvert(TransposeOperation op) {
    switch (op) {
        case TransposeOperation::NONE:
            return hipblasOperation_t::HIPBLAS_OP_N;
        case TransposeOperation::TRANSPOSE:
            return hipblasOperation_t::HIPBLAS_OP_T;
        default:
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
};

hipblasDatatype_t dtypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_FP16:
            return hipblasDatatype_t::HIPBLAS_R_16F;
        case DataType::TYPE_FP32:
            return hipblasDatatype_t::HIPBLAS_R_32F;
        default:
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
};

struct ROCmGemmDispatch {

    enum GemmImplementType {
        hipblas_basic_gemm,
        hipblas_batch_gemm,
        WeightOnlyQuantMatmulPlugin,
        invalid,
    };

    static GemmImplementType dispatch(const GemmParams& params) {
        size_t dim = params.A.dim();
        if (params.C != std::nullopt) {
            return GemmImplementType::invalid;
        }
        if (dim == 2 && params.A.isFloat() && params.B.isFloat()) {

            return GemmImplementType::hipblas_basic_gemm;
        } else if (dim > 2 && params.A.isFloat() && params.B.isFloat()) {

            return GemmImplementType::hipblas_batch_gemm;
        } else if (dim == 2 && (params.A.type() == DataType::TYPE_FP16 || params.A.type() == DataType::TYPE_BF16)
                   && params.B.type() == DataType::TYPE_QINT8) {
            return GemmImplementType::WeightOnlyQuantMatmulPlugin;
        }
        return GemmImplementType::invalid;
    }
};

struct ROCmGemmArguments {
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

    ROCmGemmArguments(const GemmParams& params) {

        Ashape = params.A.shape();
        Bshape = params.B.shape();

        if (params.transA == TransposeOperation::TRANSPOSE) {
            std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
        }

        if (params.transB == TransposeOperation::TRANSPOSE) {
            std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
        }

        if (params.C != std::nullopt) {
            Cshape = params.C.value().get().shape();
        }

        ADtype = params.A.type();
        BDtype = params.A.type();
        if (params.C != std::nullopt) {
            CDtype = params.C.value().get().type();
        }
        DDtype = (params.compute_type == DataType::TYPE_INVALID) ? params.A.type() : params.compute_type;

        dim        = params.A.dim();
        batch_size = std::accumulate(Ashape.begin(), Ashape.end() - 2, (size_t)1, std::multiplies<size_t>());

        m = Ashape[dim - 2];
        k = Ashape[dim - 1];
        n = Bshape[dim - 1];

        Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
        Dshape.insert(Dshape.end(), {m, n});

        lda      = params.A.shape()[dim - 1];
        stride_a = m * k;
        ldb      = params.B.shape()[dim - 1];
        stride_b = k * n;
        ldc      = n;
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
                  << "stride_c is : " << stride_c << "\n"
                  << std::endl;
    }
};

/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]
BufferPtr ROCmDevice::gemm(const GemmParams& params) {
    params.check();

    using GemmImplementType = ROCmGemmDispatch::GemmImplementType;
    ROCmGemmArguments arguments(params);

    if (ROCmGemmDispatch::dispatch(params) == GemmImplementType::invalid) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((arguments.DDtype == params.D->type()) && (arguments.Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              arguments.DDtype,
                              autil::StringUtil::toString(arguments.Dshape).c_str(),
                              params.D->debugString().c_str());
    } else {
        output = allocateBuffer({arguments.DDtype, arguments.Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }

    if (ROCmGemmDispatch::dispatch(params) == GemmImplementType::WeightOnlyQuantMatmulPlugin) {
        RUNTIME_ASSERT_OP_ARG(false, "weight only quant NOT supported yet.");
#if false
        if (params.A.type() == DataType::TYPE_BF16)
        {
            weight_only_matmul_plguin_->init(WeightDataType::kBF16, WeightTypeId::INT8);
        }
        size_t ws_size = weight_only_matmul_plguin_->getWorkspaceSize(arguments.m,
                                                                      arguments.n,
                                                                      arguments.k);
        auto workspace = allocateBuffer({DataType::TYPE_BYTES,
                                          {ws_size},
                                          AllocationType::DEVICE},
                                          {"workspace"});

        weight_only_matmul_plguin_->enqueue(params.A.data(),
                                            reinterpret_cast<const QBuffer&>(params.B).data(),
                                            reinterpret_cast<const QBuffer&>(params.B).scales_data(),
                                            output->data(),
                                            workspace->data(),
                                            arguments.m,
                                            arguments.n,
                                            arguments.k,
                                            stream_);
                                            
        sync_check_hip_error();
        return std::move(output);
#endif
    }

    auto A_data_type = dtypeConvert(arguments.ADtype);
    auto B_data_type = dtypeConvert(arguments.BDtype);
    auto D_data_type = dtypeConvert(arguments.DDtype);

    if (params.compute_type == DataType::TYPE_INVALID) {
        hipblasMMWrapperPtr()->setGemmConfig(A_data_type, B_data_type, D_data_type, HIPBLAS_R_32F);
    } else {
        hipblasMMWrapperPtr()->setGemmConfig(A_data_type, B_data_type, D_data_type, dtypeConvert(params.compute_type));
    }

    if (ROCmGemmDispatch::dispatch(params) == GemmImplementType::hipblas_basic_gemm) {

        const auto A    = params.A.data();
        const auto B    = params.B.data();
        auto       D    = output->data();
        auto       a_op = opConvert(params.transA);
        auto       b_op = opConvert(params.transB);

        hipblas_mm_wrapper_->Gemm(
            b_op, a_op, arguments.n, arguments.m, arguments.k, B, arguments.ldb, A, arguments.lda, D, arguments.ldc);
        sync_check_hip_error();
        return std::move(output);
    } else if (ROCmGemmDispatch::dispatch(params) == GemmImplementType::hipblas_batch_gemm) {

        // convert buffers to ptrs
        const auto A = params.A.data();
        const auto B = params.B.data();
        auto       D = output->data();

        auto a_op = opConvert(params.transA);
        auto b_op = opConvert(params.transB);

        auto A_data_type = dtypeConvert(arguments.ADtype);
        auto B_data_type = dtypeConvert(arguments.BDtype);
        auto D_data_type = dtypeConvert(arguments.DDtype);
        auto computeType = dtypeConvert(arguments.DDtype);

        hipblas_mm_wrapper_->stridedBatchedGemm(b_op,
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
        sync_check_hip_error();
        return std::move(output);
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    return std::move(output);
}

}  // namespace fastertransformer
