#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/utils/compiler_config.h"
#include "src/fastertransformer/utils/ShapeCheck.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "autil/StringUtil.h"

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
        case DataType::TYPE_BF16 : return cudaDataType_t::CUDA_R_16BF;
        case DataType::TYPE_FP32 : return cudaDataType_t::CUDA_R_32F;
#ifdef ENABLE_FP8
	case DataType::TYPE_FP8_E4M3 : return cudaDataType_t::CUDA_R_8F_E4M3;
	case DataType::TYPE_QFP8_E4M3 : return cudaDataType_t::CUDA_R_8F_E4M3;
#endif
        default: throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
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
        BDtype = params.B.type();
        if (params.C != std::nullopt) {
            CDtype = params.C.value().get().type();
        }
        DDtype = (params.compute_type == DataType::TYPE_INVALID) ?
                  params.A.type() : params.compute_type;\
        // int8 gemm
        if (ADtype == DataType::TYPE_QINT8 && BDtype == DataType::TYPE_QINT8) {
            DDtype = DataType::TYPE_FP16;
        }
        // fp8 gemm
        if (ADtype == DataType::TYPE_QFP8_E4M3 && BDtype == DataType::TYPE_QFP8_E4M3) {
            DDtype = DataType::TYPE_FP16;
        }

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

        if (ADtype == DataType::TYPE_QFP8_E4M3 && BDtype == DataType::TYPE_QFP8_E4M3) {
            float input_scale = getCudaValue(reinterpret_cast<const float*>(reinterpret_cast<const QBuffer&>(params.A).scalesData()), 0);
            float weight_scale = getCudaValue(reinterpret_cast<const float*>(reinterpret_cast<const QBuffer&>(params.B).scalesData()), 0);
            alpha = params.alpha * input_scale * weight_scale;
        } else {
            alpha = params.alpha;
        }
        
        beta = params.beta;
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

void CudaDevice::InvokeSmoothQaunt(const GemmParams& params,
                                   CudaGemmArguments arguments,
                                   BufferPtr         output) {
    bool perToken   = true;
    bool perChannel = true;
    if (reinterpret_cast<const QBuffer&>(params.B).scales().size() == 1) {
        perChannel = false;
    }
    if (reinterpret_cast<const QBuffer&>(params.A).scales().size() == 1) {
        FT_LOG_DEBUG("use static quant");
        perToken = false;
    }
    auto quant_mode =
        tensorrt_llm::common::QuantMode::fromDescription(true, true, perToken, perChannel, false, false, false, false);
    smooth_quant_plugin_->init(quant_mode, nvinfer1DtypeConvert(output->type()));

    FT_LOG_DEBUG("use int8 soomth gemm.");
    FT_CHECK_WITH_INFO(smooth_quant_plugin_->addBiasActivationEpilogueSupported(params.activationType),
     "activation type not supported: %d", int(params.activationType));
    BUFFER_DTYPE_CHECK(params.A, {DataType::TYPE_QINT8});
    BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_QINT8});
    size_t ws_size   = smooth_quant_plugin_->getWorkspaceSize(arguments.m, arguments.n, arguments.k);
    auto   workspace = allocateBuffer({DataType::TYPE_BYTES, {ws_size}, AllocationType::DEVICE}, {"workspace"});
    // TODO: support it in ppu
    smooth_quant_plugin_->enqueue(reinterpret_cast<const QBuffer&>(params.A).data(),
                                  reinterpret_cast<const QBuffer&>(params.B).data(),
                                  reinterpret_cast<const QBuffer&>(params.B).scalesData<float>(),
                                  reinterpret_cast<const QBuffer&>(params.A).scalesData<float>(),
                                  output->data(),
                                  workspace->data<char>(),
                                  params.C ? params.C.value().get().data() : nullptr,
                                  params.activationType,
                                  arguments.m,
                                  arguments.n,
                                  arguments.k,
                                  stream_);
}

void CudaDevice::InvokeWeightOnlyGemm(const GemmParams& params,
                                      CudaGemmArguments arguments,
                                      BufferPtr         output) {
    FT_LOG_DEBUG("use weight only int8 gemm.");
    FT_CHECK_WITH_INFO(params.activationType == ActivationType::Identity, "activation type should be identity");
    FT_CHECK_WITH_INFO(params.C == std::nullopt, "weightonly bias should be nullopt");
    if (reinterpret_cast<const QBuffer&>(params.B).zerosData() != nullptr) {
        FT_LOG_DEBUG("use group wise int4 gemm.");
        FT_CHECK(reinterpret_cast<const QBuffer&>(params.B).scales().dim() == 2);
        size_t kernel_dim0 = params.B.shape()[0];
        size_t scales_dim0 = reinterpret_cast<const QBuffer&>(params.B).scales().shape()[0];
        FT_CHECK((kernel_dim0 % scales_dim0 == 0));
        size_t group_size = (kernel_dim0 / scales_dim0);
        FT_CHECK((group_size == 64 || group_size == 128));
        size_t type_bits = getTypeBits(params.B.type());
        FT_CHECK((type_bits == 4 || type_bits == 8));

        weight_only_groupwise_matmul_plugin_->init(nvinfer1DtypeConvert(params.A.type()), true, group_size, type_bits);
        size_t ws_size = weight_only_groupwise_matmul_plugin_->getWorkspaceSize(arguments.m, arguments.n, arguments.k);
        auto   workspace = allocateBuffer({DataType::TYPE_BYTES, {ws_size}, AllocationType::DEVICE}, {"workspace"});

        weight_only_groupwise_matmul_plugin_->enqueue(params.A.data(),
                                                      reinterpret_cast<const QBuffer&>(params.B).data(),
                                                      reinterpret_cast<const QBuffer&>(params.B).scalesData(),
                                                      reinterpret_cast<const QBuffer&>(params.B).zerosData(),
                                                      nullptr,
                                                      output->data(),
                                                      workspace->data(),
                                                      arguments.m,
                                                      arguments.n,
                                                      arguments.k,
                                                      stream_);
    } else {
        BUFFER_DTYPE_CHECK(params.A, {DataType::TYPE_FP16, DataType::TYPE_BF16});
        BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_QINT8});
        weight_only_matmul_plugin_->init(nvinfer1DtypeConvert(params.A.type()), trt_plugins::WeightTypeId::INT8);
        FT_LOG_DEBUG("use int8 only weight gemm.");

        size_t ws_size   = weight_only_matmul_plugin_->getWorkspaceSize(arguments.m, arguments.n, arguments.k);
        auto   workspace = allocateBuffer({DataType::TYPE_BYTES, {ws_size}, AllocationType::DEVICE}, {"workspace"});

        weight_only_matmul_plugin_->enqueue(params.A.data(),
                                            reinterpret_cast<const QBuffer&>(params.B).data(),
                                            reinterpret_cast<const QBuffer&>(params.B).scalesData(),
                                            output->data(),
                                            workspace->data(),
                                            arguments.m,
                                            arguments.n,
                                            arguments.k,
                                            stream_);
    }
}

void CudaDevice::InvokeGeneralGemm(const GemmParams& params,
                                   CudaGemmArguments arguments,
                                   BufferPtr         output) {
    FT_LOG_DEBUG("use general gemm.");
    FT_CHECK_WITH_INFO(params.activationType == ActivationType::Identity, "general gemm activation type should be identity");
    FT_CHECK_WITH_INFO(params.C == std::nullopt, "general gemm bias should be nullopt");
    auto A_data_type = dtypeConvert(arguments.ADtype);
    auto B_data_type = dtypeConvert(arguments.BDtype);
    auto D_data_type = dtypeConvert(arguments.DDtype);
    auto computeType = CUDA_R_32F;
    if (params.compute_type != DataType::TYPE_INVALID) {
        computeType = dtypeConvert(params.compute_type);
    }
    const auto A    = params.A.data();
    const auto B    = params.B.data();
    auto       D    = output->data();
    auto       a_op = opConvert(params.transA);
    auto       b_op = opConvert(params.transB);

#ifdef ENABLE_FP8
    if (params.dispatch() == GemmType::QBufferA_QBufferB_BufferC_2DGemm && QBufferDtype2BufferDtype(params.A.type()) == DataType::TYPE_FP8_E4M3) {
        BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_FP8_E4M3, TYPE_QFP8_E4M3});
        cublas_mm_wrapper_->Gemm(CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 arguments.n,
                                 arguments.m,
                                 arguments.k,
                                 B,
                                 B_data_type,
                                 arguments.lda,
                                 A,
                                 A_data_type,
                                 arguments.lda,
                                 D,
                                 D_data_type,
                                 arguments.ldc,
                                 computeType,
                                 arguments.alpha,
                                 arguments.beta);

        sync_check_cuda_error();
    } else
#endif
    if (params.dispatch() == GemmType::BufferA_BufferB_BufferC_2DGemm) {
        BUFFER_DTYPE_CHECK(params.A, {DataType::TYPE_FP16, DataType::TYPE_BF16, DataType::TYPE_FP32});
        BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_FP16, DataType::TYPE_BF16, DataType::TYPE_FP32});
        cublas_mm_wrapper_->setGemmConfig(B_data_type, A_data_type, D_data_type, computeType);
        cublas_mm_wrapper_->Gemm(b_op,
                                 a_op,
                                 arguments.n,
                                 arguments.m,
                                 arguments.k,
                                 B,
                                 arguments.ldb,
                                 A,
                                 arguments.lda,
                                 D,
                                 arguments.ldc,
                                 arguments.alpha,
                                 arguments.beta);
    } else if (params.dispatch() == GemmType::BufferA_BufferB_BufferC_3DGemm) {
        BUFFER_DTYPE_CHECK(params.A, {DataType::TYPE_FP16, DataType::TYPE_BF16, DataType::TYPE_FP32});
        BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_FP16, DataType::TYPE_BF16, DataType::TYPE_FP32});
        cublas_mm_wrapper_->setGemmConfig(B_data_type, A_data_type, D_data_type, computeType);
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
                                               arguments.batch_size,
                                               arguments.alpha,
                                               arguments.beta);
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
}

    /// @brief   basic gemm ops
    /// @details D = alpha * op(A) * op(B) + beta * C
    ///          A [b, ..., m, k]
    ///          B [b, ..., k, n]
    ///          C [b, ..., m, n]
    BufferPtr CudaDevice::gemm(const GemmParams& params) {
        params.check();
        CudaGemmArguments arguments(params);

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

        if (params.dispatch() == GemmType::QBufferA_QBufferB_BufferC_2DGemm && params.A.type() == DataType::TYPE_QINT8) {
            InvokeSmoothQaunt(params, arguments, output);
        } else if (params.dispatch() == GemmType::BufferA_QBufferB_BufferC_2DGemm) {
            InvokeWeightOnlyGemm(params, arguments, output);
        } else {
            InvokeGeneralGemm(params, arguments, output);
        }
        sync_check_cuda_error();
        return output;
}

} // namespace fastertransformer

