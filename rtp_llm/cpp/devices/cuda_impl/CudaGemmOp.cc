#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/devices/CommonDefines.h"
#include "rtp_llm/cpp/kernels/layernorm_kernels.h"
#include "rtp_llm/cpp/kernels/activation_kernels.h"
#include "rtp_llm/cpp/devices/ShapeCheck.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/cuda/deep_gemm/DeepGemmPlugin.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "autil/StringUtil.h"

#include <numeric>
#include <utility>

using namespace std;

namespace rtp_llm {

cublasOperation_t opConvert(TransposeOperation op) {
    switch (op) {
        case TransposeOperation::NONE:
            return cublasOperation_t::CUBLAS_OP_N;
        case TransposeOperation::TRANSPOSE:
            return cublasOperation_t::CUBLAS_OP_T;
        default:
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
};

cudaDataType_t dtypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_FP16:
            return cudaDataType_t::CUDA_R_16F;
        case DataType::TYPE_BF16:
            return cudaDataType_t::CUDA_R_16BF;
        case DataType::TYPE_FP32:
            return cudaDataType_t::CUDA_R_32F;
#ifdef ENABLE_FP8
        case DataType::TYPE_FP8_E4M3:
            return cudaDataType_t::CUDA_R_8F_E4M3;
        case DataType::TYPE_QFP8_E4M3:
            return cudaDataType_t::CUDA_R_8F_E4M3;
#endif
        default:
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
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
            std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
        }

        if (params.transB == TransposeOperation::TRANSPOSE) {
            std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
        }

        if (params.C != std::nullopt) {
            Cshape = params.C.value().get().shape();
        }

        ADtype = params.A.type();
        BDtype = params.B.type();
        if (params.C != std::nullopt) {
            CDtype = params.C.value().get().type();
        }
        DDtype = (params.D_type == DataType::TYPE_INVALID) ?
                     ((params.compute_type == DataType::TYPE_INVALID) ? params.A.type() : params.compute_type) :
                     params.D_type;

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
        alpha    = params.alpha;
        beta     = params.beta;
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

void CudaDevice::InvokeSmoothQaunt(const GemmParams& params, CudaGemmArguments arguments, BufferPtr output) {
    bool perToken   = true;
    bool perChannel = true;
    if (reinterpret_cast<const QBuffer&>(params.B).scales().size() == 1) {
        perChannel = false;
    }
    if (reinterpret_cast<const QBuffer&>(params.A).scales().size() == 1) {
        RTP_LLM_LOG_DEBUG("use static quant");
        perToken = false;
    }
    auto quant_mode =
        tensorrt_llm::common::QuantMode::fromDescription(true, true, perToken, perChannel, false, false, false, false);
    smooth_quant_plugin_->init(quant_mode, nvinfer1DtypeConvert(output->type()));

    RTP_LLM_LOG_DEBUG("use int8 soomth gemm.");
    RTP_LLM_CHECK_WITH_INFO(smooth_quant_plugin_->addBiasActivationEpilogueSupported(params.activationType),
                            "activation type not supported: %d",
                            int(params.activationType));
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
                                  params.stream == nullptr ? stream_ : (cudaStream_t)params.stream);
}

void CudaDevice::InvokeWeightOnlyGemm(const GemmParams& params, CudaGemmArguments arguments, BufferPtr output) {
    RTP_LLM_LOG_DEBUG("use weight only int8 gemm.");
    RTP_LLM_CHECK_WITH_INFO(params.activationType == ActivationType::Identity, "activation type should be identity");
    RTP_LLM_CHECK_WITH_INFO(params.C == std::nullopt, "weightonly bias should be nullopt");
    if (reinterpret_cast<const QBuffer&>(params.B).zerosData() != nullptr) {
        RTP_LLM_LOG_DEBUG("use group wise int4 gemm.");
        RTP_LLM_CHECK(reinterpret_cast<const QBuffer&>(params.B).scales().dim() == 2);
        size_t kernel_dim0 = params.B.shape()[0];
        size_t scales_dim0 = reinterpret_cast<const QBuffer&>(params.B).scales().shape()[0];
        RTP_LLM_CHECK((kernel_dim0 % scales_dim0 == 0));
        size_t group_size = (kernel_dim0 / scales_dim0);
        RTP_LLM_CHECK((group_size == 64 || group_size == 128));
        size_t type_bits = getTypeBits(params.B.type());
        RTP_LLM_CHECK((type_bits == 4 || type_bits == 8));

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
                                                      params.stream == nullptr ? stream_ : (cudaStream_t)params.stream);
    } else {
        BUFFER_DTYPE_CHECK(params.A, {DataType::TYPE_FP16, DataType::TYPE_BF16});
        BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_QINT8});
        weight_only_matmul_plugin_->init(nvinfer1DtypeConvert(params.A.type()), trt_plugins::WeightTypeId::INT8);
        RTP_LLM_LOG_DEBUG("use int8 only weight gemm.");

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
                                            params.stream == nullptr ? stream_ : (cudaStream_t)params.stream);
    }
}

void CudaDevice::InvokeGeneralGemm(const GemmParams& params, CudaGemmArguments arguments, BufferPtr output) {
    RTP_LLM_LOG_DEBUG("use general gemm.");
    RTP_LLM_CHECK_WITH_INFO(params.activationType == ActivationType::Identity,
                            "general gemm activation type should be identity");
    RTP_LLM_CHECK_WITH_INFO(params.C == std::nullopt, "general gemm bias should be nullopt");
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
    if (params.dispatch() == GemmType::QBufferA_QBufferB_BufferC_2DGemm
        && QBufferDtype2BufferDtype(params.A.type()) == DataType::TYPE_FP8_E4M3) {
        BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_FP8_E4M3, TYPE_QFP8_E4M3});
        float* A_scale = reinterpret_cast<const QBuffer&>(params.A).scalesData<float>();
        float* B_scale = reinterpret_cast<const QBuffer&>(params.B).scalesData<float>();

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
                                 arguments.beta,
                                 A_scale,
                                 B_scale,
                                 params.math_sm_count,
                                 0,
                                 params.stream == nullptr ? stream_ : (cudaStream_t)params.stream);
        check_cuda_error();
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
                                 arguments.beta,
                                 params.math_sm_count,
                                 params.stream == nullptr ? stream_ : (cudaStream_t)params.stream);
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

void CudaDevice::InvokeDeepGemm(const GemmParams& params, CudaGemmArguments arguments, BufferPtr& output) {
    RTP_LLM_LOG_DEBUG("use deep gemm.");
    RTP_LLM_CHECK_WITH_INFO(params.activationType == ActivationType::Identity,
                            "deep gemm activation type should be identity");
    RTP_LLM_CHECK_WITH_INFO(params.C == std::nullopt, "deep gemm bias should be nullopt");
    BufferPtr quanted_input;
    BufferPtr gemm_output = output;
    if (params.A.type() != DataType::TYPE_QFP8_E4M3) {
        auto padding_size = DeepGemmPlugin::getPaddingSize(params.A.shape()[0], DeepGemmType::Normal);
        quanted_input     = quantize(QuantizeParams(
            params.A, DataType::TYPE_QFP8_E4M3, params.A.dim() - 1, QScheme::Qfp8PerTokenBlock, padding_size));
        if (initParams().profile_debug_logging_config.check_nan) {
            if (quanted_input->isQBuffer()) {
                const auto& qbuffer = reinterpret_cast<const QBuffer&>(*quanted_input);
                checkNAN(qbuffer.kernel(), "deepgemm_quanted_input_kernel_dump", nullptr, true);
                checkNAN(qbuffer.scales(), "deepgemm_quanted_input_scales_dump", nullptr, true);
            } else {
                checkNAN(*quanted_input, "deepgemm_quanted_input_dump", nullptr, true);
            }
        }
        DeepGemmPlugin::gemmFp8(*quanted_input, params.B, *gemm_output, init_params_.user_deep_gemm_num_sm, stream_);
        if (initParams().profile_debug_logging_config.check_nan) {
            checkNAN(*gemm_output, "deepgemm_gemm_output_before_slice_dump", nullptr, true);
        }
        output = gemm_output->slice(0, params.A.shape()[0], false);
        output->updateParent(gemm_output);
    } else {
        DeepGemmPlugin::gemmFp8(params.A, params.B, *gemm_output, init_params_.user_deep_gemm_num_sm, stream_);
    }
}
/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]
BufferPtr CudaDevice::gemm(const GemmParams& params) {
    params.check();
    if (initParams().profile_debug_logging_config.check_nan) {
        if (params.A.isQBuffer()) {
            const auto& qbuffer = reinterpret_cast<const QBuffer&>(params.A);
            checkNAN(qbuffer.kernel(), "gemm_A_kernel_dump", nullptr, true);
            checkNAN(qbuffer.scales(), "gemm_A_scales_dump", nullptr, true);
        } else {
            checkNAN(params.A, "gemm_A_dump", nullptr, true);
        }
        if (params.B.isQBuffer()) {
            const auto& qbuffer = reinterpret_cast<const QBuffer&>(params.B);
            checkNAN(qbuffer.kernel(), "gemm_B_kernel_dump", nullptr, true);
            checkNAN(qbuffer.scales(), "gemm_B_scales_dump", nullptr, true);
        } else {
            checkNAN(params.B, "gemm_B_dump", nullptr, true);
        }
        if (params.C.has_value()) {
            checkNAN(params.C.value().get(), "gemm_C_dump", nullptr, true);
        }
    }
    CudaGemmArguments arguments(params);

    auto is_fp8_blockwise_gemm = (params.qscheme == QScheme::Qfp8PerTokenBlock);

    BufferPtr output;
    if (!is_fp8_blockwise_gemm) {
        if (params.D) {
            output = params.D;
            RUNTIME_ASSERT_OP_ARG((arguments.DDtype == params.D->type()) && (arguments.Dshape == params.D->shape()),
                                  "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                                  arguments.DDtype,
                                  autil::StringUtil::toString(arguments.Dshape).c_str(),
                                  params.D->debugString().c_str());
        } else if (!(params.dispatch() == GemmType::BufferA_QBufferB_BufferC_2DGemm
                     && params.A.type() != DataType::TYPE_QFP8_E4M3 && params.B.type() == DataType::TYPE_QFP8_E4M3)) {
            output = allocateBuffer({arguments.DDtype, arguments.Dshape, AllocationType::DEVICE}, {"gemm_output"});
        }
    }

    if (params.dispatch() == GemmType::QBufferA_QBufferB_BufferC_2DGemm && params.A.type() == DataType::TYPE_QINT8) {
        InvokeSmoothQaunt(params, arguments, output);
    } else if (((params.dispatch() == GemmType::BufferA_QBufferB_BufferC_2DGemm
                 && params.A.type() != DataType::TYPE_QFP8_E4M3)
                || (params.qscheme == QScheme::Qfp8PerTokenBlock))
               && params.B.type() == DataType::TYPE_QFP8_E4M3) {
        auto dshape = arguments.Dshape;
        // padding to 128
        dshape[0]          = (dshape[0] + 127) / 128 * 128;
        auto padded_output = allocateBuffer({arguments.DDtype, dshape, AllocationType::DEVICE}, {"gemm_output"});
        InvokeDeepGemm(params, arguments, padded_output);
        if (params.D) {
            auto d_shape_0 = params.D->shape()[0], padded_shape_0 = padded_output->shape()[0];
            RUNTIME_ASSERT_OP_ARG(
                d_shape_0 <= padded_shape_0,
                "Gemm output D shape[0] should be less than output shape[0] in fp8 blockwise gemm case, but got [%ld] > [%ld]",
                d_shape_0,
                padded_shape_0);
            copy({*params.D, *padded_output->slice(0, d_shape_0)});
            output = params.D;
        } else {
            output = padded_output;
        }
    } else if (params.dispatch() == GemmType::BufferA_QBufferB_BufferC_2DGemm) {
        InvokeWeightOnlyGemm(params, arguments, output);
    } else {
        InvokeGeneralGemm(params, arguments, output);
    }
    check_cuda_error();
    if (initParams().profile_debug_logging_config.check_nan) {
        if (output->isQBuffer()) {
            const auto& qbuffer = reinterpret_cast<const QBuffer&>(*output);
            checkNAN(qbuffer.kernel(), "gemm_output_kernel_dump", nullptr, true);
            checkNAN(qbuffer.scales(), "gemm_output_scales_dump", nullptr, true);
        } else {
            checkNAN(*output, "gemm_output_dump", nullptr, true);
        }
    }
    return output;
}

}  // namespace rtp_llm
