#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "autil/StringUtil.h"

using namespace std;

namespace rtp_llm {
struct ROCmGemmActArguments {
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

    ROCmGemmActArguments(const GemmParams& params) {

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
};

static hipDataType dtypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_BF16:
            return hipDataType::HIP_R_16BF;
        case DataType::TYPE_FP16:
            return hipDataType::HIP_R_16F;
        case DataType::TYPE_FP32:
            return hipDataType::HIP_R_32F;
        case DataType::TYPE_FP8_E4M3:
            return hipDataType::HIP_R_8F_E4M3_FNUZ;
        default:
            ROCM_FAIL("[GEMM]: Other DataType not implemented");
    }
};

BufferPtr ROCmDevice::loraLinearWithActivation(const LoraLinearWithActivationParams& params) {
    auto gemm_params = params.lora_linear_params.gemm_params;
    auto act_params  = params.activation_params;

    hipblasLtEpilogue_t epilogue;
    if (act_params.atype == ActivationType::Gelu && act_params.qscheme == QScheme::NoQuantize) {
        epilogue = HIPBLASLT_EPILOGUE_GELU_BIAS;
    } else if (act_params.atype == ActivationType::Relu && 
               act_params.qscheme == QScheme::NoQuantize) {
        epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;

    } else if (act_params.atype == ActivationType::Identity &&
               act_params.qscheme == QScheme::NoQuantize) {
        epilogue = HIPBLASLT_EPILOGUE_BIAS;
    } else {
        // for fp8PerToken quantized Identity activation with bias is a special case currently, need to handle it separately
        if (act_params.atype == ActivationType::Identity && act_params.bias && act_params.qscheme == QScheme::Qfp8PerToken) {
          RTP_LLM_LOG_DEBUG("loraLinearWithActivation is using bias epilogue");
          gemm_params.C = params.activation_params.bias;
          gemm_params.activationType = params.activation_params.atype;
          return loraLinear(gemm_params).output;
        }
        return DeviceBase::loraLinearWithActivation(params);
    }

    if (!act_params.bias || act_params.gate || act_params.gate_bias || act_params.act_scale) {
        return DeviceBase::loraLinearWithActivation(params);
    }

    ROCmGemmActArguments arguments(gemm_params);
    BufferPtr            output;

    if (gemm_params.D) {
        output = gemm_params.D;
        RUNTIME_ASSERT_OP_ARG((arguments.DDtype == gemm_params.D->type())
                                  && (arguments.Dshape == gemm_params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              arguments.DDtype,
                              autil::StringUtil::toString(arguments.Dshape).c_str(),
                              gemm_params.D->debugString().c_str());
    } else {
        output = allocateBuffer({arguments.DDtype, arguments.Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }

    // INT8 GEMM+ GELU Fused kernel for bert
    if (gemm_params.B.type() == DataType::TYPE_QINT8 && gemm_params.A.type() == DataType::TYPE_QINT8 && act_params.atype == ActivationType::Gelu){
        const QBuffer& QA  = reinterpret_cast<const QBuffer&>(gemm_params.A);
        const QBuffer& QB  = reinterpret_cast<const QBuffer&>(gemm_params.B);
        const Buffer QD1 = QA.scales();
        const Buffer QD0 = QB.scales();

        if (QD1.size() == 1)
        {
            // QBuffer A's scale is one single value, need create a new buffer to m x 1 size of this value 
            BufferPtr hQD1 = clone({QD1, AllocationType::HOST});
            torch::Tensor hQD1Tensor = Buffer2torchTensor(hQD1);
            hQD1Tensor = hQD1Tensor.repeat({(long)QA.shape()[0], 1});
            BufferPtr temp = torchTensor2Buffer(hQD1Tensor);
            BufferPtr QD1_EXPEND = clone({*temp});

            auto ck_gemm_params = ckW8A8GemmParam({QA.kernel().data(),
                                                QB.kernel().data(),
                                                QD0.data(),
                                                QD1_EXPEND->data(),
                                                output->data(),
                                                arguments.m,
                                                arguments.n,
                                                arguments.k,
                                                arguments.k,      // arguments.stride_a,
                                                arguments.k,      // arguments.stride_b,
                                                arguments.n,      // arguments.stride_e,
                                                stream_});
            ck_w8a8_gelu_gemm_runner_->runCKW8A8GeluGemm(ck_gemm_params,gemm_params.A.type(),gemm_params.B.type());
            return output;
        }
        else
        {
            auto ck_gemm_params = ckW8A8GemmParam({QA.kernel().data(),
                                                QB.kernel().data(),
                                                QD0.data(),
                                                QD1.data(),
                                                output->data(),
                                                arguments.m,
                                                arguments.n,
                                                arguments.k,
                                                arguments.k,      // arguments.stride_a,
                                                arguments.k,      // arguments.stride_b,
                                                arguments.n,      // arguments.stride_e,
                                                stream_});
            ck_w8a8_gelu_gemm_runner_->runCKW8A8GeluGemm(ck_gemm_params,gemm_params.A.type(),gemm_params.B.type());
        }      
        return output; 
    }

    if (gemm_params.A.type() != DataType::TYPE_QFP8_E4M3 && gemm_params.B.type() == DataType::TYPE_QFP8_E4M3) {
        QBufferPtr q_hidden = std::dynamic_pointer_cast<QBuffer>(
        quantize(QuantizeParams(gemm_params.A, DataType::TYPE_QFP8_E4M3, 1, QScheme::Qfp8PerToken, 0, 0)));

        BufferPtr A_quant_buffer = q_hidden->kernelPtr();
        BufferPtr A_scales = q_hidden->scalesPtr();
        BufferPtr W_kernel = reinterpret_cast<const QBuffer&>(gemm_params.B).kernelPtr();
        BufferPtr W_scales = reinterpret_cast<const QBuffer&>(gemm_params.B).scalesPtr();
        auto A    = A_quant_buffer->data();
        auto B    = W_kernel->data();
        auto D    = output->data();

        auto A_data_type = dtypeConvert(arguments.ADtype);
        auto B_data_type = dtypeConvert(arguments.BDtype);
        auto D_data_type = dtypeConvert(arguments.DDtype);

        if (gemm_params.compute_type == DataType::TYPE_INVALID) {
            hipblasMMWrapperPtr()->setGemmConfig(dtypeConvert(A_quant_buffer->type()), dtypeConvert(W_kernel->type()), 
                                           dtypeConvert(output->type()), HIP_R_32F);
        } else {
            hipblasMMWrapperPtr()->setGemmConfig(
                A_data_type, B_data_type, D_data_type, dtypeConvert(gemm_params.compute_type));
        }

        hipblas_mm_wrapper_->GemmBiasAct(
            (gemm_params.transB == TransposeOperation::TRANSPOSE) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
            (gemm_params.transA == TransposeOperation::TRANSPOSE) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
            arguments.n,
            arguments.m,
            arguments.k,
            B,
            arguments.ldb,
            A,
            arguments.lda,
            D,
            arguments.ldc,
            act_params.bias->get().data(),
            epilogue,
            reinterpret_cast<const float*>(W_scales->data()),
            reinterpret_cast<const float*>(A_scales->data()));
    } else {
        const auto A = gemm_params.A.data();
        const auto B = gemm_params.B.data();
        auto       D = output->data();

        auto A_data_type = dtypeConvert(arguments.ADtype);
        auto B_data_type = dtypeConvert(arguments.BDtype);
        auto D_data_type = dtypeConvert(arguments.DDtype);

        if (gemm_params.compute_type == DataType::TYPE_INVALID) {
            hipblasMMWrapperPtr()->setGemmConfig(A_data_type, B_data_type, D_data_type, HIP_R_32F);
        } else {
            hipblasMMWrapperPtr()->setGemmConfig(
                A_data_type, B_data_type, D_data_type, dtypeConvert(gemm_params.compute_type));
        }

        hipblas_mm_wrapper_->GemmBiasAct(
            (gemm_params.transB == TransposeOperation::TRANSPOSE) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
            (gemm_params.transA == TransposeOperation::TRANSPOSE) ? HIPBLAS_OP_T : HIPBLAS_OP_N,
            arguments.n,
            arguments.m,
            arguments.k,
            B,
            arguments.ldb,
            A,
            arguments.lda,
            D,
            arguments.ldc,
            act_params.bias->get().data(),
            epilogue);
    }

    return output;
}
}  // namespace rtp_llm
