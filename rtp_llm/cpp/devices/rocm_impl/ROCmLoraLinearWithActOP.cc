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
        default:
            ROCM_FAIL("[GEMM]: Other DataType not implemented");
    }
};

BufferPtr ROCmDevice::loraLinearWithActivation(const LoraLinearWithActivationParams& params) {
    auto gemm_params = params.lora_linear_params.gemm_params;
    auto act_params  = params.activation_params;

    hipblasLtEpilogue_t epilogue;
    if (act_params.atype == ActivationType::Gelu) {
        epilogue = HIPBLASLT_EPILOGUE_GELU_BIAS;
    } else if (act_params.atype == ActivationType::Relu) {
        epilogue = HIPBLASLT_EPILOGUE_RELU_BIAS;

    } else if (act_params.atype == ActivationType::Identity) {
        epilogue = HIPBLASLT_EPILOGUE_BIAS;
    } else {
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

    return output;
}
}  // namespace rtp_llm
