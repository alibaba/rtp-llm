#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

namespace rtp_llm {

BufferPtr DeviceBase::mhaQKVGemm(const AttentionLayerParams& params) {
    const auto& input      = params.input;
    const auto& qkv_weight = params.weights.qkv_weight;

#if defined(__aarch64__)
    // Arm attention op only support fp32 data type
    auto qkv_gemm_params = GemmParams(input, *(qkv_weight->kernel), std::nullopt, nullptr, DataType::TYPE_FP32);
#else
    auto qkv_gemm_params =
        GemmParams(input, *(qkv_weight->kernel), std::nullopt, nullptr, DataType::TYPE_INVALID, params.compute_type);
#endif

    auto      lora_linear_params = LoraLinearParams(qkv_gemm_params, params.common.lora_input.qkv_lora_input);
    BufferPtr qkv;
    if (!params.configs.fuse_qkv_add_bias && params.weights.qkv_weight->bias) {
        ActivationParams act_params(ActivationType::Identity,
                                    nullptr,
                                    mayGetRef(params.weights.qkv_weight->bias),
                                    std::nullopt,
                                    std::nullopt,
                                    std::nullopt);
        qkv = loraLinearWithActivation(LoraLinearWithActivationParams(lora_linear_params, act_params));
    } else {
        qkv = loraLinear(LoraLinearParams(qkv_gemm_params, params.common.lora_input.qkv_lora_input)).output;
    }
    printBufferData(*qkv, "qkv");
    if (params.weights.q_norm_weight) {
        RTP_LLM_CHECK_WITH_INFO(params.weights.k_norm_weight != nullptr,
                                "q_norm_weight and k_norm_weight should both be provided");
        RTP_LLM_CHECK_WITH_INFO(params.ln_params.norm_type == NormType::rmsnorm, "qkRmsNorm only support rmsnorm");
        auto qk_rmsnorm_output = qkRmsNorm(QkRmsNormParams({qkv,
                                                            *params.weights.q_norm_weight,
                                                            *params.weights.k_norm_weight,
                                                            params.ln_params.eps,
                                                            params.configs.head_num,
                                                            params.configs.kv_head_num,
                                                            params.configs.size_per_head}));
        printBufferData(*qkv, "qkv_after_qk_norm");
    }

    return qkv;
}
}  // namespace rtp_llm