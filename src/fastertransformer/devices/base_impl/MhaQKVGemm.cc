#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"

namespace fastertransformer {

BufferPtr DeviceBase::mhaQKVGemm(const AttentionLayerParams& params) {
    const auto& input      = params.input;
    const auto& qkv_weight = params.weights.qkv_weight;

    // typically local_head_num * size_per_head + 2 * local_head_num_kv * size_per_head
    const auto qkv_merged_size = qkv_weight->kernel->shape()[1];

#if defined(__aarch64__)
    // Arm attention op only support fp32 data type
    auto qkv_gemm_params = GemmParams(input, *(qkv_weight->kernel), std::nullopt, nullptr, DataType::TYPE_FP32);
#else
    auto qkv_gemm_params = GemmParams(input, *(qkv_weight->kernel));
#endif

    auto lora_linear_params = LoraLinearParams(qkv_gemm_params, params.common.lora_input.qkv_lora_input);
    BufferPtr qkv;
    if (!params.configs.fuse_qkv_add_bias && params.weights.qkv_weight) {
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
        auto after_q_norm = layernorm(LayernormParams(
            qkv, *params.weights.q_norm_weight, params.ln_params.eps, params.ln_params.norm_type, 0, qkv_merged_size));

        qkv = std::move(after_q_norm.output);
        printBufferData(*qkv, "qkv_after_q_norm");
    }

    if (params.weights.k_norm_weight) {
        auto after_k_norm = layernorm(LayernormParams(qkv,
                                                      *params.weights.k_norm_weight,
                                                      params.ln_params.eps,
                                                      params.ln_params.norm_type,
                                                      params.configs.size_per_head * params.configs.head_num,
                                                      qkv_merged_size));

        qkv = std::move(after_k_norm.output);
        printBufferData(*qkv, "qkv_after_k_norm");
    }

    return qkv;
}
}  // namespace fastertransformer