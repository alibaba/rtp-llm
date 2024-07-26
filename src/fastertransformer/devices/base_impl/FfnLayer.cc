#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"

using namespace std;

namespace fastertransformer {

FfnLayerOutput DeviceBase::ffnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(!params.residual, "default FFN implementation does not support residual!");

    BufferPtr output;
    BufferPtr shared_expert_output;

    if (params.weights.moe_gating_weight) {
        output = moeFfnLayer(params).hidden_states;

        // deal with moe layers with parallel dense ffn layer
        if (params.weights.shared_expert) {
            shared_expert_output = ffnLayer({params.input,
                                             params.configs,
                                             *(params.weights.shared_expert),
                                             params.residual, std::nullopt, params.qscheme}).hidden_states;

            // for qwen moe
            // See https://github.com/huggingface/transformers/blob/0f67ba1d741d65b07d549daf4ee157609ce4f9c1/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L803
            if (params.weights.shared_expert_gate) {
                auto shared_gate = gemm({params.input, *(params.weights.shared_expert_gate->kernel)});
                activation({ActivationType::Sigmoid, *shared_gate});
                shared_expert_output = multiply({
                    shared_gate->reshape({shared_gate->size()}), *shared_expert_output});
            }
        }
    } else {
        auto up_gemm_params = GemmParams(params.input, *(params.weights.up_weight->kernel));
        auto up_output = loraLinear(LoraLinearParams(up_gemm_params,
                                                     *(params.weights.up_lora_weights),
                                                     params.lora_input)).output;
        printBufferData(*up_output, "ffn_up");

        if (isGatedActivation(params.configs.activation_type)) {
            auto gate_gemm_params = GemmParams(params.input, *(params.weights.gate_weight->kernel));
            auto gate_output = loraLinear(LoraLinearParams(gate_gemm_params,
                                                           *(params.weights.gate_lora_weights),
                                                           params.lora_input));

            activation({params.configs.activation_type,
                        *(up_output),
                        mayGetRef(params.weights.up_weight->bias),
                        *(gate_output.output),
                        std::nullopt,
                        mayGetRef(params.weights.act_scale)});
        } else {
            activation({params.configs.activation_type,
                        *(up_output),
                        mayGetRef(params.weights.up_weight->bias),
                        std::nullopt,
                        std::nullopt,
                        mayGetRef(params.weights.act_scale)});
        }

        if (params.qscheme != QScheme::NoQuantize) {
            auto quant_params = QuantizeParams(
                *up_output,
                DataType::TYPE_QINT8,
                1,
                params.qscheme,
                params.weights.smoother_weight ? (OptionalConstBufferRef) * (params.weights.smoother_weight->kernel) :
                                                 std::nullopt,
                std::nullopt,
                params.weights.intermediate_weight2_static_scale_weight ?
                    (OptionalConstBufferRef) * (params.weights.intermediate_weight2_static_scale_weight->kernel) :
                    std::nullopt,
                params.weights.intermediate_weight2_static_scale_reciprocal_weight ?
                    (OptionalConstBufferRef)
                        * (params.weights.intermediate_weight2_static_scale_reciprocal_weight->kernel) :
                    std::nullopt);
            up_output = quantize(quant_params);
        }

        printBufferData(*up_output, "ffn_act");
        auto down_gemm_params = GemmParams(*(up_output), *(params.weights.down_weight->kernel), nullopt, params.output);
        output = loraLinear(LoraLinearParams(down_gemm_params,
                                             *(params.weights.down_lora_weights),
                                             params.lora_input)).output;
    }

    if (shared_expert_output) {
        shared_expert_output = layernorm({
            output, nullptr, nullopt, mayGetRef(shared_expert_output)
        }).output;
    }

    printBufferData(*output, "ffn_out");
    return FfnLayerOutput({std::move(output)});
}

}; // namespace fastertransformer
