#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"

using namespace std;

namespace fastertransformer {

FfnLayerOutput DeviceBase::ffnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(!params.residual, "default FFN implementation does not support residual!");

    BufferPtr output;
    BufferPtr shared_expert_output;

    if (params.weights.moe_gating_weight) {
        output = moeFfnLayer(params).hidden_states;

        // for qwen moe
        // See https://github.com/huggingface/transformers/blob/0f67ba1d741d65b07d549daf4ee157609ce4f9c1/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L803
        if (params.weights.shared_expert) {
            auto shared_expert_params = params;
            shared_expert_output = ffnLayer({params.input,
                                             params.configs,
                                             *(params.weights.shared_expert),
                                             params.residual}).hidden_states;

            auto shared_gate = gemm({params.input, *(params.weights.shared_expert_gate->kernel)});
            activation({ActivationType::Sigmoid, *shared_gate});
            shared_expert_output = dotProduct({
                shared_gate->reshape({shared_gate->size()}), *shared_expert_output});
        }
    } else {
        const auto& input = params.input;
        const auto& up_weight = *(params.weights.up_weight->kernel);
        const auto& down_weight = *(params.weights.down_weight->kernel);

        auto up_output = loraLinear({params.input,
                                    std::nullopt,
                                    *(params.weights.up_weight),
                                    std::nullopt}).output;
        printBufferData(*up_output, "ffn_up");

        if (isGatedActivation(params.configs.activation_type)) {
            auto gate_output = loraLinear({params.input,
                                        std::nullopt,
                                        *(params.weights.gate_weight),
                                        std::nullopt});

            activation({params.configs.activation_type,
                        *(up_output),
                        mayGetRef(params.weights.up_weight->bias),
                        *(gate_output.output),
                        std::nullopt});
        } else {
            activation({params.configs.activation_type,
                        *(up_output),
                        mayGetRef(params.weights.up_weight->bias),
                        std::nullopt,
                        std::nullopt});
        }

        if (params.weights.smoother_weight != nullptr) {
            up_output = quantize(QuantizeParams(
                *up_output, *(params.weights.smoother_weight->kernel),
                std::nullopt, DataType::TYPE_QINT8, 1));
        }

        printBufferData(*up_output, "ffn_act");
        output = loraLinear({*(up_output),
                            std::nullopt,
                            *(params.weights.down_weight),
                            std::nullopt}).output;
    }

    if (shared_expert_output || params.weights.dense_layernorm) {
        shared_expert_output = layernorm({
            output, nullptr, mayGetRef(params.weights.dense_layernorm),
            mayGetRef(shared_expert_output)
        }).output;
    }

    printBufferData(*output, "ffn_out");
    return FfnLayerOutput({move(output)});
}

}; // namespace fastertransformer
