#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/utils/DevicePerfWrapper.h"
#include <numeric>

using namespace std;

namespace fastertransformer {

FfnLayerOutput DeviceBase::ffnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(!params.residual, "default FFN implementation does not support residual!");
    BufferPtr output;
    if (params.weights.moe_gating_weight) {
        RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");

        const auto& moe_conf = params.configs.moe_configs.value();
        BufferPtr shared_expert_output;
        output = moeFfnLayer(params).hidden_states;
        // deal with moe layers with parallel dense ffn layer
        if (params.weights.shared_expert) {
            shared_expert_output = allocateBufferLike({params.input}, AllocationType::DEVICE, {"shared_expert_buf"});
            shared_expert_output = prepareAllReduce({std::move(shared_expert_output), ReduceOp::Sum}).buffer;
            auto ffn_params = FfnLayerParams({params.input,
                                             params.configs,
                                             *(params.weights.shared_expert),
                                             params.residual,
                                             params.qscheme,
                                             shared_expert_output});
            ffn_params.lora_input = params.lora_input;
            shared_expert_output = ffnLayer(ffn_params).hidden_states;

            // for qwen moe
            // See https://github.com/huggingface/transformers/blob/0f67ba1d741d65b07d549daf4ee157609ce4f9c1/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L803
            if (params.weights.shared_expert_gate) {
                auto shared_gate = gemm({params.input, *(params.weights.shared_expert_gate->kernel)});
                activation({ActivationType::Sigmoid, shared_gate});
                shared_expert_output = multiply({
                        shared_gate->reshape({shared_gate->size()}), *shared_expert_output, shared_expert_output});
            }
            if (moe_conf.tp_size > 1) {
                auto wrapper = DevicePerfWrapper(this, "shared_expert_all_reduce, sizeBytes=%ld", (long)shared_expert_output->sizeBytes());
                shared_expert_output = allReduce({shared_expert_output, ReduceOp::Sum}).buffer;
            }
        }
        overlappedCommBarrier();
        printBufferData(*output, "moe_out_after_barrier");
        if (shared_expert_output) {
            // just add bias to output
            layernorm({
                output, nullptr, nullopt, mayGetRef(shared_expert_output)
            }).output;
        }
    } else {
        BufferPtr up_output;
        if (isGatedActivation(params.configs.activation_type)) {
            auto up_gemm_params = GemmParams(params.input, *(params.weights.up_weight->kernel));
            up_output = loraLinear(LoraLinearParams(up_gemm_params, params.lora_input.up_lora_input)).output;
            printBufferData(*up_output, "ffn_up");
            auto gate_gemm_params = GemmParams(params.input, *(params.weights.gate_weight->kernel));
            auto gate_output = loraLinear(LoraLinearParams(gate_gemm_params,  params.lora_input.gate_lora_input));
            printBufferData(*gate_output.output, "ffn_gate");
            activation({params.configs.activation_type,
                        up_output,
                        mayGetRef(params.weights.up_weight->bias),
                        *(gate_output.output),
                        std::nullopt,
                        mayGetRef(params.weights.act_scale)});
        } else {
            auto up_gemm_params = GemmParams(params.input, *(params.weights.up_weight->kernel));
            auto lora_linear_params = LoraLinearParams(up_gemm_params,  params.lora_input.up_lora_input);
            auto activation_params  = ActivationParams(params.configs.activation_type,
                                                      nullptr,
                                                      mayGetRef(params.weights.up_weight->bias),
                                                      std::nullopt,
                                                      std::nullopt,
                                                      mayGetRef(params.weights.act_scale));
            up_output = loraLinearWithActivation({lora_linear_params, activation_params});
        }

        if (params.qscheme != QScheme::NoQuantize) {
	        DataType quant_out_data_type = params.qscheme == QScheme::Qfp8PerTensor ? DataType::TYPE_FP8_E4M3 : DataType::TYPE_INT8;
            auto quant_params = QuantizeParams(
                *up_output,
                quant_out_data_type,
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
        output = loraLinear(LoraLinearParams(down_gemm_params, params.lora_input.down_lora_input)).output;
    }

    printBufferData(*output, "ffn_out");
    return FfnLayerOutput({std::move(output)});
}

}; // namespace fastertransformer
