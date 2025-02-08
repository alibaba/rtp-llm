// #include "devices/OpData.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/cuda/Dispatch.h"

// kernels
#include "src/fastertransformer/kernels/moe_topKSoftmax_kernels.h"

using namespace std;

namespace fastertransformer {

FfnLayerOutput ROCmDevice::moeFfnLayer(const FfnLayerParams& params) {
    // printf("[ROCM] moeFfnLayer: token_num=%d\n", params.input.shape()[0]);
    RUNTIME_ASSERT_OP_ARG(params.configs.moe_configs, "moe configs not set");

    const auto& moe_conf      = params.configs.moe_configs.value();
    const auto& hidden        = params.input;
    const auto& weights       = params.weights;
    const auto  compute_type  = hidden.type();
    const auto  weights_type  = weights.moe_down_weight->kernel->type();
    const auto  num_token     = hidden.shape()[0];
    const auto  model_dim     = hidden.shape()[1];
    const auto  num_expert    = params.weights.moe_gating_weight->kernel->shape()[1];
    const auto  inter_dim     = weights.moe_gate_weight->kernel->shape()[1];  //(size_t)moe_conf.moe_inter_padding_size;
    const auto  top_k         = moe_conf.top_k;
    const int   fused_quant   = (params.qscheme == QScheme::NoQuantize) ? 0 : 1;
    const int   gate_only     = (inter_dim == weights.moe_down_weight->kernel->shape()[2]) ? 1 : 0;
    const auto  inter_dim_per = inter_dim / (gate_only ? 1 : 2);
    // RUNTIME_ASSERT_OP_ARG(weights.moe_down_weight->kernel->shape()[2] == inter_dim_per,
    //                       "Intermedate_size not the same for gate (%d) and down(%d).",
    //                       inter_dim_per,
    //                       weights.moe_down_weight->kernel->shape()[1]);

    // TODO: cuda version also not init this
    MOEParallelismConfig parallelism_config;
    // const size_t inter_dim_per_node = inter_dim / parallelism_config.ep_size;
    const size_t num_experts_per_node = num_expert / parallelism_config.ep_size;
    const int    start_expert         = num_experts_per_node * parallelism_config.ep_rank;
    const int    end_expert           = start_expert + num_experts_per_node;
    // TODO group_size
    // auto group_size = 0;
    // if (params.weights.moe_gate_weight->kernel->isQBuffer()) {
    //     if (dynamic_cast<const QBuffer*>(params.weights.moe_gate_weight->kernel.get())->zerosData() != nullptr) {
    //         group_size =
    //             params.weights.moe_gate_weight->kernel->shape()[1]
    //             / dynamic_cast<const QBuffer*>(params.weights.moe_gate_weight->kernel.get())->zeros().shape()[1];
    //     }
    // }
    const auto gate = gemm({hidden, *params.weights.moe_gating_weight->kernel, nullopt, nullptr, DataType::TYPE_FP32});

    const size_t num_topkTokens = num_token * top_k;
    const auto   softmax_out    = allocateBuffer(
        {DataType::TYPE_FP32,
              {((num_experts_per_node & (num_experts_per_node - 1)) == 0) ? 0 : num_token * num_experts_per_node}});
    const auto topk_scales   = allocateBuffer({DataType::TYPE_FP32, {num_token, top_k}});
    const auto topk_expertID = allocateBuffer({DataType::TYPE_INT32, {num_token, top_k}});
    const auto topk_rowColID = allocateBuffer({DataType::TYPE_INT32, {num_token, top_k}}, {"topk_rowColID"});
    topkGatingSoftmax_KL(gate->data<float>(),
                         nullptr,                     // finished
                         softmax_out->data<float>(),  // softmax_out
                         topk_scales->data<float>(),
                         topk_expertID->data<int>(),
                         topk_rowColID->data<int>(),
                         num_token,
                         num_experts_per_node,
                         top_k,
                         start_expert,
                         end_expert,
                         stream_);

    // keep print cmd
    // printBufferData(hidden, "rocm_moe_input_token", nullptr, true);
    // printBufferData(*(params.weights.moe_gating_weight->kernel), "moe_gating", nullptr, true);
    // printBufferData(*gate, "topk_gate", nullptr, true);
    // printBufferData(*topk_scales, "topk_scales", nullptr, true);
    // printBufferData(*topk_expertID, "topk_expertID", nullptr, true);
    // printBufferData(*(weights.moe_gate_weight->kernel), "rocm_moe_input_gate", nullptr, true);
    // printBufferData(*(weights.moe_down_weight->kernel), "rocm_moe_input_down", nullptr, true);
    // printBufferData(*(gate), "rocm_moe_sorting_gate", nullptr, true);

    BufferPtr output = nullptr;
    if (params.output) {
        output = params.output;
    } else {
        output = allocateBuffer({compute_type, {num_token, model_dim}});
    }

    size_t block_m = 32;  // temporarily support only tile_size 32
    size_t stride  = model_dim;

    // buffer for sorting passing to expert ffn.
    size_t max_num_tokens_padded = top_k * num_token + num_expert * block_m - top_k;
    auto   sorted_token_ids_ptr  = allocateBuffer({DataType::TYPE_INT32, {max_num_tokens_padded}});
    auto   sorted_weight_ptr     = allocateBuffer({DataType::TYPE_FP32, {max_num_tokens_padded}});
    auto   sorted_expert_ids_ptr =
        allocateBuffer({DataType::TYPE_INT32, {(max_num_tokens_padded + block_m - 1) / block_m}});
    auto num_sorted_tiles_ptr = allocateBuffer({DataType::TYPE_INT32, {1}});

    // assert(inter_dim == params.weights.moe_gate_weight->kernel->shape()[1]); // not support row major.
    auto moeParams = rocmMoeParams(
        {hidden.data(),
         nullptr,  // no a scale ?
         weights.moe_gate_weight->kernel->data(),
         BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_gate_weight->kernel),
         weights.moe_down_weight->kernel->data(),
         BUFFER_GET_SCALE_IF_Q_BUFFER(weights.moe_down_weight->kernel),
         weights.smoother_weight ? BUFFER_GET_SCALE_IF_Q_BUFFER(weights.smoother_weight->kernel) : nullptr,
         output->data(),
         topk_expertID->data(),
         topk_scales->data(),
         sorted_token_ids_ptr->data(),
         sorted_weight_ptr->data(),
         sorted_expert_ids_ptr->data(),
         num_sorted_tiles_ptr->data(),
         block_m,
         model_dim,
         inter_dim_per,
         num_token,
         num_expert,
         top_k,
         stride,
         stream_});
    uint32_t ckmoe_workspace_sz = moe_runner_->runCKMoe(
        moeParams, compute_type, weights_type, params.configs.activation_type, fused_quant, gate_only);

    /*

    // todo: moe_norm not implemented, this dispatch need be fused into runCKMoe
    assert(moe_conf.has_moe_norm == false);
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(
        compute_type,
        finalizeMoeRoutingKernelLauncher,
        hidden_permuted->data(),
        output->data(),
        nullptr,  // skip_1
        nullptr,  // skip_2
        nullptr,  // bias
        topk_scales->data<float>(),
        expanded_source_row_to_expanded_dest_row->data<int>(),
        topk_expertID->data<int>(),
        num_token,
        model_dim,
        top_k,
        nullptr,  // num_valid_ptr
        parallelism_config.tp_rank,
        moe_conf.has_moe_norm ? 2 : 1,  // renormalization_mode, 0: no norm, 1: * topk_scale, 2: renorm_scale
        stream_);
    */
    // printBufferData(*(output), "rocm_moe_output", nullptr, true);
    return FfnLayerOutput({move(output)});
}

FfnLayerOutput ROCmDevice::ffnLayer(const FfnLayerParams& params) {
    RUNTIME_ASSERT_OP_ARG(!params.residual, "default FFN implementation does not support residual!");

    BufferPtr output;
    BufferPtr shared_expert_output;

    if (params.weights.moe_gating_weight) {
        output = moeFfnLayer(params).hidden_states;

        // deal with moe layers with parallel dense ffn layer
        if (params.weights.shared_expert) {
            auto ffn_params = FfnLayerParams({params.input,
                                             params.configs,
                                             *(params.weights.shared_expert),
                                             params.residual, params.qscheme});
            ffn_params.lora_input = params.lora_input;
            shared_expert_output = ffnLayer(ffn_params).hidden_states;

            // for qwen moe
            // See https://github.com/huggingface/transformers/blob/0f67ba1d741d65b07d549daf4ee157609ce4f9c1/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L803
            if (params.weights.shared_expert_gate) {
                auto shared_gate = gemm({params.input, *(params.weights.shared_expert_gate->kernel)});
                activation({ActivationType::Sigmoid, shared_gate});
                shared_expert_output = multiply({
                    shared_gate->reshape({shared_gate->size()}), *shared_expert_output});
            }
        }
    } else {
        BufferPtr up_output;
        if (isGatedActivation(params.configs.activation_type)) {
            current_stream_ = assist_stream_;
            auto up_gemm_params = GemmParams(params.input, *(params.weights.up_weight->kernel));
            up_output = loraLinear(LoraLinearParams(up_gemm_params, params.lora_input.up_lora_input)).output;
            
            current_stream_ = stream_;
            auto gate_gemm_params = GemmParams(params.input, *(params.weights.gate_weight->kernel));
            auto gate_output = loraLinear(LoraLinearParams(gate_gemm_params,  params.lora_input.gate_lora_input));

            ROCM_CHECK(hipStreamSynchronize(assist_stream_));
            current_stream_ = stream_;
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
        output = loraLinear(LoraLinearParams(down_gemm_params, params.lora_input.down_lora_input)).output;
    }

    if (shared_expert_output) {
        shared_expert_output = layernorm({
            output, nullptr, nullopt, mayGetRef(shared_expert_output)
        }).output;
    }

    printBufferData(*output, "ffn_out");
    return FfnLayerOutput({std::move(output)});
}

}  // namespace fastertransformer
