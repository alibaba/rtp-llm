#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"

#include <numeric>

using namespace std;

namespace fastertransformer {

AttentionLayerOutput DeviceBase::attentionLayer(const AttentionLayerParams& params) {
    const auto &input = params.input;
    const auto &input_lengths = params.common.input_lengths;
    const auto &sequence_lengths = params.common.sequence_lengths;

    const auto &output_weight = params.weights.output_weight;

    const auto generate_batch_size = sequence_lengths.shape()[0];
    const auto context_batch_size = input_lengths.shape()[0] - generate_batch_size;
    const auto context_token_num = params.common.context_token_num;
    const auto h_token_num = context_token_num + generate_batch_size;

    RUNTIME_ASSERT_OP_ARG(!params.residual, "default attention layer impl does not support residual!");

    const auto& layer_kv_cache = params.common.kv_cache;
    if (layer_kv_cache) {
        const auto &kv_cache = layer_kv_cache.value();
        const auto &kv_cache_offset = *kv_cache.kv_cache_offset;
        const auto &shape = kv_cache.kv_cache_offset->shape();
        RUNTIME_ASSERT_OP_ARG(
            ((shape.size() == 2) && (shape[0] == input_lengths.shape()[0])),
            "kv_cache_offset shape in attention layer should be [batch_size, block_length]"
            ", but got %s", kv_cache_offset.debugString().c_str());
        RUNTIME_ASSERT_OP_ARG(
                kv_cache.k_cache_buffer && kv_cache.v_cache_buffer,
                "kv cache buffer should has value when use kv_cache_offset");
        const auto& k_cache_shape = kv_cache.k_cache_buffer->shape();
        const auto& v_cache_shape = kv_cache.v_cache_buffer->shape();
        RUNTIME_ASSERT_OP_ARG(
                ((k_cache_shape.size() == 4) && (v_cache_shape.size() == 4) && \
                 (k_cache_shape[0] == v_cache_shape[0]) && (k_cache_shape[1] == v_cache_shape[1]) && \
                 (k_cache_shape[2] == v_cache_shape[2]) && (k_cache_shape[3] == v_cache_shape[3]) && \
                 (k_cache_shape[1] == params.configs.kv_head_num) && \
                 (k_cache_shape[2] == params.configs.tokens_per_block) && \
                 (k_cache_shape[3] == params.configs.size_per_head)),
                "kv cache buffer check shape failed. k_cache_buffer: %s, v_cache_buffer: %s",
                kv_cache.k_cache_buffer->debugString().c_str(),
                kv_cache.v_cache_buffer->debugString().c_str());
        if (kv_cache.k_scale_buffer) {
            const auto& k_scale_shape = kv_cache.k_scale_buffer->shape();
            const auto& v_scale_shape = kv_cache.v_scale_buffer->shape();
            RUNTIME_ASSERT_OP_ARG(
                    ((k_scale_shape.size() == 3) && (v_scale_shape.size() == 3) && \
                     (k_scale_shape[0] == v_scale_shape[0]) && (k_scale_shape[1] == v_scale_shape[1]) && \
                     (k_scale_shape[2] == v_scale_shape[2]) && (k_cache_shape[0] == k_scale_shape[0]) && \
                     (k_scale_shape[1] == params.configs.kv_head_num) && \
                     (k_scale_shape[2] == params.configs.tokens_per_block)),
                    "kv scale check buffer failed. k_scale_buffer: %s, v_scale_buffer: %s",
                    kv_cache.k_scale_buffer->debugString().c_str(),
                    kv_cache.v_scale_buffer->debugString().c_str());
        }
    }

    // typically local_head_num * size_per_head
    const auto qkv_hidden_size = output_weight->kernel->shape()[0];

    // NOTE: Cuda implementation fused adding qkv_weight->bias in invokeAddFusedQKVBiasTranspose kernel call.
    // other devices need to be careful about this.
    // maybe add a device property here.
    auto qkv = params.configs.use_mla ? mlaQKVGemm(params): mhaQKVGemm(params);

    // attention layer output is preallocated to avoid memory fragmentation
    // note that this output is returned and further used as residual
    auto dtype = (input.isQBuffer() ? qkv->type() : input.type());
    auto output = params.output ? params.output
                : allocateBuffer({dtype, {h_token_num, output_weight->kernel->shape()[1]}},
                                 {"attn_layer_out"});

#if defined(__aarch64__)
    // Arm attention op only support fp32 data type
    auto qkv_output = allocateBuffer({DataType::TYPE_FP32, {h_token_num, qkv_hidden_size}}, {"qkv_output"});
#else
    auto qkv_output = allocateBuffer({dtype, {h_token_num, qkv_hidden_size}}, {"qkv_output"});
#endif 

    auto kv_cache_offset = layer_kv_cache ? layer_kv_cache->kv_cache_offset : nullptr;
    if (generate_batch_size) {
        auto generate_qkv = qkv->view(0, generate_batch_size);
        auto generate_output = qkv_output->view(0, generate_batch_size);
        if (layer_kv_cache) {
            params.common.kv_cache->kv_cache_offset = kv_cache_offset->slice(0, generate_batch_size);
        }
        decoderSelfAttention({params.layer_id, generate_qkv, generate_output, params.common, params.weights, params.configs});
    }
    if (context_batch_size) {
        auto context_qkv = qkv->view(generate_batch_size, context_token_num);
        auto context_output = qkv_output->view(generate_batch_size, context_token_num);
        if (layer_kv_cache) {
            params.common.kv_cache->kv_cache_offset = kv_cache_offset->slice(generate_batch_size, context_batch_size);
        }
        contextAttention({params.layer_id, context_qkv, context_output, params.common, params.weights, params.configs});
    }
    printBufferData(*qkv_output, "qkv_output");

    if(params.qscheme != QScheme::NoQuantize) {
        OptionalConstBufferRef smoother_weight =
            params.weights.smoother_weight ? (OptionalConstBufferRef) * (params.weights.smoother_weight->kernel) :
                                             std::nullopt;

        OptionalConstBufferRef shift_weight = (params.weights.shift_weight == nullptr) ?
                                                  nullopt :
                                                  (OptionalConstBufferRef)*params.weights.shift_weight->kernel;

        OptionalConstBufferRef static_scale_weight =
            params.weights.static_quant_weight ?
                (OptionalConstBufferRef) * (params.weights.static_quant_weight->kernel) :
                std::nullopt;

        OptionalConstBufferRef static_scale_reciprocal_weight =
            params.weights.static_scale_reciprocal_weight ?
                (OptionalConstBufferRef) * (params.weights.static_scale_reciprocal_weight->kernel) :
                std::nullopt;

        auto quant_params = QuantizeParams(
            *qkv_output,
            DataType::TYPE_QINT8,
            1,
            params.qscheme,
            smoother_weight,
            shift_weight,
            static_scale_weight,
            static_scale_reciprocal_weight);

        BufferPtr quantized_attention_output = quantize(quant_params);

        auto output_gemm_params = GemmParams(*quantized_attention_output, *(output_weight->kernel), nullopt, output);
        loraLinear(LoraLinearParams(output_gemm_params, params.common.lora_input.out_lora_input)).output;
    } else {
        auto output_gemm_params = GemmParams(*qkv_output, *(output_weight->kernel), nullopt, output, dtype);
        loraLinear(LoraLinearParams(output_gemm_params, params.common.lora_input.out_lora_input)).output;
    }

    return {std::move(output)};
}

}; // namespace fastertransformer
