#include "maga_transformer/cpp/models/GptModel.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"

using namespace std;

namespace rtp_llm {

GptModel::GptModel(const GptModelInitParams& params)
    : device_(params.device)
    , weights_(params.weights)
    , description_(params.description)
    {};

void getPaddingOffsetAndCuSeqLens(int32_t*       padding_offset,
                                  int32_t*       cu_seqlens,
                                  const int32_t* sequence_length,
                                  const int32_t  batch_size,
                                  const int32_t  max_seq_len)
{
    // do cumulated sum
    int32_t        total_seq_len        = 0;
    int32_t        cum_offset           = 0;
    int32_t        index                = 0;
    for (int32_t i = 0; i < batch_size; i++) {
        const int32_t seq_len = sequence_length[i];
        cu_seqlens[i] = total_seq_len;
        for (int32_t j = 0; j < seq_len; j++) {
            padding_offset[index] = cum_offset;
            index++;
        }
        cum_offset += max_seq_len - seq_len;
        total_seq_len += seq_len;
    }
    cu_seqlens[batch_size] = total_seq_len;
}

void checkKvBlocksShape(const BufferPtr& input_kv_blocks) {
    if (!input_kv_blocks) {
        return;
    }
    RUNTIME_ASSERT_OP_ARG(
        input_kv_blocks->shape().size() == 4,
        "kv_cache_blocks shape should be [layer_num, 2, batch_size, block_length].");
}

AttentionCommonInputs GptModel::prepareAttentionInputs(const GptModelInputs& inputs) {
    const auto decoder_batch_size = inputs.sequence_lengths->shape()[0];
    const auto context_batch_size = inputs.input_lengths->shape()[0] - decoder_batch_size;
    const auto max_context_seq_len = context_batch_size ? *std::max_element(
        inputs.input_lengths->data<int32_t>() + decoder_batch_size,
        inputs.input_lengths->data<int32_t>() + decoder_batch_size + context_batch_size) : 0;
    const auto max_decoder_seq_len = decoder_batch_size ? *std::max_element(
        inputs.sequence_lengths->data<int32_t>(),
        inputs.sequence_lengths->data<int32_t>() + decoder_batch_size) : 0;

    std::vector<int32_t> cu_seqlens_data(context_batch_size + 1);
    std::vector<int32_t> padding_offset_data(inputs.combo_tokens->shape()[0]);
    getPaddingOffsetAndCuSeqLens(
        padding_offset_data.data(),
        cu_seqlens_data.data(),
        inputs.input_lengths->dataWithOffset<int32_t>(decoder_batch_size),
        context_batch_size,
        max_context_seq_len);

    RUNTIME_ASSERT_OP_ARG(
        (cu_seqlens_data[context_batch_size] == inputs.combo_tokens->shape()[0]),
        "cu_seqlens is not consistent with combo_tokens.");
    checkKvBlocksShape(inputs.kv_cache_blocks);
    checkKvBlocksShape(inputs.kv_cache_scales);

    AttentionCommonInputs attention_inputs({
        *inputs.input_lengths,
        *inputs.sequence_lengths
    });
    attention_inputs.cu_seqlens = device_->clone({*vector2Buffer(cu_seqlens_data)});
    attention_inputs.padding_offset = device_->clone({*vector2Buffer(padding_offset_data)});
    attention_inputs.decoder_batch_size = decoder_batch_size;
    attention_inputs.context_batch_size = context_batch_size;
    attention_inputs.context_max_seq_len = max_context_seq_len;
    attention_inputs.decoder_max_seq_len = max_decoder_seq_len;
    attention_inputs.position_ids = inputs.position_ids;
    attention_inputs.attention_mask = inputs.attention_mask;
    return move(attention_inputs);
}

GptModelOutputs GptModel::forward(const GptModelInputs& inputs) {
    const auto hidden_type = datatype_enum::TYPE_BF16;
    const auto norm_type = description_.norm_type;
    const auto norm_eps = description_.layernorm_eps;

    const auto batch_size = inputs.input_lengths->shape()[0];
    const auto& combo_tokens = inputs.combo_tokens;
    const auto& embedding_table = weights_.embedding->kernel;
    const auto hidden_size = embedding_table->shape()[1];

    // word embedding lookup
    auto hidden = device_->embeddingLookup({*combo_tokens, *embedding_table, nullopt, nullopt});

    // pre layernorm
    if (weights_.pre_decoder_layernorm) {
        device_->layernorm(LayernormParams(
            *hidden, *hidden, nullopt, norm_type, *(weights_.pre_decoder_layernorm), norm_eps));
    }

    // prepare resources for all layers
    auto attention_common_inputs = prepareAttentionInputs(inputs);
    auto& input_kv_blocks = inputs.kv_cache_blocks;
    auto& input_kv_scales = inputs.kv_cache_scales;
    RUNTIME_ASSERT_OP_ARG(input_kv_blocks, "kv_cache_blocks is required for GPT model.");

    printBufferData(*hidden, "input_hidden");

    // layers
    const auto layer_num = weights_.layers.size();
    for (int i = 0; i < layer_num; ++i) {
        const auto& layer = weights_.layers[i];

        auto residual = device_->allocateBuffer({hidden->type(), hidden->shape()}, {});
        device_->copy({*residual, *hidden});
        if (layer.pre_layernorm) {
            device_->layernorm(LayernormParams(
                *hidden, *hidden, nullopt, norm_type,
                *layer.pre_layernorm, norm_eps));
        }

        auto layer_kv_blocks = (*input_kv_blocks)[i];
        attention_common_inputs.kv_cache_blocks = layer_kv_blocks;
        auto attn_output = device_->attentionLayer(AttentionLayerParams({
            *hidden,
            description_.attention_conf,
            layer.self_attention_weights,
            attention_common_inputs,
        }));
        auto attn_hidden = move(attn_output.hidden_states);
        printBufferData(*attn_hidden, "layer_" + to_string(i) + "_attn_output");

        if (layer.post_layernorm) {
            // attn_hidden = attn_hidden + residual
            // hidden = layernorm(attn_hidden)
            device_->layernorm(LayernormParams(
                *attn_hidden, *hidden, *attn_hidden,
                norm_type, mayGetRef(layer.post_layernorm), norm_eps, *residual));
            residual.swap(attn_hidden);
        } else {
            hidden.swap(attn_hidden);
        }

        printBufferData(*hidden, "layer_" + to_string(i) + "_ffn_input");
        auto ffn_output = device_->ffnLayer(FfnLayerParams({
            *hidden,
            layer.ffn_weights,
            description_.activation_type,
        }));
        hidden.swap(ffn_output.hidden_states);
        printBufferData(*hidden, "layer_" + to_string(i) + "_ffn_output");

        // TODO: maybe move this layernorm to ffn layer
        device_->layernorm(LayernormParams(
            *hidden, *hidden, nullopt,
            norm_type, mayGetRef(layer.post_ffn_layernorm), norm_eps, *residual
        ));
        printBufferData(*hidden, "layer_" + to_string(i) + "_final_hidden");
    }

    // final layernorm
    if (weights_.final_layernorm) {
        device_->layernorm(LayernormParams(
            *hidden, *hidden, nullopt, norm_type, *(weights_.final_layernorm), norm_eps));
    }
    printBufferData(*hidden, "final_hidden");

    // lm head
    const auto& lm_head = weights_.lm_head;
    const auto& padded_vocab_size = lm_head->kernel->shape()[0];
    auto final_hidden = device_->gemm(GemmParams(*hidden, *(lm_head->kernel)));

    return {move(final_hidden)};
}

} // namespace rtp_llm

