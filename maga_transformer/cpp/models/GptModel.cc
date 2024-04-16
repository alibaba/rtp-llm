#include "maga_transformer/cpp/models/GptModel.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"

using namespace std;

namespace rtp_llm {

GptModel::GptModel(const GptModelInitParams& params)
    : device_(params.device)
    , weights_(params.weights)
    , description_(params.description)
    , attention_configs_(AttentionConfigs({}))
    {};

void getPaddingOffsetAndCuSeqLens(int32_t*       padding_offset,
                                  int32_t*       cu_seqlens,
                                  const int32_t* sequence_length,
                                  const int32_t  batch_size)
{
    // do cumulated sum
    int32_t        total_seq_len        = 0;
    int32_t        cum_offset           = 0;
    int32_t        index                = 0;
    const auto max_seq_len = *std::max_element(sequence_length, sequence_length + batch_size);
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

AttentionCommonInputs GptModel::prepareAttentionInputs(const GptModelInputs& inputs) {
    AttentionCommonInputs attention_inputs({
        inputs.input_lengths,
        inputs.sequence_lengths
    });

    const auto decoder_batch_size = inputs.sequence_lengths.shape()[0];
    const auto context_batch_size = inputs.input_lengths.shape()[0] - decoder_batch_size;

    std::vector<int32_t> cu_seqlens_data(context_batch_size + 1);
    std::vector<int32_t> padding_offset_data(inputs.combo_tokens.shape()[0]);
    getPaddingOffsetAndCuSeqLens(
        padding_offset_data.data(),
        cu_seqlens_data.data(),
        inputs.input_lengths.dataWithOffset<int32_t>(decoder_batch_size),
        context_batch_size);
    if (!(cu_seqlens_data[context_batch_size] == inputs.combo_tokens.shape()[0])) {
        throw OpException(
            {OpErrorType::ERROR_INVALID_ARGS, "cu_seqlens is not consistent with combo_tokens."});
    }
    attention_inputs.cu_seqlens = device_->clone({vector2Buffer(cu_seqlens_data)});
    attention_inputs.padding_offset = device_->clone({vector2Buffer(padding_offset_data)});
    attention_inputs.kv_cache_blocks = inputs.kv_cache_blocks;
    attention_inputs.position_ids = inputs.position_ids;
    attention_inputs.attention_mask = inputs.attention_mask;
    return move(attention_inputs);
}

GptModelOutputs GptModel::forward(const GptModelInputs& inputs) {
    const auto hidden_type = datatype_enum::TYPE_BF16;
    const auto norm_type = description_.norm_type;
    const auto norm_eps = description_.layernorm_eps;

    const auto batch_size = inputs.input_lengths.shape()[0];
    const auto& combo_tokens = inputs.combo_tokens;
    const auto& embedding_table = weights_.embedding->kernel;
    const auto hidden_size = embedding_table->shape()[1];

    // word embedding lookup
    auto hidden = device_->embeddingLookup({combo_tokens, *embedding_table, nullopt, nullopt});

    // pre layernorm
    if (weights_.pre_decoder_layernorm) {
        device_->layernorm(LayernormParams(
            *hidden, *hidden, nullopt, norm_type, *(weights_.pre_decoder_layernorm), norm_eps));
    }

    // layers
    const auto layer_num = weights_.layers.size();
    for (int i = 0; i < layer_num; ++i) {
        const auto& layer = weights_.layers[i];

        auto residual = device_->allocateBuffer({hidden->type(), hidden->shape()}, {});
        device_->copy({*residual, *hidden});
        device_->layernorm(LayernormParams(
            *hidden, *hidden, nullopt, norm_type,
            mayGetRef(layer.self_attention_weights.pre_attention_layernorm), norm_eps));

        AttentionCommonInputs attention_inputs({
            inputs.input_lengths,
            inputs.sequence_lengths
            // inputs.kv_cache_blocks,
            // nullopt,
            // nullopt,
            // nullopt,
            // nullopt,
            // nullopt,
            // nullopt,
            // nullopt,
            // nullopt,
            // nullopt,
            // nullopt
        });

        auto attn_output = device_->attentionLayer(AttentionLayerParams({
            *hidden,
            attention_configs_,
            layer.self_attention_weights,
            attention_inputs,
        }));
        auto attn_hidden = move(attn_output.hidden_states);

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

        auto ffn_output = device_->ffnLayer(FfnLayerParams({
            *hidden,
            layer.ffn_weights,
            description_.activation_type,
        }));
        hidden.swap(ffn_output.hidden_states);

        // TODO: maybe move this layernorm to ffn layer
        device_->layernorm(LayernormParams(
            *hidden, *hidden, nullopt,
            norm_type, mayGetRef(layer.post_ffn_layernorm), norm_eps, *residual
        ));
    }

    // final layernorm
    if (weights_.final_layernorm) {
        device_->layernorm(LayernormParams(
            *hidden, *hidden, nullopt, norm_type, *(weights_.final_layernorm), norm_eps));
    }

    // lm head
    const auto& lm_head = weights_.lm_head;
    const auto& padded_vocab_size = lm_head->kernel->shape()[0];
    auto final_hidden = device_->gemm(GemmParams(*hidden, *(lm_head->kernel)));

    return {move(final_hidden)};
}

} // namespace rtp_llm

