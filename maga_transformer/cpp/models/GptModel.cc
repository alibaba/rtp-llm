#include "maga_transformer/cpp/models/GptModel.h"

using namespace std;

namespace rtp_llm {

GptModel::GptModel(const GptModelInitParams& params)
    : device_(params.device)
    , weights_(params.weights)
    , description_(params.description)
    , attention_configs_(AttentionConfigs({
        (PositionEmbeddingStyle)description_.rotary_embedding_style,
        description_.rotary_embedding_dim,
        description_.rotary_embedding_base,
    }))
    {};

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
            inputs.kv_cache_blocks,
            nullopt,
            inputs.input_lengths,
            inputs.sequence_lengths,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt,
            nullopt
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

