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

    const auto& input_ids = inputs.input_ids;
    const auto batch_size = input_ids.shape()[0];

    const auto& combo_tokens = inputs.combo_tokens;
    const auto cumulated_seq_len = combo_tokens.shape()[0];
    const auto& embedding_table = weights_.embedding->kernel;
    const auto hidden_size = embedding_table->shape()[1];

    // word embedding lookup
    auto hidden = device_->embeddingLookup({combo_tokens, *embedding_table, nullopt, nullopt});

    // pre layernorm
    if (weights_.pre_decoder_layernorm) {
        auto r = device_->layernorm(LayernormParams(
            norm_type, *move(hidden), nullopt, nullopt, nullopt,
            *(weights_.pre_decoder_layernorm), norm_eps));
    }

    // layers
    const auto layer_num = weights_.layers.size();
    for (int i = 0; i < layer_num; ++i) {
        const auto& layer = weights_.layers[i];

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
        auto& attn_hidden = attn_output.hidden_states;

        // TODO: maybe move this layernorm to attention layer
        auto normed_attn = device_->layernorm(LayernormParams(
            norm_type, *attn_hidden, *attn_hidden, *hidden, nullopt,
            *(layer.self_attention_weights.attention_layernorm), norm_eps));

        attn_hidden.reset();
        hidden = move(normed_attn.norm_output);

        auto ffn_output = device_->ffnLayer(FfnLayerParams({
            *hidden,
            layer.ffn_weights,
            description_.activation_type,
        }));
        auto& ffn_hidden = ffn_output.hidden_states;

        // TODO: maybe move this layernorm to ffn layer
        auto normed_ffn_hidden = device_->layernorm(LayernormParams(
            norm_type, *ffn_hidden, *ffn_hidden, *hidden, nullopt,
            *(layer.ffn_weights.dense_layernorm), norm_eps));
        hidden = move(normed_ffn_hidden.norm_output);
    }

    // final layernorm
    if (weights_.final_layernorm) {
        hidden  = device_->layernorm(LayernormParams(
            norm_type, *hidden, nullopt, nullopt, nullopt,
            *(weights_.pre_decoder_layernorm), norm_eps)).norm_output;
    }

    // lm head
    const auto& lm_head = weights_.lm_head;
    const auto& padded_vocab_size = lm_head->kernel->shape()[0];
    auto final_hidden = device_->gemm(GemmParams(*hidden, *(lm_head->kernel)));

    return {move(final_hidden)};
}

} // namespace rtp_llm

