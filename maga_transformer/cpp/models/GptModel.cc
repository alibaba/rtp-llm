#include "maga_transformer/cpp/models/GptModel.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include <memory>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

GptModel::GptModel(const GptModelInitParams& params)
    : device_(params.device)
    , device_props_(params.device->getDeviceProperties())
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

void GptModel::prepareAttentionInputs(
        const GptModelInputs& inputs,
        AttentionCommonInputs& attention_inputs) {

    const auto& input_lengths = inputs.input_lengths;
    const auto& sequence_lengths = inputs.sequence_lengths;
    const auto decoder_batch_size = sequence_lengths->shape()[0];
    const auto context_batch_size = input_lengths->shape()[0] - decoder_batch_size;
    const auto max_context_seq_len = context_batch_size ? *std::max_element(
        input_lengths->data<int32_t>() + decoder_batch_size,
        input_lengths->data<int32_t>() + decoder_batch_size + context_batch_size) : 0;
    const auto max_decoder_seq_len = decoder_batch_size ? *std::max_element(
        sequence_lengths->data<int32_t>(),
        sequence_lengths->data<int32_t>() + decoder_batch_size) : 0;

    std::vector<int32_t> cu_seqlens_data(context_batch_size + 1);
    std::vector<int32_t> padding_offset_data(inputs.combo_tokens->shape()[0]);
    getPaddingOffsetAndCuSeqLens(
        padding_offset_data.data(),
        cu_seqlens_data.data(),
        input_lengths->dataWithOffset<int32_t>(decoder_batch_size),
        context_batch_size,
        max_context_seq_len);

    RUNTIME_ASSERT_OP_ARG(
        (cu_seqlens_data[context_batch_size] + decoder_batch_size == inputs.combo_tokens->shape()[0]),
        "combo_tokens is not consistent with input lengths, "
        "there are %d tokens in context plus %d tokens in decoder batch, but got %d input tokens.",
        cu_seqlens_data[context_batch_size], decoder_batch_size, inputs.combo_tokens->shape()[0]);
    checkKvBlocksShape(inputs.kv_cache_blocks);
    checkKvBlocksShape(inputs.kv_cache_scales);

    attention_inputs.cu_seqlens = device_->clone(
        {*vector2Buffer(cu_seqlens_data), AllocationType::DEVICE, {"cu_seqlens"}});
    attention_inputs.padding_offset = device_->clone(
        {*vector2Buffer(padding_offset_data), AllocationType::DEVICE, {"padding_offset"}});
    attention_inputs.decoder_batch_size = decoder_batch_size;
    attention_inputs.context_batch_size = context_batch_size;
    attention_inputs.context_max_seq_len = max_context_seq_len;
    attention_inputs.decoder_max_seq_len = max_decoder_seq_len;
    attention_inputs.context_token_num = cu_seqlens_data[context_batch_size];
    attention_inputs.position_ids = inputs.position_ids;
    attention_inputs.attention_mask = inputs.attention_mask;
}

GptModelOutputs GptModel::forward(const GptModelInputs& inputs) {
    const auto hidden_type = datatype_enum::TYPE_BF16;
    const auto norm_type = description_.norm_type;
    const auto norm_eps = description_.layernorm_eps;

    const auto batch_size = inputs.input_lengths->shape()[0];
    const auto combo_tokens = device_->clone(
        {*inputs.combo_tokens, AllocationType::DEVICE, {"combo_tokens"}});
    const auto input_lengths = device_->clone({*inputs.input_lengths});
    const auto sequence_lengths = device_->clone({*inputs.sequence_lengths});

    const auto& embedding_table = weights_.embedding->kernel;
    const auto hidden_size = embedding_table->shape()[1];

    const BufferPtr combo_position_ids = inputs.combo_position_ids ? device_->clone({*inputs.combo_position_ids}): nullptr;
    const BufferPtr combo_tokens_type_ids = inputs.combo_tokens_type_ids ? device_->clone({*inputs.combo_tokens_type_ids}): nullptr;

    // word embedding lookup
    auto hidden = device_->embeddingLookup({
            *combo_tokens, *embedding_table,
            combo_position_ids ? (OptionalConstBufferRef)*combo_position_ids: nullopt,
            combo_position_ids ? (OptionalConstBufferRef)*weights_.position_encoding->kernel: nullopt,
            combo_tokens_type_ids ? (OptionalConstBufferRef)*combo_tokens_type_ids: nullopt,
            combo_tokens_type_ids ? (OptionalConstBufferRef)*weights_.token_type_embedding->kernel: nullopt});

    // pre layernorm
    if (weights_.pre_decoder_layernorm) {
        device_->layernorm(LayernormParams(
            *hidden, *hidden, nullopt, norm_type, *(weights_.pre_decoder_layernorm), norm_eps));
    }

    // prepare resources for all layers
    AttentionCommonInputs attention_common_inputs({
        *input_lengths,
        *sequence_lengths
    });

    prepareAttentionInputs(inputs, attention_common_inputs);
    BufferPtr input_kv_blocks;
    if (inputs.kv_cache_blocks) {
        input_kv_blocks = device_->clone({*inputs.kv_cache_blocks, AllocationType::DEVICE, {"kv_block_ptrs"}});
    }

    printBufferData(*hidden, "input_hidden");

    // layers
    const auto layer_num = weights_.layers.size();
    for (int i = 0; i < layer_num; ++i) {
        const auto& layer = weights_.layers[i];

        auto attn_out_buf = device_->allocateBuffer({hidden->type(), hidden->shape()}, {"attn_out_buf"});
        auto residual = hidden;
        if (layer.pre_layernorm) {
            residual = device_->clone({*hidden, AllocationType::DEVICE, {"residual"}});
            device_->layernorm(LayernormParams(
                *hidden, *hidden, nullopt, norm_type,
                *layer.pre_layernorm, norm_eps));
        }

        BufferPtr layer_kv_blocks_ptr;
        if (input_kv_blocks) {
            auto layer_buffer = (*input_kv_blocks)[i];
            layer_kv_blocks_ptr = std::make_unique<Buffer>(layer_buffer.where(), layer_buffer.type(), layer_buffer.shape(), layer_buffer.data());
            attention_common_inputs.kv_cache_blocks = *layer_kv_blocks_ptr;
        }

        auto attn_output = device_->attentionLayer(AttentionLayerParams({
            *hidden,
            move(attn_out_buf),
            description_.attention_conf,
            layer.self_attention_weights,
            attention_common_inputs,
            device_props_.attn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt
        }));
        auto attn_hidden = move(attn_output.hidden_states);
        printBufferData(*attn_hidden, "layer_" + to_string(i) + "_attn_output");

        if (layer.post_layernorm) {
            // attn_hidden = attn_hidden + residual
            // hidden = layernorm(attn_hidden)
            device_->layernorm(LayernormParams(
                *attn_hidden, *hidden, *attn_hidden,
                norm_type, ft::mayGetRef(layer.post_layernorm), norm_eps,
                device_props_.attn_fuse_add_residual ? nullopt : (OptionalConstBufferRef)*residual,
                nullopt, ft::mayGetRef(layer.self_attention_weights.output_weight->bias)));
            if (description_.post_layernorm) {
                residual = device_->clone({*hidden, AllocationType::DEVICE, {"residual"}});
            } else {
                residual = attn_hidden;
            }
        } else {
            hidden = move(attn_hidden);
        }

        printBufferData(*hidden, "layer_" + to_string(i) + "_ffn_input");
        auto ffn_output = device_->ffnLayer(FfnLayerParams({
            *hidden,
            layer.ffn_weights,
            description_.activation_type,
            device_props_.ffn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt
        }));
        hidden = ffn_output.hidden_states;
        printBufferData(*hidden, "layer_" + to_string(i) + "_ffn_output");

        // TODO: maybe move this layernorm to ffn layer
        device_->layernorm(LayernormParams(
            *hidden, *hidden, nullopt,
            norm_type, ft::mayGetRef(layer.post_ffn_layernorm), norm_eps,
            device_props_.ffn_fuse_add_residual ? nullopt : (OptionalConstBufferRef)*residual,
            nullopt, ft::mayGetRef(layer.ffn_weights.down_weight->bias)));

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
    if (lm_head) {
        // gen last token hidden
        auto last_hidden = device_->allocateBufferLike(hidden->view(0, batch_size));
        device_->copy({last_hidden->view(0, attention_common_inputs.decoder_batch_size), hidden->view(0, attention_common_inputs.decoder_batch_size)});
        if (attention_common_inputs.context_batch_size) {
            auto context_last_hidden = device_->select({
                    hidden->view(attention_common_inputs.decoder_batch_size, combo_tokens->size() - attention_common_inputs.decoder_batch_size),
                    0,
                    attention_common_inputs.cu_seqlens->view(1, attention_common_inputs.context_batch_size)});
            device_->copy({last_hidden->view(attention_common_inputs.decoder_batch_size, attention_common_inputs.context_batch_size), *context_last_hidden});
        }

        printBufferData(*last_hidden, "last_hidden");

        auto logits = device_->gemm(GemmParams(
            *last_hidden, *(lm_head->kernel), nullopt, nullptr,
            ft::DataType::TYPE_FP32, TransposeOperation::NONE, TransposeOperation::TRANSPOSE));
        // logits is too big, tmp not print default
        // printBufferData(*logits, "logits");

        return {std::move(logits), std::move(last_hidden), std::move(hidden)};
    } else {
        return {nullptr, nullptr, std::move(hidden)};
    }
}

} // namespace rtp_llm

