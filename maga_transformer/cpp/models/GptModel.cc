#include "maga_transformer/cpp/models/GptModel.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/models/W.h"
#include "src/fastertransformer/utils/assert_utils.h"
#include <memory>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

GptModel::GptModel(const GptModelInitParams& params)
    : device_(params.device)
    , device_props_(params.device->getDeviceProperties())
    , weights_(params.weights)
    , description_(params.description)
    {}

void getPaddingOffsetAndCuSeqLens(int32_t*       padding_offset,
                                  int32_t*       cu_seqlens,
                                  const int32_t* sequence_length,
                                  const int32_t* prefix_length,
                                  const int32_t  batch_size,
                                  const int32_t  max_seq_len)
{
    // do cumulated sum
    int32_t        total_seq_len        = 0;
    int32_t        cum_offset           = 0;
    int32_t        index                = 0;
    for (int32_t i = 0; i < batch_size; i++) {
        int32_t seq_len = sequence_length[i];
        if (prefix_length) {
            seq_len += prefix_length[i];
        }
        cu_seqlens[i] = total_seq_len;
        if (padding_offset) {
            for (int32_t j = 0; j < seq_len; j++) {
                padding_offset[index] = cum_offset;
                index++;
            }
        }
        cum_offset += max_seq_len - seq_len;
        total_seq_len += seq_len;
    }
    cu_seqlens[batch_size] = total_seq_len;
}

void checkKvBlocksShape(const BufferPtr& input_kv_offset) {
    if (!input_kv_offset) {
        return;
    }
    RUNTIME_ASSERT_OP_ARG(
        input_kv_offset->shape().size() == 2,
        "kv_cache_blocks shape should be [batch_size, block_length].");
}

BufferPtr GptModel::tpSyncEmbeddingOrLogits(const BufferPtr& buffer) {
    const auto tp_size = device_props_.tp_size;
    const auto tp_rank = device_props_.tp_rank;
    const auto buffer_shape = buffer->shape();
    const auto local_size = buffer->size();
    auto all_data = device_->allocateBuffer({buffer->type(), {buffer_shape[0], buffer_shape[1] * tp_size}});
    auto buffer_view = buffer->reshape({buffer->size()});
    auto all_data_1d = all_data->reshape({all_data->size()});
    device_->copy({all_data_1d.view(local_size * tp_rank, local_size), buffer_view});
    device_->syncAndCheck();
    device_->allGather({{all_data}});
    auto ret = device_->transpose({all_data->reshape({tp_size, buffer_shape[0], buffer_shape[1]})});
    ret->updateShape({buffer_shape[0], buffer_shape[1] * tp_size});
    return ret;
}

void GptModel::prepareAttentionInputs(
        const GptModelInputs& inputs,
        AttentionCommonInputs& attention_inputs)
{
    if (inputs.kv_cache_offset) {
        checkKvBlocksShape(inputs.kv_cache_offset);
        KvCacheInfo kv_cache;
        kv_cache.kv_cache_offset = device_->clone({*inputs.kv_cache_offset, AllocationType::DEVICE, {"kv_cache_offset"}});
        attention_inputs.kv_cache = kv_cache;
    }

    const auto& input_lengths = inputs.input_lengths;
    const auto& sequence_lengths = inputs.sequence_lengths;
    const auto& prefix_lengths = inputs.prefix_lengths;
    const auto decoder_batch_size = sequence_lengths->shape()[0];
    const auto context_batch_size = input_lengths->shape()[0] - decoder_batch_size;
    const auto max_context_seq_len = context_batch_size ? *std::max_element(
        input_lengths->data<int32_t>() + decoder_batch_size,
        input_lengths->data<int32_t>() + decoder_batch_size + context_batch_size) : 0;
    FT_CHECK_WITH_INFO(!prefix_lengths || prefix_lengths->size() == context_batch_size, "prefix_lengths size %d is not equal to context batch size %d.", prefix_lengths->size(), context_batch_size);
    attention_inputs.max_prefix_length = context_batch_size && prefix_lengths ? *std::max_element(
        prefix_lengths->data<int32_t>(),
        prefix_lengths->data<int32_t>() + prefix_lengths->size()) : 0;
    const auto max_decoder_seq_len = decoder_batch_size ? *std::max_element(
        sequence_lengths->data<int32_t>(),
        sequence_lengths->data<int32_t>() + decoder_batch_size) : 0;

    std::vector<int32_t> cu_seqlens_data(context_batch_size + 1);
    std::vector<int32_t> padding_offset_data(inputs.combo_tokens->shape()[0]);
    getPaddingOffsetAndCuSeqLens(
        padding_offset_data.data(),
        cu_seqlens_data.data(),
        input_lengths->dataWithOffset<int32_t>(decoder_batch_size),
        nullptr,
        context_batch_size,
        max_context_seq_len);

    RUNTIME_ASSERT_OP_ARG(
        (cu_seqlens_data[context_batch_size] + decoder_batch_size == inputs.combo_tokens->shape()[0]),
        "combo_tokens is not consistent with input lengths, "
        "there are %d tokens in context plus %ld tokens in decoder batch, but got %ld input tokens.",
        cu_seqlens_data[context_batch_size], decoder_batch_size, inputs.combo_tokens->shape()[0]);

    attention_inputs.cu_seqlens = device_->clone(
        {*vector2Buffer(cu_seqlens_data), AllocationType::DEVICE, {"cu_seqlens"}});
    if (attention_inputs.max_prefix_length) {
        attention_inputs.prefix_prompt_lengths = device_->clone(*prefix_lengths);
        std::vector<int32_t> cu_kv_seqlens_data(context_batch_size + 1);
        getPaddingOffsetAndCuSeqLens(
                nullptr,
                cu_kv_seqlens_data.data(),
                input_lengths->dataWithOffset<int32_t>(decoder_batch_size),
                prefix_lengths->data<int32_t>(),
                context_batch_size,
                max_context_seq_len);
        attention_inputs.cu_kv_seqlens = device_->clone(
            {*vector2Buffer(cu_kv_seqlens_data), AllocationType::DEVICE, {"cu_kv_seqlens"}});
    } else {
        attention_inputs.cu_kv_seqlens = attention_inputs.cu_seqlens;
    }
    attention_inputs.padding_offset = device_->clone(
        {*vector2Buffer(padding_offset_data), AllocationType::DEVICE, {"padding_offset"}});
    attention_inputs.decoder_batch_size = decoder_batch_size;
    attention_inputs.context_batch_size = context_batch_size;
    attention_inputs.context_max_seq_len = max_context_seq_len;
    attention_inputs.decoder_max_seq_len = max_decoder_seq_len;
    attention_inputs.context_token_num = cu_seqlens_data[context_batch_size];
    if (weights_.linear_bias_slopes) {
        attention_inputs.linear_bias_slopes = weights_.linear_bias_slopes->kernel;
    }
    attention_inputs.attention_mask = inputs.attention_mask;
}


/*
 *          ┌───────────┐
 *          │  hidden   │
 *          └─────┬─────┘
 *                │
 *                │
 *        ┌───────▼───────┐
 *        │ pre_layernorm?├─────────┐
 *        └───────┬───────┘         │
 *                │                 │
 *          ┌─────▼─────┐           │
 *          │ attention │           │
 *          └─────┬─────┘           │
 *                │                 │
 *        ┌───────▼───────┐         │
 * ┌──────┤post_attn_norm?◄─────────┘
 * │      └───────┬───────┘
 * │              │
 * │         ┌────▼────┐
 * │         │   mlp   │
 * │         └────┬────┘
 * │              │
 * │         ┌────▼────┐
 * └─────────►   add   │
 *           └────┬────┘
 *                │
 *          ┌─────▼─────┐
 *          │ layernorm │
 *          └───────────┘
 */
GptModelOutputs GptModel::forward(const GptModelInputs& inputs) {
    const auto norm_type = description_.norm_type;
    const auto norm_eps = description_.layernorm_eps;

    const auto combo_tokens = device_->clone(
        {*inputs.combo_tokens, AllocationType::DEVICE, {"combo_tokens"}});
    const auto input_lengths = device_->clone({*inputs.input_lengths});
    const auto sequence_lengths = device_->clone({*inputs.sequence_lengths});

    const auto& embedding_table = weights_.embedding->kernel;

    const BufferPtr combo_position_ids = inputs.combo_position_ids ? device_->clone({*inputs.combo_position_ids}): nullptr;
    const BufferPtr combo_tokens_type_ids = inputs.combo_tokens_type_ids ? device_->clone({*inputs.combo_tokens_type_ids}): nullptr;    

    const BufferPtr text_tokens_mask = inputs.multimodal_features ? 
        device_->clone({*inputs.text_tokens_mask, AllocationType::DEVICE, {"text_tokens_mask"}}) : nullptr;
    const BufferPtr mm_feature_locs = inputs.mm_features_locs ? inputs.mm_features_locs: nullptr;

    // lora input
    OptionalLoraInput lora_input = std::nullopt;
    if (inputs.lora_ids != nullptr && inputs.lora_input_lengths != nullptr) {
        lora_input = (OptionalLoraInput)LoraInput({inputs.lora_ids, inputs.lora_input_lengths});
    }

    // word embedding lookup
    auto hidden = device_->embeddingLookup({
            *combo_tokens, *embedding_table, description_.input_embedding_scalar,
            text_tokens_mask ? (OptionalConstBufferRef)*text_tokens_mask : nullopt,
            combo_position_ids ? (OptionalConstBufferRef)*combo_position_ids: nullopt,
            weights_.position_encoding ? (OptionalConstBufferRef)*weights_.position_encoding->kernel: nullopt,
            combo_tokens_type_ids ? (OptionalConstBufferRef)*combo_tokens_type_ids: nullopt,
            weights_.token_type_embedding ? (OptionalConstBufferRef)*weights_.token_type_embedding->kernel: nullopt});
    const auto dtype = hidden->type();
    if (device_props_.tp_size > 1) {
        hidden = tpSyncEmbeddingOrLogits(hidden);
    }

    if (inputs.multimodal_features) {
        hidden = device_->multimodalEmbedding({
                hidden,
                inputs.multimodal_features ? (OptionalConstVecBufferPtrRef)inputs.multimodal_features : nullopt, 
                mm_feature_locs ? (OptionalConstBufferRef)*mm_feature_locs: nullopt
        });
    }
        

    // TODO: fix me
    ft::QScheme qscheme = QScheme::NoQuantize;
    if (weights_.layers[0].post_layernorm && weights_.layers[0].post_layernorm->static_scale != nullptr) {
        qscheme = QScheme::Qint8PerTensor;
    } else if (weights_.layers[0].self_attention_weights.smoother_weight != nullptr) {
        qscheme = QScheme::Qint8PerChannelLastAxis;
    }

    // pre layernorm    
    BufferPtr pre_decoder_residual = nullptr;
    if (qscheme != QScheme::NoQuantize && weights_.pre_decoder_layernorm) {
        pre_decoder_residual = device_->allocateBufferLike(*hidden);
    }
    printBufferData(*hidden, "before decoder layernorm hidden");
    if (weights_.pre_decoder_layernorm) {    
        auto decoder_input = device_->layernorm(LayernormParams(hidden,
                                                                pre_decoder_residual,
                                                                *weights_.pre_decoder_layernorm,
                                                                nullopt,
                                                                nullopt,
                                                                nullopt,
                                                                0.f,
                                                                norm_eps,
                                                                true,
                                                                pre_decoder_residual != nullptr,
                                                                norm_type,
                                                                qscheme));
        hidden = std::move(decoder_input.output);
    }

    // prepare resources for all layers
    auto kv_cache_offset = inputs.kv_cache_offset;
    AttentionCommonInputs attention_common_inputs({
        *input_lengths,
        *sequence_lengths
    });

    prepareAttentionInputs(inputs, attention_common_inputs);
    attention_common_inputs.lora_input = lora_input;
    attention_common_inputs.position_ids = combo_position_ids;

    printBufferData(*hidden, "input_hidden");
    // layers
    const int layer_num = weights_.layers.size();
    for (int i = 0; i < layer_num; ++i) {
        const auto& layer = weights_.layers[i];

        // here hidden->dtype maybe int8, so use dytpe of embedding lookup result instead
        auto attn_out_buf = device_->allocateBuffer({dtype, hidden->shape()}, {"attn_out_buf"});
        // Note: for custom all reduce, prepareAllReduce will replace the original attn_out_buf with 
        // a new custom_ar_comm buffer. Here we must make sure that attn_out_buf is not released or replaced by 
        // other buffer before the actual allreduce operations. Otherwise, it will raise an error in custom ar.
        attn_out_buf = device_->prepareAllReduce({std::move(attn_out_buf), ReduceOp::Sum}).buffer;
        auto residual = pre_decoder_residual ? pre_decoder_residual : hidden;
        printBufferData(*residual, "in residual");
        BufferPtr residual2 = nullptr;
        if (layer.pre_layernorm) {
            residual = device_->clone({*hidden, AllocationType::DEVICE, {"residual"}});
            auto pre_layernorm_output = device_->layernorm(LayernormParams(hidden,
                                                                            nullptr,
                                                                            *layer.pre_layernorm,
                                                                            std::nullopt,
                                                                            std::nullopt,
                                                                            std::nullopt,
                                                                            0.f,
                                                                            norm_eps,
                                                                            true,
                                                                            false,
                                                                            norm_type,
                                                                            qscheme));
            hidden = std::move(pre_layernorm_output.output);
        }

        if (kv_cache_offset) {
            // NOTE: these values in each layer are overwritten.
            attention_common_inputs.kv_cache->kv_cache_offset = kv_cache_offset;
            attention_common_inputs.kv_cache->k_cache_buffer = inputs.k_cache_buffer->index(i);
            attention_common_inputs.kv_cache->v_cache_buffer = inputs.v_cache_buffer->index(i);
            if (inputs.k_scale_buffer) {
                attention_common_inputs.kv_cache->k_scale_buffer = inputs.k_scale_buffer->index(i);
                attention_common_inputs.kv_cache->v_scale_buffer = inputs.v_scale_buffer->index(i);
            }
        }

        auto attn_output = device_->attentionLayer(AttentionLayerParams({
            *hidden,
            move(attn_out_buf),
            description_.attention_conf,
            layer.self_attention_weights,
            attention_common_inputs,
            device_props_.attn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt,
            {description_.layernorm_eps, description_.norm_type},
            qscheme
        }));
        auto attn_hidden = std::move(attn_output.hidden_states);
        if (device_props_.tp_size > 1) {
            // Note: for custom all reduce, allReduce will allocate a new buffer and replace the original attn_hidden with it
            attn_hidden = device_->allReduce({std::move(attn_hidden), ReduceOp::Sum}).buffer;
        }
        printBufferData(*attn_hidden, "layer_" + to_string(i) + "_attn_output");

        if (layer.post_layernorm) {
            // attn_hidden = attn_hidden + residual
            // hidden = layernorm(attn_hidden)
            printBufferData(*residual, "post layernorm residual");
            auto post_layernorm_params = LayernormParams(attn_hidden,
                                                         attn_hidden,
                                                         ft::mayGetRef(layer.post_layernorm),
                                                         device_props_.attn_fuse_add_residual ? nullopt : (OptionalConstBufferRef)*residual,
                                                         nullopt,
                                                         ft::mayGetRef(layer.self_attention_weights.output_weight->bias),
                                                         0.f,
                                                         norm_eps,
                                                         false,
                                                         description_.post_layernorm,
                                                         norm_type,
                                                         qscheme);

            auto post_layernorm_output = device_->layernorm(post_layernorm_params);
            hidden = std::move(post_layernorm_output.output);
            if (!layer.post_layernorm_2) {
                attn_hidden = std::move(post_layernorm_output.before_norm_output);
                residual = attn_hidden;
            }
            printBufferData(*residual, "post_layernorm_residual");    
        } else {
            residual2 = attn_hidden;
        }

        if (layer.post_layernorm_2) {
            // attn_hidden = attn_hidden + residual
            // hidden = layernorm(attn_hidden)
            auto post_layernorm_params = LayernormParams(hidden,
                                                         hidden,
                                                         ft::mayGetRef(layer.post_layernorm_2),
                                                         *residual,
                                                         nullopt,
                                                         nullopt,
                                                         0.f,
                                                         norm_eps,
                                                         false,
                                                         description_.post_layernorm,
                                                         norm_type,
                                                         qscheme);

            auto post_layernorm_output = device_->layernorm(post_layernorm_params);
            hidden = std::move(post_layernorm_output.output);
            attn_hidden = std::move(post_layernorm_output.before_norm_output);
            residual = attn_hidden;
        }

        printBufferData(*hidden, "layer_" + to_string(i) + "_ffn_input");
        auto ffn_output_buf = device_->allocateBuffer({dtype, hidden->shape()}, {"ffn_out_buf"});
        // Note: for custom all reduce, prepareAllReduce will replace the original attn_out_buf with 
        // a new custom_ar_comm buffer. Here we must make sure that attn_out_buf is not released or replaced by 
        // other buffer before the actual allreduce operations. Otherwise, it will raise an error in custom ar.
        ffn_output_buf = device_->prepareAllReduce({std::move(ffn_output_buf), ReduceOp::Sum}).buffer;
        auto ffn_output = device_->ffnLayer(FfnLayerParams({
            *hidden,
            description_.ffn_conf,
            layer.ffn_weights,
            device_props_.ffn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt,
            lora_input,
            qscheme,
            std::move(ffn_output_buf),
        }));
        hidden = ffn_output.hidden_states;
        if (device_props_.tp_size > 1) {
            // Note: for custom all reduce, allReduce will allocate a new buffer and replace the original attn_hidden with it
            hidden = device_->allReduce({std::move(hidden), ReduceOp::Sum}).buffer;
        }
        printBufferData(*hidden, "layer_" + to_string(i) + "_ffn_output");

        // TODO: maybe move this layernorm to ffn layer
        auto ffn_layernorm_output = device_->layernorm(LayernormParams(hidden,
                                                                       pre_decoder_residual,
                                                                       ft::mayGetRef(layer.post_ffn_layernorm),
                                                                       device_props_.ffn_fuse_add_residual ? nullopt : (OptionalConstBufferRef)*residual,
                                                                       (residual2 == nullptr) ? nullopt : (OptionalConstBufferRef)*residual2,
                                                                       ft::mayGetRef(WEIGHT_MAY_GET_BIAS(layer.ffn_weights.down_weight)),
                                                                       1.0f,
                                                                       norm_eps,
                                                                       true,                                                                       
                                                                       description_.post_layernorm,
                                                                       norm_type,
                                                                       ((i == layer_num - 1) || (!layer.post_ffn_layernorm)) ? QScheme::NoQuantize: qscheme));
        hidden = std::move(ffn_layernorm_output.output);
        printBufferData(*hidden, "layer_" + to_string(i) + "_final_hidden");
    }

    // final layernorm
    if (weights_.final_layernorm) {
        auto final_layernorm = device_->layernorm(LayernormParams(hidden,
                                                                  nullptr,
                                                                  ft::mayGetRef(weights_.final_layernorm),
                                                                  nullopt,
                                                                  nullopt,
                                                                  nullopt,
                                                                  0.f,
                                                                  norm_eps,
                                                                  true,
                                                                  false,
                                                                  norm_type));
        hidden = std::move(final_layernorm.output);
    }
    printBufferData(*hidden, "final_hidden");

    // lm head
    const auto& lm_head = weights_.lm_head;
    if (lm_head) {
        // gen last token hidden
        auto last_hidden = attention_common_inputs.context_batch_size && !inputs.need_all_logits
                         ? device_->select({*hidden, *device_->clone({*inputs.lm_output_indexes})})
                         : hidden;

        printBufferData(*last_hidden, "last_hidden");

        auto logits = device_->gemm(GemmParams(
            *last_hidden, *(lm_head->kernel), nullopt, nullptr,
            ft::DataType::TYPE_FP32, TransposeOperation::NONE, TransposeOperation::TRANSPOSE));
        if (device_props_.tp_size > 1) {
            logits = tpSyncEmbeddingOrLogits(logits);
        }
        // logits is too big, tmp not print default
        // printBufferData(*logits, "logits");
        if (inputs.need_all_logits) {
            auto last_logits = device_->select({*logits, *device_->clone({*inputs.lm_output_indexes})});
            return {std::move(last_logits), std::move(last_hidden), std::move(hidden), std::move(logits)};
        }
        return {std::move(logits), std::move(last_hidden), std::move(hidden), nullptr};
    } else {
        return {nullptr, nullptr, std::move(hidden)};
    }
}


void GptModel::addLoRA(const int64_t lora_id,
                       const std::vector<LoraMap>& lora_a,
                       const std::vector<LoraMap>& lora_b)
{
    auto layer_num = weights_.layers.size();
    FT_CHECK_WITH_INFO(((lora_a.size() == layer_num) && (lora_b.size() == layer_num)),
        "lora_a/lora b layer num[%d]/[%d] must be equal to layer num[%d]",
        lora_a.size(), lora_b.size(), layer_num);

    auto helper_func = [&lora_id](std::string& adapter_name,
                                  std::shared_ptr<LoraWeightsMap> layer_weights,
                                  ConstBufferPtr lora_a,
                                  ConstBufferPtr lora_b)
    {
        layer_weights->setLoRAWeight(lora_id, lora_a, lora_b);
    };
    std::vector<std::string> adapter_set = {W::attn_qkv_w, W::attn_o_w, W::ffn_w1, W::ffn_w2, W::ffn_w3};

    for (int i = 0; i < int(layer_num); i++) {
        for (auto adapter_name : adapter_set) {
            if (lora_a[i].find(adapter_name) == lora_a[i].end()) {
                continue;
            }

            if (adapter_name == W::attn_qkv_w) {
                helper_func(adapter_name,
                            weights_.layers[i].self_attention_weights.qkv_lora_weights,
                            lora_a[i].at(adapter_name),
                            lora_b[i].at(adapter_name));
            } else if (adapter_name == W::attn_o_w) {
                helper_func(adapter_name,
                            weights_.layers[i].self_attention_weights.output_lora_weights,
                            lora_a[i].at(adapter_name),
                            lora_b[i].at(adapter_name));
            } else if (adapter_name == W::ffn_w1) {
                helper_func(adapter_name,
                            weights_.layers[i].ffn_weights.gate_lora_weights,
                            lora_a[i].at(adapter_name),
                            lora_b[i].at(adapter_name));
            } else if (adapter_name == W::ffn_w2) {
                helper_func(adapter_name,
                            weights_.layers[i].ffn_weights.down_lora_weights,
                            lora_a[i].at(adapter_name),
                            lora_b[i].at(adapter_name));
            } else if (adapter_name == W::ffn_w3) {
                helper_func(adapter_name,
                            weights_.layers[i].ffn_weights.up_lora_weights,
                            lora_a[i].at(adapter_name),
                            lora_b[i].at(adapter_name));
            } else {
 	        FT_FAIL("unknown lora W %s", adapter_name.c_str());
            }
        }
    }

}

void GptModel::removeLoRA(const int64_t lora_id) {
    int layer_num = weights_.layers.size();

    auto helper_func = [&lora_id](std::shared_ptr<LoraWeightsMap> layer_weights) {
        layer_weights->removeLoRAWeight(lora_id);
    };
    std::vector<std::string> adapter_set = {W::attn_qkv_w, W::attn_o_w, W::ffn_w1, W::ffn_w2, W::ffn_w3};
    for (int i = 0; i < layer_num; i++) {
        for (auto adapter_name : adapter_set) {
            if (adapter_name == W::attn_qkv_w) {
                helper_func(weights_.layers[i].self_attention_weights.qkv_lora_weights);
            } else if (adapter_name == W::attn_o_w) {
                helper_func(weights_.layers[i].self_attention_weights.output_lora_weights);
            } else if (adapter_name == W::ffn_w1) {
                helper_func(weights_.layers[i].ffn_weights.gate_lora_weights);
            } else if (adapter_name == W::ffn_w2) {
                helper_func(weights_.layers[i].ffn_weights.down_lora_weights);
            } else if (adapter_name == W::ffn_w3) {
                helper_func(weights_.layers[i].ffn_weights.up_lora_weights);
            } else {
                FT_FAIL("unknown lora W %s", adapter_name.c_str());
            }
        }
    }
}


} // namespace rtp_llm
