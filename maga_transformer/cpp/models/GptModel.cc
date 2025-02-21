#include "maga_transformer/cpp/models/GptModel.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/OpData.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/models/W.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include "maga_transformer/cpp/utils/StringUtil.h"
#include <memory>


using namespace std;
using namespace fastertransformer;
using namespace rtp_llm;

namespace rtp_llm {

string GptModelInputs::debugString() const {
    if (!Logger::getEngineLogger().isDebugMode()) {
        return "";
    }
    std::stringstream debug_string;
    debug_string << "GptModelInputs { "
                    << "combo_tokens: " << combo_tokens->debugStringWithData<int32_t>()
                    << ", input_lengths: " << input_lengths->debugStringWithData<int32_t>()
                    << ", sequence_lengths: " << sequence_lengths->debugStringWithData<int32_t>()
                    << ", prefix_lengths: " << prefix_lengths->debugStringWithData<int32_t>();
    if (combo_position_ids) {
        debug_string << ", combo_position_ids: " << combo_position_ids->debugStringWithData<int32_t>();
    }
    if (lora_ids) {
        debug_string << ", lora_ids: " << lora_ids->debugStringWithData<int32_t>();
    }
    if (lora_input_lengths) {
        debug_string << ", lora_input_lengths: " << lora_input_lengths->debugStringWithData<int32_t>();
    }
    if (kv_cache_block_id) {
        debug_string << ", kv_cache_block_id: " << kv_cache_block_id->debugStringWithData<int32_t>();
    }
    if (attention_mask) {
        debug_string << ", attention_mask: " << attention_mask->debugString();
    }
    if (request_id) {
        debug_string << ", request_id: " << request_id->debugStringWithData<int64_t>();
    }
    if (request_pd_separation) {
        debug_string << ", request_pd_separation: " << request_pd_separation->debugStringWithData<bool>();
    }
    if (cache_keys) {
        debug_string << ", cache_keys: " << cache_keys->debugStringWithData<int64_t>();
    }
    debug_string << ", block_size: " << block_size;
    debug_string << ", pd_separation: " << pd_separation;
    debug_string << "}";
    return debug_string.str();
}

GptModel::GptModel(const GptModelInitParams& params)
    : device_(params.device)
    , device_props_(params.device->getDeviceProperties())
    , weights_(params.weights)
    , layer_num_(params.weights.layers.size())
    , description_(params.description)
    {
        if (params.kv_cache_buffer) {
            k_cache_buffer_ = params.kv_cache_buffer->k_blocks;
            v_cache_buffer_ = params.kv_cache_buffer->v_blocks;
            if (params.kv_cache_buffer->k_scale) {
                k_scale_buffer_ = params.kv_cache_buffer->k_scale;
                v_scale_buffer_ = params.kv_cache_buffer->v_scale;
            }
        }
        if (abs(description_.residual_scalar - 1.0) > 1e-6) {
            residual_scale_fp32_ = device_->clone({*vector2Buffer(vector<float>{(float)description_.residual_scalar})});
            residual_scale_ = residual_scale_fp32_;
        }
    }

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
        ft::DataType dtype,
        AttentionCommonInputs& attention_inputs)
{
    attention_inputs.warmup = inputs.warmup;
    if (!inputs.warmup && inputs.pd_separation) {
        FT_CHECK_WITH_INFO(inputs.input_lengths && inputs.prefix_lengths && inputs.kv_cache_block_id, "failed to get information for pd seperation store cache");
        CacheStoreInputs cache_store_inputs({inputs.input_lengths, inputs.prefix_lengths, inputs.kv_cache_block_id});
        attention_inputs.cache_store_inputs = cache_store_inputs;
    }
    if (inputs.kv_cache_block_id) {
        checkKvBlocksShape(inputs.kv_cache_block_id);
        KvCacheInfo kv_cache;
        kv_cache.layer_num = layer_num_;
        kv_cache.kv_cache_block_id = device_->clone({*inputs.kv_cache_block_id, AllocationType::DEVICE, {"kv_cache_block_id"}});
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

    auto prep_output = device_->prepareModelRun({
            description_.attention_conf,
            inputs.sequence_lengths,
            inputs.kv_cache_block_id,
            dtype,
            context_batch_size,
            decoder_batch_size,
            (bool)k_cache_buffer_,
            attention_inputs.max_prefix_length > 0,
            (bool)weights_.linear_bias_slopes,
        });
    if (inputs.cache_keys) {
        vector<int64_t> cache_keys_vec = ft::buffer2vector<int64_t>(*inputs.cache_keys);
        attention_inputs.cache_keys = transVectorToString(cache_keys_vec);
    }
    attention_inputs.flash_infer_attn_params.swap(prep_output.flash_infer_attn_params);
    attention_inputs.request_id = inputs.request_id;
    attention_inputs.request_pd_separation = inputs.request_pd_separation;
    attention_inputs.block_size = inputs.block_size;
    attention_inputs.scale_block_size = inputs.scale_block_size;
    attention_inputs.pd_separation = inputs.pd_separation;

    if (context_batch_size && prep_output.need_mask) {
        attention_inputs.attention_mask = device_->attentionMask({
                inputs.input_lengths->view(decoder_batch_size, context_batch_size),
                *inputs.prefix_lengths,
                dtype,
                description_.attention_conf.mask_type == ft::AttentionMaskType::causalMask
            });
    }
}

GptLayerInputs GptModel::forwardPreLayers(const GptModelInputs& inputs) {
    const auto combo_tokens = device_->clone(
        {*inputs.combo_tokens, AllocationType::DEVICE, {"combo_tokens"}});

    const auto& embedding_table = weights_.embedding->kernel;

    const BufferPtr combo_position_ids = inputs.combo_position_ids ? device_->clone({*inputs.combo_position_ids}): nullptr;
    const BufferPtr combo_tokens_type_ids = inputs.combo_tokens_type_ids ? device_->clone({*inputs.combo_tokens_type_ids}): nullptr;

    const BufferPtr text_tokens_mask = inputs.multimodal_features ?
        device_->clone({*inputs.text_tokens_mask, AllocationType::DEVICE, {"text_tokens_mask"}}) : nullptr;
    const BufferPtr mm_feature_locs = inputs.mm_features_locs ? inputs.mm_features_locs: nullptr;

    // word embedding lookup
    auto hidden = device_->embeddingLookup({
            *combo_tokens, *embedding_table, description_.input_embedding_scalar,
            text_tokens_mask ? (OptionalConstBufferRef)*text_tokens_mask : nullopt,
            combo_position_ids ? (OptionalConstBufferRef)*combo_position_ids: nullopt,
            weights_.position_encoding ? (OptionalConstBufferRef)*weights_.position_encoding->kernel: nullopt,
            combo_tokens_type_ids ? (OptionalConstBufferRef)*combo_tokens_type_ids: nullopt,
            weights_.token_type_embedding ? (OptionalConstBufferRef)*weights_.token_type_embedding->kernel: nullopt});
    const auto dtype = hidden->type();
    if (residual_scale_fp32_ && residual_scale_->type() != dtype) {
        residual_scale_ = device_->convert({residual_scale_fp32_, dtype});
    }
    if (device_props_.tp_size > 1) {
        hidden = tpSyncEmbeddingOrLogits(hidden);
    }

    // pre layernorm
    BufferPtr pre_decoder_residual = nullptr;
    if (description_.act_qscheme != QScheme::NoQuantize && weights_.pre_decoder_layernorm) {
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
                                                                description_.layernorm_eps,
                                                                true,
                                                                pre_decoder_residual != nullptr,
                                                                description_.norm_type,
                                                                description_.act_qscheme));
        hidden = std::move(decoder_input.output);
    }

    if (inputs.multimodal_features) {
        hidden = device_->multimodalEmbedding({
                hidden,
                (OptionalConstVecBufferPtrRef)inputs.multimodal_features,
                mm_feature_locs ? (OptionalConstBufferRef)*mm_feature_locs: nullopt
        });
    }

    printBufferData(*hidden, "input_hidden");

    // prepare resources for all layers
    AttentionCommonInputs attention_common_inputs({
        device_->clone({*inputs.input_lengths}),
        device_->clone({*inputs.sequence_lengths}),
    });
    prepareAttentionInputs(inputs, dtype, attention_common_inputs);
    attention_common_inputs.position_ids = combo_position_ids;

    return {move(hidden), move(pre_decoder_residual), inputs.dp_token_nums, move(attention_common_inputs), dtype};
}

GptLayerOutputs GptModel::forwardGptLayer(
    GptLayerInputs inputs,
    const int32_t layer_id,
    ft::lora::LoraModelInputPtr lora_model_input)
{
    auto hidden = inputs.hidden;
    auto pre_decoder_residual = inputs.pre_decoder_residual;
    auto attention_common_inputs = move(inputs.attention_common_inputs);

    attention_common_inputs.layer_id = layer_id;
    const auto& layer = weights_.layers[layer_id];
    ft::QScheme act_qscheme = description_.act_qscheme;

    // here hidden->dtype maybe int8, so use dytpe of embedding lookup result instead
    auto attn_out_buf = device_->allocateBuffer({inputs.dtype, hidden->shape()}, {"attn_out_buf"});
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
                                                                        description_.layernorm_eps,
                                                                        true,
                                                                        false,
                                                                        description_.norm_type,
                                                                        act_qscheme));
        hidden = std::move(pre_layernorm_output.output);
    }

    if (k_cache_buffer_) {
        attention_common_inputs.kv_cache->k_cache_buffer = k_cache_buffer_->index(layer_id);
        attention_common_inputs.kv_cache->v_cache_buffer = v_cache_buffer_->index(layer_id);
        if (k_scale_buffer_) {
            attention_common_inputs.kv_cache->k_scale_buffer = k_scale_buffer_->index(layer_id);
            attention_common_inputs.kv_cache->v_scale_buffer = v_scale_buffer_->index(layer_id);
        }
    }
    if (lora_model_input) {
        attention_common_inputs.lora_input = lora_model_input->getAttentionLayerLoraInput(layer_id);
    }

    auto attn_output = device_->attentionLayer(AttentionLayerParams({
        layer_id,
        *hidden,
        move(attn_out_buf),
        description_.attention_conf,
        layer.self_attention_weights,
        attention_common_inputs,
        device_props_.attn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt,
        {description_.layernorm_eps, description_.norm_type},
        act_qscheme
    }));
    auto attn_hidden = std::move(attn_output.hidden_states);
    if (device_props_.tp_size > 1) {
        // Note: for custom all reduce, allReduce will allocate a new buffer and replace the original attn_hidden with it
        attn_hidden = device_->allReduce({std::move(attn_hidden), ReduceOp::Sum}).buffer;
    }
    if (residual_scale_) {
        attn_hidden = device_->multiply({*residual_scale_, *attn_hidden});
    }
    printBufferData(*attn_hidden, "layer_" + to_string(layer_id) + "_attn_output");

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
                                                        description_.layernorm_eps,
                                                        false,
                                                        description_.post_layernorm,
                                                        description_.norm_type,
                                                        act_qscheme);

        auto post_layernorm_output = device_->layernorm(post_layernorm_params);
        hidden = std::move(post_layernorm_output.output);
        attn_hidden = std::move(post_layernorm_output.before_norm_output);
        residual = attn_hidden;
        printBufferData(*residual, "post_layernorm_residual");
    } else {
        residual2 = attn_hidden;
    }

    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_ffn_input");
    auto ffn_output_buf = device_->allocateBuffer({inputs.dtype, hidden->shape()}, {"ffn_out_buf"});
    // Note: for custom all reduce, prepareAllReduce will replace the original attn_out_buf with
    // a new custom_ar_comm buffer. Here we must make sure that attn_out_buf is not released or replaced by
    // other buffer before the actual allreduce operations. Otherwise, it will raise an error in custom ar.
    ffn_output_buf = device_->prepareAllReduce({std::move(ffn_output_buf), ReduceOp::Sum}).buffer;
    auto ffn_layer_params = FfnLayerParams({*hidden, description_.ffn_conf,
                                            layer.ffn_weights,
                                            device_props_.ffn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt,
                                            inputs.dp_token_nums ? (OptionalConstBufferRef)*inputs.dp_token_nums : nullopt,
                                            act_qscheme,
                                            std::move(ffn_output_buf)});
    if (lora_model_input) {
        ffn_layer_params.lora_input =  lora_model_input->getFfnLayerLoraInput(layer_id);
    }
    auto ffn_output = device_->ffnLayer(ffn_layer_params);
    hidden = ffn_output.hidden_states;
    if (device_props_.tp_size > 1 && !(device_props_.dp_size > 1 && layer.ffn_weights.moe_gating_weight)) {
        // Note: for custom all reduce, allReduce will allocate a new buffer and replace the original attn_hidden with it
        hidden = device_->allReduce({std::move(hidden), ReduceOp::Sum}).buffer;
    }
    if (residual_scale_) {
        hidden = device_->multiply({*residual_scale_, *hidden});
    }
    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_ffn_output");

    // TODO: maybe move this layernorm to ffn layer
    auto ffn_layernorm_output = device_->layernorm(LayernormParams(hidden,
                                                                    pre_decoder_residual,
                                                                    ft::mayGetRef(layer.post_ffn_layernorm),
                                                                    device_props_.ffn_fuse_add_residual ? nullopt : (OptionalConstBufferRef)*residual,
                                                                    (residual2 == nullptr) ? nullopt : (OptionalConstBufferRef)*residual2,
                                                                    ft::mayGetRef(WEIGHT_MAY_GET_BIAS(layer.ffn_weights.down_weight)),
                                                                    1.0f,
                                                                    description_.layernorm_eps,
                                                                    true,
                                                                    description_.post_layernorm,
                                                                    description_.norm_type,
                                                                    ((layer_id == layer_num_ - 1) || (!layer.post_ffn_layernorm)) ? QScheme::NoQuantize: act_qscheme));
    hidden = std::move(ffn_layernorm_output.output);
    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_final_hidden");

    return {hidden, pre_decoder_residual};
}

GptModelOutputs GptModel::forwardPostLayers(
    const ft::BufferPtr input,
    const bool has_context_request,
    const bool need_all_logits,
    const ft::BufferPtr lm_output_indexes)
{
    auto hidden = input;
    if (weights_.final_layernorm) {
        auto final_layernorm = device_->layernorm(LayernormParams(hidden,
                                                                  nullptr,
                                                                  ft::mayGetRef(weights_.final_layernorm),
                                                                  nullopt,
                                                                  nullopt,
                                                                  nullopt,
                                                                  0.f,
                                                                  description_.layernorm_eps,
                                                                  true,
                                                                  false,
                                                                  description_.norm_type));
        hidden = std::move(final_layernorm.output);
    }
    printBufferData(*hidden, "final_hidden");

    const auto& lm_head = weights_.lm_head;
    // in case host buffer destruct before async clone finished
    device_->syncAndCheck();
    if (lm_head) {
        // gen last token hidden
        auto last_hidden = has_context_request && !need_all_logits
                         ? device_->select({*hidden, *device_->clone({*lm_output_indexes})})
                         : hidden;

        printBufferData(*last_hidden, "last_hidden");

        auto logits = device_->gemm(GemmParams(
            *last_hidden, *(lm_head->kernel), nullopt, nullptr,
            ft::DataType::TYPE_FP32, TransposeOperation::NONE, TransposeOperation::TRANSPOSE));
        if (device_props_.tp_size > 1) {
            logits = tpSyncEmbeddingOrLogits(logits);
        }
        // TODO(xinfei.sxf) calculate softmax_result
        ft::BufferPtr softmax_result;
        // logits is too big, tmp not print default
        // printBufferData(*logits, "logits");
        if (need_all_logits) {
            auto last_logits = device_->select({*logits, *device_->clone({*lm_output_indexes})});
            return {std::move(last_logits), std::move(last_hidden), std::move(hidden), std::move(logits), std::move(softmax_result)};
        }
        return {std::move(logits), std::move(last_hidden), std::move(hidden), nullptr, std::move(softmax_result)};
    } else {
        return {nullptr, nullptr, std::move(hidden)};
    }
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
    auto layer_inputs = forwardPreLayers(inputs);

    for (int32_t i = 0; i < layer_num_; ++i) {
        auto layer_outputs = forwardGptLayer(layer_inputs, i, inputs.lora_model_input);
        layer_inputs.hidden = move(layer_outputs.hidden);
        layer_inputs.pre_decoder_residual = move(layer_outputs.pre_decoder_residual);
    }

    return forwardPostLayers(layer_inputs.hidden, layer_inputs.attention_common_inputs.context_batch_size,
                             inputs.need_all_logits, inputs.lm_output_indexes);
}

void dpAndTpSyncModelInputs(GptModelInputs &inputs, ft::DeviceBase* device) {
    if (device->getDeviceProperties().dp_size > 1) {
        inputs.dp_token_nums = device->allocateBuffer(
                {ft::DataType::TYPE_UINT32,
                         {device->getDeviceProperties().dp_size},
                         ft::AllocationType::HOST});
        if (device->getDeviceProperties().tp_rank == 0) {
            *(inputs.dp_token_nums->dataWithOffset<uint32_t>(device->getDeviceProperties().dp_rank)) = inputs.combo_tokens->shape()[0];
            device->allGather({{inputs.dp_token_nums}, ParallelMode::DP});
        }
        if (device->getDeviceProperties().tp_size <= 1) {
            device->syncCommunication(false);
        }
    }
    if (device->getDeviceProperties().tp_size <= 1) {
        return;
    }
    const size_t shape_hints_size = GptModelInputIndex::gptModelInputLength;
    auto shape_hints = device->allocateBuffer({ft::DataType::TYPE_INT32, {shape_hints_size}, ft::AllocationType::HOST});
    auto shape_hints_ptr = shape_hints->data<int32_t>();
    shape_hints_ptr[GptModelInputIndex::comboTokens] = inputs.combo_tokens.get() ? inputs.combo_tokens->size() : 0;
    shape_hints_ptr[GptModelInputIndex::inputLengths] = inputs.input_lengths.get() ? inputs.input_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::sequenceLengths] = inputs.sequence_lengths.get() ? inputs.sequence_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::prefixLengths] = inputs.prefix_lengths.get() ? inputs.prefix_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch] = inputs.kv_cache_block_id.get() ? inputs.kv_cache_block_id->shape()[1] : 0;
    shape_hints_ptr[GptModelInputIndex::lmOutputIndexes] = inputs.lm_output_indexes.get() ? inputs.lm_output_indexes->size() : 0;
    shape_hints_ptr[GptModelInputIndex::comboPositionIds] = inputs.combo_position_ids.get() ? inputs.combo_position_ids->size() : 0;
    shape_hints_ptr[GptModelInputIndex::loraIds] = inputs.lora_ids.get() ? inputs.lora_ids->size() : 0;
    shape_hints_ptr[GptModelInputIndex::loraInputLengths] = inputs.lora_input_lengths.get() ? inputs.lora_input_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::textTokensMask] = inputs.text_tokens_mask.get() ? inputs.text_tokens_mask->size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs] = inputs.mm_features_locs.get() ? inputs.mm_features_locs->size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] = inputs.multimodal_features.has_value() ? inputs.multimodal_features.value().size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesSize] = shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] ? inputs.multimodal_features.value()[0]->shape()[1] : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype] = shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] ? (std::uint8_t)inputs.multimodal_features.value()[0]->type() : 0;
    shape_hints_ptr[GptModelInputIndex::needAllLogits] = inputs.need_all_logits;
    device->broadcast({{shape_hints}, 0});
    device->syncCommunication(false);
    device->syncAndCheck();

    // multimodal features shape broadcast
    ft::BufferPtr mm_features_shape;
    int32_t* mm_features_shape_ptr = nullptr;
    inputs.need_all_logits = shape_hints_ptr[GptModelInputIndex::needAllLogits];
    const size_t mm_features_num = shape_hints_ptr[GptModelInputIndex::mmFeaturesNum];
    if (mm_features_num) {
        mm_features_shape =
            device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::mmFeaturesNum]}, ft::AllocationType::HOST});
        mm_features_shape_ptr = mm_features_shape->data<int32_t>();
        for (auto i = 0; i < mm_features_num; ++i) {
            mm_features_shape_ptr[i] = inputs.multimodal_features.has_value() ? inputs.multimodal_features.value()[i]->shape()[0] : 0;
        }
        device->broadcast({{mm_features_shape}, 0});
        device->syncCommunication(false);
        device->syncAndCheck();
    }

    auto max_blocks = (size_t)shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch];
    auto combo_position_ids_size = shape_hints_ptr[GptModelInputIndex::comboPositionIds];
    auto text_tokens_mask_size = shape_hints_ptr[GptModelInputIndex::textTokensMask];
    auto mm_features_locs_size = shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs];

    if (device->getDeviceProperties().tp_rank) {
        auto context_batch_size = (size_t)shape_hints_ptr[GptModelInputIndex::prefixLengths];

        inputs.combo_tokens = device->allocateBuffer(
            {ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::comboTokens]}, ft::AllocationType::HOST});
        inputs.input_lengths = device->allocateBuffer(
            {ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::inputLengths]}, ft::AllocationType::HOST});
        inputs.sequence_lengths = device->allocateBuffer(
            {ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::sequenceLengths]}, ft::AllocationType::HOST});
        inputs.prefix_lengths = device->allocateBuffer(
             {ft::DataType::TYPE_INT32, {context_batch_size}, ft::AllocationType::HOST});
        if (max_blocks != 0) {
            inputs.kv_cache_block_id = device->allocateBuffer(
                    {ft::DataType::TYPE_INT32,
                    {(size_t)shape_hints_ptr[GptModelInputIndex::inputLengths], max_blocks}, ft::AllocationType::HOST});
            inputs.cache_keys = device->allocateBuffer(
                    {ft::DataType::TYPE_INT64, {context_batch_size, max_blocks}, ft::AllocationType::HOST});
        }
        inputs.request_id = device->allocateBuffer(
            {ft::DataType::TYPE_INT64, {context_batch_size}, ft::AllocationType::HOST});
        inputs.request_pd_separation = device->allocateBuffer(
            {ft::DataType::TYPE_BOOL, {context_batch_size}, ft::AllocationType::HOST});
        inputs.lm_output_indexes = device->allocateBuffer(
            {ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::lmOutputIndexes]}, ft::AllocationType::HOST});
        if (combo_position_ids_size) {
            inputs.combo_position_ids = device->allocateBuffer(
                {ft::DataType::TYPE_INT32, {(size_t)combo_position_ids_size}, ft::AllocationType::HOST});
        }
        if (shape_hints_ptr[GptModelInputIndex::loraIds]) {
            inputs.lora_ids = device->allocateBuffer(
                {ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::loraIds]}, ft::AllocationType::HOST});
        }
        if (shape_hints_ptr[GptModelInputIndex::loraInputLengths]) {
            inputs.lora_input_lengths = device->allocateBuffer(
                {ft::DataType::TYPE_INT32, {(size_t)shape_hints_ptr[GptModelInputIndex::loraInputLengths]}, ft::AllocationType::HOST});
        }
        if (text_tokens_mask_size) {
            inputs.text_tokens_mask = device->allocateBuffer(
                {ft::DataType::TYPE_INT32, {(size_t)text_tokens_mask_size}, ft::AllocationType::HOST});
        }
        if (mm_features_locs_size) {
            inputs.mm_features_locs = device->allocateBuffer(
                {ft::DataType::TYPE_INT32, {(size_t)mm_features_locs_size}, ft::AllocationType::HOST});
        }
        if (mm_features_num) {
            std::vector<ft::BufferPtr> mm_features;
            for (auto mm_index = 0; mm_index < mm_features_num; ++mm_index) {
                mm_features.emplace_back(
                    device->allocateBuffer(
                        {(ft::DataType)shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype],
                         {(size_t)mm_features_shape_ptr[mm_index], (size_t)shape_hints_ptr[GptModelInputIndex::mmFeaturesSize]},
                         ft::AllocationType::HOST}));
            }
            inputs.multimodal_features = std::move(mm_features);
        }
    }

    std::vector<ft::BufferPtr> buffers;
    buffers.emplace_back(inputs.combo_tokens);
    buffers.emplace_back(inputs.input_lengths);
    buffers.emplace_back(inputs.sequence_lengths);
    buffers.emplace_back(inputs.prefix_lengths);
    if (max_blocks) {
        buffers.emplace_back(inputs.kv_cache_block_id);
        buffers.emplace_back(inputs.cache_keys);
    }
    buffers.emplace_back(inputs.request_id);
    buffers.emplace_back(inputs.request_pd_separation);
    buffers.emplace_back(inputs.lm_output_indexes);
    if (combo_position_ids_size) {
        buffers.emplace_back(inputs.combo_position_ids);
    }
    buffers.emplace_back(inputs.lora_ids);
    buffers.emplace_back(inputs.lora_input_lengths);
    if (text_tokens_mask_size) {
        buffers.emplace_back(inputs.text_tokens_mask);
    }
    if (mm_features_locs_size) {
        buffers.emplace_back(inputs.mm_features_locs);
    }
    if (mm_features_num) {
        for (auto& mm_feature: inputs.multimodal_features.value()) {
            buffers.emplace_back(mm_feature);
        }
    }
    if (device->getDeviceProperties().dp_size > 1) {
        buffers.emplace_back(inputs.dp_token_nums);
    }
    device->broadcast({buffers, 0});
    device->syncAndCheck();
}

} // namespace rtp_llm
