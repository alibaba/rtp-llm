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
#include "src/fastertransformer/devices/utils/DevicePerfWrapper.h"
#include <algorithm>
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
    debug_string << ", k block_size: " << k_block_size;
    debug_string << ", v block_size: " << v_block_size;
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
    device_->allGather({{all_data}});
    auto ret = device_->transpose({all_data->reshape({tp_size, buffer_shape[0], buffer_shape[1]})});
    ret->updateShape({buffer_shape[0], buffer_shape[1] * tp_size});
    return ret;
}

ft::AttentionCommonInputs GptModel::prepareAttentionInputs(
        const GptModelInputs& inputs,
        ft::DataType dtype,
        ft::BufferPtr combo_position_ids)
{
    AttentionCommonInputs attention_inputs({
        device_->clone({*inputs.input_lengths}),
        device_->clone({*inputs.sequence_lengths}),
    });
    attention_inputs.position_ids = combo_position_ids;
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

    // RUNTIME_ASSERT_OP_ARG(
    //     (cu_seqlens_data[context_batch_size] + decoder_batch_size == inputs.combo_tokens->shape()[0]),
    //     "combo_tokens is not consistent with input lengths, "
    //     "there are %d tokens in context plus %ld tokens in decoder batch, but got %ld input tokens.",
    //     cu_seqlens_data[context_batch_size], decoder_batch_size, inputs.combo_tokens->shape()[0]);

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

    FT_LOG_DEBUG("prepare model run sequence lengths: %s, input_lengths: %s, kv cache: %s, context batch size: %ld, decoder batch size: %ld",
                inputs.sequence_lengths->debugStringWithData<int32_t>().c_str(),
                inputs.input_lengths->debugStringWithData<int32_t>().c_str(),
                inputs.kv_cache_block_id ? inputs.kv_cache_block_id->debugString().c_str() : "NULL",
                context_batch_size, decoder_batch_size);
    auto prep_output = device_->prepareModelRun({
            description_.attention_conf,
            inputs.sequence_lengths,
            inputs.input_lengths,
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
    attention_inputs.k_block_size = inputs.k_block_size;
    attention_inputs.v_block_size = inputs.v_block_size;
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

    return attention_inputs;
}

MicroBatchPlan GptModel::planMicroBatches(const GptModelInputs& inputs) {
    if (!device_props_.enable_layer_micro_batch) {
        FT_LOG_DEBUG("micro batch disable when enable_layer_micro_batch is false");
        return {false, {}};
    }

    const auto& input_lengths = inputs.input_lengths;
    const auto& sequence_lengths = inputs.sequence_lengths;
    const auto decoder_batch_size = sequence_lengths->shape()[0];
    const auto context_batch_size = input_lengths->shape()[0] - decoder_batch_size;

    // NOTE: when context batch size > 0, to keep micro batching behavior consistent within dp ranks,
    // we still need to enable micro batching, but send empty query to the second micro batch
    if (context_batch_size || decoder_batch_size < 2) {
        FT_LOG_DEBUG("micro batch disable when context batch size > 0");
        return {true, {decoder_batch_size, 0}};
    }

    // NOTE: for now, we simply split the decode batch into two micro batches equally
    // TODO: design better split strategy that consider the computational workload of each request
    const auto batch_0_size = decoder_batch_size / 2;
    const auto batch_1_size = decoder_batch_size - batch_0_size;
    FT_LOG_DEBUG("micro batch enable, split decoder batch into two micro batches %ld, %ld",
                batch_0_size, batch_1_size);
    return {true, {batch_0_size, batch_1_size}};
}

vector<LayerMicroBatchInputs> GptModel::prepareMicroBatchInputs(
    const GptModelInputs& inputs,
    const BufferPtr& hidden,
    const BufferPtr& pre_decoder_residual,
    const ft::DataType dtype,
    const MicroBatchPlan& micro_batch_plan)
{
    vector<LayerMicroBatchInputs> micro_batch_inputs;
    size_t sliced_token_idx = 0;

    const auto decoder_batch_size = inputs.sequence_lengths->shape()[0];
    const auto context_batch_size = inputs.input_lengths->shape()[0] - decoder_batch_size;

    if (context_batch_size || decoder_batch_size < 2) {
        // if context request exists, then micro bacthing is de facto disabled
        // we put everything into the first micro batch, and send empty query to the second micro batch
        auto attention_common_inputs = prepareAttentionInputs(inputs, dtype, nullptr);
        micro_batch_inputs.push_back({hidden, pre_decoder_residual, attention_common_inputs});

        // The fake query
        GptModelInputs fake_inputs;
        fake_inputs.kv_cache_block_id = nullptr;
        fake_inputs.combo_tokens = inputs.combo_tokens->slice(0, 1);
        fake_inputs.input_lengths = device_->allocateBuffer({DataType::TYPE_INT32, {1}, AllocationType::HOST});
        fake_inputs.input_lengths->data<int32_t>()[0] = 1;
        fake_inputs.sequence_lengths = device_->allocateBuffer({DataType::TYPE_INT32, {0}, AllocationType::HOST});
        auto fake_hidden = device_->allocateBuffer({dtype, {1, hidden->shape()[1]}});
        auto attention_common_inputs_fake = prepareAttentionInputs(fake_inputs, dtype, nullptr);
        micro_batch_inputs.push_back({move(fake_hidden), nullptr, move(attention_common_inputs_fake)});
    } else {
        // if context request does not exist, do normal micro batching
        for (size_t i = 0; i < micro_batch_plan.decoder_sizes.size(); ++i) {
            const auto& decode_batch_size = micro_batch_plan.decoder_sizes[i];
            GptModelInputs micro_model_inputs = inputs;
            micro_model_inputs.combo_tokens = inputs.combo_tokens->slice(sliced_token_idx, decode_batch_size);
            micro_model_inputs.input_lengths = inputs.input_lengths->slice(sliced_token_idx, decode_batch_size);
            micro_model_inputs.sequence_lengths = inputs.sequence_lengths->slice(sliced_token_idx, decode_batch_size);
            micro_model_inputs.kv_cache_block_id = inputs.kv_cache_block_id->slice(sliced_token_idx, decode_batch_size);
            auto micro_hidden = hidden->slice(sliced_token_idx, decode_batch_size);
            auto micro_pre_decoder_residual = pre_decoder_residual ? pre_decoder_residual->slice(sliced_token_idx, decode_batch_size) : nullptr;
            auto attention_common_inputs = prepareAttentionInputs(micro_model_inputs, dtype, nullptr);
            micro_batch_inputs.push_back({
                move(micro_hidden), move(micro_pre_decoder_residual), move(attention_common_inputs)});
            sliced_token_idx += decode_batch_size;
        }
    }
    return micro_batch_inputs;
}

GptLayerInputs GptModel::forwardPreLayers(const GptModelInputs& inputs) {
    DevicePerfWrapper wrapper(device_, "forwardPreLayers");
    bool enable_sp = device_->getDeviceProperties().enable_sp;
    size_t token_num = inputs.combo_tokens->shape()[0];
    size_t pad_token_num = token_num;
    size_t pad_mod_num = device_props_.tp_size * max((size_t)1, device_props_.m_split);
    if (token_num <= pad_mod_num) {
        enable_sp = false;
    }
    if (enable_sp && token_num % pad_mod_num != 0) {
        pad_token_num = token_num + (pad_mod_num - token_num % pad_mod_num);
        BufferPtr combo_tokens = inputs.combo_tokens;
        BufferPtr pad_combo_tokens = device_->allocateBuffer({combo_tokens->type(), {pad_token_num}, AllocationType::HOST},{"pad_combo_tokens"});
        device_->bufMemset(*pad_combo_tokens, 0);
        device_->copy({pad_combo_tokens->view(0, token_num), *combo_tokens});
        inputs.combo_tokens = pad_combo_tokens;
        printBufferData(*combo_tokens, {"combo_tokens"});
        printBufferData(*pad_combo_tokens, {"pad_combo_tokens"});
    }
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

    if (device_props_.overlap_comm_type == 2) {
        const auto& layer0 = weights_.layers[0];
        FT_CHECK_WITH_INFO(description_.act_qscheme == QScheme::NoQuantize || description_.act_qscheme == QScheme::Qint8PerToken, "ring p2p overlap only supports bf16/fp16 or w8a8");
        const size_t max_batch_seq_len = autil::EnvUtil::getEnv("MAX_CONTEXT_BATCH_SIZE", 1) * device_->initParams().max_seq_len;
        const size_t attn_rs_hidden = layer0.self_attention_weights.output_weight->kernel->shape()[1];
        const size_t ffn_rs_hidden = layer0.ffn_weights.down_weight->kernel->shape()[1];
        const size_t attn_ag_hidden = layer0.self_attention_weights.qkv_weight->kernel->shape()[0];
        const size_t ffn_ag_hidden = layer0.ffn_weights.gate_weight->kernel->shape()[0];
        DataType rs_output_type = dtype;
        DataType ag_input_type = description_.act_qscheme == QScheme::NoQuantize ? dtype : DataType::TYPE_QINT8;
        bool enable_per_token_scale = description_.act_qscheme == QScheme::Qint8PerToken;
        bool enable_ffn_tp = enable_sp && device_props_.ffn_tp_size > 1;
        device_->prepareCommBuffer({max_batch_seq_len, attn_rs_hidden, ffn_rs_hidden, attn_ag_hidden, ffn_ag_hidden, rs_output_type, ag_input_type, enable_per_token_scale, enable_ffn_tp});
    }

    auto micro_batch_plan = planMicroBatches(inputs);
    if (micro_batch_plan.enable) {
        auto micro_batch_inputs = prepareMicroBatchInputs(
            inputs, hidden, pre_decoder_residual, dtype,  micro_batch_plan);
        return {move(hidden), move(pre_decoder_residual), AttentionCommonInputs(), dtype, micro_batch_inputs, enable_sp, token_num, pad_token_num};
    } else {
        // prepare resources for all layers
        auto attention_common_inputs = prepareAttentionInputs(inputs, dtype, combo_position_ids);
        return {move(hidden), move(pre_decoder_residual), move(attention_common_inputs), dtype, {}, enable_sp, token_num, pad_token_num};
    }
}

GptLayerOutputs GptModel::forwardMicroBatchedLayers(
        const GptLayerInputs& layer_inputs, const GptModelInputs& inputs)
{
    std::vector<GptLayerInputs> micro_batch_layer_inputs;
    for (auto& micro_batch_input : layer_inputs.micro_batch_inputs) {
        micro_batch_layer_inputs.push_back({
            micro_batch_input.hidden,
            micro_batch_input.pre_decoder_residual,
            micro_batch_input.attention_common_inputs,
            layer_inputs.dtype
        });
    }

    std::vector<LastLayerDeferedParams> last_layer_defered_params(micro_batch_layer_inputs.size());

    for (int32_t i = 0; i < layer_num_; ++i) {
        const auto& layer = weights_.layers[i];
        bool moe_layer = weights_.layers[i].ffn_weights.moe_gate_weight != nullptr;

        // dense layer does not need micro batching.
        if (!moe_layer) {
            for (auto& layer_input : micro_batch_layer_inputs) {
                auto layer_outputs = forwardGptLayer(layer_input, i, inputs.lora_model_input);
                layer_input.hidden = move(layer_outputs.hidden);
            }
            continue;
        }

        std::vector<EpFfnInputs> ep_inputs;
        for (size_t micro_batch_idx = 0; micro_batch_idx < micro_batch_layer_inputs.size(); ++micro_batch_idx) {
            auto& layer_input = micro_batch_layer_inputs[micro_batch_idx];
            auto batch_ep_input = forwardAttentionAndMoeGate(
                layer_input, last_layer_defered_params[micro_batch_idx], i);
            ep_inputs.push_back(move(batch_ep_input));
        }

        std::vector<EpFfnOutputs> ep_outputs;
        for (size_t micro_batch_idx = 0; micro_batch_idx < micro_batch_layer_inputs.size(); ++micro_batch_idx) {
            if (micro_batch_idx > 0 && ep_inputs[micro_batch_idx].dispatch_output.comm_barrier_hook) {
                FT_LOG_DEBUG("synchronize barrier event for layer %ld, micro batch %ld", i, micro_batch_idx);
                ep_inputs[micro_batch_idx].dispatch_output.comm_barrier_hook->hook_sync();
            }

            DevicePerfWrapper wrapper(device_, "mb_moe_layer_" + std::to_string(i) + "_idx_" + std::to_string(micro_batch_idx));
            // auto& layer_input = micro_batch_layer_inputs[micro_batch_idx];
            auto& batch_ep_input = ep_inputs[micro_batch_idx];
            const auto& ffn_params = batch_ep_input.moe_ffn_params;
            const auto& dispatched_output = batch_ep_input.dispatch_output;

            auto out = device_->moeFfnAndCombine(ffn_params, dispatched_output);
            auto output = out.hidden_states;
            printBufferData(*output, "moe_ffn_ep_out");

            ep_outputs.push_back(EpFfnOutputs({output, move(out.comm_barrier_hook)}));
        }

        for (size_t micro_batch_idx = 0; micro_batch_idx < micro_batch_layer_inputs.size(); ++micro_batch_idx) {
            // last layer: add residual and shared expert output
            auto& layer_input = micro_batch_layer_inputs[micro_batch_idx];
            auto& batch_ep_input = ep_inputs[micro_batch_idx];
            auto& batch_ep_output = ep_outputs[micro_batch_idx];

            if (i == layer_num_ - 1) {
                if (batch_ep_output.comm_barrier_hook) {
                    FT_LOG_DEBUG("synchronize barrier event for layer %ld, micro batch %ld", i, micro_batch_idx);
                    batch_ep_output.comm_barrier_hook->hook_sync();
                }
                auto& output = batch_ep_output.hidden;

                printBufferData(*output, "layer_" + to_string(i) + "_ffn_output");

                auto ffn_layernorm_output = device_->layernorm({
                    output,
                    nullptr,
                    ft::mayGetRef(layer.post_ffn_layernorm),
                    ft::mayGetRef(batch_ep_input.residual),
                    ft::mayGetRef(batch_ep_input.shared_expert_output),
                    nullopt,
                    1.0f,
                    description_.layernorm_eps,
                    true,
                    description_.post_layernorm,
                    description_.norm_type,
                    QScheme::NoQuantize
                });
                layer_input.hidden = move(ffn_layernorm_output.output);
                printBufferData(*layer_input.hidden, "layer_" + to_string(i) + "_final_hidden");
            } else {
                // not last layer: defer add residual and bias to next layer
                last_layer_defered_params[micro_batch_idx].residual = batch_ep_input.residual;
                last_layer_defered_params[micro_batch_idx].shared_expert_output = batch_ep_input.shared_expert_output;
                last_layer_defered_params[micro_batch_idx].post_ffn_layernorm_weights = layer.post_ffn_layernorm;
                last_layer_defered_params[micro_batch_idx].comm_barrier_hook = move(batch_ep_output.comm_barrier_hook);
                layer_input.hidden = move(batch_ep_output.hidden);
            }
        }
    }

    const auto& hidden = layer_inputs.hidden;
    size_t copy_from_token_idx = 0;
    if ((inputs.sequence_lengths->shape()[0] == inputs.input_lengths->shape()[0]) && inputs.input_lengths->shape()[0] > 1) {
        for (size_t i = 0; i < micro_batch_layer_inputs.size(); ++i) {
            const auto& micro_batch_hidden = micro_batch_layer_inputs[i].hidden;
            const auto micro_batch_token_num = micro_batch_hidden->shape()[0];
            const auto target_hidden = hidden->slice(copy_from_token_idx, micro_batch_token_num);
            device_->copy({*target_hidden, *micro_batch_hidden});
            copy_from_token_idx += micro_batch_token_num;
        }
    } else {
        device_->copy({*hidden, *(micro_batch_layer_inputs[0].hidden)});
    }

    return {hidden, nullptr};
}

GptLayerOutputs GptModel::forwardGptLayer(
    GptLayerInputs inputs,
    const int32_t layer_id,
    ft::lora::LoraModelInputPtr lora_model_input)
{
    auto pre_decoder_residual = inputs.pre_decoder_residual;
    auto attention_block_output = forwardAttentionBlock(inputs, layer_id, lora_model_input);

    auto hidden = move(attention_block_output.hidden);
    auto residual = move(attention_block_output.residual);
    auto residual2 = move(attention_block_output.residual2);
    const auto& layer = weights_.layers[layer_id];

    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_ffn_input");
    bool enable_sp = inputs.enable_sp;
    size_t rank_pad_token_num = enable_sp ? inputs.pad_token_num / device_props_.tp_size : hidden->shape()[0];
    auto ffn_output_buf = device_->allocateBuffer({inputs.dtype, {rank_pad_token_num, hidden->shape()[1]}}, {"ffn_out_buf"});
    if (!enable_sp) {
        // Note: for custom all reduce, prepareAllReduce will replace the original attn_out_buf with
        // a new custom_ar_comm buffer. Here we must make sure that attn_out_buf is not released or replaced by
        // other buffer before the actual allreduce operations. Otherwise, it will raise an error in custom ar.
        ffn_output_buf = device_->prepareAllReduce({std::move(ffn_output_buf), ReduceOp::Sum}).buffer;
    }
    auto ffn_layer_params = FfnLayerParams({*hidden, description_.ffn_conf,
                                            layer.ffn_weights,
                                            device_props_.ffn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt,
                                            description_.act_qscheme,
                                            std::move(ffn_output_buf),
                                            enable_sp});
    if (lora_model_input) {
        ffn_layer_params.lora_input =  lora_model_input->getFfnLayerLoraInput(layer_id);
    }
    auto ffn_output = device_->ffnLayer(ffn_layer_params);
    hidden = ffn_output.hidden_states;
    if (device_props_.ffn_tp_size > 1 && !layer.ffn_weights.moe_gating_weight && !enable_sp) {
        // Note: for custom all reduce, allReduce will allocate a new buffer and replace the original attn_hidden with it
        auto wrapper = DevicePerfWrapper(device_, "post_ffn_all_reduce, sizeBytes=%ld", (long)hidden->sizeBytes());
        hidden = device_->allReduce({std::move(hidden), ReduceOp::Sum, false, ParallelMode::FFN_TP}).buffer;
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
                                                                    ((layer_id == layer_num_ - 1) || (!layer.post_ffn_layernorm)) ? QScheme::NoQuantize: description_.act_qscheme));
    hidden = std::move(ffn_layernorm_output.output);
    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_final_hidden");

    return {hidden, pre_decoder_residual};
}

AttentionBlockOutputs GptModel::forwardAttentionBlock(
        const GptLayerInputs& inputs,
        const int32_t layer_id,
        ft::lora::LoraModelInputPtr lora_model_input)
{
    auto hidden = inputs.hidden;
    auto pre_decoder_residual = inputs.pre_decoder_residual;
    auto attention_common_inputs = move(inputs.attention_common_inputs);
    DevicePerfWrapper wrapper(device_, "attention_block_layer_" + std::to_string(layer_id) + "_bs_" + std::to_string(hidden->shape()[0]));

    attention_common_inputs.layer_id = layer_id;
    const auto& layer = weights_.layers[layer_id];
    bool enable_sp = inputs.enable_sp;

    // here hidden->dtype maybe int8, so use dytpe of embedding lookup result instead
    size_t rank_pad_token_num = enable_sp ? inputs.pad_token_num / device_props_.tp_size : hidden->shape()[0];
    BufferPtr attn_out_buf = device_->allocateBuffer({inputs.dtype, {rank_pad_token_num, hidden->shape()[1]}}, {"attn_out_buf"});
    if (!enable_sp) {
        // Note: for custom all reduce, prepareAllReduce will replace the original attn_out_buf with
        // a new custom_ar_comm buffer. Here we must make sure that attn_out_buf is not released or replaced by
        // other buffer before the actual allreduce operations. Otherwise, it will raise an error in custom ar.
        attn_out_buf = device_->prepareAllReduce({std::move(attn_out_buf), ReduceOp::Sum}).buffer;
    }
    auto residual = pre_decoder_residual ? pre_decoder_residual : hidden;
    printBufferData(*residual, "in residual");
    BufferPtr residual2 = nullptr;
    BufferPtr cloned_hidden = nullptr;
    if (layer.pre_layernorm) {
        cloned_hidden = device_->clone({*hidden, AllocationType::DEVICE, {"residual"}});
        residual = cloned_hidden;
        auto pre_layernorm_output = device_->layernorm(LayernormParams(hidden,
                                                                        nullptr,
                                                                        *layer.pre_layernorm,
                                                                        std::nullopt,
                                                                        std::nullopt,
                                                                        std::nullopt,
                                                                        0.f,
                                                                        description_.layernorm_eps,
                                                                        false,
                                                                        false,
                                                                        description_.norm_type,
                                                                        description_.act_qscheme,
                                                                        layer_id > 0 ? true: false,
                                                                        false));
           
        int m_split = device_props_.m_split;
        size_t overlap_comm_type = device_props_.overlap_comm_type;
        if (enable_sp && layer_id == 0) {
            if (overlap_comm_type == 1 && m_split > 0) {
                vector<int> selected_indices;
                selected_indices.reserve(rank_pad_token_num);
                size_t m = inputs.pad_token_num;
                size_t m_chunk = m / m_split;
                if (m > 128) {
                    m_chunk = (m / m_split + 127) & ~127;
                }
                size_t tp_rank = device_props_.tp_rank;
                size_t round = m_chunk / device_props_.tp_size;

                size_t offset = tp_rank * round;
                for (size_t i = 0; i < rank_pad_token_num; i++) {
                    selected_indices.push_back( (i / round) * m_chunk + i % round + offset);
                }
                // printBufferData(*vector2Buffer(selected_indices), "selected_indices");
                residual = device_->select({*residual, *device_->clone({*vector2Buffer(selected_indices)})});
            } else {
                residual = residual->slice(rank_pad_token_num * device_props_.tp_rank, rank_pad_token_num);
            }
        }
        
        hidden = std::move(pre_layernorm_output.output);
    }

    if (k_cache_buffer_ && attention_common_inputs.kv_cache) {
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
    AttentionLayerOutput attn_output;
    auto attn_params = AttentionLayerParams({
            layer_id,
            *hidden,
            move(attn_out_buf),
            description_.attention_conf,
            layer.self_attention_weights,
            attention_common_inputs,
            device_props_.attn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt,
            {description_.layernorm_eps, description_.norm_type},
            description_.act_qscheme,
            enable_sp,
            inputs.pad_token_num
    });
    if (description_.attention_conf.use_mla && device_->mla_ops_type != ft::MlaOpsType::MHA) {
        attn_output = device_->mlaAttentionLayer(attn_params);
    } else {
        attn_output = device_->attentionLayer(attn_params);
    }

    auto attn_hidden = std::move(attn_output.hidden_states);
    if (device_props_.tp_size > 1 && !enable_sp) {
        // Note: for custom all reduce, allReduce will allocate a new buffer and replace the original attn_hidden with it
        auto wrapper = DevicePerfWrapper(device_, "allReduce, sizeBytes=%ld", (long)attn_hidden->sizeBytes());
        attn_hidden = device_->allReduce({std::move(attn_hidden), ReduceOp::Sum}).buffer;
    }
    if (residual_scale_) {
        attn_hidden = device_->multiply({*residual_scale_, *attn_hidden});
    }
    printBufferData(*attn_hidden, "layer_" + to_string(layer_id) + "_attn_output");

    if (layer.post_layernorm) {
        // attn_hidden = attn_hidden + residual
        // hidden = layernorm(attn_hidden)
        printBufferData(*residual, "before post layernorm residual");
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
                                                        description_.act_qscheme,
                                                        false,
                                                        true);

        auto post_layernorm_output = device_->layernorm(post_layernorm_params);
        hidden = std::move(post_layernorm_output.output);
        attn_hidden = std::move(post_layernorm_output.before_norm_output);
        residual = attn_hidden;
        printBufferData(*residual, "after post layernorm residual");
    } else {
        residual2 = attn_hidden;
    }

    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_ffn_input");
    return {hidden, residual, residual2};
}

EpFfnInputs GptModel::forwardAttentionAndMoeGate(
        const GptLayerInputs& inputs,
        LastLayerDeferedParams& last_layer_defered_params,
        const int32_t layer_id)
{
    if (last_layer_defered_params.comm_barrier_hook) {
        FT_LOG_DEBUG("synchronize previous layer barrier event for layer %ld", layer_id);
        move(last_layer_defered_params.comm_barrier_hook)->hook_sync();
    }

    auto hidden = inputs.hidden;
    auto pre_decoder_residual = inputs.pre_decoder_residual;
    const auto& layer = weights_.layers[layer_id];

    if (last_layer_defered_params.residual || last_layer_defered_params.shared_expert_output || last_layer_defered_params.post_ffn_layernorm_weights) {
        printBufferData(*hidden, "layer_" + to_string(layer_id - 1) + "_ffn_output_defered");

        auto prev_ffn_layernorm_output = device_->layernorm({
            hidden,
            nullptr,
            ft::mayGetRef(last_layer_defered_params.post_ffn_layernorm_weights),
            ft::mayGetRef(last_layer_defered_params.residual),
            ft::mayGetRef(last_layer_defered_params.shared_expert_output),
            nullopt,
            1.0f,
            description_.layernorm_eps,
            true,
            description_.post_layernorm,
            description_.norm_type,
            layer.post_ffn_layernorm ? description_.act_qscheme : QScheme::NoQuantize
        });
        hidden = move(prev_ffn_layernorm_output.output);
        printBufferData(*hidden, "layer_" + to_string(layer_id - 1) + "_final_hidden_defered");
    }

    DevicePerfWrapper wrapper(device_, "mb_forwardGptLayer_" + std::to_string(layer_id) + "_bs_" + std::to_string(hidden->shape()[0]));

    GptLayerInputs real_layer_inputs = inputs;
    real_layer_inputs.hidden = hidden;
    auto attention_block_output = forwardAttentionBlock(real_layer_inputs, layer_id, nullptr);
    hidden = move(attention_block_output.hidden);
    auto residual = move(attention_block_output.residual);
    auto residual2 = move(attention_block_output.residual2);

    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_ffn_input");
    auto ffn_output_buf = device_->allocateBuffer({inputs.dtype, hidden->shape()}, {"ffn_out_buf"});
    auto ffn_layer_params = FfnLayerParams({*hidden, description_.ffn_conf,
                                            layer.ffn_weights,
                                            device_props_.ffn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt,
                                            description_.act_qscheme,
                                            std::move(ffn_output_buf)});

    auto shared_expert_output = device_->moeSharedExpert(ffn_layer_params).hidden_states;
    MoeGateSelectOutput gate_output = device_->moeGateSelect(ffn_layer_params);
    MoeDispatchOutput dispatched_output = device_->epDispatch({
        ffn_layer_params.input,
        *gate_output.expert_ids,
        *gate_output.expert_scales,
        description_.ffn_conf.moe_configs.value(),
        device_props_.enable_comm_overlap,
        description_.act_qscheme
    });

    return {hidden, residual, shared_expert_output, move(ffn_layer_params), move(gate_output), move(dispatched_output)};
}

GptLayerOutputs GptModel::forwardMoeFfn(const GptLayerOutputs& inputs, const int32_t layer_id) {
    return inputs;
}

GptModelOutputs GptModel::forwardPostLayers(
    ft::BufferPtr input,
    const bool has_context_request,
    const bool need_all_logits,
    const ft::BufferPtr lm_output_indexes,
    bool enable_sp,
    size_t token_num)
{
    DevicePerfWrapper wrapper(device_, "forwardPostLayers");
    BufferPtr all_gather_output = nullptr;
    if (enable_sp && device_props_.tp_size > 1) {
        all_gather_output = device_->allocateBuffer({input->type(), {input->shape()[0] * device_props_.tp_size, input->shape()[1]}}, {"all_gather_output"});
        size_t m = all_gather_output->shape()[0];
        int m_split = device_props_.m_split;
        size_t overlap_comm_type = device_props_.overlap_comm_type;
        if (overlap_comm_type == 1 && m_split > 0) {
            size_t token_idx = 0;
            size_t ag_token_idx = 0;
            size_t m_chunk = m / m_split;
            if (m > 128) {
                m_chunk = (m / m_split + 127) & ~127;
            }
            while (token_idx < m) {
                const auto micro_batch_tokens = std::min(m - token_idx, m_chunk);
                const auto ag_micro_batch_tokens = micro_batch_tokens / device_props_.tp_size;
                auto micro_batch_recv_buffer = all_gather_output->slice(token_idx, micro_batch_tokens);
                auto micro_ag_send_buffer = input->slice(ag_token_idx, ag_micro_batch_tokens);
                device_->allGather({{micro_batch_recv_buffer}, ParallelMode::TP, {micro_ag_send_buffer}, false});
                token_idx += micro_batch_tokens;
                ag_token_idx += ag_micro_batch_tokens;
            }
        } else {
            device_->allGather({{all_gather_output}, ParallelMode::TP, {input}, false});
        }

        size_t pad_mod_num = device_props_.tp_size * max((size_t)1, device_props_.m_split);
        if (token_num % pad_mod_num != 0) {
            input = device_->clone({all_gather_output->view(0, token_num), AllocationType::DEVICE});
        } else {
            input = all_gather_output;            
        }
    }

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
    if (lm_head) {
        // gen last token hidden
        printBufferData(*lm_output_indexes, "lm_output_indexes");
        auto last_hidden = has_context_request && !need_all_logits
                         ? device_->select({*hidden, *device_->clone({*lm_output_indexes})})
                         : hidden;

        printBufferData(*last_hidden, "last_hidden");

        auto logits = device_->gemm(GemmParams(
            *last_hidden, *(lm_head->kernel), nullopt, nullptr,
            ft::DataType::TYPE_FP32, TransposeOperation::NONE, TransposeOperation::TRANSPOSE));
        printBufferData(*logits, "logits");
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

GptModelOutputs GptModel::forward(const GptModelInputs& inputs) {
    auto layer_inputs = forwardPreLayers(inputs);

    GptLayerOutputs layer_outputs;

    if (layer_inputs.micro_batch_inputs.size()) {
        layer_outputs = forwardMicroBatchedLayers(layer_inputs, inputs);
    } else {
        for (int32_t i = 0; i < layer_num_; ++i) {
            layer_outputs = forwardGptLayer(layer_inputs, i, inputs.lora_model_input);
            layer_inputs.hidden = layer_outputs.hidden;
            layer_inputs.pre_decoder_residual = layer_outputs.pre_decoder_residual;
        }
    }

    auto outputs = forwardPostLayers(
        layer_outputs.hidden,
        inputs.input_lengths->shape()[0] != inputs.sequence_lengths->shape()[0],
        inputs.need_all_logits,
        inputs.lm_output_indexes,
        layer_inputs.enable_sp,
        layer_inputs.token_num);

    // make sure cpu buffers out lives gpu exec
    outputs.captured_values = make_shared<GptLayerInputs>(layer_inputs);
    return outputs;
}

void dpAndTpSyncModelInputs(GptModelInputs &inputs, ft::DeviceBase* device) {
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
                         ft::AllocationType::DEVICE}));
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
    device->broadcast({buffers, 0});
    device->syncAndCheck();
}

} // namespace rtp_llm
