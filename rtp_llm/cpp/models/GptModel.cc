#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/devices/OpData.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/models/Eagle3Model.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/devices/utils/DevicePerfWrapper.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <algorithm>
#include <memory>

using namespace std;

using namespace rtp_llm;

namespace rtp_llm {

GptModel::GptModel(const GptModelInitParams& params):
    device_(params.device),
    device_props_(params.device->getDeviceProperties()),
    layer_num_(params.weights.layers.size()),
    description_(params.description),
    weights_(params.weights),
    model_id_(params.model_id) {
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
        residual_scale_      = residual_scale_fp32_;
    }

    if (params.description.ffn_conf.moe_configs.has_value()) {
        auto moe_conf         = params.description.ffn_conf.moe_configs.value();
        overall_expert_stats_ = device_->createMoeExpertStates(
            {layer_num_, moe_conf.ep_size, moe_conf.expert_num, moe_conf.expert_num + moe_conf.extra_expert_num});
    }
}

void getPaddingOffsetAndCuSeqLens(int32_t*       padding_offset,
                                  int32_t*       cu_seqlens,
                                  const int32_t* sequence_length,
                                  const int32_t* prefix_length,
                                  const int32_t  batch_size,
                                  const int32_t  max_seq_len) {
    // do cumulated sum
    int32_t total_seq_len = 0;
    int32_t cum_offset    = 0;
    int32_t index         = 0;
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
    RUNTIME_ASSERT_OP_ARG(input_kv_offset->shape().size() == 2,
                          "kv_cache_blocks shape should be [batch_size, block_length].");
}

BufferPtr GptModel::tpSyncEmbeddingOrLogits(const BufferPtr& buffer) {
    const auto tp_size      = device_props_.tp_size;
    const auto tp_rank      = device_props_.tp_rank;
    const auto buffer_shape = buffer->shape();
    const auto local_size   = buffer->size();
    auto       all_data     = device_->allocateBuffer({buffer->type(), {buffer_shape[0], buffer_shape[1] * tp_size}});
    auto       buffer_view  = buffer->reshape({buffer->size()});
    auto       all_data_1d  = all_data->reshape({all_data->size()});
    device_->copy({all_data_1d.view(local_size * tp_rank, local_size), buffer_view});
    device_->allGather({{all_data}});
    device_->checkError();
    auto ret = device_->transpose({all_data->reshape({tp_size, buffer_shape[0], buffer_shape[1]})});
    device_->checkError();
    ret->updateShape({buffer_shape[0], buffer_shape[1] * tp_size});
    return ret;
}

rtp_llm::AttentionCommonInputs GptModel::prepareAttentionInputs(const GptModelInputs& inputs,
                                                                rtp_llm::DataType     attn_dtype,
                                                                rtp_llm::BufferPtr    combo_position_ids) {
    DevicePerfWrapper     wrapper(device_, "cpp model prepareAttentionInputs");
    AttentionCommonInputs attention_inputs({
        device_->clone({*inputs.input_lengths}),
        device_->clone({*inputs.sequence_lengths}),
    });
    attention_inputs.position_ids = combo_position_ids;
    if (inputs.kv_cache_block_id) {
        checkKvBlocksShape(inputs.kv_cache_block_id);
        KvCacheInfo kv_cache;
        kv_cache.layer_num = layer_num_;
        kv_cache.kv_cache_block_id =
            device_->clone({*inputs.kv_cache_block_id, AllocationType::DEVICE, {"kv_cache_block_id"}});
        attention_inputs.kv_cache = kv_cache;
    }
    const auto& input_lengths      = inputs.input_lengths;
    const auto& sequence_lengths   = inputs.sequence_lengths;
    const auto& prefix_lengths     = inputs.prefix_lengths;
    const auto  decoder_batch_size = sequence_lengths->shape()[0];
    const auto  context_batch_size = input_lengths->shape()[0] - decoder_batch_size;
    const auto  max_context_seq_len =
        context_batch_size ?
             *std::max_element(input_lengths->data<int32_t>() + decoder_batch_size,
                              input_lengths->data<int32_t>() + decoder_batch_size + context_batch_size) :
             0;
    RTP_LLM_CHECK_WITH_INFO(!prefix_lengths || prefix_lengths->size() == context_batch_size,
                            "prefix_lengths size %d is not equal to context batch size %d.",
                            prefix_lengths->size(),
                            context_batch_size);
    attention_inputs.max_prefix_length =
        context_batch_size && prefix_lengths ?
            *std::max_element(prefix_lengths->data<int32_t>(),
                              prefix_lengths->data<int32_t>() + prefix_lengths->size()) :
            0;
    const auto max_decoder_seq_len = decoder_batch_size ?
                                         *std::max_element(sequence_lengths->data<int32_t>(),
                                                           sequence_lengths->data<int32_t>() + decoder_batch_size) :
                                         0;

    std::vector<int32_t> cu_seqlens_data(context_batch_size + 1);
    std::vector<int32_t> padding_offset_data(inputs.combo_tokens->shape()[0]);
    getPaddingOffsetAndCuSeqLens(padding_offset_data.data(),
                                 cu_seqlens_data.data(),
                                 input_lengths->dataWithOffset<int32_t>(decoder_batch_size),
                                 nullptr,
                                 context_batch_size,
                                 max_context_seq_len);
    device_->checkError();

    // RUNTIME_ASSERT_OP_ARG(
    //     (cu_seqlens_data[context_batch_size] + decoder_batch_size == inputs.combo_tokens->shape()[0]),
    //     "combo_tokens is not consistent with input lengths, "
    //     "there are %d tokens in context plus %ld tokens in decoder batch, but got %ld input tokens.",
    //     cu_seqlens_data[context_batch_size], decoder_batch_size, inputs.combo_tokens->shape()[0]);

    attention_inputs.cu_seqlens =
        device_->clone({*vector2Buffer(cu_seqlens_data), AllocationType::DEVICE, {"cu_seqlens"}});
    if (attention_inputs.max_prefix_length) {
        attention_inputs.prefix_prompt_lengths = device_->clone(*prefix_lengths);
        std::vector<int32_t> cu_kv_seqlens_data(context_batch_size + 1);
        getPaddingOffsetAndCuSeqLens(nullptr,
                                     cu_kv_seqlens_data.data(),
                                     input_lengths->dataWithOffset<int32_t>(decoder_batch_size),
                                     prefix_lengths->data<int32_t>(),
                                     context_batch_size,
                                     max_context_seq_len);

        std::vector<uint32_t> kv_seqlens_data(context_batch_size);
        for (int i = 0; i < context_batch_size; i++) {
            kv_seqlens_data[i] = cu_kv_seqlens_data[i + 1] - cu_kv_seqlens_data[i];
        }

        attention_inputs.cu_kv_seqlens =
            device_->clone({*vector2Buffer(cu_kv_seqlens_data), AllocationType::DEVICE, {"cu_kv_seqlens"}});
        attention_inputs.kv_seqlens =
            device_->clone({*vector2Buffer(kv_seqlens_data), AllocationType::DEVICE, {"kv_seqlens"}});
        attention_inputs.context_total_kv_length = cu_kv_seqlens_data[context_batch_size];
    } else {
        attention_inputs.cu_kv_seqlens = attention_inputs.cu_seqlens;
        std::vector<uint32_t> kv_seqlens_data(context_batch_size, 0);
        attention_inputs.kv_seqlens =
            device_->clone({*vector2Buffer(kv_seqlens_data), AllocationType::DEVICE, {"kv_seqlens"}});
        attention_inputs.context_total_kv_length = cu_seqlens_data[context_batch_size];
    }
    device_->checkError();

    attention_inputs.padding_offset =
        device_->clone({*vector2Buffer(padding_offset_data), AllocationType::DEVICE, {"padding_offset"}});
    attention_inputs.decoder_batch_size  = decoder_batch_size;
    attention_inputs.context_batch_size  = context_batch_size;
    attention_inputs.context_max_seq_len = max_context_seq_len;
    attention_inputs.decoder_max_seq_len = max_decoder_seq_len;
    attention_inputs.context_token_num   = cu_seqlens_data[context_batch_size];
    if (weights_.linear_bias_slopes) {
        attention_inputs.linear_bias_slopes = weights_.linear_bias_slopes->kernel;
    }

    RTP_LLM_LOG_DEBUG(
        "prepare model run sequence lengths: %s, input_lengths: %s, kv cache: %s, context batch size: %ld, decoder batch size: %ld",
        inputs.sequence_lengths->debugStringWithData<int32_t>().c_str(),
        inputs.input_lengths->debugStringWithData<int32_t>().c_str(),
        inputs.kv_cache_block_id ? inputs.kv_cache_block_id->debugString().c_str() : "NULL",
        context_batch_size,
        decoder_batch_size);
    auto prep_output =
        device_->prepareModelRun({description_.attention_conf,
                                  inputs.prefix_lengths,
                                  inputs.sequence_lengths,
                                  inputs.input_lengths,
                                  inputs.kv_cache_block_id,
                                  attention_inputs.kv_cache ? attention_inputs.kv_cache->kv_cache_block_id : nullptr,
                                  k_cache_buffer_,
                                  attn_dtype,
                                  context_batch_size,
                                  decoder_batch_size,
                                  attention_inputs.max_prefix_length > 0,
                                  (bool)weights_.linear_bias_slopes});
    device_->checkError();

    attention_inputs.decode_flash_infer_attn.swap(prep_output.decode_flash_infer_attn);
    attention_inputs.prefill_flash_infer_attn.swap(prep_output.prefill_flash_infer_attn);
    attention_inputs.decode_trt_attn.swap(prep_output.decode_trt_attn);
    attention_inputs.prefill_trt_attn.swap(prep_output.prefill_trt_attn);
    attention_inputs.decode_aiter_attn.swap(prep_output.decode_aiter_attn);
    if (!inputs.warmup && inputs.pd_separation) {
        RTP_LLM_CHECK_WITH_INFO(inputs.input_lengths && inputs.prefix_lengths && inputs.kv_cache_block_id,
                                "failed to get information for pd seperation store cache");
        vector<int64_t> cache_keys_vec;
        if (inputs.cache_keys) {
            cache_keys_vec = rtp_llm::buffer2vector<int64_t>(*inputs.cache_keys);
        }
        CacheStoreInputs cache_store_inputs({
            inputs.input_lengths,
            inputs.prefix_lengths,
            inputs.kv_cache_block_id,
            attention_inputs.context_batch_size,
            attention_inputs.decoder_batch_size,
            inputs.request_id,
            inputs.request_pd_separation,
            transVectorToString(cache_keys_vec),
            inputs.seq_size_per_block,
            inputs.k_block_size,
            inputs.v_block_size,
            inputs.scale_block_size,
            inputs.pd_separation,
            model_id_,
            inputs.decode_entrance,
            inputs.warmup,
        });
        attention_inputs.cache_store_inputs = cache_store_inputs;
    }

    if (context_batch_size && prep_output.need_mask) {
        attention_inputs.attention_mask =
            device_->attentionMask({inputs.input_lengths->view(decoder_batch_size, context_batch_size),
                                    *inputs.prefix_lengths,
                                    attn_dtype,
                                    description_.attention_conf.mask_type == rtp_llm::AttentionMaskType::causalMask});
    }

    return attention_inputs;
}

MicroBatchPlan GptModel::planMicroBatches(const GptModelInputs& inputs) {
    if (!int(device_props_.enable_layer_micro_batch)) {
        RTP_LLM_LOG_DEBUG("micro batch disable when enable_layer_micro_batch is false");
        return {false, {}};
    }

    const auto& input_lengths      = inputs.input_lengths;
    const auto& sequence_lengths   = inputs.sequence_lengths;
    const auto  decoder_batch_size = sequence_lengths->shape()[0];
    const auto  context_batch_size = input_lengths->shape()[0] - decoder_batch_size;

    if (decoder_batch_size + context_batch_size < 2) {
        RTP_LLM_LOG_DEBUG("micro batch disable when batch size %ld is less than 2",
                          decoder_batch_size + context_batch_size);
        return {false, {}};
    }

    // TODO: design better split strategy that consider the computational workload of each request

    // disable micro batching if both context and decoder query exists.
    if (context_batch_size && decoder_batch_size) {
        if (layer_num_ == 1) {
            size_t total_token_num = decoder_batch_size;
            for (size_t i = 0; i < context_batch_size; i++) {
                total_token_num += input_lengths->data<int32_t>()[i + decoder_batch_size];
            }
            RTP_LLM_LOG_DEBUG("total_token_num %d, decode_batch_size %d, context_batch_size",
                              total_token_num,
                              decoder_batch_size,
                              context_batch_size);
            size_t context_batch_0_size = 0;
            size_t context_batch_1_size = 0;
            size_t decode_batch_0_size  = 0;
            size_t decode_batch_1_size  = 0;
            if (total_token_num > decoder_batch_size * 2) {
                decode_batch_0_size        = decoder_batch_size;
                decode_batch_1_size        = 0;
                size_t acc_token_num       = decoder_batch_size;
                size_t context_split_point = 0;
                for (context_split_point = 0; context_split_point < context_batch_size; context_split_point++) {
                    acc_token_num += input_lengths->data<int32_t>()[context_split_point + decoder_batch_size];
                    if (acc_token_num * 2 >= total_token_num) {
                        break;
                    }
                }
                context_batch_0_size = context_split_point;
                context_batch_1_size = context_batch_size - context_split_point;
            } else {
                decode_batch_0_size  = total_token_num / 2;
                decode_batch_1_size  = decoder_batch_size - total_token_num / 2;
                context_batch_0_size = 0;
                context_batch_1_size = context_batch_size;
            }
            RTP_LLM_LOG_DEBUG("split [c]%d:[d]%d in micro batch 0 and [c]%d:[d]%d in micro batch 1",
                              context_batch_0_size,
                              decode_batch_0_size,
                              context_batch_1_size,
                              decode_batch_1_size);
            return MicroBatchPlan{
                true, {{context_batch_0_size, decode_batch_0_size}, {context_batch_1_size, decode_batch_1_size}}};
        } else {
            RTP_LLM_LOG_DEBUG("split context in micro batch 0, decode in micro batch 1 disabled!");
            return {false, {}};
        }
    }

    const auto batch_size_to_split = context_batch_size ? context_batch_size : decoder_batch_size;
    const auto micro_batch_0_size  = (batch_size_to_split + 1) / 2;
    const auto micro_batch_1_size  = batch_size_to_split - micro_batch_0_size;

    RTP_LLM_LOG_DEBUG("split micro batch size %ld, %ld", micro_batch_0_size, micro_batch_1_size);
    return context_batch_size ? MicroBatchPlan{true, {{micro_batch_0_size, 0}, {micro_batch_1_size, 0}}} :
                                MicroBatchPlan{true, {{0, micro_batch_0_size}, {0, micro_batch_1_size}}};
}

std::pair<vector<GptModelInputs>, vector<TokenSliceInfo>>
GptModel::splitInputsIntoMicroBatches(const GptModelInputs& inputs, const MicroBatchPlan& micro_batch_plan) {
    vector<GptModelInputs> micro_batch_inputs;
    vector<TokenSliceInfo> token_slice_recipes;
    size_t                 sliced_token_idx       = 0;
    size_t                 sliced_lm_output_index = 0;
    size_t                 sliced_batch_idx       = 0;  // for input_lengths and kv cache block id
    size_t                 decode_batch_idx       = 0;  // for sequence_lengths
    size_t                 prefill_batch_idx      = 0;  // for lm_output_indexes and prefix_lengths

    if (!micro_batch_plan.enable) {
        RTP_LLM_LOG_DEBUG("micro batch disable when enable is false, use fake");
        // we put everything into the first micro batch, and send empty query to the second micro batch
        micro_batch_inputs.push_back(inputs);

        // The fake query
        GptModelInputs fake_inputs;
        fake_inputs.kv_cache_block_id = nullptr;
        fake_inputs.combo_tokens      = inputs.combo_tokens->slice(0, 1);
        fake_inputs.input_lengths     = device_->allocateBuffer({DataType::TYPE_INT32, {1}, AllocationType::HOST});
        fake_inputs.input_lengths->data<int32_t>()[0] = 1;
        fake_inputs.sequence_lengths = device_->allocateBuffer({DataType::TYPE_INT32, {0}, AllocationType::HOST});
        fake_inputs.prefix_lengths   = device_->allocateBuffer({DataType::TYPE_INT32, {1}, AllocationType::HOST});
        fake_inputs.prefix_lengths->data<int32_t>()[0] = 0;
        auto fake_hidden =
            device_->allocateBuffer({description_.data_type, {1, description_.attention_conf.hidden_size}});
        micro_batch_inputs.push_back(fake_inputs);
    } else {
        // TODO(wangyin.yx): refact this splitting method, extract common code
        for (size_t i = 0; i < micro_batch_plan.batch_infos.size(); ++i) {
            // Notes: p_micro_batch_size and d_micro_batch_size are continuous batch
            const auto& p_micro_batch_size = micro_batch_plan.batch_infos[i].prefill_num;
            const auto& d_micro_batch_size = micro_batch_plan.batch_infos[i].decoder_num;
            // RUNTIME_ASSERT_OP_ARG(!(p_micro_batch_size && d_micro_batch_size),
            //     "one micro batch can not contain both p and d tokens, but got %ld and %ld",
            //     p_micro_batch_size, d_micro_batch_size);
            RTP_LLM_LOG_DEBUG(
                "micro batch index %ld, prefill size %ld, decode size %ld", i, p_micro_batch_size, d_micro_batch_size);

            if (d_micro_batch_size && p_micro_batch_size) {
                GptModelInputs micro_model_inputs = inputs;
                size_t         total_batch_size   = d_micro_batch_size + p_micro_batch_size;
                RTP_LLM_LOG_DEBUG("d and p slice from %ld %ld %ld %ld",
                                  sliced_token_idx,
                                  sliced_batch_idx,
                                  decode_batch_idx,
                                  prefill_batch_idx);
                micro_model_inputs.input_lengths = inputs.input_lengths->slice(sliced_batch_idx, total_batch_size);
                micro_model_inputs.sequence_lengths =
                    inputs.sequence_lengths->slice(decode_batch_idx, d_micro_batch_size);
                micro_model_inputs.kv_cache_block_id =
                    inputs.kv_cache_block_id->slice(sliced_batch_idx, total_batch_size);
                micro_model_inputs.prefix_lengths = inputs.prefix_lengths->slice(prefill_batch_idx, p_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask ? inputs.attention_mask->slice(sliced_batch_idx, total_batch_size) : nullptr;
                micro_model_inputs.lm_output_lengths =
                    inputs.lm_output_lengths->slice(sliced_batch_idx, total_batch_size);
                int32_t slice_token_num =
                    std::accumulate(micro_model_inputs.input_lengths->data<int32_t>() + d_micro_batch_size,
                                    micro_model_inputs.input_lengths->data<int32_t>() + total_batch_size,
                                    0)
                    + d_micro_batch_size;
                int32_t slice_lm_output_num =
                    std::accumulate(micro_model_inputs.lm_output_lengths->data<int32_t>(),
                                    micro_model_inputs.lm_output_lengths->data<int32_t>() + total_batch_size,
                                    0);
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes->slice(sliced_lm_output_index, slice_lm_output_num);
                micro_model_inputs.combo_tokens = inputs.combo_tokens->slice(sliced_token_idx, slice_token_num);
                micro_model_inputs.request_id =
                    inputs.request_id ? inputs.request_id->slice(prefill_batch_idx, p_micro_batch_size) : nullptr;
                micro_model_inputs.request_pd_separation =
                    inputs.request_pd_separation ?
                        inputs.request_pd_separation->slice(prefill_batch_idx, p_micro_batch_size) :
                        nullptr;
                micro_model_inputs.cache_keys =
                    inputs.cache_keys ? inputs.cache_keys->slice(prefill_batch_idx, p_micro_batch_size) : nullptr;

                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, (size_t)slice_token_num});

                micro_batch_inputs.push_back(micro_model_inputs);

                sliced_lm_output_index += slice_lm_output_num;
                sliced_token_idx += slice_token_num;
                sliced_batch_idx += total_batch_size;
                prefill_batch_idx += p_micro_batch_size;
                decode_batch_idx += d_micro_batch_size;
                RTP_LLM_LOG_DEBUG(
                    "micro batch %ld sliced context and decode, batch idx %ld, token idx %ld, prefill batch idx %d, decode batch idx %d",
                    i,
                    sliced_batch_idx,
                    sliced_token_idx,
                    prefill_batch_idx,
                    decode_batch_idx);
            } else if (d_micro_batch_size) {
                GptModelInputs micro_model_inputs = inputs;
                RTP_LLM_LOG_DEBUG("d slice from %ld %ld %ld", sliced_token_idx, sliced_batch_idx, decode_batch_idx);
                micro_model_inputs.combo_tokens  = inputs.combo_tokens->slice(sliced_token_idx, d_micro_batch_size);
                micro_model_inputs.input_lengths = inputs.input_lengths->slice(sliced_batch_idx, d_micro_batch_size);
                micro_model_inputs.sequence_lengths =
                    inputs.sequence_lengths->slice(decode_batch_idx, d_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask ? inputs.attention_mask->slice(sliced_batch_idx, d_micro_batch_size) :
                                            nullptr;
                micro_model_inputs.kv_cache_block_id =
                    inputs.kv_cache_block_id->slice(sliced_batch_idx, d_micro_batch_size);
                micro_model_inputs.prefix_lengths =
                    device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {0}, rtp_llm::AllocationType::HOST}, {});
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes->slice(sliced_batch_idx, d_micro_batch_size);

                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, d_micro_batch_size});

                micro_batch_inputs.push_back(micro_model_inputs);

                sliced_token_idx += d_micro_batch_size;
                sliced_batch_idx += d_micro_batch_size;
                decode_batch_idx += d_micro_batch_size;
                sliced_lm_output_index += d_micro_batch_size;
                RTP_LLM_LOG_DEBUG("micro batch %ld sliced decode, batch idx %ld, token idx %ld",
                                  i,
                                  sliced_batch_idx,
                                  sliced_token_idx);
            } else {
                GptModelInputs micro_model_inputs = inputs;
                RTP_LLM_LOG_DEBUG("p slice from %ld %ld %ld", sliced_token_idx, sliced_batch_idx, prefill_batch_idx);
                micro_model_inputs.input_lengths = inputs.input_lengths->slice(sliced_batch_idx, p_micro_batch_size);
                micro_model_inputs.kv_cache_block_id =
                    inputs.kv_cache_block_id->slice(sliced_batch_idx, p_micro_batch_size);
                micro_model_inputs.prefix_lengths = inputs.prefix_lengths->slice(prefill_batch_idx, p_micro_batch_size);
                micro_model_inputs.attention_mask =
                    inputs.attention_mask ? inputs.attention_mask->slice(sliced_batch_idx, p_micro_batch_size) :
                                            nullptr;
                micro_model_inputs.sequence_lengths =
                    device_->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {0}, rtp_llm::AllocationType::HOST}, {});
                micro_model_inputs.lm_output_lengths =
                    inputs.lm_output_lengths->slice(sliced_batch_idx, p_micro_batch_size);
                int32_t slice_token_num =
                    std::accumulate(micro_model_inputs.input_lengths->data<int32_t>(),
                                    micro_model_inputs.input_lengths->data<int32_t>() + p_micro_batch_size,
                                    0);
                int32_t slice_lm_output_num =
                    std::accumulate(micro_model_inputs.lm_output_lengths->data<int32_t>(),
                                    micro_model_inputs.lm_output_lengths->data<int32_t>() + p_micro_batch_size,
                                    0);
                micro_model_inputs.lm_output_indexes =
                    inputs.lm_output_indexes->slice(sliced_lm_output_index, slice_lm_output_num);
                micro_model_inputs.combo_tokens = inputs.combo_tokens->slice(sliced_token_idx, slice_token_num);
                micro_model_inputs.request_id =
                    inputs.request_id ? inputs.request_id->slice(prefill_batch_idx, p_micro_batch_size) : nullptr;
                micro_model_inputs.request_pd_separation =
                    inputs.request_pd_separation ?
                        inputs.request_pd_separation->slice(prefill_batch_idx, p_micro_batch_size) :
                        nullptr;
                micro_model_inputs.cache_keys =
                    inputs.cache_keys ? inputs.cache_keys->slice(prefill_batch_idx, p_micro_batch_size) : nullptr;

                token_slice_recipes.emplace_back(TokenSliceInfo{sliced_token_idx, (size_t)slice_token_num});

                micro_batch_inputs.push_back(micro_model_inputs);
                sliced_lm_output_index += slice_lm_output_num;
                sliced_token_idx += slice_token_num;
                sliced_batch_idx += p_micro_batch_size;
                prefill_batch_idx += p_micro_batch_size;
                RTP_LLM_LOG_DEBUG("micro batch %ld sliced context, batch idx %ld, token idx %ld",
                                  i,
                                  sliced_batch_idx,
                                  sliced_token_idx);
            }
        }
    }
    return {micro_batch_inputs, token_slice_recipes};
}

vector<LayerMicroBatchInputs> GptModel::prepareMicroBatchInputs(const GptModelInputs&   inputs,
                                                                const BufferPtr&        hidden,
                                                                const BufferPtr&        pre_decoder_residual,
                                                                const rtp_llm::DataType attn_dtype,
                                                                const MicroBatchPlan&   micro_batch_plan) {

    auto [split_inputs, token_recipes] = splitInputsIntoMicroBatches(inputs, micro_batch_plan);

    vector<LayerMicroBatchInputs> final_inputs;
    final_inputs.reserve(split_inputs.size());

    for (size_t i = 0; i < split_inputs.size(); ++i) {
        const auto& micro_inputs = split_inputs[i];

        auto attention_common_inputs = prepareAttentionInputs(micro_inputs, attn_dtype, nullptr);

        if (!micro_batch_plan.enable) {
            final_inputs.emplace_back(LayerMicroBatchInputs{
                micro_inputs.combo_tokens, hidden, pre_decoder_residual, std::move(attention_common_inputs)});
            if (i > 0) {
                // The second micro-batch is a fake/dummy input used when micro-batching is disabled.
                // This is a workaround to maintain the original structure, which results in some tight coupling.
                final_inputs.back().fake = true;
            }
        } else {
            const auto& recipe       = token_recipes[i];
            auto        micro_hidden = hidden ? hidden->slice(recipe.offset, recipe.count) : nullptr;
            auto        micro_residual =
                pre_decoder_residual ? pre_decoder_residual->slice(recipe.offset, recipe.count) : nullptr;

            final_inputs.emplace_back(LayerMicroBatchInputs{
                micro_inputs.combo_tokens, micro_hidden, micro_residual, std::move(attention_common_inputs)});
        }
    }

    return final_inputs;
}

EmbeddingPostOutput GptModel::embeddingPost(const BufferPtr& hidden_states, const GptModelInputs& inputs) {
    return {hidden_states, nullptr};
};

GptLayerInputs GptModel::forwardPreLayers(const GptModelInputs& inputs) {
    DevicePerfWrapper wrapper(device_, "forwardPreLayers");
    bool              enable_sp     = device_->getDeviceProperties().enable_sp;
    size_t            token_num     = inputs.combo_tokens->shape()[0];
    size_t            pad_token_num = token_num;
    size_t            pad_mod_num   = device_props_.tp_size * max((size_t)1, device_props_.m_split);
    if (token_num <= pad_mod_num) {
        enable_sp = false;
    }
    if (enable_sp && token_num % pad_mod_num != 0) {
        pad_token_num              = token_num + (pad_mod_num - token_num % pad_mod_num);
        BufferPtr combo_tokens     = inputs.combo_tokens;
        BufferPtr pad_combo_tokens = device_->allocateBuffer(
            {combo_tokens->type(), {pad_token_num}, AllocationType::HOST}, {"pad_combo_tokens"});
        device_->bufMemset(*pad_combo_tokens, 0);
        device_->copy({pad_combo_tokens->view(0, token_num), *combo_tokens});
        inputs.combo_tokens = pad_combo_tokens;
        printBufferData(*combo_tokens, {"combo_tokens"});
        printBufferData(*pad_combo_tokens, {"pad_combo_tokens"});
    }
    device_->checkError();

    // Performance timing for data preparation operations
    auto       start_time   = std::chrono::high_resolution_clock::now();
    const auto combo_tokens = device_->clone({*inputs.combo_tokens, AllocationType::DEVICE, {"combo_tokens"}});
    auto       end_time     = std::chrono::high_resolution_clock::now();
    auto       duration     = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    RTP_LLM_LOG_INFO("GptModel combo_tokens clone time: %ld microseconds", duration.count());

    const auto& embedding_table = weights_.embedding->kernel;

    start_time = std::chrono::high_resolution_clock::now();
    const BufferPtr combo_position_ids =
        inputs.combo_position_ids ? device_->clone({*inputs.combo_position_ids}) : nullptr;
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    RTP_LLM_LOG_INFO("GptModel combo_position_ids clone time: %ld microseconds", duration.count());

    start_time = std::chrono::high_resolution_clock::now();
    const BufferPtr combo_tokens_type_ids =
        inputs.combo_tokens_type_ids ? device_->clone({*inputs.combo_tokens_type_ids}) : nullptr;
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    RTP_LLM_LOG_INFO("GptModel combo_tokens_type_ids clone time: %ld microseconds", duration.count());

    start_time = std::chrono::high_resolution_clock::now();
    const BufferPtr text_tokens_mask =
        inputs.multimodal_features ?
            device_->clone({*inputs.text_tokens_mask, AllocationType::DEVICE, {"text_tokens_mask"}}) :
            nullptr;
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    RTP_LLM_LOG_INFO("GptModel text_tokens_mask clone time: %ld microseconds", duration.count());

    const BufferPtr mm_feature_locs       = inputs.mm_features_locs ? inputs.mm_features_locs : nullptr;
    const BufferPtr input_embeddings_locs = inputs.input_embeddings_locs ? inputs.input_embeddings_locs : nullptr;

    // word embedding lookup
    start_time  = std::chrono::high_resolution_clock::now();
    auto hidden = device_->embeddingLookup(
        {*combo_tokens,
         *embedding_table,
         description_.input_embedding_scalar,
         text_tokens_mask ? (OptionalConstBufferRef)*text_tokens_mask : nullopt,
         combo_position_ids ? (OptionalConstBufferRef)*combo_position_ids : nullopt,
         weights_.position_encoding ? (OptionalConstBufferRef)*weights_.position_encoding->kernel : nullopt,
         combo_tokens_type_ids ? (OptionalConstBufferRef)*combo_tokens_type_ids : nullopt,
         weights_.token_type_embedding ? (OptionalConstBufferRef)*weights_.token_type_embedding->kernel : nullopt});
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    RTP_LLM_LOG_INFO("GptModel embeddingLookup time: %ld microseconds", duration.count());
    if (residual_scale_fp32_ && residual_scale_->type() != hidden->type()) {
        residual_scale_ = device_->convert({residual_scale_fp32_, hidden->type()});
    }
    device_->checkError();

    if (device_props_.tp_size > 1) {
        hidden = tpSyncEmbeddingOrLogits(hidden);
    }
    device_->checkError();

    EmbeddingPostOutput output = embeddingPost(hidden, inputs);
    hidden                     = output.hidden;
    device_->checkError();
    if (inputs.input_embeddings) {
        hidden =
            device_->inputEmbedding({hidden,
                                     (OptionalConstVecBufferPtrRef)inputs.input_embeddings,
                                     input_embeddings_locs ? (OptionalConstBufferRef)*input_embeddings_locs : nullopt});
    }

    auto hidden_dtype = hidden->type();
    auto attn_dtype   = hidden_dtype;

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
        hidden             = std::move(decoder_input.output);
    }
    device_->checkError();
    if (hidden != nullptr)
        printBufferData(*hidden, "before embedding hidden");
    if (mm_feature_locs != 0)
        printBufferData(*mm_feature_locs, "mm_feature_locs");
    if (inputs.multimodal_features) {
        std::vector<rtp_llm::BufferPtr> mm_features;
        bool                            features_need_quantize = false;
        if (inputs.multimodal_features.value()[0]->type() != hidden->type() && hidden->isQBuffer()) {
            features_need_quantize                 = true;
            ConstBufferPtr static_scale_reciprocal = nullptr;
            if (description_.act_qscheme == QScheme::Qint8PerTensor && weights_.pre_decoder_layernorm) {
                auto norm_weight        = weights_.pre_decoder_layernorm;
                static_scale_reciprocal = norm_weight->static_scale_reciprocal;
            }
            for (auto& mm_feature : inputs.multimodal_features.value()) {
                auto quantized_feature = device_->quantize(
                    {*mm_feature,
                     hidden->type(),
                     1,
                     description_.act_qscheme,
                     nullopt,
                     nullopt,
                     nullopt,
                     static_scale_reciprocal ? (OptionalConstBufferRef)*static_scale_reciprocal : nullopt});
                if (quantized_feature->isQBuffer()) {
                    quantized_feature->updateTypeAndShape(QBufferDtype2BufferDtype(quantized_feature->type()),
                                                          quantized_feature->shape());
                }
                mm_features.emplace_back(quantized_feature);
            }
        }
        bool     hidden_is_qbuffer = false;
        DataType hidden_qbuffer_dt = hidden->type();
        if (hidden->isQBuffer()) {
            hidden_is_qbuffer = true;
            hidden->updateTypeAndShape(QBufferDtype2BufferDtype(hidden_qbuffer_dt), hidden->shape());
        }

        hidden = device_->multimodalEmbedding({hidden,
                                               features_need_quantize ?
                                                   (OptionalConstVecBufferPtrRef)mm_features :
                                                   (OptionalConstVecBufferPtrRef)inputs.multimodal_features,
                                               mm_feature_locs ? (OptionalConstBufferRef)*mm_feature_locs : nullopt});
        if (hidden_is_qbuffer) {
            hidden->updateTypeAndShape(hidden_qbuffer_dt, hidden->shape());
        }
    }

    if (description_.act_qscheme == QScheme::Qint8PerTensor && !(hidden->isQBuffer())) {
        auto norm_weight             = weights_.pre_decoder_layernorm;
        auto static_scale_reciprocal = norm_weight->static_scale_reciprocal;
        hidden =
            device_->quantize({*hidden,
                               DataType::TYPE_INT8,
                               1,
                               description_.act_qscheme,
                               nullopt,
                               nullopt,
                               nullopt,
                               static_scale_reciprocal ? (OptionalConstBufferRef)*static_scale_reciprocal : nullopt});
    }

    device_->checkError();

    printBufferData(*hidden, "input_hidden");

    if (device_props_.overlap_comm_type == 2) {
        const auto& layer0 = weights_.layers[0];
        RTP_LLM_CHECK_WITH_INFO(description_.act_qscheme == QScheme::NoQuantize
                                    || description_.act_qscheme == QScheme::Qint8PerToken
                                    || description_.act_qscheme == Qfp8PerTensor,
                                "ring p2p overlap only supports bf16/fp16 or w8a8 or fp8 per block");
        const size_t max_batch_seq_len =
            device_->initParams().fifo_scheduler_config.max_context_batch_size * device_->initParams().max_seq_len;
        const size_t attn_rs_hidden         = layer0.self_attention_weights.output_weight->kernel->shape()[1];
        const size_t ffn_rs_hidden          = layer0.ffn_weights.down_weight->kernel->shape()[1];
        const size_t attn_ag_hidden         = layer0.self_attention_weights.qkv_weight->kernel->shape()[0];
        const size_t ffn_ag_hidden          = layer0.ffn_weights.gate_up_weight->kernel->shape()[0];
        DataType     rs_output_type         = hidden->type();
        DataType     ag_input_type          = attn_dtype;
        bool         enable_per_token_scale = description_.act_qscheme == QScheme::Qint8PerToken;
        bool         enable_ffn_tp          = enable_sp && device_props_.ffn_tp_size > 1;
        device_->prepareCommBuffer({max_batch_seq_len,
                                    attn_rs_hidden,
                                    ffn_rs_hidden,
                                    attn_ag_hidden,
                                    ffn_ag_hidden,
                                    rs_output_type,
                                    ag_input_type,
                                    enable_per_token_scale,
                                    enable_ffn_tp});
    }
    device_->checkError();
    if (device_->initParams().profile_debug_logging_config.check_nan) {
        (void)device_->checkNAN(*hidden);
    }
    if (int(device_props_.enable_layer_micro_batch)) {
        auto micro_batch_plan = planMicroBatches(inputs);
        auto micro_batch_inputs =
            prepareMicroBatchInputs(inputs, hidden, pre_decoder_residual, attn_dtype, micro_batch_plan);
        return {move(hidden),
                move(pre_decoder_residual),
                AttentionCommonInputs(),
                hidden_dtype,
                micro_batch_inputs,
                enable_sp,
                token_num,
                pad_token_num,
                output.residual};
    } else {
        // prepare resources for all layers
        auto attention_common_inputs = prepareAttentionInputs(inputs, attn_dtype, combo_position_ids);
        return {move(hidden),
                move(pre_decoder_residual),
                move(attention_common_inputs),
                hidden_dtype,
                {},
                enable_sp,
                token_num,
                pad_token_num,
                output.residual};
    }
}

vector<GptLayerInputs> GptModel::forwardPrefillMicroBatchedLayers(vector<GptLayerInputs>  micro_batch_layer_inputs,
                                                                  std::vector<BufferPtr>& eagle3_selected_hidden) {
    std::vector<LastLayerDeferedParams> last_layer_defered_params(micro_batch_layer_inputs.size());

    for (int32_t i = 0; i < layer_num_; ++i) {
        const auto& layer     = weights_.layers[i];
        bool        moe_layer = weights_.layers[i].ffn_weights.moe_gate_weight != nullptr;

        // dense layer does not need micro batching.
        if (!moe_layer) {
            for (auto& layer_input : micro_batch_layer_inputs) {
                auto layer_outputs = forwardGptLayer(layer_input, i, nullptr);
                layer_input.hidden = move(layer_outputs.hidden);
                if (dynamic_cast<Eagle3Model*>(this) == nullptr && device_props_.is_eagle3
                    && device_props_.eagle3_selected_layer.count(i) > 0) {
                    eagle3_selected_hidden.push_back(device_->clone({*layer_input.hidden, AllocationType::DEVICE}));
                }
            }
            continue;
        }

        std::vector<EpFfnInputs> ep_inputs;
        for (size_t micro_batch_idx = 0; micro_batch_idx < micro_batch_layer_inputs.size(); ++micro_batch_idx) {
            if (micro_batch_idx) {
                device_->holdBufferRecycle();
            }

            auto& layer_input         = micro_batch_layer_inputs[micro_batch_idx];
            bool  capture_last_hidden = dynamic_cast<Eagle3Model*>(this) == nullptr && device_props_.is_eagle3
                                       && device_props_.eagle3_selected_layer.count(i - 1) > 0;
            auto batch_ep_input = forwardAttentionAndMoeGate(
                layer_input, last_layer_defered_params[micro_batch_idx], i, micro_batch_idx, capture_last_hidden);
            if (capture_last_hidden) {
                eagle3_selected_hidden.push_back(
                    device_->clone({*batch_ep_input.last_layer_hidden, AllocationType::DEVICE}));
            }

            if (micro_batch_idx == 0) {
                // to overlap shared with combine
                auto shared_expert_output = device_->moeSharedExpert(batch_ep_input.moe_ffn_params).hidden_states;
                batch_ep_input.shared_expert_output = shared_expert_output;
            }

            batch_ep_input.compute_event = device_->createTorchEvent();
            ep_inputs.push_back(move(batch_ep_input));
        }

        std::vector<MoeDispatchOutput> dispatch_outputs;
        std::vector<MoeOutputs>        moe_ffn_outputs;
        for (size_t micro_batch_idx = 0; micro_batch_idx < micro_batch_layer_inputs.size(); ++micro_batch_idx) {
            auto&       ep_input         = ep_inputs[micro_batch_idx];
            const auto& ffn_layer_params = ep_input.moe_ffn_params;
            const auto& gate_output      = ep_input.gate_output;

            auto dispatched_output = device_->epDispatch({
                ep_input.quantized_hidden ? *(ep_input.quantized_hidden) : *(ep_input.hidden),
                *gate_output.expert_ids,
                *gate_output.expert_scales,
                description_.ffn_conf.moe_configs.value(),
                device_props_.enable_comm_overlap,
                description_.act_qscheme,
                ffn_layer_params.expert_stats,
                move(ep_input.compute_event),
            });

            device_->releaseBufferRecycleHold();
            device_->holdBufferRecycle();

            printBufferData(*dispatched_output.hidden, "layer_" + to_string(i) + "_dispatch_output");
            dispatch_outputs.push_back(dispatched_output);

            DevicePerfWrapper wrapper(device_,
                                      "mb_moe_layer_" + std::to_string(i) + "_idx_" + std::to_string(micro_batch_idx));
            // auto& layer_input = micro_batch_layer_inputs[micro_batch_idx];
            auto&       batch_ep_input = ep_inputs[micro_batch_idx];
            const auto& ffn_params     = batch_ep_input.moe_ffn_params;

            auto hidden_states = dispatched_output.hidden;
            if (device_->initParams().profile_debug_logging_config.check_nan) {
                (void)device_->checkNAN(*hidden_states);
            }
            auto moe_ffn_params = FfnLayerParams(
                {*hidden_states, ffn_params.configs, ffn_params.weights, ffn_params.residual, ffn_params.qscheme});
            prepareExpertStats(i, moe_ffn_params);

            if (dispatched_output.comm_barrier_hook) {
                dispatched_output.comm_barrier_hook->hook_sync();
                dispatched_output.comm_barrier_hook = nullptr;
            }

            hidden_states = device_
                                ->moeFfn(moe_ffn_params,
                                         {dispatched_output.expert_ids,
                                          dispatched_output.expert_scales,
                                          nullptr,
                                          dispatched_output.deep_ep_ll_output})
                                .hidden_states;
            device_->checkError();
            if (device_->initParams().profile_debug_logging_config.check_nan) {
                (void)device_->checkNAN(*hidden_states);
            }
            // shared experts to overlap combine
            if (micro_batch_idx) {
                auto shared_expert_output =
                    device_->moeSharedExpert(ep_inputs[micro_batch_idx].moe_ffn_params).hidden_states;
                ep_inputs[micro_batch_idx].shared_expert_output = shared_expert_output;
            }
            device_->checkError();

            auto compute_event = device_->createTorchEvent();
            moe_ffn_outputs.push_back({move(hidden_states), move(compute_event)});
            printBufferData(*hidden_states, "layer_" + to_string(i) + "_combine_input");
        }

        std::vector<EpFfnOutputs> ep_outputs;
        for (size_t micro_batch_idx = 0; micro_batch_idx < micro_batch_layer_inputs.size(); ++micro_batch_idx) {
            auto&       batch_ep_input    = ep_inputs[micro_batch_idx];
            const auto& ffn_params        = batch_ep_input.moe_ffn_params;
            auto&       dispatched_output = dispatch_outputs[micro_batch_idx];
            const auto& moe_conf          = ffn_params.configs.moe_configs.value();

            // const auto overlap = (!micro_batch_idx) ? device_props_.enable_comm_overlap : false;

            auto combine_out = device_->epCombine({
                moe_ffn_outputs[micro_batch_idx].hidden,
                dispatched_output.indices,
                ffn_params.output,
                dispatched_output.input_split_sizes,
                dispatched_output.output_split_sizes,
                moe_conf,
                ffn_params.input.shape()[0],
                // overlap, // device_props_.enable_comm_overlap,
                device_props_.enable_comm_overlap,
                dispatched_output.deep_ep_output,
                dispatched_output.deep_ep_ll_output,
                std::make_shared<MoeGateSelectOutput>(batch_ep_input.gate_output),
                dispatched_output.expert_ids,
                dispatched_output.expert_scales,
                move(moe_ffn_outputs[micro_batch_idx].compute_event),
            });
            device_->checkError();
            printBufferData(*combine_out.all_output, "layer_" + to_string(i) + "_combine_output");

            auto hook   = nullptr;
            auto output = combine_out.all_output;

            ep_outputs.push_back(EpFfnOutputs({output, move(combine_out), move(hook)}));

            if (micro_batch_idx == 0) {
                device_->releaseBufferRecycleHold();
            }
        }

        for (size_t micro_batch_idx = 0; micro_batch_idx < micro_batch_layer_inputs.size(); ++micro_batch_idx) {
            // last layer: add residual and shared expert output
            auto& layer_input     = micro_batch_layer_inputs[micro_batch_idx];
            auto& batch_ep_input  = ep_inputs[micro_batch_idx];
            auto& batch_ep_output = ep_outputs[micro_batch_idx];

            if (i == layer_num_ - 1) {
                if (batch_ep_output.combine_output.comm_barrier_hook) {
                    batch_ep_output.combine_output.comm_barrier_hook->hook_sync();
                }
                auto output = batch_ep_output.hidden;
                output      = device_->gatherCombineOutput(batch_ep_output.combine_output).hidden_states;

                printBufferData(*output, "layer_" + to_string(i) + "_ffn_output");

                auto ffn_layernorm_output = device_->layernorm({output,
                                                                nullptr,
                                                                rtp_llm::mayGetRef(layer.post_ffn_layernorm),
                                                                rtp_llm::mayGetRef(batch_ep_input.residual),
                                                                rtp_llm::mayGetRef(batch_ep_input.shared_expert_output),
                                                                nullopt,
                                                                1.0f,
                                                                description_.layernorm_eps,
                                                                true,
                                                                description_.post_layernorm,
                                                                description_.norm_type,
                                                                QScheme::NoQuantize});
                device_->checkError();
                layer_input.hidden = move(ffn_layernorm_output.output);
                printBufferData(*layer_input.hidden, "layer_" + to_string(i) + "_final_hidden");
            } else {
                // not last layer: defer add residual and bias to next layer
                last_layer_defered_params[micro_batch_idx].residual             = batch_ep_input.residual;
                last_layer_defered_params[micro_batch_idx].shared_expert_output = batch_ep_input.shared_expert_output;
                last_layer_defered_params[micro_batch_idx].post_ffn_layernorm_weights = layer.post_ffn_layernorm;
                if (last_layer_defered_params[micro_batch_idx].combine_output) {
                    last_layer_defered_params[micro_batch_idx].combine_output.value().params.expert_ids    = nullptr;
                    last_layer_defered_params[micro_batch_idx].combine_output.value().params.expert_scales = nullptr;
                    last_layer_defered_params[micro_batch_idx].combine_output                              = nullopt;
                }
                last_layer_defered_params[micro_batch_idx].combine_output    = move(batch_ep_output.combine_output);
                last_layer_defered_params[micro_batch_idx].comm_barrier_hook = move(batch_ep_output.comm_barrier_hook);
                layer_input.hidden                                           = move(batch_ep_output.hidden);
            }
        }
    }
    return micro_batch_layer_inputs;
}

vector<GptLayerInputs> GptModel::forwardDecodeMicroBatchedLayers(vector<GptLayerInputs>  micro_batch_layer_inputs,
                                                                 std::vector<BufferPtr>& eagle3_selected_hidden) {

    std::vector<LastLayerDeferedParams> last_layer_defered_params_vec(micro_batch_layer_inputs.size());
    for (int32_t i = 0; i < layer_num_; ++i) {
        const auto& layer     = weights_.layers[i];
        bool        moe_layer = layer.ffn_weights.moe_gate_weight != nullptr;

        // dense layer does not need micro batching.
        if (!moe_layer) {
            for (auto& layer_input : micro_batch_layer_inputs) {
                auto layer_outputs = forwardGptLayer(layer_input, i, nullptr);
                device_->checkError();
                layer_input.hidden = move(layer_outputs.hidden);
                if (dynamic_cast<Eagle3Model*>(this) == nullptr && device_props_.is_eagle3
                    && device_props_.eagle3_selected_layer.count(i) > 0) {
                    eagle3_selected_hidden.push_back(device_->clone({*layer_input.hidden, AllocationType::DEVICE}));
                }
            }
            continue;
        }

        for (size_t micro_batch_idx = 0; micro_batch_idx < micro_batch_layer_inputs.size(); ++micro_batch_idx) {
            auto& layer_input               = micro_batch_layer_inputs[micro_batch_idx];
            auto& last_layer_defered_params = last_layer_defered_params_vec[micro_batch_idx];

            auto last_layer_moe_ret = device_->stealMoEInsertionRet();
            // qwen moe has no shared expert, so we can not use it to check if the moe insertion is valid.
            if (layer.ffn_weights.shared_expert) {
                RUNTIME_ASSERT_OP_ARG(bool(last_layer_defered_params.shared_expert_output) == bool(last_layer_moe_ret),
                                      "moe insegrtion return should only be null if no previous layer.");
            }
            if (last_layer_defered_params.combine_output) {
                last_layer_defered_params.combine_output = nullopt;
            }
            last_layer_defered_params.combine_output =
                last_layer_moe_ret ? std::optional<rtp_llm::MoeCombineOutput>(last_layer_moe_ret->combine_output) :
                                     nullopt;

            bool capture_last_hidden = dynamic_cast<Eagle3Model*>(this) == nullptr && device_props_.is_eagle3
                                       && device_props_.eagle3_selected_layer.count(i - 1) > 0;
            auto ep_input = forwardAttentionAndMoeGate(
                layer_input, last_layer_defered_params, i, micro_batch_idx, capture_last_hidden);
            if (capture_last_hidden) {
                eagle3_selected_hidden.push_back(device_->clone({*ep_input.last_layer_hidden, AllocationType::DEVICE}));
            }

            // call combine hook sync
            auto& previous_moe_ret = device_->getMoEInsertionRet();
            if (previous_moe_ret && previous_moe_ret->combine_output.comm_barrier_hook) {
                previous_moe_ret->combine_output.comm_barrier_hook->hook_sync();
                previous_moe_ret->combine_output.comm_barrier_hook = nullptr;
            }

            MoeDispatchOutput dispatched_output = device_->epDispatch({
                ep_input.moe_ffn_params.input,
                *ep_input.gate_output.expert_ids,
                *ep_input.gate_output.expert_scales,
                description_.ffn_conf.moe_configs.value(),
                device_props_.enable_comm_overlap,
                description_.act_qscheme,
                ep_input.moe_ffn_params.expert_stats,
                std::move(ep_input.compute_event),
            });

            // set moe insertion params
            device_->setMoEInsertion(MoEInsertionParams(dispatched_output,
                                                        ep_input.moe_ffn_params,
                                                        std::make_shared<MoeGateSelectOutput>(ep_input.gate_output),
                                                        ep_input.hidden->shape()[0]));
            last_layer_defered_params.residual                   = ep_input.residual;
            last_layer_defered_params.post_ffn_layernorm_weights = layer.post_ffn_layernorm;

            // call shared
            auto shared_expert_output = device_->moeSharedExpert(ep_input.moe_ffn_params).hidden_states;
            device_->checkError();
            last_layer_defered_params.shared_expert_output = shared_expert_output;
        }
    }

    // deal with last layer
    auto mb0_moe_insertion_ret                      = device_->stealMoEInsertionRet();
    last_layer_defered_params_vec[0].combine_output = nullopt;
    last_layer_defered_params_vec[0].combine_output = mb0_moe_insertion_ret->combine_output;

    // last layer last micro batch
    device_->computeInsertedMoE();
    device_->checkError();
    auto moe_insertion_ret = device_->stealMoEInsertionRet();
    moe_insertion_ret->combine_output.comm_barrier_hook->hook_sync();
    last_layer_defered_params_vec.back().combine_output = nullopt;
    last_layer_defered_params_vec.back().combine_output = move(moe_insertion_ret->combine_output);

    for (size_t micro_batch_idx = 0; micro_batch_idx < micro_batch_layer_inputs.size(); ++micro_batch_idx) {
        auto& layer_input               = micro_batch_layer_inputs[micro_batch_idx];
        auto& last_layer_defered_params = last_layer_defered_params_vec[micro_batch_idx];

        auto output = device_->gatherCombineOutput(last_layer_defered_params.combine_output.value()).hidden_states;

        auto ffn_layernorm_output =
            device_->layernorm({output,
                                nullptr,
                                rtp_llm::mayGetRef(last_layer_defered_params.post_ffn_layernorm_weights),
                                rtp_llm::mayGetRef(last_layer_defered_params.residual),
                                rtp_llm::mayGetRef(last_layer_defered_params.shared_expert_output),
                                nullopt,
                                1.0f,
                                description_.layernorm_eps,
                                true,
                                description_.post_layernorm,
                                description_.norm_type,
                                QScheme::NoQuantize});
        device_->checkError();
        layer_input.hidden = move(ffn_layernorm_output.output);
        printBufferData(*layer_input.hidden, "mb_" + to_string(micro_batch_idx) + "_final_hidden");
    }

    return micro_batch_layer_inputs;
}

GptLayerOutputs GptModel::forwardMicroBatchedLayers(const GptLayerInputs&   layer_inputs,
                                                    const GptModelInputs&   inputs,
                                                    std::vector<BufferPtr>& eagle3_selected_hidden) {
    std::vector<GptLayerInputs> micro_batch_layer_inputs;
    for (auto& micro_batch_input : layer_inputs.micro_batch_inputs) {
        micro_batch_layer_inputs.push_back({micro_batch_input.hidden,
                                            micro_batch_input.pre_decoder_residual,
                                            micro_batch_input.attention_common_inputs,
                                            layer_inputs.dtype});
    }
    if (device_props_.enable_layer_micro_batch == MicroBatchType::DS_PREFILL) {
        micro_batch_layer_inputs = forwardPrefillMicroBatchedLayers(micro_batch_layer_inputs, eagle3_selected_hidden);
    } else if (device_props_.enable_layer_micro_batch == MicroBatchType::DS_DECODE) {
        micro_batch_layer_inputs = forwardDecodeMicroBatchedLayers(micro_batch_layer_inputs, eagle3_selected_hidden);
    } else {
        RUNTIME_ASSERT_OP_ARG(
            false, "micro batch type %d is not supported", int(device_props_.enable_layer_micro_batch));
    }
    device_->checkError();

    const auto& hidden              = layer_inputs.hidden;
    size_t      copy_from_token_idx = 0;
    if (!layer_inputs.micro_batch_inputs[1].fake) {
        for (size_t i = 0; i < micro_batch_layer_inputs.size(); ++i) {
            const auto& micro_batch_hidden    = micro_batch_layer_inputs[i].hidden;
            const auto  micro_batch_token_num = micro_batch_hidden->shape()[0];
            const auto  target_hidden         = hidden->slice(copy_from_token_idx, micro_batch_token_num);
            device_->copy({*target_hidden, *micro_batch_hidden});
            copy_from_token_idx += micro_batch_token_num;
        }
        printBufferData(*hidden, "micor_batched_final_hidden");
    } else {
        device_->copy({*hidden, *(micro_batch_layer_inputs[0].hidden)});
        printBufferData(*hidden, "non-micor_batched_final_hidden");
    }

    return {hidden, nullptr};
}

GptLayerOutputs GptModel::forwardGptLayer(GptLayerInputs                          inputs,
                                          const int32_t                           layer_id,
                                          const rtp_llm::lora::LoraModelInputPtr& lora_model_input) {
    DevicePerfWrapper wrapper(device_, "forwardGptLayer_token_num_%d", inputs.hidden ? inputs.hidden->shape()[0] : 0);
    auto              pre_decoder_residual   = inputs.pre_decoder_residual;
    auto              attention_block_output = forwardAttentionBlock(inputs, layer_id, lora_model_input);

    auto        hidden    = move(attention_block_output.hidden);
    auto        residual  = move(attention_block_output.residual);
    auto        residual2 = move(attention_block_output.residual2);
    const auto& layer     = weights_.layers[layer_id];
    BufferPtr   moe_gating;

    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_ffn_input");
    bool   enable_sp          = inputs.enable_sp;
    size_t rank_pad_token_num = enable_sp ? inputs.pad_token_num / device_props_.tp_size : hidden->shape()[0];
    auto   ffn_output_buf =
        device_->allocateBuffer({inputs.dtype, {rank_pad_token_num, hidden->shape()[1]}}, {"ffn_out_buf"});
    if (!enable_sp) {
        // Note: for custom all reduce, prepareAllReduce will replace the original attn_out_buf with
        // a new custom_ar_comm buffer. Here we must make sure that attn_out_buf is not released or replaced by
        // other buffer before the actual allreduce operations. Otherwise, it will raise an error in custom ar.
        ffn_output_buf = device_->prepareAllReduce({std::move(ffn_output_buf), ReduceOp::Sum}).buffer;
    }
    if (device_->initParams().profile_debug_logging_config.check_nan) {
        (void)device_->checkNAN(*hidden);
    }
    auto ffn_layer_params =
        FfnLayerParams({*hidden,
                        description_.ffn_conf,
                        layer.ffn_weights,
                        device_props_.ffn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt,
                        description_.act_qscheme,
                        description_.compute_type,
                        std::move(ffn_output_buf),
                        enable_sp,
                        inputs.need_moe_gating});

    // expert stats
    prepareExpertStats(layer_id, ffn_layer_params);

    if (lora_model_input) {
        ffn_layer_params.lora_input = lora_model_input->getFfnLayerLoraInput(layer_id);
    }
    auto ffn_output = device_->ffnLayer(ffn_layer_params);
    device_->checkError();
    hidden = ffn_output.hidden_states;
    if (device_->initParams().profile_debug_logging_config.check_nan) {
        (void)device_->checkNAN(*hidden);
    }
    if (inputs.need_moe_gating) {
        moe_gating = std::move(ffn_output.moe_gating);
    }
    if (device_props_.ffn_tp_size > 1 && (!layer.ffn_weights.moe_gating_weight || device_props_.use_all_gather)
        && !enable_sp) {
        {
            // Note: for custom all reduce, allReduce will allocate a new buffer and replace the original attn_hidden
            // with it
            auto wrapper = DevicePerfWrapper(device_, "post_ffn_all_reduce, sizeBytes=%ld", (long)hidden->sizeBytes());
            hidden       = device_->allReduce({std::move(hidden), ReduceOp::Sum, false, ParallelMode::FFN_TP}).buffer;
        }
    }
    device_->checkError();
    if (residual_scale_) {
        hidden = device_->multiply({*residual_scale_, *hidden});
    }
    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_ffn_output");

    // TODO: maybe move this layernorm to ffn layer
    auto ffn_act_qscheme =
        ((layer_id == layer_num_ - 1) || (!layer.post_ffn_layernorm)) ? QScheme::NoQuantize : description_.act_qscheme;
    auto ffn_layernorm_output = device_->layernorm(
        LayernormParams(hidden,
                        pre_decoder_residual,
                        rtp_llm::mayGetRef(layer.post_ffn_layernorm),
                        device_props_.ffn_fuse_add_residual ? nullopt : (OptionalConstBufferRef)*residual,
                        (residual2 == nullptr) ? nullopt : (OptionalConstBufferRef)*residual2,
                        rtp_llm::mayGetRef(WEIGHT_MAY_GET_BIAS(layer.ffn_weights.down_weight)),
                        1.0f,
                        description_.layernorm_eps,
                        true,
                        description_.post_layernorm,
                        description_.norm_type,
                        ffn_act_qscheme));
    if (layer.post_ffn_layernorm && ffn_act_qscheme == QScheme::Qint8PerTensor
        && !(ffn_layernorm_output.output->isQBuffer())) {
        auto norm_weight             = layer.post_ffn_layernorm;
        auto static_scale_reciprocal = norm_weight->static_scale_reciprocal;
        ffn_layernorm_output.output =
            device_->quantize({*ffn_layernorm_output.output,
                               DataType::TYPE_INT8,
                               1,
                               description_.act_qscheme,
                               nullopt,
                               nullopt,
                               nullopt,
                               static_scale_reciprocal ? (OptionalConstBufferRef)*static_scale_reciprocal : nullopt});
    }
    device_->checkError();
    hidden = std::move(ffn_layernorm_output.output);
    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_final_hidden");

    return {hidden, pre_decoder_residual, moe_gating};
}

AttentionBlockOutputs GptModel::forwardAttentionBlock(const GptLayerInputs&                   inputs,
                                                      const int32_t                           layer_id,
                                                      const rtp_llm::lora::LoraModelInputPtr& lora_model_input,
                                                      const LastLayerDeferedParams&           last_layer_defered_params,
                                                      bool                                    capture_last_hidden) {
    auto              hidden                  = inputs.hidden;
    auto              pre_decoder_residual    = inputs.pre_decoder_residual;
    auto              attention_common_inputs = move(inputs.attention_common_inputs);
    DevicePerfWrapper wrapper(
        device_, "attention_block_layer_" + std::to_string(layer_id) + "_bs_" + std::to_string(hidden->shape()[0]));

    if (last_layer_defered_params.combine_output) {
        if (last_layer_defered_params.combine_output.value().comm_barrier_hook) {
            last_layer_defered_params.combine_output.value().comm_barrier_hook->hook_sync();
        }

        printBufferData(*(last_layer_defered_params.combine_output.value().all_output),
                        "layer_" + to_string(layer_id - 1) + "_combine_output_defered");
        hidden = device_->gatherCombineOutput(last_layer_defered_params.combine_output.value()).hidden_states;
    }
    device_->checkError();

    if (attention_common_inputs.cache_store_inputs.has_value()) {
        attention_common_inputs.cache_store_inputs.value().layer_id = layer_id;
    }
    const auto& layer     = weights_.layers[layer_id];
    bool        enable_sp = inputs.enable_sp;

    // here hidden->dtype maybe int8, so use dytpe of embedding lookup result instead
    size_t    rank_pad_token_num   = enable_sp ? inputs.pad_token_num / device_props_.tp_size : hidden->shape()[0];
    size_t    attn_out_hidden_size = dynamic_cast<Eagle3Model*>(this) ? hidden->shape()[1] / 2 : hidden->shape()[1];
    BufferPtr attn_out_buf =
        device_->allocateBuffer({inputs.dtype, {rank_pad_token_num, attn_out_hidden_size}}, {"attn_out_buf"});
    if (!enable_sp) {
        // Note: for custom all reduce, prepareAllReduce will replace the original attn_out_buf with
        // a new custom_ar_comm buffer. Here we must make sure that attn_out_buf is not released or replaced by
        // other buffer before the actual allreduce operations. Otherwise, it will raise an error in custom ar.
        attn_out_buf = device_->prepareAllReduce({std::move(attn_out_buf), ReduceOp::Sum}).buffer;
    }
    auto residual = pre_decoder_residual ? pre_decoder_residual : hidden;
    if (device_->initParams().use_deepep_moe) {
        // avoid attention o gemm copy
        attn_out_buf.reset();
    }
    printBufferData(*residual, "in residual");
    BufferPtr residual2         = nullptr;
    BufferPtr hidden_to_slice   = nullptr;  // for sp and overlap comm type 2
    BufferPtr last_layer_hidden = nullptr;
    if (layer.pre_layernorm) {
        if (inputs.residual) {
            // this branch for qwen3 eagle3
            residual = device_->clone({*inputs.residual, AllocationType::DEVICE, {"residual"}});
            const_cast<GptLayerInputs&>(inputs).residual = nullptr;
        } else {
            if (last_layer_defered_params.residual) {
                residual = device_->allocateBufferLike(*hidden, AllocationType::DEVICE, {"residual"});
            } else {
                residual = device_->clone({*hidden, AllocationType::DEVICE, {"residual"}});
            }
        }
        int    m_split           = device_props_.m_split;
        size_t overlap_comm_type = device_props_.overlap_comm_type;
        auto   pre_layernorm_output =
            device_->layernorm(LayernormParams(hidden,
                                               residual,
                                               *layer.pre_layernorm,
                                               rtp_llm::mayGetRef(last_layer_defered_params.residual),
                                               rtp_llm::mayGetRef(last_layer_defered_params.shared_expert_output),
                                               std::nullopt,
                                               0.f,
                                               description_.layernorm_eps,
                                               false,
                                               false,
                                               description_.norm_type,
                                               description_.act_qscheme,
                                               layer_id > 0 ? true : false,
                                               false));

        if (capture_last_hidden) {
            last_layer_hidden = residual;
        }

        if (enable_sp && layer_id == 0) {
            if (overlap_comm_type == 1 && m_split > 0) {
                vector<int> selected_indices;
                selected_indices.reserve(rank_pad_token_num);
                size_t m       = inputs.pad_token_num;
                size_t m_chunk = m / m_split;
                if (m > 128) {
                    m_chunk = (m / m_split + 127) & ~127;
                }
                size_t tp_rank = device_props_.tp_rank;
                size_t round   = m_chunk / device_props_.tp_size;

                size_t offset = tp_rank * round;
                for (size_t i = 0; i < rank_pad_token_num; i++) {
                    selected_indices.push_back((i / round) * m_chunk + i % round + offset);
                }
                // printBufferData(*vector2Buffer(selected_indices), "selected_indices");
                residual = device_->select({*residual, *device_->clone({*vector2Buffer(selected_indices)})});
            } else {
                hidden_to_slice = residual;
                residual        = residual->slice(rank_pad_token_num * device_props_.tp_rank, rank_pad_token_num);
            }
        }
        if (description_.act_qscheme == QScheme::Qint8PerTensor && !(pre_layernorm_output.output->isQBuffer())) {
            auto norm_weight             = layer.pre_layernorm;
            auto static_scale_reciprocal = norm_weight->static_scale_reciprocal;
            pre_layernorm_output.output  = device_->quantize(
                {*pre_layernorm_output.output,
                  DataType::TYPE_INT8,
                  1,
                  description_.act_qscheme,
                  nullopt,
                  nullopt,
                  nullopt,
                 static_scale_reciprocal ? (OptionalConstBufferRef)*static_scale_reciprocal : nullopt});
        }
        hidden = std::move(pre_layernorm_output.output);
    } else if (last_layer_defered_params.residual || last_layer_defered_params.shared_expert_output) {
        // NOTE(wangyin): this branch is not used for now, might be errornous
        residual                       = device_->clone({*hidden, AllocationType::DEVICE, {"residual"}});
        auto prev_ffn_layernorm_output = device_->layernorm({
            hidden,
            nullptr,
            std::nullopt,  // post_ffn_layernorm_weights
            rtp_llm::mayGetRef(last_layer_defered_params.residual),
            rtp_llm::mayGetRef(last_layer_defered_params.shared_expert_output),
        });
        hidden                         = std::move(prev_ffn_layernorm_output.output);
    }

    printBufferData(*hidden, "pre layer norm hidden");

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
    if (device_->initParams().profile_debug_logging_config.check_nan) {
        (void)device_->checkNAN(*hidden);
    }
    auto attn_params =
        AttentionLayerParams({layer_id,
                              *hidden,
                              move(attn_out_buf),
                              description_.attention_conf,
                              layer.self_attention_weights,
                              attention_common_inputs,
                              device_props_.attn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt,
                              {description_.layernorm_eps, description_.norm_type},
                              description_.act_qscheme,
                              description_.compute_type,
                              enable_sp,
                              inputs.pad_token_num});
    if (description_.attention_conf.use_mla && device_->mla_ops_type != rtp_llm::MlaOpsType::MHA) {
        attn_output = device_->mlaAttentionLayer(attn_params);
    } else {
        attn_output = device_->attentionLayer(attn_params);
    }
    device_->checkError();

    auto attn_hidden = std::move(attn_output.hidden_states);
    if (device_->initParams().profile_debug_logging_config.check_nan) {
        (void)device_->checkNAN(*attn_hidden);
    }
    if (device_props_.tp_size > 1 && !enable_sp) {
        // Note: for custom all reduce, allReduce will allocate a new buffer and replace the original attn_hidden with
        // it
        auto wrapper = DevicePerfWrapper(device_, "allReduce, sizeBytes=%ld", (long)attn_hidden->sizeBytes());
        attn_hidden  = device_->allReduce({std::move(attn_hidden), ReduceOp::Sum}).buffer;
    }
    if (residual_scale_) {
        attn_hidden = device_->multiply({*residual_scale_, *attn_hidden});
    }
    printBufferData(*attn_hidden, "layer_" + to_string(layer_id) + "_attn_output");

    if (layer.post_layernorm) {
        // attn_hidden = attn_hidden + residual
        // hidden = layernorm(attn_hidden)
        printBufferData(*residual, "before post layernorm residual");
        auto post_layernorm_params =
            LayernormParams(attn_hidden,
                            attn_hidden,
                            rtp_llm::mayGetRef(layer.post_layernorm),
                            device_props_.attn_fuse_add_residual ? nullopt : (OptionalConstBufferRef)*residual,
                            nullopt,
                            rtp_llm::mayGetRef(layer.self_attention_weights.output_weight->bias),
                            0.f,
                            description_.layernorm_eps,
                            false,
                            description_.post_layernorm,
                            description_.norm_type,
                            description_.act_qscheme,
                            false,
                            true);

        auto post_layernorm_output = device_->layernorm(post_layernorm_params);
        if (description_.act_qscheme == QScheme::Qint8PerTensor && !(post_layernorm_output.output->isQBuffer())) {
            auto norm_weight             = layer.post_layernorm;
            auto static_scale_reciprocal = norm_weight->static_scale_reciprocal;
            post_layernorm_output.output = device_->quantize(
                {*post_layernorm_output.output,
                 DataType::TYPE_INT8,
                 1,
                 description_.act_qscheme,
                 nullopt,
                 nullopt,
                 nullopt,
                 static_scale_reciprocal ? (OptionalConstBufferRef)*static_scale_reciprocal : nullopt});
        }
        device_->checkError();
        hidden      = std::move(post_layernorm_output.output);
        attn_hidden = std::move(post_layernorm_output.before_norm_output);
        residual    = attn_hidden;
        printBufferData(*residual, "after post layernorm residual");
    } else {
        residual2 = attn_hidden;
    }

    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_ffn_input");
    return {hidden, residual, residual2, last_layer_hidden};
}

EpFfnInputs GptModel::forwardAttentionAndMoeGate(const GptLayerInputs&   inputs,
                                                 LastLayerDeferedParams& last_layer_defered_params,
                                                 const int32_t           layer_id,
                                                 const size_t            micro_batch_idx,
                                                 bool                    capture_last_hidden) {
    // TODO(zhangjianning.zjn) support returning moe_gating when need_moe_gating is true
    auto        hidden               = inputs.hidden;
    auto        pre_decoder_residual = inputs.pre_decoder_residual;
    const auto& layer                = weights_.layers[layer_id];

    DevicePerfWrapper wrapper(
        device_, "mb_forwardGptLayer_" + std::to_string(layer_id) + "_bs_" + std::to_string(hidden->shape()[0]));

    auto attention_block_output =
        forwardAttentionBlock(inputs, layer_id, nullptr, last_layer_defered_params, capture_last_hidden);
    hidden         = move(attention_block_output.hidden);
    auto residual  = move(attention_block_output.residual);
    auto residual2 = move(attention_block_output.residual2);

    printBufferData(*hidden, "layer_" + to_string(layer_id) + "_ffn_input");
    auto ffn_output_buf = device_->allocateBuffer({inputs.dtype, hidden->shape()}, {"ffn_out_buf"});
    auto ffn_layer_params =
        FfnLayerParams({*hidden,
                        description_.ffn_conf,
                        layer.ffn_weights,
                        device_props_.ffn_fuse_add_residual ? (OptionalConstBufferRef)*residual : nullopt,
                        description_.act_qscheme,
                        description_.compute_type,
                        std::move(ffn_output_buf)});

    prepareExpertStats(layer_id, ffn_layer_params);

    MoeGateSelectOutput gate_output = device_->moeGateSelect(ffn_layer_params);
    device_->checkError();
    RTP_LLM_LOG_DEBUG("call layer %ld micro batch ep dispatch batch size = %ld", layer_id, hidden->shape()[0]);

    BufferPtr shared_expert_output = nullptr;

    printBufferData(ffn_layer_params.input, "layer_" + to_string(layer_id) + "_ep_dispatch_input");
    printBufferData(*gate_output.expert_ids, "layer_" + to_string(layer_id) + "_expert_ids");
    if (gate_output.expert_scales) {
        printBufferData(*gate_output.expert_scales, "layer_" + to_string(layer_id) + "_expert_scales");
    }

    BufferPtr quantized_hidden = nullptr;
    if (description_.act_qscheme == QScheme::Qfp8PerTokenBlock) {
        quantized_hidden = device_->quantize({*hidden, DataType::TYPE_QFP8_E4M3, 1, description_.act_qscheme});
    }

    return {hidden,
            quantized_hidden,
            residual,
            shared_expert_output,
            move(ffn_layer_params),
            move(gate_output),
            nullptr,
            attention_block_output.last_layer_hidden};
}

GptLayerOutputs GptModel::forwardMoeFfn(const GptLayerOutputs& inputs, const int32_t layer_id) {
    return inputs;
}

GptModelOutputs GptModel::forwardPostLayers(rtp_llm::BufferPtr       input,
                                            const bool               has_context_request,
                                            const bool               need_all_logits,
                                            const rtp_llm::BufferPtr lm_output_indexes,
                                            bool                     enable_sp,
                                            size_t                   token_num,
                                            const GptModelInputs&    inputs,
                                            rtp_llm::BufferPtr       merged_eagle3_hidden,
                                            bool                     skip_final_layernorm) {
    DevicePerfWrapper wrapper(device_, "forwardPostLayers");
    BufferPtr         all_gather_output = nullptr;
    if (enable_sp && device_props_.tp_size > 1) {
        all_gather_output = device_->allocateBuffer(
            {input->type(), {input->shape()[0] * device_props_.tp_size, input->shape()[1]}}, {"all_gather_output"});
        size_t m                 = all_gather_output->shape()[0];
        int    m_split           = device_props_.m_split;
        size_t overlap_comm_type = device_props_.overlap_comm_type;
        if (overlap_comm_type == 1 && m_split > 0) {
            size_t token_idx    = 0;
            size_t ag_token_idx = 0;
            size_t m_chunk      = m / m_split;
            if (m > 128) {
                m_chunk = (m / m_split + 127) & ~127;
            }
            while (token_idx < m) {
                const auto micro_batch_tokens      = std::min(m - token_idx, m_chunk);
                const auto ag_micro_batch_tokens   = micro_batch_tokens / device_props_.tp_size;
                auto       micro_batch_recv_buffer = all_gather_output->slice(token_idx, micro_batch_tokens);
                auto       micro_ag_send_buffer    = input->slice(ag_token_idx, ag_micro_batch_tokens);
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
    if (weights_.final_layernorm && !skip_final_layernorm) {
        auto final_layernorm = device_->layernorm(LayernormParams(hidden,
                                                                  nullptr,
                                                                  rtp_llm::mayGetRef(weights_.final_layernorm),
                                                                  nullopt,
                                                                  nullopt,
                                                                  nullopt,
                                                                  0.f,
                                                                  description_.layernorm_eps,
                                                                  true,
                                                                  false,
                                                                  description_.norm_type));
        hidden               = std::move(final_layernorm.output);
    }
    printBufferData(*hidden, "final_hidden");

    const auto& lm_head = weights_.lm_head;

    if (lm_head) {
        // gen last token hidden
        printBufferData(*lm_output_indexes, "lm_output_indexes");
        auto last_hidden = has_context_request && !need_all_logits ?
                               device_->select({*hidden, *device_->clone({*lm_output_indexes})}) :
                               hidden;

        printBufferData(*last_hidden, "last_hidden");

        auto logits = device_->gemm(GemmParams(*last_hidden,
                                               *(lm_head->kernel),
                                               nullopt,
                                               nullptr,
                                               rtp_llm::DataType::TYPE_FP32,
                                               rtp_llm::DataType::TYPE_FP32,
                                               TransposeOperation::NONE,
                                               TransposeOperation::TRANSPOSE));
        printBufferData(*logits, "logits");
        if (device_props_.tp_size > 1) {
            logits = tpSyncEmbeddingOrLogits(logits);
        }
        if (device_->initParams().profile_debug_logging_config.check_nan) {
            (void)device_->checkNAN(*last_hidden);
            (void)device_->checkNAN(*logits);
        }
        // TODO(xinfei.sxf) calculate softmax_result
        rtp_llm::BufferPtr softmax_result;
        // logits is too big, tmp not print default
        // printBufferData(*logits, "logits");
        if (need_all_logits) {
            auto last_logits = device_->select({*logits, *device_->clone({*lm_output_indexes})});
            return {std::move(last_logits),
                    std::move(last_hidden),
                    std::move(hidden),
                    std::move(logits),
                    std::move(softmax_result)};
        }

        hidden = merged_eagle3_hidden ? merged_eagle3_hidden : hidden;
        return {std::move(logits), std::move(last_hidden), std::move(hidden), nullptr, std::move(softmax_result)};
    } else {
        return {nullptr, nullptr, std::move(hidden)};
    }
}

GptModelOutputs GptModel::forward(const GptModelInputs& inputs) {
    DevicePerfWrapper wrapper(device_, "forward [tp=%d, dp=%d]", device_props_.tp_size, device_props_.dp_size);
    cleanExpertStats();
    auto layer_inputs = forwardPreLayers(inputs);

    GptLayerOutputs        layer_outputs;
    std::vector<BufferPtr> eagle3_selected_hidden;
    std::vector<BufferPtr> moe_gating;
    if (inputs.need_moe_gating) {
        moe_gating.reserve(layer_num_);
    }
    if (int(device_props_.enable_layer_micro_batch) && layer_inputs.micro_batch_inputs.size() > 0) {
        // TODO(zhangjianning.zjn) support return moe_gating in micro batch
        layer_outputs = forwardMicroBatchedLayers(layer_inputs, inputs, eagle3_selected_hidden);
    } else {
        layer_inputs.need_moe_gating = inputs.need_moe_gating;
        for (int32_t i = 0; i < layer_num_; ++i) {
            layer_outputs                     = forwardGptLayer(layer_inputs, i, inputs.lora_model_input);
            layer_inputs.hidden               = layer_outputs.hidden;
            layer_inputs.pre_decoder_residual = layer_outputs.pre_decoder_residual;
            if (inputs.need_moe_gating) {
                moe_gating.push_back(std::move(layer_outputs.moe_gating));
            }
            if (dynamic_cast<Eagle3Model*>(this) == nullptr && device_props_.is_eagle3
                && device_props_.eagle3_selected_layer.count(i) > 0) {
                eagle3_selected_hidden.push_back(device_->clone({*layer_inputs.hidden, AllocationType::DEVICE}));
            }
        }
    }

    BufferPtr merged_eagle3_hidden = mergeEagle3HiddenState(layer_inputs, eagle3_selected_hidden);
    if (device_->initParams().profile_debug_logging_config.check_nan) {
        (void)device_->checkNAN(*layer_outputs.hidden);
    }
    auto outputs = forwardPostLayers(layer_outputs.hidden,
                                     inputs.input_lengths->shape()[0] != inputs.sequence_lengths->shape()[0],
                                     inputs.need_all_logits,
                                     inputs.lm_output_indexes,
                                     layer_inputs.enable_sp,
                                     layer_inputs.token_num,
                                     inputs,
                                     merged_eagle3_hidden);

    // make sure cpu buffers out lives gpu exec
    outputs.captured_values = make_shared<GptLayerInputs>(layer_inputs);
    outputs.moe_gating      = std::move(moe_gating);
    return outputs;
}

void GptModel::prepareExpertStats(const size_t layer_id, rtp_llm::FfnLayerParams& ffn_layer_params) {
    OptionalExpertStats layer_expert_stats = nullopt;
    if (overall_expert_stats_.log_exp_num != 0) {
        layer_expert_stats = ExpertStats({layer_id,
                                          overall_expert_stats_.ep_size,
                                          overall_expert_stats_.log_exp_num,
                                          overall_expert_stats_.phy_exp_num,
                                          overall_expert_stats_.stats_buf});
    }
    ffn_layer_params.expert_stats = layer_expert_stats;
}

void GptModel::cleanExpertStats() {
    if (overall_expert_stats_.log_exp_num != 0) {
        device_->cleanMoeExpertStates(overall_expert_stats_);
    }
}

bool GptModel::containMoeLayer() {
    for (int32_t i = 0; i < layer_num_; ++i) {
        if (weights_.layers[i].ffn_weights.moe_gate_weight != nullptr) {
            return true;
        }
    }
    return false;
}

BufferPtr GptModel::mergeEagle3HiddenState(const GptLayerInputs&   layer_inputs,
                                           std::vector<BufferPtr>& eagle3_selected_hidden) {
    if (eagle3_selected_hidden.empty()) {
        return nullptr;
    }

    std::vector<torch::Tensor> eagle3_selected_hidden_tensor;
    for (int i = 0; i < eagle3_selected_hidden.size(); i++) {
        eagle3_selected_hidden_tensor.push_back(Buffer2torchTensor(eagle3_selected_hidden[i], false));
        // printBufferData(*eagle3_selected_hidden[i], "eagle3_selected_hidden_tensor");
    }

    size_t        micro_batch_size = layer_inputs.micro_batch_inputs.size();
    torch::Tensor merged_hidden_states_tensor;
    if (micro_batch_size == 0) {
        merged_hidden_states_tensor = torch::cat(eagle3_selected_hidden_tensor, -1);
    } else {
        std::vector<std::vector<torch::Tensor>> grouped_hidden_states(micro_batch_size);
        std::vector<torch::Tensor>              merged_row_hidden_states(micro_batch_size);
        for (size_t i = 0; i < eagle3_selected_hidden.size(); i++) {
            grouped_hidden_states[i % micro_batch_size].push_back(Buffer2torchTensor(eagle3_selected_hidden[i], false));
        }
        for (size_t i = 0; i < merged_row_hidden_states.size(); i++) {
            merged_row_hidden_states[i] = torch::cat(grouped_hidden_states[i], -1);
        }
        merged_hidden_states_tensor = torch::cat(merged_row_hidden_states, 0);
    }

    return device_->clone({*torchTensor2Buffer(merged_hidden_states_tensor), AllocationType::DEVICE});
}

void tpSyncModelInputs(GptModelInputs& inputs, rtp_llm::DeviceBase* device) {
    if (device->getDeviceProperties().tp_size <= 1) {
        return;
    }
    const size_t shape_hints_size = GptModelInputIndex::gptModelInputLength;
    auto         shape_hints =
        device->allocateBuffer({rtp_llm::DataType::TYPE_INT32, {shape_hints_size}, rtp_llm::AllocationType::HOST});
    auto shape_hints_ptr                              = shape_hints->data<int32_t>();
    shape_hints_ptr[GptModelInputIndex::comboTokens]  = inputs.combo_tokens.get() ? inputs.combo_tokens->size() : 0;
    shape_hints_ptr[GptModelInputIndex::inputLengths] = inputs.input_lengths.get() ? inputs.input_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::sequenceLengths] =
        inputs.sequence_lengths.get() ? inputs.sequence_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::prefixLengths] =
        inputs.prefix_lengths.get() ? inputs.prefix_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch] =
        inputs.kv_cache_block_id.get() ? inputs.kv_cache_block_id->shape()[1] : 0;
    shape_hints_ptr[GptModelInputIndex::kvCacheUpdateCopyNum] =
        inputs.kv_cache_update_mapping.get() ? inputs.kv_cache_update_mapping->shape()[0] : 0;
    shape_hints_ptr[GptModelInputIndex::lmOutputIndexes] =
        inputs.lm_output_indexes.get() ? inputs.lm_output_indexes->size() : 0;
    shape_hints_ptr[GptModelInputIndex::lmOutputLengthes] =
        inputs.lm_output_lengths.get() ? inputs.lm_output_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::comboPositionIds] =
        inputs.combo_position_ids.get() ? inputs.combo_position_ids->size() : 0;
    shape_hints_ptr[GptModelInputIndex::loraIds] = inputs.lora_ids.get() ? inputs.lora_ids->size() : 0;
    shape_hints_ptr[GptModelInputIndex::loraInputLengths] =
        inputs.lora_input_lengths.get() ? inputs.lora_input_lengths->size() : 0;
    shape_hints_ptr[GptModelInputIndex::textTokensMask] =
        inputs.text_tokens_mask.get() ? inputs.text_tokens_mask->size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs] =
        inputs.mm_features_locs.get() ? inputs.mm_features_locs->size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] =
        inputs.multimodal_features.has_value() ? inputs.multimodal_features.value().size() : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesSize] =
        shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] ? inputs.multimodal_features.value()[0]->shape()[1] : 0;
    shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype] =
        shape_hints_ptr[GptModelInputIndex::mmFeaturesNum] ?
            (std::uint8_t)inputs.multimodal_features.value()[0]->type() :
            0;
    shape_hints_ptr[GptModelInputIndex::needAllLogits] = inputs.need_all_logits;
    shape_hints_ptr[GptModelInputIndex::mtpHiddenStates] =
        inputs.last_hidden_states.get() ? inputs.last_hidden_states->size() : 0;
    shape_hints_ptr[GptModelInputIndex::mtpHiddenStatesDtype] =
        shape_hints_ptr[GptModelInputIndex::mtpHiddenStates] ? (std::uint8_t)inputs.last_hidden_states->type() : 0;
    shape_hints_ptr[GptModelInputIndex::skipRun] = inputs.skip_run;
    device->broadcast({{shape_hints}, 0});
    device->syncCommunication(false);
    device->syncAndCheck();

    // multimodal features shape broadcast
    rtp_llm::BufferPtr mm_features_shape;
    int32_t*           mm_features_shape_ptr = nullptr;
    inputs.need_all_logits                   = shape_hints_ptr[GptModelInputIndex::needAllLogits];
    inputs.skip_run                          = shape_hints_ptr[GptModelInputIndex::skipRun];
    if (inputs.skip_run) {
        return;
    }
    const size_t mm_features_num = shape_hints_ptr[GptModelInputIndex::mmFeaturesNum];
    if (mm_features_num) {
        mm_features_shape     = device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                        {(size_t)shape_hints_ptr[GptModelInputIndex::mmFeaturesNum]},
                                                        rtp_llm::AllocationType::HOST});
        mm_features_shape_ptr = mm_features_shape->data<int32_t>();
        for (auto i = 0; i < mm_features_num; ++i) {
            mm_features_shape_ptr[i] =
                inputs.multimodal_features.has_value() ? inputs.multimodal_features.value()[i]->shape()[0] : 0;
        }
        device->broadcast({{mm_features_shape}, 0});
        device->syncCommunication(false);
        device->syncAndCheck();
    }

    auto max_blocks              = (size_t)shape_hints_ptr[GptModelInputIndex::maxBlocksPerBatch];
    auto combo_position_ids_size = shape_hints_ptr[GptModelInputIndex::comboPositionIds];
    auto text_tokens_mask_size   = shape_hints_ptr[GptModelInputIndex::textTokensMask];
    auto mm_features_locs_size   = shape_hints_ptr[GptModelInputIndex::mmFeaturesLocs];
    auto hidden_states_size      = shape_hints_ptr[GptModelInputIndex::mtpHiddenStates];

    if (device->getDeviceProperties().tp_rank) {
        auto context_batch_size = (size_t)shape_hints_ptr[GptModelInputIndex::prefixLengths];

        inputs.combo_tokens  = device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                       {(size_t)shape_hints_ptr[GptModelInputIndex::comboTokens]},
                                                       rtp_llm::AllocationType::HOST});
        inputs.input_lengths = device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                       {(size_t)shape_hints_ptr[GptModelInputIndex::inputLengths]},
                                                       rtp_llm::AllocationType::HOST});
        inputs.sequence_lengths =
            device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                    {(size_t)shape_hints_ptr[GptModelInputIndex::sequenceLengths]},
                                    rtp_llm::AllocationType::HOST});
        inputs.prefix_lengths = device->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {context_batch_size}, rtp_llm::AllocationType::HOST});
        if (max_blocks != 0) {
            inputs.kv_cache_block_id =
                device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                        {(size_t)shape_hints_ptr[GptModelInputIndex::inputLengths], max_blocks},
                                        rtp_llm::AllocationType::HOST});
            if (inputs.pd_separation) {
                inputs.cache_keys = device->allocateBuffer(
                    {rtp_llm::DataType::TYPE_INT64, {context_batch_size, max_blocks}, rtp_llm::AllocationType::HOST});
            }
            inputs.kv_cache_update_mapping =
                device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                        {(size_t)shape_hints_ptr[GptModelInputIndex::kvCacheUpdateCopyNum], 2},
                                        rtp_llm::AllocationType::HOST});
        }
        inputs.request_id = device->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT64, {context_batch_size}, rtp_llm::AllocationType::HOST});
        inputs.request_pd_separation =
            device->allocateBuffer({rtp_llm::DataType::TYPE_BOOL, {context_batch_size}, rtp_llm::AllocationType::HOST});
        inputs.lm_output_indexes =
            device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                    {(size_t)shape_hints_ptr[GptModelInputIndex::lmOutputIndexes]},
                                    rtp_llm::AllocationType::HOST});
        inputs.lm_output_lengths =
            device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                    {(size_t)shape_hints_ptr[GptModelInputIndex::lmOutputLengthes]},
                                    rtp_llm::AllocationType::HOST});
        if (combo_position_ids_size) {
            inputs.combo_position_ids = device->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {(size_t)combo_position_ids_size}, rtp_llm::AllocationType::HOST});
        }
        if (shape_hints_ptr[GptModelInputIndex::loraIds]) {
            inputs.lora_ids = device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                                      {(size_t)shape_hints_ptr[GptModelInputIndex::loraIds]},
                                                      rtp_llm::AllocationType::HOST});
        }
        if (shape_hints_ptr[GptModelInputIndex::loraInputLengths]) {
            inputs.lora_input_lengths =
                device->allocateBuffer({rtp_llm::DataType::TYPE_INT32,
                                        {(size_t)shape_hints_ptr[GptModelInputIndex::loraInputLengths]},
                                        rtp_llm::AllocationType::HOST});
        }
        if (shape_hints_ptr[GptModelInputIndex::mtpHiddenStates]) {
            auto hidden_states_dim0 = (size_t)shape_hints_ptr[GptModelInputIndex::comboTokens];
            auto hidden_states_dim1 = (size_t)hidden_states_size / hidden_states_dim0;
            RTP_LLM_CHECK(hidden_states_size % hidden_states_dim0 == 0);
            inputs.last_hidden_states =
                device->allocateBuffer({(rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mtpHiddenStatesDtype],
                                        {hidden_states_dim0, hidden_states_dim1},
                                        rtp_llm::AllocationType::DEVICE});
        }
        if (text_tokens_mask_size) {
            inputs.text_tokens_mask = device->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {(size_t)text_tokens_mask_size}, rtp_llm::AllocationType::HOST});
        }
        if (mm_features_locs_size) {
            inputs.mm_features_locs = device->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {(size_t)mm_features_locs_size}, rtp_llm::AllocationType::HOST});
        }
        if (mm_features_num) {
            std::vector<rtp_llm::BufferPtr> mm_features;
            for (auto mm_index = 0; mm_index < mm_features_num; ++mm_index) {
                mm_features.emplace_back(
                    device->allocateBuffer({(rtp_llm::DataType)shape_hints_ptr[GptModelInputIndex::mmFeaturesDtype],
                                            {(size_t)mm_features_shape_ptr[mm_index],
                                             (size_t)shape_hints_ptr[GptModelInputIndex::mmFeaturesSize]},
                                            rtp_llm::AllocationType::DEVICE}));
            }
            inputs.multimodal_features = std::move(mm_features);
        }
    }

    std::vector<rtp_llm::BufferPtr> buffers;
    buffers.emplace_back(inputs.combo_tokens);
    buffers.emplace_back(inputs.input_lengths);
    buffers.emplace_back(inputs.sequence_lengths);
    buffers.emplace_back(inputs.prefix_lengths);
    if (max_blocks) {
        buffers.emplace_back(inputs.kv_cache_block_id);
        if (inputs.pd_separation) {
            buffers.emplace_back(inputs.cache_keys);
        }
        buffers.emplace_back(inputs.kv_cache_update_mapping);
    }
    buffers.emplace_back(inputs.request_id);
    buffers.emplace_back(inputs.request_pd_separation);
    buffers.emplace_back(inputs.lm_output_indexes);
    buffers.emplace_back(inputs.lm_output_lengths);
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
        for (auto& mm_feature : inputs.multimodal_features.value()) {
            buffers.emplace_back(mm_feature);
        }
    }
    if (hidden_states_size) {
        buffers.emplace_back(inputs.last_hidden_states);
    }
    device->broadcast({buffers, 0});
    device->syncAndCheck();
}

}  // namespace rtp_llm
