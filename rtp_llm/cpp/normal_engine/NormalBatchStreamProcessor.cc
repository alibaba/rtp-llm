#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <limits>
#include <utility>
#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "rtp_llm/cpp/models/Sampler.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"

using namespace std;

namespace rtp_llm {

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto context_streams = stream_groups.contextStreams();
    auto decode_streams  = stream_groups.decodeStreams();
    RTP_LLM_LOG_DEBUG(
        "context_streams size = %d, decode_streams size = %d", context_streams.size(), decode_streams.size());
    GptModelInputs model_input;
    const size_t   current_tokens_size      = stream_groups.modelExecuteTokenSize();
    const size_t   total_batch_size         = stream_groups.totalModelBatchSize();
    const size_t   total_decode_batch_size  = stream_groups.totalDecodeBatchSize();
    const size_t   total_context_batch_size = stream_groups.totalContextBatchSize();
    const size_t   total_block_copy_num     = stream_groups.totalBlockUpdateCopyNum();
    const size_t   max_block_size           = stream_groups.maxBlockSize();
    const size_t   multimodal_features_len  = stream_groups.mmFeaturesLen();

    const bool has_multimodal_input = is_multimodal_ && stream_groups.has_multimodal_input();
    const bool need_cal_position_id = (mm_position_ids_style_ != PositionIdsStyle::DEFAULT) || has_positional_encoding_;

    model_input.combo_tokens = CACHED_HOST_BUF(TYPE_INT32, {current_tokens_size});
    if (max_block_size) {
        model_input.kv_cache_block_id       = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size, max_block_size});
        model_input.kv_cache_update_mapping = CACHED_HOST_BUF(TYPE_INT32, {total_block_copy_num, 2});
        model_input.cache_keys              = CACHED_HOST_BUF(TYPE_INT64, {total_context_batch_size, max_block_size});
    }
    model_input.request_id            = CACHED_HOST_BUF(TYPE_INT64, {total_context_batch_size});
    model_input.request_pd_separation = CACHED_HOST_BUF(TYPE_BOOL, {total_context_batch_size});
    model_input.input_lengths         = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size});
    model_input.lora_ids              = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size});
    model_input.lora_input_lengths    = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size});
    model_input.sequence_lengths      = CACHED_HOST_BUF(TYPE_INT32, {total_decode_batch_size});
    model_input.lm_output_indexes     = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size});
    model_input.lm_output_lengths     = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size});
    model_input.prefix_lengths        = CACHED_HOST_BUF(TYPE_INT32, {total_context_batch_size});
    if (need_cal_position_id) {
        model_input.combo_position_ids = CACHED_HOST_BUF(TYPE_INT32, {current_tokens_size * position_id_len_factor_});
    }
    if (has_multimodal_input) {
        model_input.text_tokens_mask = CACHED_HOST_BUF(TYPE_INT32, {current_tokens_size});
        model_input.mm_features_locs = CACHED_HOST_BUF(TYPE_INT32, {multimodal_features_len});
    }
    model_input.k_block_size       = k_block_size_;
    model_input.v_block_size       = v_block_size_;
    model_input.seq_size_per_block = seq_size_per_block_;
    model_input.scale_block_size   = scale_block_size_;
    model_input.pd_separation      = role_type_ == RoleType::PREFILL;
    model_input.warmup             = warm_up_;
    model_input.decode_entrance    = decode_entrance_;

    int* merged_tokens      = (int*)model_input.combo_tokens->data();
    int* input_lengths      = (int*)model_input.input_lengths->data();
    int* lora_ids           = (int*)model_input.lora_ids->data();
    int* lora_input_lengths = (int*)model_input.lora_input_lengths->data();
    int* sequence_lengths   = (int*)model_input.sequence_lengths->data();
    int* lm_output_indexes  = (int*)model_input.lm_output_indexes->data();
    int* lm_output_lengths  = (int*)model_input.lm_output_lengths->data();
    int* prefix_lengths     = (int*)model_input.prefix_lengths->data();
    int* combo_position_ids = need_cal_position_id ? (int*)model_input.combo_position_ids->data() : nullptr;
    int* merged_text_mask   = has_multimodal_input ? (int*)model_input.text_tokens_mask->data() : nullptr;
    int* mm_features_locs   = has_multimodal_input ? (int*)model_input.mm_features_locs->data() : nullptr;
    int  batch_idx          = 0;
    int  input_vocab_size   = input_vocab_size_ ? input_vocab_size_ : vocab_size_;

    auto* kv_cache_update_mapping =
        model_input.kv_cache_update_mapping ? (BlockIdPair*)model_input.kv_cache_update_mapping->data() : nullptr;
    const auto add_cache_update_copy = [&](const auto& update_mapping) {
        size_t update_copy_num = update_mapping.size();
        std::memcpy(kv_cache_update_mapping, update_mapping.data(), update_copy_num * sizeof(BlockIdPair));
        kv_cache_update_mapping += update_copy_num;
    };

    if (merged_text_mask) {
        std::fill(merged_text_mask, merged_text_mask + current_tokens_size, 1);
    }

    for (const auto& stream : decode_streams) {
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto current_batch_size     = stream->currentBatchSize();

        const auto& kv_cache = stream->kvCache();
        RTP_LLM_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        RTP_LLM_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());

        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());

            auto currentTokens = stream->currentExecuteTokens(i);
            if (currentTokens[0] >= input_vocab_size) {
                std::ostringstream error_msg;
                error_msg << "stream [" << stream->streamId() << "] token_id " << currentTokens[0]
                          << " exceed vocab_size " << input_vocab_size;
                return absl::InvalidArgumentError(error_msg.str());
            }
            merged_tokens[batch_idx]    = currentTokens[0];
            input_lengths[batch_idx]    = stream->inputLength();
            sequence_lengths[batch_idx] = stream->seqLength() - 1;  // need remove
            if (need_cal_position_id) {
                stream->generateNextPositionId(combo_position_ids + batch_idx * position_id_len_factor_, device_);
            }
            lora_ids[batch_idx]           = stream->loraId();
            lora_input_lengths[batch_idx] = 1;
            lm_output_indexes[batch_idx]  = batch_idx;
            lm_output_lengths[batch_idx]  = 1;
            if (max_block_size) {
                std::memcpy((*model_input.kv_cache_block_id)[batch_idx].data(),
                            kv_cache.batch_block_id[i].data(),
                            kv_cache.batch_block_id[i].size() * sizeof(int));
            }
            batch_idx += 1;
        }

        if (max_block_size) {
            add_cache_update_copy(stream->streamCacheResource().getKVBlockUpdateMapping());
        }

        stream->step();
    }

    std::vector<rtp_llm::BufferPtr> gathered_mm_features;
    int                             token_idx          = batch_idx;
    int                             cum_output_seq_len = batch_idx;
    int                             mm_feature_index   = 0;

    for (const auto& stream : context_streams) {
        // context stream也需要batch运行是为了fallback的场景和perf test的场景
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto current_batch_size     = stream->currentBatchSize();

        const auto& kv_cache = stream->kvCache();
        if (enable_detail_log_) {
            RTP_LLM_LOG_DEBUG("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_DEBUG("context stream: %s", stream->debugString().c_str());
        } else {
            RTP_LLM_LOG_TRACE("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_TRACE("context stream: %s", stream->debugString().c_str());
        }

        // TODO(xinfei.sxf) deal with adjusted common seq len.
        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());

            auto input_tokens = stream->currentExecuteTokens(i);
            auto input_masks  = stream->textTokensMask();
            memcpy(merged_tokens + token_idx, input_tokens.data(), input_tokens.size() * sizeof(int));
            cum_output_seq_len += input_tokens.size();

            for (int index = 0; index < input_tokens.size(); ++index) {
                if (input_tokens[index] >= input_vocab_size && (index >= input_masks.size() || input_masks[index])) {
                    std::ostringstream error_msg;
                    error_msg << "stream [" << stream->streamId() << "] token_id " << input_tokens[index]
                              << " exceed vocab_size " << input_vocab_size;
                    return absl::InvalidArgumentError(error_msg.str());
                }
            }

            input_lengths[batch_idx]                            = input_tokens.size();
            prefix_lengths[batch_idx - total_decode_batch_size] = stream->prefixLength();
            lm_output_indexes[batch_idx]                        = cum_output_seq_len - 1;
            lm_output_lengths[batch_idx]                        = 1;

            if (has_multimodal_input) {
                std::vector<torch::Tensor> mm_features = stream->multimodalFeatures();
                rtp_llm::BufferPtr         mm_locs     = stream->multimodalLocations();
                if (mm_locs != nullptr) {
                    for (int i = 0; i < mm_locs->size(); ++i) {
                        mm_features_locs[mm_feature_index] =
                            *mm_locs->dataWithOffset<int>(i) + token_idx - stream->reuseLength();
                        mm_feature_index++;
                    }
                    for (auto& mm_feature : mm_features) {
                        auto feature_buffer = torchTensor2Buffer(mm_feature);
                        if (feature_buffer->where() != rtp_llm::MemoryType::MEMORY_GPU) {
                            gathered_mm_features.emplace_back(device_->clone({*feature_buffer}));
                        } else {
                            gathered_mm_features.emplace_back(feature_buffer);
                        }
                    }
                    auto text_token_mask = stream->textTokensMask();
                    memcpy(merged_text_mask + token_idx, text_token_mask.data(), text_token_mask.size() * sizeof(int));
                }
            }

            if (need_cal_position_id) {
                auto context_pos_ids = stream->generateContextPositionIds(device_);
                memcpy(combo_position_ids + token_idx * position_id_len_factor_,
                       context_pos_ids->dataWithOffset<int>(stream->reuseLength() * position_id_len_factor_),
                       (context_pos_ids->size() - stream->reuseLength() * position_id_len_factor_)
                           * context_pos_ids->typeSize());
            }
            lora_ids[batch_idx]           = stream->loraId();
            lora_input_lengths[batch_idx] = input_lengths[batch_idx];
            if (max_block_size) {
                std::memcpy((*model_input.kv_cache_block_id)[batch_idx].data(),
                            kv_cache.batch_block_id[i].data(),
                            kv_cache.batch_block_id[i].size() * sizeof(int));
                if (role_type_ == RoleType::PREFILL && stream->hasCacheKeys()) {
                    std::memcpy((*model_input.cache_keys)[batch_idx - total_decode_batch_size].data(),
                                stream->cacheKeys(i).data(),
                                stream->cacheKeys(i).size() * sizeof(int64_t));
                }
            }
            *(model_input.request_id->dataWithOffset<int64_t>(batch_idx - total_decode_batch_size)) =
                stream->streamId();
            *(model_input.request_pd_separation->dataWithOffset<bool>(batch_idx - total_decode_batch_size)) =
                stream->queryPdSep();
            batch_idx += 1;
            token_idx += input_tokens.size();
        }

        if (max_block_size) {
            add_cache_update_copy(stream->streamCacheResource().getKVBlockUpdateMapping());
        }

        stream->step();
    }

    if (is_multimodal_ && gathered_mm_features.size() > 0) {
        model_input.multimodal_features = std::move(gathered_mm_features);
    }
    return model_input;
}

absl::StatusOr<SamplerInputs> NormalBatchStreamProcessor::gatherSamplerInput(
    const StreamGroups& stream_groups, const GptModelInputs& model_inputs, const GptModelOutputs& model_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_CHECK(!stream_groups.empty());
    auto all_streams          = stream_groups.allStreams();
    auto total_batch_size_in  = stream_groups.totalSamplerBatchSizeIn();
    auto total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    bool return_all_probs     = stream_groups.needReturnAllProbs();

    SamplerInputs sampler_inputs =
        allocateSamplerInputs(stream_groups, total_batch_size_in, total_batch_size_out, model_inputs.sequence_lengths);
    setCommonSamplerInputs(sampler_inputs, all_streams);

    setLogitsProcessorInputs(sampler_inputs, all_streams);

    size_t total_decode_batch_size_in = 0;
    int    batch_idx                  = 0;
    bool   return_logits              = false;
    bool   calculate_softmax_probs    = false;
    bool   need_tiling                = false;
    for (auto& stream : all_streams) {
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        complete_seq_len   = complete_token_ids->shape()[1];
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = stream->currentBatchSize();
        auto        sampler_batch_size =
            stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();

        for (int i = 0; i < sampler_batch_size; ++i) {
            int cur_batch = std::min(i, current_batch_size - 1);
            memcpy(sampler_inputs.token_ids->dataWithOffset<int32_t>((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids->dataWithOffset<int32_t>(cur_batch * complete_seq_len),
                   seq_len * sizeof(int));
            sampler_inputs.finished_mask->data<bool>()[batch_idx] = stream->isDoneWithoutLock(cur_batch);
            batch_idx += 1;
        }
        need_tiling |= stream->needTilingForSampling();
        if (!stream->isContextStream()) {
            total_decode_batch_size_in += sampler_batch_size;
        }
        return_logits |= stream->returnLogits();
        calculate_softmax_probs |= stream->calculateSoftmaxProbs();
        RTP_LLM_LOG_DEBUG("stream [%ld], complete token ids = [%s]",
                          stream->streamId(),
                          complete_token_ids->debugStringWithData<int32_t>(sampler_inputs.step).c_str());
        RTP_LLM_LOG_DEBUG("stream [%ld], sampler inputs token ids = [%s]",
                          stream->streamId(),
                          sampler_inputs.token_ids->debugStringWithData<int32_t>().c_str());
    }

    auto vocab_size           = model_output.logits->shape()[1];
    sampler_inputs.vocab_size = vocab_size;
    if (return_all_probs) {
        sampler_inputs.all_probs = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {total_batch_size_in, vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
        device_->bufMemset(*sampler_inputs.all_probs, 0);
    }

    // copy logits when needs tiling or returning logits
    if (need_tiling) {
        // TODO(zhangjianning.zjn): tiling for logits should be moved after logits processors
        sampler_inputs.logits = device_->allocateBuffer(
            {model_output.logits->type(), {total_batch_size_in, vocab_size}, rtp_llm::AllocationType::DEVICE}, {});
        device_->copy({sampler_inputs.logits->view(0, total_decode_batch_size_in),
                       model_output.logits->view(0, total_decode_batch_size_in)});
        size_t input_offset = total_decode_batch_size_in, logits_offset = total_decode_batch_size_in;
        for (auto& stream : stream_groups.contextStreams()) {
            auto sampler_batch_size =
                stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();
            for (int i = 0; i < sampler_batch_size; ++i) {
                device_->copy(
                    {sampler_inputs.logits->view(input_offset, 1), model_output.logits->view(logits_offset, 1)});
                input_offset += 1;
            }
            logits_offset += 1;
        }
    } else if (return_logits || calculate_softmax_probs) {
        sampler_inputs.logits = device_->clone({*model_output.logits, rtp_llm::AllocationType::DEVICE});
    } else {
        sampler_inputs.logits = model_output.logits;
    }

    RTP_LLM_LOG_DEBUG("sampler inputs logits [%s]",
                      device_->clone({*sampler_inputs.logits, rtp_llm::AllocationType::HOST})
                          ->debugStringWithData<float>(10)
                          .c_str());

    RTP_LLM_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

SamplerInputs NormalBatchStreamProcessor::allocateSamplerInputs(const StreamGroups&       stream_groups,
                                                                size_t                    total_batch_size_in,
                                                                size_t                    total_batch_size_out,
                                                                const rtp_llm::BufferPtr& sequence_lengths) const {
    // TODO(xinfei.sxf) don't sample for chunk stream
    SamplerInputs sampler_inputs;
    sampler_inputs.step             = stream_groups.maxSeqLen();
    sampler_inputs.batch_size       = total_batch_size_in;
    sampler_inputs.batch_size_out   = total_batch_size_out;
    sampler_inputs.sequence_lengths = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size_in});
    sampler_inputs.logits_processor_states_ptr.reset();
    sampler_inputs.input_lengths        = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size_in});
    sampler_inputs.num_beams_in         = CACHED_HOST_BUF(TYPE_UINT64, {total_batch_size_in});
    sampler_inputs.num_beams_out        = CACHED_HOST_BUF(TYPE_UINT64, {total_batch_size_in});
    sampler_inputs.top_k                = CACHED_HOST_BUF(TYPE_UINT32, {total_batch_size_in});
    sampler_inputs.top_p                = CACHED_HOST_BUF(TYPE_FP32, {total_batch_size_in});
    sampler_inputs.temperature          = CACHED_HOST_BUF(TYPE_FP32, {total_batch_size_in});
    sampler_inputs.random_seeds         = CACHED_HOST_BUF(TYPE_UINT64, {total_batch_size_in});
    sampler_inputs.repetition_penalty   = CACHED_HOST_BUF(TYPE_FP32, {total_batch_size_in});
    sampler_inputs.presence_penalty     = CACHED_HOST_BUF(TYPE_FP32, {total_batch_size_in});
    sampler_inputs.frequency_penalty    = CACHED_HOST_BUF(TYPE_FP32, {total_batch_size_in});
    sampler_inputs.min_lengths          = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size_in});
    sampler_inputs.no_repeat_ngram_size = CACHED_HOST_BUF(TYPE_INT32, {total_batch_size_in});
    sampler_inputs.do_sample            = CACHED_HOST_BUF(TYPE_BOOL, {total_batch_size_in});
    sampler_inputs.finished_mask        = CACHED_HOST_BUF(TYPE_BOOL, {total_batch_size_in});
    if (stream_groups.needReturnCumLogProbs()) {
        sampler_inputs.cum_log_probs = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {total_batch_size_in}, rtp_llm::AllocationType::HOST}, {});
    }
    sampler_inputs.token_ids = device_->allocateBuffer(
        {rtp_llm::DataType::TYPE_INT32, {total_batch_size_in, sampler_inputs.step + 1}, rtp_llm::AllocationType::HOST},
        {});
    return sampler_inputs;
}

void NormalBatchStreamProcessor::setCommonSamplerInputs(SamplerInputs&                sampler_inputs,
                                                        std::list<GenerateStreamPtr>& all_streams,
                                                        bool                          score_batch) const {
    int*      input_lengths        = sampler_inputs.input_lengths->data<int32_t>();
    int*      sequence_lengths     = sampler_inputs.sequence_lengths->data<int32_t>();
    uint64_t* num_beams_in         = sampler_inputs.num_beams_in->data<uint64_t>();
    uint64_t* num_beams_out        = sampler_inputs.num_beams_out->data<uint64_t>();
    uint32_t* top_k                = sampler_inputs.top_k->data<uint32_t>();
    float*    top_p                = sampler_inputs.top_p->data<float>();
    float*    temperature          = sampler_inputs.temperature->data<float>();
    uint64_t* random_seeds         = sampler_inputs.random_seeds->data<uint64_t>();
    float*    repetition_penalty   = sampler_inputs.repetition_penalty->data<float>();
    float*    presence_penalty     = sampler_inputs.presence_penalty->data<float>();
    float*    frequency_penalty    = sampler_inputs.frequency_penalty->data<float>();
    int32_t*  min_lengths          = sampler_inputs.min_lengths->data<int32_t>();
    int32_t*  no_repeat_ngram_size = sampler_inputs.no_repeat_ngram_size->data<int32_t>();
    bool*     do_sample            = sampler_inputs.do_sample->data<bool>();

    int  batch_idx       = 0;
    bool has_random_seed = false;
    for (auto& stream : all_streams) {
        int sampler_batch_size;
        if (score_batch) {
            sampler_batch_size = stream->scoreLen();
        } else if (stream->needTilingForSampling()) {
            sampler_batch_size = stream->nextBatchSize();
        } else {
            sampler_batch_size = stream->currentBatchSize();
        }
        if (sampler_inputs.cum_log_probs) {
            const auto& cum_log_probs = stream->cumLogProbs();
            memcpy(sampler_inputs.cum_log_probs->dataWithOffset<float>(batch_idx),
                   cum_log_probs->data(),
                   cum_log_probs->sizeBytes());
        }
        for (int i = 0; i < sampler_batch_size; ++i) {
            input_lengths[batch_idx]      = stream->inputLength();
            sequence_lengths[batch_idx]   = stream->seqLength();
            num_beams_in[batch_idx]       = stream->currentNumBeams();
            num_beams_out[batch_idx]      = stream->nextNumBeams();
            top_k[batch_idx]              = stream->generateConfig()->top_k;
            top_p[batch_idx]              = stream->generateConfig()->top_p;
            temperature[batch_idx]        = stream->generateConfig()->temperature;
            repetition_penalty[batch_idx] = stream->generateConfig()->repetition_penalty;
            presence_penalty[batch_idx]   = stream->generateConfig()->presence_penalty;
            frequency_penalty[batch_idx]  = stream->generateConfig()->frequency_penalty;
            min_lengths[batch_idx]        = stream->generateConfig()->min_new_tokens;
            do_sample[batch_idx]          = stream->generateConfig()->do_sample;
            if (!do_sample[batch_idx]) {
                top_k[batch_idx]       = 1;
                top_p[batch_idx]       = 1;
                temperature[batch_idx] = 1;
            }
            if (stream->generateConfig()->random_seed.has_value()) {
                random_seeds[batch_idx] = stream->generateConfig()->random_seed.value();
                has_random_seed         = true;
            } else {
                std::random_device                          rd;
                std::mt19937_64                             gen(rd());
                std::uniform_int_distribution<std::int64_t> distrib(0, std::numeric_limits<std::int64_t>::max());
                random_seeds[batch_idx] = distrib(gen);
            }
            no_repeat_ngram_size[batch_idx] = stream->generateConfig()->no_repeat_ngram_size.value_or(0);
            batch_idx += 1;
        }
    }
    if (!has_random_seed) {
        sampler_inputs.random_seeds.reset();
    }
}

void NormalBatchStreamProcessor::setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                                          std::list<GenerateStreamPtr>& all_streams,
                                                          bool                          score_batch) const {
    LogitsProcessorStatesPtr state_ptr = std::make_shared<LogitsProcessorStates>();
    // TODO(zhangjianning.zjn): tiling for logits should be moved after logits processors
    std::for_each(all_streams.begin(), all_streams.end(), [&state_ptr, idx = 0](auto& stream) mutable {
        const auto batch_size = stream->needTilingForSampling() ? stream->nextBatchSize() : stream->currentBatchSize();
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            state_ptr->insert(processor, idx, idx + batch_size);
        }
        idx += batch_size;
    });
    sampler_inputs.logits_processor_states_ptr = state_ptr;
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups& stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto& model_output      = merge_outputs.model_output;
    const auto& sampler_output    = merge_outputs.sampler_output;
    const auto& new_all_token_ids = sampler_output.token_ids;
    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", new_all_token_ids->debugStringWithData<int32_t>().c_str());
    const size_t token_stride         = new_all_token_ids->shape()[1];
    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == new_all_token_ids->shape()[0]);
    int  batch_idx_in     = 0;
    int  batch_idx_out    = 0;
    int  token_offset     = 0;
    bool return_all_probs = stream_groups.needReturnAllProbs();
    auto new_tokens_all   = CACHED_HOST_BUF(TYPE_INT32, {(size_t)total_batch_size_out, (size_t)1});

    for (auto& stream : stream_groups.allStreams()) {
        if (stream->isChunkStream()) {
            continue;
        }
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();

        auto token_size              = stream->currentExecuteTokenSize();
        auto batch_new_all_token_ids = new_all_token_ids->slice(batch_idx_out, next_batch_size);

        bool has_beam_search = stream->currentNumBeams() > 1 || stream->nextNumBeams() > 1;
        bool has_var_batch   = stream->currentBatchSize() != stream->nextBatchSize();

        // construct mapping from output batches to input batches
        BufferPtr src_batch_indices;
        if (has_beam_search) {
            // beam search
            src_batch_indices = sampler_output.beam_index->slice(batch_idx_out, next_batch_size);
        } else if (has_var_batch) {
            // from context stream to decode straem, there might be other cases in future
            src_batch_indices = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {(size_t)next_batch_size}, rtp_llm::AllocationType::HOST}, {});
            device_->bufMemset(*src_batch_indices, 0);
        }
        const auto get_src_idx = [&](int32_t dst_idx) {
            return src_batch_indices ? src_batch_indices->data<int32_t>()[dst_idx] : dst_idx;
        };

        // construct update info
        BufferPtr batch_hidden_states = nullptr;
        if (stream->generateConfig()->return_hidden_states) {
            batch_hidden_states = model_output.hidden_states->slice(batch_idx_in, cur_batch_size);
        }

        BufferPtr batch_logits = nullptr;
        if (stream->returnLogits() || stream->calculateSoftmaxProbs() || has_beam_search) {
            batch_logits = model_output.logits->slice(batch_idx_in, cur_batch_size);
        }

        BufferPtr all_probs = nullptr;
        if (return_all_probs) {
            all_probs = sampler_output.all_probs->slice(batch_idx_out, next_batch_size, false);
            all_probs->updateParent(sampler_output.all_probs);
        };

        BufferPtr batch_cum_log_probs;
        if (sampler_output.cum_log_probs) {
            batch_cum_log_probs = sampler_output.cum_log_probs->slice(batch_idx_out, next_batch_size);
        }

        BufferPtr loss;
        if (stream->calculateLoss()) {
            auto               all_logits = model_output.all_logits->view(token_offset, token_size - 1);
            auto               tokens     = stream->currentExecuteTokens(0);
            rtp_llm::BufferPtr label      = device_->clone({{rtp_llm::MemoryType::MEMORY_CPU,
                                                             rtp_llm::DataType::TYPE_INT32,
                                                             {tokens.size() - 1},
                                                             tokens.data() + 1}});
            loss                          = device_->loss({all_logits, *label});
        }

        BufferPtr all_hidden_states = nullptr;
        if (stream->needReturnHiddenStates()) {
            all_hidden_states = model_output.all_hidden_states->slice(token_offset, token_size, false);
            all_hidden_states->updateParent(model_output.all_hidden_states);
        }

        BufferPtr new_tokens = new_tokens_all->slice(batch_idx_out, next_batch_size);
        for (size_t i = 0; i < next_batch_size; ++i) {
            new_tokens->data<int32_t>()[i] =
                new_all_token_ids->data<int32_t>()[(batch_idx_out + i) * token_stride + token_stride - 1];
        }

        BufferPtr batch_softmax_result;
        BufferPtr current_softmax_result;
        if (stream->calculateSoftmaxProbs()) {
            current_softmax_result = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_FP32, {(size_t)next_batch_size, (size_t)1}, rtp_llm::AllocationType::HOST},
                {});
            batch_softmax_result =
                device_->softmax({batch_logits, std::nullopt, std::nullopt, 1.0f, DataType::TYPE_FP32, std::nullopt});
            for (int i = 0; i < next_batch_size; ++i) {
                device_->copy({(*current_softmax_result)[i],
                               (*batch_softmax_result)[get_src_idx(i)].view(new_tokens->data<int32_t>()[i], 1)});
            }
        }

        for (int i = 0; i < cur_batch_size; ++i) {
            if (sampler_output.success && !(*(sampler_output.success->dataWithOffset<bool>(batch_idx_in + i)))) {
                stream->setStop(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
            }
        }

        RTP_LLM_LOG_DEBUG(
            "stream [%ld], new_tokens = [%s]", stream->streamId(), new_tokens->debugStringWithData<int32_t>().c_str());

        stream->update({has_beam_search ? batch_new_all_token_ids : new_tokens,
                        1,
                        batch_hidden_states,
                        batch_logits,
                        current_softmax_result,
                        batch_cum_log_probs,
                        all_probs,
                        loss,
                        src_batch_indices,
                        all_hidden_states});

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += token_size;
    }

    RTP_LLM_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

}  // namespace rtp_llm
