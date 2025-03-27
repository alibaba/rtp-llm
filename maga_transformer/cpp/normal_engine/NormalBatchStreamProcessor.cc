#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <limits>
#include <utility>
#include "c10/core/DeviceType.h"
#include "c10/core/ScalarType.h"
#include "maga_transformer/cpp/models/Sampler.h"
#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include "src/fastertransformer/core/Types.h"
#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"

using namespace std;
using namespace fastertransformer;
namespace rtp_llm {

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto           context_streams = stream_groups.contextStreams();
    auto           decode_streams  = stream_groups.decodeStreams();
    FT_LOG_DEBUG("context_streams size = %d, decode_streams size = %d", context_streams.size(), decode_streams.size());
    GptModelInputs model_input;
    size_t         current_tokens_size      = stream_groups.modelExecuteTokenSize();
    size_t         total_batch_size         = stream_groups.totalModelBatchSize();
    size_t         total_decode_batch_size  = stream_groups.totalDecodeBatchSize();
    size_t         total_context_batch_size  = stream_groups.totalContextBatchSize();
    size_t         max_block_size           = stream_groups.maxBlockSize();
    size_t         multimodal_features_len  = stream_groups.mmFeaturesLen();

    const bool has_multimodal_input = is_multimodal_ && stream_groups.has_multimodal_input();
    const bool need_cal_position_id = (mm_position_ids_style_ != PositionIdsStyle::DEFAULT) || has_positional_encoding_;

    model_input.combo_tokens =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {current_tokens_size}, ft::AllocationType::HOST}, {});
    if (max_block_size) {
        model_input.kv_cache_block_id = device_->allocateBuffer(
                {ft::DataType::TYPE_INT32, {total_batch_size, max_block_size}, ft::AllocationType::HOST}, {});
        model_input.cache_keys = device_->allocateBuffer(
            {ft::DataType::TYPE_INT64, {total_context_batch_size, max_block_size}, ft::AllocationType::HOST}, {});
    }
    model_input.request_id = device_->allocateBuffer(
            {ft::DataType::TYPE_INT64, {total_context_batch_size}, ft::AllocationType::HOST}, {});
    model_input.request_pd_separation = device_->allocateBuffer(
            {ft::DataType::TYPE_BOOL, {total_context_batch_size}, ft::AllocationType::HOST}, {});
    model_input.input_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    model_input.lora_ids =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    model_input.lora_input_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    model_input.sequence_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_decode_batch_size}, ft::AllocationType::HOST}, {});
    model_input.lm_output_indexes =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    model_input.prefix_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_context_batch_size}, ft::AllocationType::HOST}, {});
    if (need_cal_position_id) {
        model_input.combo_position_ids =
            device_->allocateBuffer({ft::DataType::TYPE_INT32, {current_tokens_size * position_id_len_factor_}, ft::AllocationType::HOST}, {});
    }
    if (has_multimodal_input) {
        model_input.text_tokens_mask =
            device_->allocateBuffer({ft::DataType::TYPE_INT32, {current_tokens_size}, ft::AllocationType::HOST}, {});
        model_input.mm_features_locs =
            device_->allocateBuffer({ft::DataType::TYPE_INT32, {multimodal_features_len}, ft::AllocationType::HOST}, {});
    }
    model_input.k_block_size = k_block_size_;
    model_input.v_block_size = v_block_size_;
    model_input.scale_block_size = scale_block_size_;
    model_input.pd_separation = pd_separation_;
    model_input.warmup = warm_up_;

    int*      merged_tokens    = (int*)model_input.combo_tokens->data();
    int*      input_lengths    = (int*)model_input.input_lengths->data();
    int*      lora_ids         = (int*)model_input.lora_ids->data();
    int*      lora_input_lengths = (int*)model_input.lora_input_lengths->data();
    int*      sequence_lengths = (int*)model_input.sequence_lengths->data();
    int*      lm_output_indexes = (int*)model_input.lm_output_indexes->data();
    int*      prefix_lengths   = (int*)model_input.prefix_lengths->data();
    int*      combo_position_ids = need_cal_position_id ? (int*)model_input.combo_position_ids->data() : nullptr;
    int*      merged_text_mask = has_multimodal_input ? (int*)model_input.text_tokens_mask->data() : nullptr;
    int*      mm_features_locs = has_multimodal_input ? (int*)model_input.mm_features_locs->data() : nullptr;
    int       batch_idx        = 0;
    int       input_vocab_size = input_vocab_size_ ? input_vocab_size_ : vocab_size_;

    if (merged_text_mask) {
        std::fill(merged_text_mask, merged_text_mask + current_tokens_size, 1);
    }

    for (const auto& stream : decode_streams) {
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto current_batch_size = stream->batchSize();
        auto kv_cache = stream->kvCache();
        FT_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        FT_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());
        for (auto i = 0; i < current_batch_size; ++i) {
            auto currentTokens      = stream->currentExecuteTokens(i);
            if (currentTokens[0] >= input_vocab_size) {
                std::ostringstream error_msg;
                error_msg << "stream [" << stream->streamId() << "] token_id " << currentTokens[0] << " exceed vocab_size " << input_vocab_size;
                return absl::InvalidArgumentError(error_msg.str());
            }
            merged_tokens[batch_idx] = currentTokens[0];
            input_lengths[batch_idx]    = stream->inputLength();
            sequence_lengths[batch_idx] = stream->seqLength() - 1; // need remove
            if (need_cal_position_id) {
                stream->generateNextPositionId(combo_position_ids + batch_idx * position_id_len_factor_);
            }
            lora_ids[batch_idx]         = stream->loraId();
            lora_input_lengths[batch_idx] = 1;
            lm_output_indexes[batch_idx] = batch_idx;
            if (max_block_size) {
                std::memcpy((*model_input.kv_cache_block_id)[batch_idx].data(),
                            kv_cache.batch_block_id[i].data(),
                            kv_cache.batch_block_id[i].size() * sizeof(int));
            }
            batch_idx += 1;
        }
        stream->step();
    }

    std::vector<ft::BufferPtr> gathered_mm_features;
    int token_idx = batch_idx;
    int cum_output_seq_len = batch_idx;
    int mm_feature_index = 0;

    for (const auto& stream : context_streams) {
        // context stream也需要batch运行是为了fallback的场景和perf test的场景
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto current_batch_size = stream->batchSize();
        auto kv_cache                 = stream->kvCache();
        FT_LOG_DEBUG("context kv_cache: %s", kv_cache.debugString().c_str());
        FT_LOG_DEBUG("context stream: %s", stream->debugString().c_str());

        // TODO(xinfei.sxf) deal with adjusted common seq len.
        for (auto i = 0; i < current_batch_size; ++i) {
            auto input_tokens    = stream->currentExecuteTokens(i);
            auto input_masks     = stream->textTokensMask();
            memcpy(merged_tokens + token_idx, input_tokens.data(), input_tokens.size() * sizeof(int));
            cum_output_seq_len += input_tokens.size();

            for (int index = 0; index < input_tokens.size(); ++index) {
                if (input_tokens[index] >= input_vocab_size && (index >= input_masks.size() || input_masks[index])) {
                    std::ostringstream error_msg;
                    error_msg << "stream [" << stream->streamId() << "] token_id " << input_tokens[index] << " exceed vocab_size " << input_vocab_size;
                    return absl::InvalidArgumentError(error_msg.str());
                }
            }

            input_lengths[batch_idx] = input_tokens.size();
            prefix_lengths[batch_idx - total_decode_batch_size] = stream->prefixLength();
            lm_output_indexes[batch_idx] = cum_output_seq_len - 1;

            if (has_multimodal_input) {
                std::vector<torch::Tensor> mm_features = stream->multimodalFeatures();
                ft::BufferPtr mm_locs = stream->multimodalLocations();
                if (mm_locs != nullptr) {
                    for (int i = 0;i < mm_locs->size(); ++i) {
                        mm_features_locs[mm_feature_index] = *mm_locs->dataWithOffset<int>(i) + token_idx - stream->reuseLength();
                        mm_feature_index++;
                    }
                    for (auto& mm_feature: mm_features) {
                        auto feature_buffer = torchTensor2Buffer(mm_feature);
                        if (feature_buffer->where() != ft::MemoryType::MEMORY_GPU) {
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
                       (context_pos_ids->size() - stream->reuseLength() * position_id_len_factor_) * context_pos_ids->typeSize());
            }
            lora_ids[batch_idx]           = stream->loraId();
            lora_input_lengths[batch_idx] = input_lengths[batch_idx];
            if (max_block_size) {
                std::memcpy((*model_input.kv_cache_block_id)[batch_idx].data(),
                            kv_cache.batch_block_id[i].data(),
                            kv_cache.batch_block_id[i].size() * sizeof(int));
                if (stream->hasCacheKeys()) {
                    std::memcpy((*model_input.cache_keys)[batch_idx - total_decode_batch_size].data(),
                        stream->cacheKeys(i).data(),
                        stream->cacheKeys(i).size() * sizeof(int64_t));
                }
            }
            *(model_input.request_id->dataWithOffset<int64_t>(batch_idx - total_decode_batch_size)) = stream->streamId();
            *(model_input.request_pd_separation->dataWithOffset<bool>(batch_idx - total_decode_batch_size)) = stream->queryPdSep();
            batch_idx += 1;
            token_idx += input_tokens.size();
        }

        stream->step();
    }

    if (is_multimodal_ && gathered_mm_features.size() > 0) {
        model_input.multimodal_features = std::move(gathered_mm_features);
    }
    return model_input;
}

absl::StatusOr<SamplerInputs>
NormalBatchStreamProcessor::gatherSamplerInput(const StreamGroups&    stream_groups,
                                               const GptModelInputs&  model_inputs,
                                               const GptModelOutputs& model_output) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    FT_CHECK(!stream_groups.empty());
    const auto& context_streams = stream_groups.contextStreams();
    size_t total_decode_batch_size = stream_groups.totalDecodeBatchSize();
    auto all_streams = stream_groups.allStreams();
    auto total_batch_size = stream_groups.totalSamplerBatchSize();
    bool return_all_probs = stream_groups.needReturnAllProbs();

    SamplerInputs sampler_inputs = allocateSamplerInputs(stream_groups, total_batch_size, model_inputs.sequence_lengths);
    setCommonSamplerInputs(sampler_inputs, all_streams);

    setThinkModeLogitsProcessorInputs(sampler_inputs, all_streams);

    int batch_idx   = 0;
    bool return_logits = false;
    bool calculate_softmax_probs = false;
    for (auto& stream : all_streams) {
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        complete_seq_len   = complete_token_ids->shape()[1];
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = stream->tileNum();

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(sampler_inputs.token_ids->dataWithOffset<int32_t>((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids->dataWithOffset<int32_t>(i * complete_seq_len),
                   seq_len * sizeof(int));
            batch_idx += 1;
        }
        return_logits |= stream->returnLogits();
        calculate_softmax_probs |= stream->calculateSoftmaxProbs();
        FT_LOG_DEBUG("stream [%d], complete token ids = [%s]", stream->streamId(), complete_token_ids->debugStringWithData<int32_t>(sampler_inputs.step).c_str());
        FT_LOG_DEBUG("stream [%d], sampler inputs token ids = [%s]", stream->streamId(), sampler_inputs.token_ids->debugStringWithData<int32_t>().c_str());
    }

    auto vocab_size = model_output.logits->shape()[1];
    sampler_inputs.vocab_size = vocab_size;
    if (return_all_probs) {
        sampler_inputs.all_probs = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size, vocab_size}, ft::AllocationType::DEVICE}, {});
        device_->bufMemset(*sampler_inputs.all_probs, 0);
    }

    batch_idx = 0;
    // need copy logits when has tile or return logits
    if (return_logits || calculate_softmax_probs || (context_streams.size() && total_batch_size > all_streams.size())) {
        sampler_inputs.logits = device_->allocateBuffer({model_output.logits->type(), {total_batch_size, vocab_size}, ft::AllocationType::DEVICE}, {});
        device_->copy({sampler_inputs.logits->view(0, total_decode_batch_size), model_output.logits->view(0, total_decode_batch_size)});
    } else {
        sampler_inputs.logits = model_output.logits;
    }
    batch_idx += total_decode_batch_size;
    size_t logits_offset = batch_idx;
    for (auto& stream : context_streams) {
        auto current_batch_size = stream->tileNum();
        for (int i = 0; i < current_batch_size; ++i) {
            device_->copy({sampler_inputs.logits->view(batch_idx, 1), model_output.logits->view(logits_offset, 1)});
            batch_idx += 1;
        }
        logits_offset += 1;
    }

    FT_LOG_DEBUG("sampler inputs logits [%s]",
                device_->clone({*sampler_inputs.logits, ft::AllocationType::HOST})->debugStringWithData<float>(10).c_str());

    FT_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

SamplerInputs NormalBatchStreamProcessor::allocateSamplerInputs(const StreamGroups& stream_groups, size_t total_batch_size, const ft::BufferPtr& sequence_lengths) const {
    // TODO(xinfei.sxf) don't sample for chunk stream
    SamplerInputs sampler_inputs;
    sampler_inputs.step   = stream_groups.maxSeqLen();;
    sampler_inputs.batch_size = total_batch_size;
    sampler_inputs.sequence_lengths = device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.grammars.clear();
    sampler_inputs.beam_search_sequence_lengths = device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.beam_index   =  device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    // TODO(lidongjin.ldj) use bufMemset after arm/amd support this op.
    // eg: device_->bufMemset(*sampler_inputs.beam_index, 0);
    sampler_inputs.input_lengths = device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.num_beams     = device_->allocateBuffer({ft::DataType::TYPE_UINT64, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.top_k         = device_->allocateBuffer({ft::DataType::TYPE_UINT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.top_p         = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.temperature   = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.random_seeds  = device_->allocateBuffer({ft::DataType::TYPE_UINT64, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.repetition_penalty = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.min_lengths   = device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.no_repeat_ngram_size = device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    if (stream_groups.needReturnCumLogProbs()) {
        sampler_inputs.cum_log_probs = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}, ft::AllocationType::HOST}, {});
    }
    sampler_inputs.token_ids = device_->allocateBuffer(
            {ft::DataType::TYPE_INT32, {total_batch_size, sampler_inputs.step + 1}, ft::AllocationType::HOST}, {});
    return sampler_inputs;
}

void NormalBatchStreamProcessor::setCommonSamplerInputs(SamplerInputs& sampler_inputs, std::list<GenerateStreamPtr>& all_streams, bool score_batch) const {
    int* input_lengths        = sampler_inputs.input_lengths->data<int32_t>();
    int* sequence_lengths     = sampler_inputs.sequence_lengths->data<int32_t>();
    uint64_t* num_beams       = sampler_inputs.num_beams->data<uint64_t>();
    uint32_t* top_k           = sampler_inputs.top_k->data<uint32_t>();
    float* top_p              = sampler_inputs.top_p->data<float>();
    float* temperature        = sampler_inputs.temperature->data<float>();
    uint64_t* random_seeds    = sampler_inputs.random_seeds->data<uint64_t>();
    float* repetition_penalty = sampler_inputs.repetition_penalty->data<float>();
    int32_t* min_lengths      = sampler_inputs.min_lengths->data<int32_t>();
    int32_t* no_repeat_ngram_size = sampler_inputs.no_repeat_ngram_size->data<int32_t>();
    int* beam_search_sequence_lengths = sampler_inputs.beam_search_sequence_lengths->data<int32_t>();

    int batch_idx   = 0;
    bool has_random_seed = false;
    for (auto& stream : all_streams) {
        int        current_batch_size;
        if (!score_batch) {
            current_batch_size = stream->tileNum();
        } else {
            current_batch_size = stream->scoreLen();
        }
        if (sampler_inputs.cum_log_probs) {
            const auto& cum_log_probs      = stream->cumLogProbs();
            memcpy(sampler_inputs.cum_log_probs->dataWithOffset<float>(batch_idx), cum_log_probs->data(), cum_log_probs->sizeBytes());
        }
        for (int i = 0; i < current_batch_size; ++i) {
            input_lengths[batch_idx]      = stream->inputLength();
            sequence_lengths[batch_idx]   = stream->seqLength();
            beam_search_sequence_lengths[batch_idx]  = stream->seqLength();
            // TODO(xinfei.sxf) fix num beams after sampler support
            num_beams[batch_idx]          = stream->numBeams();
            top_k[batch_idx]              = stream->generateConfig()->top_k;
            top_p[batch_idx]              = stream->generateConfig()->top_p;
            temperature[batch_idx]        = stream->generateConfig()->temperature;
            repetition_penalty[batch_idx] = stream->generateConfig()->repetition_penalty;
            min_lengths[batch_idx]        = stream->generateConfig()->min_new_tokens;
            if (stream->generateConfig()->random_seed.has_value()) {
                random_seeds[batch_idx]   = stream->generateConfig()->random_seed.value();
                has_random_seed = true;
            } else {
                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<std::int64_t> distrib(0, std::numeric_limits<std::int64_t>::max());
                random_seeds[batch_idx]   = distrib(gen);
            }
            no_repeat_ngram_size[batch_idx] = stream->generateConfig()->no_repeat_ngram_size.value_or(0);
            batch_idx += 1;
        }
    }
    if (!has_random_seed) {
        sampler_inputs.random_seeds.reset();
    }
}


void NormalBatchStreamProcessor::setThinkModeLogitsProcessorInputs(SamplerInputs& sampler_inputs, std::list<GenerateStreamPtr>& all_streams, bool score_batch) const {
    
    int batch_idx   = 0;
    std::deque<bool> think_modes;
    std::vector<int> max_thinking_tokens;
    std::vector<std::vector<int>> end_think_token_ids;
    std::vector<std::shared_ptr<StringContainDFA<size_t, int>>> think_status_dfa_ptrs;

    for (auto& stream : all_streams) {
        int current_batch_size;
        if (!score_batch) {
            current_batch_size = stream->tileNum();
        } else {
            current_batch_size = stream->scoreLen();
        }
        for (int i = 0; i < current_batch_size; ++i) {
            think_modes.push_back(stream->thinkMode());
            max_thinking_tokens.push_back(stream->maxThinkingTokens());
            end_think_token_ids.push_back(stream->endThinkTokenIds());
            think_status_dfa_ptrs.push_back(stream->thinkEndStatusDfa(i));            
            batch_idx += 1;
        }
    }

    FT_CHECK(think_modes.size() == sampler_inputs.batch_size);
    FT_CHECK(max_thinking_tokens.size() == sampler_inputs.batch_size);
    FT_CHECK(end_think_token_ids.size() == sampler_inputs.batch_size);
    FT_CHECK(think_status_dfa_ptrs.size() == sampler_inputs.batch_size);

    BaseLogitsProcessorPtr processor_ptr = std::make_shared<ThinkModeLogitsProcessor>(device_, think_modes, max_thinking_tokens, end_think_token_ids, think_status_dfa_ptrs);

    sampler_inputs.grammars.push_back(processor_ptr);
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups&                  stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto& model_output      = merge_outputs.model_output;
    const auto& sampler_output    = merge_outputs.sampler_output;
    const auto& new_all_token_ids = sampler_output.token_ids;
    FT_LOG_DEBUG("new_all_token_ids = [%s]", new_all_token_ids->debugStringWithData<int32_t>().c_str());
    const size_t step = new_all_token_ids->shape()[1];
    size_t total_batch_size  = stream_groups.totalSamplerBatchSize();
    FT_CHECK(total_batch_size == new_all_token_ids->shape()[0]);
    int batch_idx = 0;
    int offset = 0;
    int token_offset = 0;
    bool return_all_probs = stream_groups.needReturnAllProbs();
    ft::BufferPtr new_tokens_all = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)total_batch_size, (size_t)1}, ft::AllocationType::HOST}, {});

    for (auto& stream : stream_groups.allStreams()) {
        if (stream->isChunkStream()) {
            continue;
        }
        auto current_batch_size = stream->tileNum();
        auto new_tokens = new_tokens_all->slice(batch_idx, current_batch_size);
        auto token_size = stream->currentExecuteTokenSize();
        auto batch = stream->isContextStream() ? 1 : current_batch_size;
        auto batch_logits = model_output.logits->slice(offset, batch);
        auto batch_hidden_states = model_output.hidden_states->slice(offset, batch);
        BufferPtr batch_cum_log_probs;
        if (sampler_output.cum_log_probs) {
            batch_cum_log_probs = sampler_output.cum_log_probs->slice(batch_idx, current_batch_size);
        }
        auto all_probs = return_all_probs ? sampler_output.all_probs->slice(batch_idx, current_batch_size) : nullptr;
        BufferPtr loss = nullptr;
        BufferPtr beam_index = (sampler_output.beam_index == nullptr) ? nullptr : sampler_output.beam_index->slice(batch_idx, current_batch_size);
        if (stream->calculateLoss()) {
            auto all_logits = model_output.all_logits->view(token_offset, token_size - 1);
            auto tokens = stream->currentExecuteTokens(0);
            ft::BufferPtr label = device_->clone({{ft::MemoryType::MEMORY_CPU, ft::DataType::TYPE_INT32, {tokens.size() - 1}, tokens.data() + 1}});
            loss = device_->loss({all_logits, *label});
        }
        BufferPtr batch_softmax_result;
        BufferPtr current_softmax_result;
        if (stream->calculateSoftmaxProbs()) {
            current_softmax_result = device_->allocateBuffer({ft::DataType::TYPE_FP32, {(size_t)current_batch_size, (size_t)1}, ft::AllocationType::HOST}, {});
            batch_softmax_result = device_->softmax({batch_logits, std::nullopt, std::nullopt, 1.0f, DataType::TYPE_FP32, std::nullopt});
        }
        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(new_tokens->dataWithOffset<int32_t>(i), new_all_token_ids->dataWithOffset<int32_t>(batch_idx * step + step - 1), sizeof(int32_t));
            if (stream->calculateSoftmaxProbs()) {
                device_->copy({(*current_softmax_result)[i], (*batch_softmax_result)[i].view(*(new_tokens->dataWithOffset<int32_t>(i)), 1)});
            }
            if (sampler_output.success && !(*(sampler_output.success->dataWithOffset<bool>(batch_idx)))) {
                stream->setStop(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
            }
            batch_idx += 1;
        }
        FT_LOG_DEBUG("stream [%d], new_tokens = [%s]", stream->streamId(), new_tokens->debugStringWithData<int32_t>().c_str());
        if (stream->numBeams() > 1 && beam_index != nullptr) {
            StreamUpdateInfo update_info{new_all_token_ids, 1, batch_hidden_states, batch_logits,
                    current_softmax_result, batch_cum_log_probs, all_probs, loss};
            stream->update(update_info);
            stream->beamSearchKvCacheUpdate(beam_index);
        } else {
            stream->update({new_tokens, 1, batch_hidden_states, batch_logits,
                    current_softmax_result, batch_cum_log_probs, all_probs, loss});
        }
        offset += batch;
        token_offset += token_size;
    }
    FT_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

}  // namespace rtp_llm
