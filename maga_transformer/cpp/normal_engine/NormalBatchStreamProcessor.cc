#include <cstring>
#include <random>
#include <limits>

#include "src/fastertransformer/utils/assert_utils.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/utils/KvCacheUtils.h"

using namespace std;
using namespace fastertransformer;
namespace rtp_llm {

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto           context_streams = stream_groups.contextStreams();
    auto           decode_streams  = stream_groups.decodeStreams();
    GptModelInputs model_input;
    size_t         context_batch_size       = context_streams.size();
    size_t         current_tokens_size      = stream_groups.modelExecuteTokenSize();
    size_t         total_batch_size         = stream_groups.totalModelBatchSize();
    size_t         total_decode_batch_size  = stream_groups.totalDecodeBatchSize();
    size_t         max_block_size           = stream_groups.maxBlockSize();
    model_input.combo_tokens =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {current_tokens_size}, ft::AllocationType::HOST}, {});
    model_input.kv_cache_blocks = device_->allocateBuffer(
        {ft::DataType::TYPE_UINT64, {num_layers_, total_batch_size, 2, max_block_size}, ft::AllocationType::HOST}, {});
    if (use_int8_kv_cache_) {
        model_input.kv_cache_scales = device_->allocateBuffer(
            {ft::DataType::TYPE_UINT64, {num_layers_, total_batch_size, 2, max_block_size}, ft::AllocationType::HOST},
            {});
    }
    model_input.input_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    model_input.lora_ids =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    model_input.sequence_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_decode_batch_size}, ft::AllocationType::HOST}, {});
    model_input.prefix_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    int*      merged_tokens    = (int*)model_input.combo_tokens->data();
    int*      input_lengths    = (int*)model_input.input_lengths->data();
    int*      lora_ids          = (int*)model_input.lora_ids->data();
    int*      sequence_lengths = (int*)model_input.sequence_lengths->data();
    int*      prefix_lengths   = (int*)model_input.prefix_lengths->data();
    uint64_t* kv_cache_blocks  = (uint64_t*)model_input.kv_cache_blocks->data();
    uint64_t* kv_cache_scales  = use_int8_kv_cache_ ? (uint64_t*)model_input.kv_cache_scales->data() : nullptr;
    int       batch_idx        = 0;

    for (const auto& stream : decode_streams) {
        auto currentTokens      = stream->currentExecuteTokens();
        auto current_batch_size = stream->batchSize();
        memcpy(merged_tokens + batch_idx, currentTokens.data(), currentTokens.size() * sizeof(int));
        auto kv_cache = stream->kvCache();
        FT_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        FT_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());
        for (auto i = 0; i < current_batch_size; ++i) {
            input_lengths[batch_idx]    = stream->inputLength();
            sequence_lengths[batch_idx] = stream->seqLength() - 1; // need remove
            prefix_lengths[batch_idx]   = 0;
            lora_ids[batch_idx]         = stream->loraId();
            memcpyKvCache(kv_cache_blocks,
                          kv_cache.k_ptr[i],
                          kv_cache.v_ptr[i],
                          num_layers_,
                          max_block_size,
                          total_batch_size,
                          batch_idx);
            if (use_int8_kv_cache_) {
                memcpyKvCache(kv_cache_scales,
                              kv_cache.k_scale_ptr[i],
                              kv_cache.v_scale_ptr[i],
                              num_layers_,
                              max_block_size,
                              total_batch_size,
                              batch_idx);
            }
            batch_idx += 1;
        }
    }

    int token_idx = batch_idx;
    for (const auto& stream : context_streams) {
        auto input_tokens    = stream->currentExecuteTokens();
        memcpy(merged_tokens + token_idx, input_tokens.data(), input_tokens.size() * sizeof(int));
        token_idx += input_tokens.size();
        input_lengths[batch_idx]  = stream->contextLength();
        prefix_lengths[batch_idx] = stream->reuseLength();
        lora_ids[batch_idx]       = stream->loraId();
        auto kv_cache             = stream->kvCache();
        FT_LOG_DEBUG("context kv_cache: %s", kv_cache.debugString().c_str());
        FT_LOG_DEBUG("context stream: %s", stream->debugString().c_str());
        memcpyKvCache(kv_cache_blocks,
                      kv_cache.k_ptr[0],
                      kv_cache.v_ptr[0],
                      num_layers_,
                      max_block_size,
                      total_batch_size,
                      batch_idx);
        if (use_int8_kv_cache_) {
            memcpyKvCache(kv_cache_scales,
                          kv_cache.k_scale_ptr[0],
                          kv_cache.v_scale_ptr[0],
                          num_layers_,
                          max_block_size,
                          total_batch_size,
                          batch_idx);
        }
        batch_idx += 1;
    }
    FT_LOG_DEBUG("input_lengths: %s", model_input.input_lengths->debugStringWithData<int32_t>().c_str());
    FT_LOG_DEBUG("sequence_lengths: %s", model_input.sequence_lengths->debugStringWithData<int32_t>().c_str());

    createAttentionMask(stream_groups, model_input);
    return model_input;
}

// TODO(xinfei.sxf) fmha enable的判断支持动态化，现在是靠静态的判断，不太好。
void NormalBatchStreamProcessor::createAttentionMask(const StreamGroups& stream_groups, GptModelInputs& model_input) const {
    if (!need_attention_mask_) {
        return;
    }
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    auto           context_streams     = stream_groups.contextStreams();
    size_t         context_batch_size  = context_streams.size();
    size_t         max_context_seq_len = stream_groups.maxContextSeqLen();
    size_t         max_reuse_len       = stream_groups.maxReuseLength();

    DataType target_data_type = getDataType(data_type_);
    DataType data_type = TYPE_FP32;

    // TODO(xinfei.sxf)考虑在fallback的情况下，sequence len的使用
    auto attention_mask =
        device_->allocateBuffer({data_type,
            {context_batch_size, max_context_seq_len, max_context_seq_len + max_reuse_len}, ft::AllocationType::HOST}, {});

    // TODO(xinfei.sxf) memset 0 in device base when api is ready
    for (size_t i = 0; i < context_batch_size; i++) {
        for (size_t j = 0; j < max_context_seq_len; j++) {
            for (size_t k = 0; k <= max_context_seq_len + max_reuse_len; k++) {
                switch (data_type) {
                    #define ATTENTION_MASK_VALUE(ft_type) \
                    case ft_type: { \
                        typedef DataTypeTraits<ft_type>::type cppType; \
                        auto data = reinterpret_cast<cppType (*) \
                            [context_batch_size][max_context_seq_len][max_context_seq_len + max_reuse_len]>((void*)attention_mask->data()); \
                        (*data)[i][j][k] = 0.0; \
                        break; \
                    }

                    ATTENTION_MASK_VALUE(TYPE_FP32)
                    default:
                        throw std::runtime_error("wrong data type.");
                }
            }
        }
    }

    for (size_t i = 0; i < context_batch_size; i++) {
        auto context_len = (*std::next(context_streams.begin(), i))->contextLength();
        auto reuse_len = (*std::next(context_streams.begin(), i))->reuseLength();
        for (size_t j = 0; j < context_len; j++) {
            for (size_t k = 0; k <= reuse_len + j; k++) {
                switch (data_type) {
                    #define ATTENTION_MASK_VALUE(ft_type) \
                    case ft_type: { \
                        typedef DataTypeTraits<ft_type>::type cppType; \
                        auto data = reinterpret_cast<cppType (*) \
                            [context_batch_size][max_context_seq_len][max_context_seq_len + max_reuse_len]>((void*)attention_mask->data()); \
                        (*data)[i][j][k] = 1.0; \
                        break; \
                    }

                    ATTENTION_MASK_VALUE(TYPE_FP32)
                    default:
                        throw std::runtime_error("wrong data type.");
                }
            }
        }
    }

    FT_LOG_DEBUG("attention_mask = [%s]", attention_mask->debugStringWithData<float>().c_str());
    auto convert_attention_mask = device_->convert({attention_mask, target_data_type});
    auto attention_mask_gpu = device_->allocateBuffer(
        {convert_attention_mask->type(), convert_attention_mask->shape()}, {"attn_mask"});
    device_->copy({*attention_mask_gpu, *convert_attention_mask});

    FT_LOG_DEBUG("create attention_mask done");

    model_input.attention_mask = attention_mask_gpu;
}

absl::StatusOr<SamplerInputs>
NormalBatchStreamProcessor::gatherSamplerInput(const StreamGroups&    stream_groups,
                                               const GptModelInputs&  model_inputs,
                                               const GptModelOutputs& model_output) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    assert(!stream_groups.empty());
    const auto& context_streams = stream_groups.contextStreams();
    const auto& decode_streams = stream_groups.decodeStreams();
    size_t total_decode_batch_size = stream_groups.totalDecodeBatchSize();
    auto all_streams = stream_groups.allStreams();

    SamplerInputs sampler_inputs;
    sampler_inputs.step   = stream_groups.maxSeqLen();;
    auto total_batch_size = stream_groups.totalSamplerBatchSize();
    sampler_inputs.batch_size = total_batch_size;
    sampler_inputs.sequence_lengths = model_inputs.sequence_lengths;
    sampler_inputs.input_lengths = device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.num_beams     = device_->allocateBuffer({ft::DataType::TYPE_UINT64, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.top_k         = device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.top_p         = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.temperature   = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.random_seeds  = device_->allocateBuffer({ft::DataType::TYPE_UINT64, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.repetition_penalty = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.min_lengths   = device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.cum_log_probs = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.token_ids = device_->allocateBuffer(
            {ft::DataType::TYPE_INT32, {total_batch_size, sampler_inputs.step + 1}, ft::AllocationType::HOST}, {});

    int* input_lengths        = sampler_inputs.input_lengths->data<int32_t>();
    uint64_t* num_beams       = sampler_inputs.num_beams->data<uint64_t>();
    int32_t* top_k            = sampler_inputs.top_k->data<int32_t>();
    float* top_p              = sampler_inputs.top_p->data<float>();
    float* temperature        = sampler_inputs.temperature->data<float>();
    uint64_t* random_seeds    = sampler_inputs.random_seeds->data<uint64_t>();
    float* repetition_penalty = sampler_inputs.repetition_penalty->data<float>();
    int32_t* min_lengths      = sampler_inputs.min_lengths->data<int32_t>();

    int batch_idx   = 0;
    for (auto& stream : all_streams) {
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        complete_seq_len   = complete_token_ids->shape()[1];
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = stream->batchSize();
        const auto& cum_log_probs      = stream->cumLogProbs();

        memcpy(sampler_inputs.cum_log_probs->dataWithOffset<float>(batch_idx), cum_log_probs->data(), cum_log_probs->sizeBytes());

        for (int i = 0; i < current_batch_size; ++i) {
            input_lengths[batch_idx]      = stream->inputLength();
            // TODO(xinfei.sxf) fix num beams after sampler support
            num_beams[batch_idx]          = 1;
            top_k[batch_idx]              = stream->generateConfig()->top_k;
            top_p[batch_idx]              = stream->generateConfig()->top_p;
            temperature[batch_idx]        = stream->generateConfig()->temperature;
            repetition_penalty[batch_idx] = stream->generateConfig()->repetition_penalty;
            min_lengths[batch_idx]        = stream->generateConfig()->min_new_tokens;
            if (stream->generateConfig()->random_seed.has_value()) {
                random_seeds[batch_idx]   = stream->generateConfig()->random_seed.value();
            } else {
                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<std::int64_t> distrib(0, std::numeric_limits<std::int64_t>::max());
                random_seeds[batch_idx]   = distrib(gen);
            }
            memcpy(sampler_inputs.token_ids->dataWithOffset<int32_t>((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids->dataWithOffset<int32_t>(i * complete_seq_len),
                   seq_len * sizeof(int));
            batch_idx += 1;
        }

        FT_LOG_DEBUG("stream [%d], complete token ids = [%s]", stream->streamId(), complete_token_ids->debugStringWithData<int32_t>(sampler_inputs.step).c_str());
        FT_LOG_DEBUG("stream [%d], sampler inputs token ids = [%s]", stream->streamId(), sampler_inputs.token_ids->debugStringWithData<int32_t>().c_str());
    }

    auto vocab_size = model_output.logits->shape()[1];
    sampler_inputs.logits = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size, vocab_size}, ft::AllocationType::DEVICE}, {});

    batch_idx = 0;
    device_->copy({sampler_inputs.logits->view(0, total_decode_batch_size), model_output.logits->view(0, total_decode_batch_size)});
    batch_idx += total_decode_batch_size;
    size_t logits_offset = batch_idx;
    for (auto& stream : context_streams) {
        auto current_batch_size = stream->batchSize();
        for (int i = 0; i < current_batch_size; ++i) {
            device_->copy({sampler_inputs.logits->view(batch_idx, 1), model_output.logits->view(logits_offset, 1)});
            batch_idx += 1;
        }
        logits_offset += 1;
    }

    FT_LOG_DEBUG("sampler inputs logits [%s]",
                device_->clone({*sampler_inputs.logits, ft::AllocationType::HOST})->debugStringWithData<float>(10).c_str());

    FT_LOG_DEBUG("gatherSamplerInput done");
    return move(sampler_inputs);
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups&                  stream_groups,
                                                  const SamplerInputs&                 sampler_inputs,
                                                  const std::unique_ptr<MergedOutput>& merge_outputs) const {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto& model_output      = merge_outputs->model_output;
    const auto& sampler_output    = merge_outputs->sampler_output;
    const auto& new_all_token_ids = sampler_output.token_ids;
    FT_LOG_DEBUG("new_all_token_ids = [%s]", new_all_token_ids->debugStringWithData<int32_t>().c_str());
    const size_t step = new_all_token_ids->shape()[1];
    size_t total_batch_size  = stream_groups.totalSamplerBatchSize();
    FT_CHECK(total_batch_size == new_all_token_ids->shape()[0]);
    auto token_ids_cpu =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, new_all_token_ids->shape(), ft::AllocationType::HOST}, {});
    int batch_idx = 0;
    int offset = 0;
    for (auto& stream : stream_groups.allStreams()) {
        auto current_batch_size = stream->batchSize();
        auto batch = stream->isContextStream() ? 1 : current_batch_size;
        auto batch_logits = model_output.logits->view(offset, batch);
        auto batch_hidden_states = model_output.hidden_states->view(offset, batch);
        auto batch_cum_log_probs = sampler_output.cum_log_probs->view(batch_idx, current_batch_size);
        offset += batch;
        ft::BufferPtr new_tokens = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)current_batch_size, (size_t)1}, ft::AllocationType::HOST}, {});
        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(new_tokens->dataWithOffset<int32_t>(i), new_all_token_ids->dataWithOffset<int32_t>(batch_idx * step + step - 1), sizeof(int32_t));
            batch_idx += 1;
        }
        FT_LOG_DEBUG("stream [%d], new_tokens = [%s]", stream->streamId(), new_tokens->debugStringWithData<int32_t>().c_str());
        stream->update(new_tokens, 1, batch_hidden_states, batch_logits, batch_cum_log_probs);
    }
    FT_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

}  // namespace rtp_llm
