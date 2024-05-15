#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/utils/KvCacheUtils.h"
#include "src/fastertransformer/core/Types.h"
#include "maga_transformer/cpp/utils/TimeUtility.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include <cstring>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    // assert(!stream_groups.empty());
    auto           context_streams = stream_groups.contextStreams();
    auto           decode_streams  = stream_groups.decodeStreams();
    GptModelInputs model_input;
    size_t         context_batch_size  = context_streams.size();
    size_t         current_tokens_size = stream_groups.modelExecuteTokenSize();
    size_t         total_batch_size    = stream_groups.totalModelBatchSize();
    size_t         max_block_size      = stream_groups.maxBlockSize();
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
    model_input.sequence_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {decode_streams.size()}, ft::AllocationType::HOST}, {});
    model_input.prefix_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    int*      merged_tokens    = (int*)model_input.combo_tokens->data();
    int*      input_lengths    = (int*)model_input.input_lengths->data();
    int*      sequence_lengths = (int*)model_input.sequence_lengths->data();
    int*      prefix_lengths   = (int*)model_input.prefix_lengths->data();
    uint64_t* kv_cache_blocks  = (uint64_t*)model_input.kv_cache_blocks->data();
    uint64_t* kv_cache_scales  = use_int8_kv_cache_ ? (uint64_t*)model_input.kv_cache_scales->data() : nullptr;
    int       batch_idx        = 0;

    for (auto& stream : decode_streams) {
        auto currentTokens      = stream->currentExecuteTokens();
        auto current_batch_size = stream->batchSize();
        // TODO(xinfei.sxf) consider batch
        memcpy(merged_tokens + batch_idx, currentTokens.data(), currentTokens.size() * sizeof(int));
        auto kv_cache = stream->kvCache();
        FT_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        FT_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());
        for (auto i = 0; i < current_batch_size; ++i) {
            input_lengths[batch_idx]    = stream->inputLength();
            sequence_lengths[batch_idx] = stream->seqLength() - 1; // need remove
            // TODO(xinfei.sxf) decode stream 还需要设置prefix len吗？
            prefix_lengths[batch_idx]   = stream->reuseLength();
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
    for (auto& stream : context_streams) {
        auto input_tokens    = stream->currentExecuteTokens();
        auto block_and_scale = stream->kvCache();
        memcpy(merged_tokens + token_idx, input_tokens.data(), input_tokens.size() * sizeof(int));
        token_idx += input_tokens.size();
        input_lengths[batch_idx]  = stream->inputLength() - stream->reuseLength();
        prefix_lengths[batch_idx] = stream->reuseLength();
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
    createAttentionMask(stream_groups, model_input);
    return model_input;
}

// TODO(xinfei.sxf) fmha enable的判断支持动态化，现在是靠静态的判断，不太好。
void NormalBatchStreamProcessor::createAttentionMask(const StreamGroups& stream_groups, GptModelInputs& model_input) const {
    if (!need_attention_mask_) {
        return;
    }

    auto           context_streams     = stream_groups.contextStreams();
    size_t         context_batch_size  = context_streams.size();
    size_t         max_context_seq_len = stream_groups.maxContextSeqLen();
    size_t         max_reuse_len       = stream_groups.maxReuseLength();

    DataType target_data_type = getDataType(data_type_);
    DataType data_type = TYPE_FP32;

    // TODO(xinfei.sxf) set 0 in device base
    auto attention_mask =
        device_->allocateBuffer({data_type,
            {context_batch_size, max_context_seq_len, max_context_seq_len + max_reuse_len}, ft::AllocationType::HOST}, {});
    for (size_t i = 0; i < context_batch_size; i++) {
        auto seq_len = (*std::next(context_streams.begin(), i))->seqLength();
        auto reuse_len = (*std::next(context_streams.begin(), i))->reuseLength();
        for (size_t j = 0; j < seq_len; j++) {
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
    auto attention_mask_gpu =
        device_->allocateBuffer({attention_mask->type(), attention_mask->shape()});
    device_->copy({*attention_mask_gpu, *attention_mask});
    // TODO(xinfei.sxf) add convert to target data type
    model_input.attention_mask = std::move(attention_mask_gpu);
}

absl::StatusOr<SamplerInputs>
NormalBatchStreamProcessor::gatherSamplerInput(const StreamGroups&    stream_groups,
                                               const GptModelOutputs& model_output) const {
    assert(!stream_groups.empty());
    SamplerInputs sampler_inputs;
    int    max_seq_len      = stream_groups.maxSeqLen();
    sampler_inputs.step     = max_seq_len;
    size_t total_batch_size = stream_groups.totalSamplerBatchSize();

    sampler_inputs.top_k         = device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.top_p         = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.temperature   = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.num_beams     = device_->allocateBuffer({ft::DataType::TYPE_UINT64, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.random_seeds  = device_->allocateBuffer({ft::DataType::TYPE_UINT64, {total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.cum_log_probs = device_->allocateBuffer({ft::DataType::TYPE_FP32, {total_batch_size}}, {});

    sampler_inputs.batch_size = total_batch_size;
    sampler_inputs.token_ids = device_->allocateBuffer(
            {ft::DataType::TYPE_INT32, {total_batch_size, sampler_inputs.step + 1}, ft::AllocationType::HOST}, {});

    int                     batch_idx   = 0;
    int32_t*                token_ids   = sampler_inputs.token_ids->data<int32_t>();
    list<GenerateStreamPtr> all_streams = stream_groups.allStreams();
    sampler_inputs.sequence_lengths =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, {total_batch_size}, ft::AllocationType::HOST}, {});
    int*      sequence_lengths = sampler_inputs.sequence_lengths->data<int32_t>();

    int32_t* top_k            = sampler_inputs.top_k->data<int32_t>();
    float* top_p              = sampler_inputs.top_p->data<float>();
    float* temperature        = sampler_inputs.temperature->data<float>();
    uint64_t* num_beams       = sampler_inputs.num_beams->data<uint64_t>();
    uint64_t* random_seeds    = sampler_inputs.random_seeds->data<uint64_t>();
    for (auto& stream : all_streams) {
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        complete_seq_len   = complete_token_ids->shape()[1];
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = stream->batchSize();
        for (int i = 0; i < current_batch_size; ++i) {
            num_beams[batch_idx]        = 1;
            random_seeds[batch_idx]     = TimeUtility::currentTimeInMicroSeconds();
            top_k[batch_idx]            = stream->generateConfig()->top_k.value_or(0);
            top_p[batch_idx]            = stream->generateConfig()->top_p.value_or(0.95);
            temperature[batch_idx]      = stream->generateConfig()->temperature.value_or(1.0);
            sequence_lengths[batch_idx] = stream->seqLength();
            memcpy(sampler_inputs.token_ids->dataWithOffset<int32_t>((batch_idx) * sampler_inputs.step),
                   complete_token_ids->dataWithOffset<int32_t>(i * complete_seq_len),
                   seq_len * sizeof(int));
            batch_idx += 1;
        }

        FT_LOG_DEBUG("stream [%d], complete_token_ids = [%s]\n", stream->streamId(), complete_token_ids->debugStringWithData<int32_t>(sampler_inputs.step).c_str());
        FT_LOG_DEBUG("stream [%d], sampler_inputs = [%s]\n", stream->streamId(), complete_token_ids->debugStringWithData<int32_t>(sampler_inputs.step).c_str());
    }

    sampler_inputs.logits.reset(new ft::Buffer(ft::MemoryType::MEMORY_GPU,
                                               ft::DataType::TYPE_FP32,
                                               model_output.logits->shape(),
                                               model_output.logits->data()));
    return sampler_inputs;
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups&                  stream_groups,
                                                  const std::unique_ptr<MergedOutput>& merge_outputs) const {
    const auto& model_output      = merge_outputs->model_output;
    const auto& sampler_output    = merge_outputs->sampler_output;
    const auto& new_all_token_ids = sampler_output.token_ids;
    const size_t step = new_all_token_ids->shape()[1];
    size_t      total_batch_size  = stream_groups.totalSamplerBatchSize();
    // assert(total_batch_size == new_all_token_ids->size());
    assert(total_batch_size == new_all_token_ids->shape()[0]);
    auto token_ids_cpu =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, new_all_token_ids->shape(), ft::AllocationType::HOST}, {});
    int batch_idx = 0;
    for (auto& stream : stream_groups.allStreams()) {
        auto          current_batch_size = stream->batchSize();
        ft::BufferPtr new_tokens = device_->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)current_batch_size, (size_t)1}, ft::AllocationType::HOST}, {});
        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(new_tokens->dataWithOffset<int32_t>(i), new_all_token_ids->dataWithOffset<int32_t>(batch_idx * step + step - 1), sizeof(int32_t));
            batch_idx += 1;
        }
        FT_LOG_DEBUG("stream [%d], new_tokens = [%s]\n", stream->streamId(), new_tokens->debugStringWithData<int32_t>(step).c_str());
        stream->update(new_tokens, 1, false, nullopt, nullopt, nullopt);
    }
    return absl::OkStatus();
}

}  // namespace rtp_llm
