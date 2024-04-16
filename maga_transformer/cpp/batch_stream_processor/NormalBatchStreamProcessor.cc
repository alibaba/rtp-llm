#include "maga_transformer/cpp/batch_stream_processor/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/common/status_util.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include <cstring>

using namespace std;

namespace rtp_llm {

void memcpyKvCache(uint64_t*                    kv_cache_blocks,
                   const vector<vector<void*>>& k_ptr,
                   const vector<vector<void*>>& v_ptr,
                   int                          layer_nums,
                   int                          max_block_size,
                   int                          total_batch_size,
                   int                          batch_idx) {
    assert(k_ptr.size() == v_ptr.size() && layer_nums == k_ptr.size());
    const size_t layer_stride = total_batch_size * 2 * max_block_size;
    const size_t batch_begin  = batch_idx * 2 * max_block_size;
    for (size_t layer_id = 0; layer_id < layer_nums; ++layer_id) {
        memcpy(kv_cache_blocks + layer_id * layer_stride + batch_begin,
               k_ptr[layer_id].data(),
               k_ptr[layer_id].size() * sizeof(int64_t));
        memcpy(kv_cache_blocks + layer_id * layer_stride + batch_begin + max_block_size,
               v_ptr[layer_id].data(),
               v_ptr[layer_id].size() * sizeof(int64_t));
    }
}

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    assert(!stream_groups.empty());
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
        memcpy(merged_tokens + batch_idx, currentTokens.data(), currentTokens.size() * sizeof(int));
        auto kv_cache = stream->kvCache();
        for (auto i = 0; i < current_batch_size; ++i) {
            input_lengths[batch_idx]    = stream->inputLength();
            sequence_lengths[batch_idx] = stream->seqLength();
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
    RETURN_IF_STATUS_ERROR(createAttentionMask(model_input));
    return model_input;
}

absl::StatusOr<SamplerInputs>
NormalBatchStreamProcessor::gatherSamplerInput(const StreamGroups&    stream_groups,
                                               const GptModelOutputs& model_output) const {
    assert(!stream_groups.empty());

    // const auto generate_config = (*stream_groups.begin())->generateConfig();
    SamplerInputs sampler_inputs;
    // auto device = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
    int    max_seq_len      = stream_groups.maxSeqLen();
    size_t total_batch_size = stream_groups.totalSamplerBatchSize();
    auto   token_ids_cpu    = device_->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)max_seq_len + 10, total_batch_size}, ft::AllocationType::HOST}, {});
    sampler_inputs.token_ids = device_->allocateBuffer(
        {ft::DataType::TYPE_INT32, {(size_t)max_seq_len + 10, total_batch_size}, ft::AllocationType::DEVICE}, {});

    int                     batch_idx   = 0;
    int*                    token_ids   = (int*)token_ids_cpu->data();
    list<GenerateStreamPtr> all_streams = stream_groups.allStreams();
    for (auto& stream : all_streams) {
        const auto& complete_token_ids = stream->completeTokenIds();
        auto        complete_seq_len   = complete_token_ids->shape()[1];
        auto        seq_len            = stream->seqLength();
        auto        current_batch_size = stream->batchSize();
        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(token_ids + (batch_idx + i) * max_seq_len,
                   (int*)complete_token_ids->data() + i * complete_seq_len,
                   seq_len * sizeof(int));
        }
        batch_idx += current_batch_size;
    }
    cudaMemcpyAsync(sampler_inputs.token_ids->data(),
                    token_ids_cpu->data(),
                    token_ids_cpu->sizeBytes(),
                    cudaMemcpyHostToDevice,
                    dynamic_cast<CudaDevice*>(device_)->stream());

    sampler_inputs.logits.reset(new ft::Buffer(ft::MemoryType::MEMORY_GPU,
                                               ft::DataType::TYPE_FP16,
                                               model_output.logits->shape(),
                                               model_output.logits->data()));
    sampler_inputs.step       = max_seq_len;
    sampler_inputs.batch_size = total_batch_size;
    sampler_inputs.top_k      = device_->allocateBuffer({ft::DataType::TYPE_INT32, {1}, ft::AllocationType::HOST}, {});
    int* top_k                = (int*)sampler_inputs.top_k->data();
    top_k[0]                  = 1;
    return sampler_inputs;
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups&                  stream_groups,
                                                  const std::unique_ptr<MergedOutput>& merge_outputs) const {
    const auto& model_output      = merge_outputs->model_output;
    const auto& sampler_output    = merge_outputs->sampler_output;
    const auto& new_all_token_ids = sampler_output.token_ids;
    size_t      total_batch_size  = stream_groups.totalSamplerBatchSize();
    // assert(total_batch_size == new_all_token_ids->size());
    assert(total_batch_size == new_all_token_ids->shape()[1]);
    auto token_ids_cpu =
        device_->allocateBuffer({ft::DataType::TYPE_INT32, new_all_token_ids->shape(), ft::AllocationType::HOST}, {});
    cudaMemcpyAsync(token_ids_cpu->data(),
                    new_all_token_ids->data(),
                    new_all_token_ids->sizeBytes(),
                    cudaMemcpyDeviceToHost,
                    dynamic_cast<CudaDevice*>(device_)->stream());
    int token_idx = 0;
    for (auto& stream : stream_groups.allStreams()) {
        auto          current_batch_size = stream->batchSize();
        auto          fake_input         = std::optional<ft::BufferPtr>();
        ft::BufferPtr new_tokens;
        new_tokens.reset(new Buffer(ft::MemoryType::MEMORY_CPU_PINNED,
                                    ft::DataType::TYPE_INT32,
                                    {(size_t)1, (size_t)current_batch_size},
                                    (int*)token_ids_cpu->data() + token_idx));
        stream->update(new_tokens, 1, false, fake_input, fake_input, fake_input, fake_input);
        token_idx += current_batch_size;
    }
    return absl::OkStatus();
}

}  // namespace rtp_llm
