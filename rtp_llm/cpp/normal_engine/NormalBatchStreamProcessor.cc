#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"

namespace rtp_llm {

NormalBatchStreamProcessor::NormalBatchStreamProcessor(
    const ModelConfig&                 model_config,
    const PDSepConfig&                 pd_sep_config,
    const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
    const CacheConfig&                 cache_config,
    bool                               warm_up) {
    incremental_cache_store_ = pd_sep_config.enable_chunkwise_cache_transfer && pd_sep_config.cache_store_rdma_mode;
    if (pd_sep_config.enable_chunkwise_cache_transfer && !pd_sep_config.cache_store_rdma_mode) {
        RTP_LLM_LOG_INFO("chunkwise transfer requires RDMA; TCP publishes all cache blocks at terminal prefill");
    }
    model_input_gatherer_config_.num_layers              = model_config.num_layers;
    model_input_gatherer_config_.vocab_size              = model_config.vocab_size;
    model_input_gatherer_config_.input_vocab_size        = model_config.input_vocab_size;
    model_input_gatherer_config_.has_positional_encoding = model_config.has_positional_encoding;
    model_input_gatherer_config_.is_multimodal           = model_config.mm_model_config.is_multimodal;
    model_input_gatherer_config_.mm_position_ids_style =
        static_cast<PositionIdsStyle>(model_config.mm_model_config.mm_position_ids_style);
    model_input_gatherer_config_.position_id_len_factor     = model_config.attn_config.rope_config.index_factor;
    model_input_gatherer_config_.role_type                  = pd_sep_config.role_type;
    model_input_gatherer_config_.decode_entrance            = pd_sep_config.decode_entrance;
    model_input_gatherer_config_.block_stride_bytes         = cache_config.kv_block_stride_bytes;
    model_input_gatherer_config_.scale_stride_bytes         = cache_config.kv_scale_stride_bytes;
    model_input_gatherer_config_.seq_size_per_block         = cache_config.seq_size_per_block;
    model_input_gatherer_config_.kernel_seq_size_per_block  = cache_config.kernel_seq_size_per_block;
    model_input_gatherer_config_.kernel_blocks_per_kv_block = cache_config.kernelBlocksPerKvBlock();
    model_input_gatherer_config_.kv_cache_group_nums        = cache_config.groupNums();
    model_input_gatherer_config_.use_opaque_kv_cache_store  = cache_config.use_opaque_kv_cache_store;
    if (model_input_gatherer_config_.kv_cache_group_nums > 0) {
        model_input_gatherer_config_.kv_cache_group_types = cache_config.groupTypesSnapshot();
        model_input_gatherer_config_.kv_cache_group_tags  = cache_config.groupTagsSnapshot();
    }
    model_input_gatherer_config_.warm_up                 = warm_up;
    model_input_gatherer_config_.enable_detail_log       = profiling_debug_logging_config.enable_detail_log;
    model_input_gatherer_config_.enable_model_inputs_log = profiling_debug_logging_config.enable_model_inputs_log;

    model_input_gatherer_   = std::make_unique<NormalModelInputGatherer>(model_input_gatherer_config_);
    sampler_input_gatherer_ = std::make_unique<NormalSamplerInputGatherer>();
    output_dispatcher_      = std::make_unique<NormalOutputDispatcher>(model_config.output_vocab_ids);
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups& stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    return output_dispatcher_->dispatch(stream_groups, merge_outputs);
}

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups,
                                                                            TensorHolder&       host_holder) const {
    auto result = model_input_gatherer_->gather(stream_groups, host_holder);
    if (result.ok()) {
        auto status = prepareCacheStorePublishPlan(stream_groups, result.value(), 0);
        if (!status.ok()) {
            // The stream already owns the failure. Still synchronize skip_run
            // so non-root TP ranks do not enter a forward without rank 0.
            result->skip_run = true;
        }
    }
    return result;
}

absl::Status NormalBatchStreamProcessor::prepareCacheStorePublishPlan(
    const StreamGroups& streams, GptModelInputs& inputs, size_t model_id) const {
    inputs.cache_store_publish_begin_tokens = torch::Tensor();
    inputs.cache_store_publish_end_tokens = torch::Tensor();
    inputs.cache_store_publish_terminal = torch::Tensor();
    inputs.cache_store_incremental = false;
    if (inputs.warmup || inputs.is_fake_stream || !inputs.pd_separation) {
        return absl::OkStatus();
    }
    bool chunked = false;
    for (const auto& stream : streams.contextStreams()) {
        chunked |= stream->chunkedPrefillEnabled() && stream->queryPdSep();
    }
    if (!chunked) {
        return absl::OkStatus();
    }
    const auto rows = static_cast<int64_t>(streams.totalContextBatchSize());
    inputs.cache_store_publish_begin_tokens = torch::zeros({rows}, torch::kInt32).pin_memory();
    inputs.cache_store_publish_end_tokens = torch::empty({rows}, torch::kInt32).pin_memory();
    inputs.cache_store_publish_terminal = torch::empty({rows}, torch::kBool).pin_memory();
    inputs.cache_store_incremental = incremental_cache_store_;
    const auto lengths = inputs.input_lengths.cpu();
    const auto prefixes = inputs.prefix_lengths.cpu();
    const auto decode_rows = lengths.numel() - rows;
    int64_t row = 0;
    for (const auto& stream : streams.contextStreams()) {
        const auto progress = stream->cacheStorePublishProgress(model_id);
        if (model_id == 0 && stream->queryPdSep() && stream->resourceContext().cache_manager) {
            stream->holdKVCacheForPDSep();
        }
        for (int b = 0; b < stream->currentBatchSize(); ++b, ++row) {
            const int prefix = prefixes[row].item<int32_t>();
            const int end = prefix + lengths[decode_rows + row].item<int32_t>();
            if (stream->queryPdSep() && (progress.terminal_committed || progress.committed_window_end > prefix)) {
                stream->reportError(ErrorCode::CACHE_STORE_STORE_FAILED,
                                    "chunkwise cache publication cannot rewind; retry the PD request");
                return absl::FailedPreconditionError("chunkwise cache publication cannot rewind");
            }
            inputs.cache_store_publish_begin_tokens[row] = progress.committed_window_end;
            inputs.cache_store_publish_end_tokens[row] = end;
            inputs.cache_store_publish_terminal[row] = !stream->isMiddleChunk();
        }
    }
    return absl::OkStatus();
}

void NormalBatchStreamProcessor::commitCacheStorePublishPlan(
    const StreamGroups& streams, const GptModelInputs& inputs, size_t model_id) const {
    if (!inputs.cache_store_publish_begin_tokens.defined()) {
        return;
    }
    int64_t row = 0;
    for (const auto& stream : streams.contextStreams()) {
        if (stream->queryPdSep() && stream->isActive()) {
            stream->commitCacheStorePublication(model_id,
                inputs.cache_store_publish_begin_tokens[row].item<int32_t>(),
                inputs.cache_store_publish_end_tokens[row].item<int32_t>(),
                inputs.cache_store_publish_terminal[row].item<bool>());
        }
        row += stream->currentBatchSize();
    }
}

absl::StatusOr<SamplerInputs> NormalBatchStreamProcessor::gatherSamplerInput(
    const StreamGroups& stream_groups, const GptModelInputs& model_inputs, const GptModelOutputs& model_output) const {
    return sampler_input_gatherer_->gather(stream_groups, model_inputs, model_output);
}

absl::StatusOr<torch::Tensor> NormalBatchStreamProcessor::gatherKvCacheKernelBlockId(const StreamGroups& stream_groups,
                                                                                     TensorHolder& host_holder) const {
    return model_input_gatherer_->gatherKvCacheKernelBlockId(stream_groups, host_holder);
}

SamplerInputs NormalBatchStreamProcessor::allocateSamplerInputs(const StreamGroups& stream_groups,
                                                                size_t              total_batch_size_in,
                                                                size_t              total_batch_size_out,
                                                                size_t              propose_step) const {
    return sampler_input_gatherer_->allocateSamplerInputs(
        stream_groups, total_batch_size_in, total_batch_size_out, propose_step);
}

void NormalBatchStreamProcessor::fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                                         std::list<GenerateStreamPtr>& all_streams,
                                                         bool                          score_batch,
                                                         size_t                        propose_step) const {
    sampler_input_gatherer_->fillSamplerCommonInputs(sampler_inputs, all_streams, score_batch, propose_step);
}

void NormalBatchStreamProcessor::setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                                          std::list<GenerateStreamPtr>& all_streams,
                                                          bool                          score_batch) const {
    sampler_input_gatherer_->setLogitsProcessorInputs(sampler_inputs, all_streams, score_batch);
}

}  // namespace rtp_llm
