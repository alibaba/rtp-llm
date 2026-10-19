#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"

namespace rtp_llm {

NormalBatchStreamProcessor::NormalBatchStreamProcessor(
    const ModelConfig&                 model_config,
    const PDSepConfig&                 pd_sep_config,
    const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
    const CacheConfig&                 cache_config,
    bool                               warm_up,
    int                                async_worker_count) {
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
    model_input_gatherer_config_.seq_size_per_block         = cache_config.seq_size_per_block;
    model_input_gatherer_config_.kernel_seq_size_per_block  = cache_config.seq_size_per_block;
    model_input_gatherer_config_.kernel_blocks_per_kv_block = 1;
    model_input_gatherer_config_.kv_cache_group_nums        = cache_config.groupNums();
    model_input_gatherer_config_.use_opaque_kv_cache_store  = cache_config.use_opaque_kv_cache_store;
    if (model_input_gatherer_config_.kv_cache_group_nums > 0) {
        // Scalar layout metadata is meaningful only for a single group.
        // Multi-group consumers read their tagged specs and block tables.
        model_input_gatherer_config_.kernel_seq_size_per_block = 0;
        if (cache_config.groupNums() == 1) {
            const auto& group                                      = cache_config.topology().groups().front();
            model_input_gatherer_config_.block_stride_bytes        = group.kvBlockStrideBytes();
            model_input_gatherer_config_.scale_stride_bytes        = group.kvScaleStrideBytes();
            model_input_gatherer_config_.kernel_seq_size_per_block = group.kernelSeqSizePerBlock();
        }
        model_input_gatherer_config_.kernel_blocks_per_kv_block = cache_config.topology().maxKernelBlocksPerKvBlock();
        const auto& topology_groups                             = cache_config.topology().groups();
        model_input_gatherer_config_.kv_cache_group_types.reserve(topology_groups.size());
        model_input_gatherer_config_.kv_cache_group_tags = cache_config.groupTags();
        for (const auto& group : topology_groups) {
            model_input_gatherer_config_.kv_cache_group_types.push_back(group.policy.group_type);
        }
    }
    model_input_gatherer_config_.warm_up                 = warm_up;
    model_input_gatherer_config_.enable_detail_log       = profiling_debug_logging_config.enable_detail_log;
    model_input_gatherer_config_.enable_model_inputs_log = profiling_debug_logging_config.enable_model_inputs_log;

    model_input_gatherer_   = std::make_unique<NormalModelInputGatherer>(model_input_gatherer_config_);
    sampler_input_gatherer_ = std::make_unique<NormalSamplerInputGatherer>();
    output_dispatcher_ = std::make_unique<NormalOutputDispatcher>(model_config.output_vocab_ids, async_worker_count);
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups& stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    return output_dispatcher_->dispatch(stream_groups, merge_outputs);
}

absl::Status
NormalBatchStreamProcessor::dispatchPrefillOnly(const StreamGroups&              stream_groups,
                                                GenerationPrefillCudaGraphStatus generation_prefill_cuda_graph_status,
                                                const GptModelOutputs*           model_output) const {
    const auto custom_output = model_output && model_output->custom_output.defined() ?
                                   model_output->custom_output.to(torch::kCPU) :
                                   torch::Tensor();
    int        expected_rows = 0;
    for (const auto& stream : stream_groups.contextStreams()) {
        if (stream->generateInput()->custom_output_token_position >= stream->prefixLength()) {
            expected_rows += stream->currentBatchSize();
        }
    }
    std::string output_error = model_output ? model_output->custom_output_error : "";
    if (expected_rows > 0 && output_error.empty()
        && (!custom_output.defined() || custom_output.dim() != 2 || custom_output.size(0) != expected_rows)) {
        output_error = "decision output row count does not match selected sequences";
    }
    int output_offset = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        const bool has_custom_output =
            stream->isContextStream()
            && stream->generateInput()->custom_output_token_position >= stream->prefixLength();
        const int row_offset = output_offset;
        if (has_custom_output) {
            // Gather includes selected rows even if their requests fail or are
            // cancelled during execution. Consume that row before skipping the
            // response so later requests keep their original output positions.
            output_offset += stream->currentBatchSize();
        }
        if (stream->hasError()) {
            continue;
        }
        torch::Tensor row_output;
        if (has_custom_output) {
            if (!output_error.empty()) {
                stream->reportError(ErrorCode::EXECUTION_EXCEPTION, output_error);
                continue;
            }
            row_output = custom_output.narrow(0, row_offset, stream->currentBatchSize());
        }
        auto empty_token_ids = torch::empty({stream->currentBatchSize(), 0}, torch::kInt32);
        stream->update({empty_token_ids,
                        /*num_new_tokens=*/0,
                        /*hidden_states=*/torch::Tensor(),
                        /*logits=*/torch::Tensor(),
                        /*softmax_probs=*/torch::Tensor(),
                        /*cum_log_probs=*/torch::Tensor(),
                        /*all_probs=*/torch::Tensor(),
                        /*loss=*/torch::Tensor(),
                        /*src_batch_indices=*/torch::Tensor(),
                        /*all_hidden_states=*/torch::Tensor(),
                        /*update_remote_generate=*/false,
                        /*force_update_info=*/false,
                        /*prompt_logits=*/std::nullopt,
                        /*error_info=*/std::nullopt,
                        generation_prefill_cuda_graph_status,
                        row_output});
    }
    return absl::OkStatus();
}

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups,
                                                                            TensorHolder&       host_holder) const {
    return model_input_gatherer_->gather(stream_groups, host_holder);
}

absl::StatusOr<SamplerInputs> NormalBatchStreamProcessor::gatherSamplerInput(
    const StreamGroups& stream_groups, const GptModelInputs& model_inputs, const GptModelOutputs& model_output) const {
    return sampler_input_gatherer_->gather(stream_groups, model_inputs, model_output);
}

absl::StatusOr<torch::Tensor> NormalBatchStreamProcessor::gatherKvCacheKernelBlockId(
    const StreamGroups& stream_groups, const std::vector<std::string>& group_tags, TensorHolder& host_holder) const {
    return model_input_gatherer_->gatherKvCacheKernelBlockId(stream_groups, group_tags, host_holder);
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
