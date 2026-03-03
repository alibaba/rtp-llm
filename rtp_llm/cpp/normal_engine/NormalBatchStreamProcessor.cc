#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"

namespace rtp_llm {

NormalBatchStreamProcessor::NormalBatchStreamProcessor(
    const ModelConfig&                 model_config,
    const PDSepConfig&                 pd_sep_config,
    const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
    const CacheConfig&                 cache_config,
    bool                               warm_up) {
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
    model_input_gatherer_config_.layer_to_kv_cache_group_id = cache_config.layer_to_group_id;
    model_input_gatherer_config_.kv_cache_group_types       = cache_config.group_types;
    model_input_gatherer_config_.warm_up                    = warm_up;
    model_input_gatherer_config_.enable_detail_log          = profiling_debug_logging_config.enable_detail_log;

    model_input_gatherer_   = std::make_unique<NormalModelInputGatherer>(model_input_gatherer_config_);
    sampler_input_gatherer_ = std::make_unique<NormalSamplerInputGatherer>();
    output_dispatcher_      = std::make_unique<NormalOutputDispatcher>();
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups& stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    return output_dispatcher_->dispatch(stream_groups, merge_outputs);
}

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups) const {
    return model_input_gatherer_->gather(stream_groups);
}

absl::StatusOr<SamplerInputs> NormalBatchStreamProcessor::gatherSamplerInput(
    const StreamGroups& stream_groups, const GptModelInputs& model_inputs, const GptModelOutputs& model_output) const {
    return sampler_input_gatherer_->gather(stream_groups, model_inputs, model_output);
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
