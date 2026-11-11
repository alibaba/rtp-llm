#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"

#include <algorithm>
#include <unordered_set>

namespace rtp_llm {

namespace {

struct OwnedGroup {
    const std::string& tag;
    const GroupBase&   group;
};

void fillModelInputGathererMetadata(NormalModelInputGathererConfig& config,
                                    const std::vector<OwnedGroup>&  groups,
                                    size_t                          seq_size_per_block) {
    config.seq_size_per_block         = seq_size_per_block;
    config.kernel_seq_size_per_block  = seq_size_per_block;
    config.kernel_blocks_per_kv_block = 1;
    config.block_stride_bytes         = 0;
    config.scale_stride_bytes         = 0;
    config.kv_cache_group_tags.clear();
    config.kv_cache_group_types.clear();
    config.kv_cache_group_nums = groups.size();

    std::unordered_set<std::string> seen_tags;
    size_t                          max_kernel_blocks_per_kv_block = 1;
    for (const auto& owned : groups) {
        RTP_LLM_CHECK_WITH_INFO(!owned.tag.empty() && seen_tags.insert(owned.tag).second,
                                "cache group tags must be non-empty and unique: tag=%s",
                                owned.tag.c_str());
        config.kv_cache_group_tags.push_back(owned.tag);
        config.kv_cache_group_types.push_back(owned.group.policy.group_type);
        max_kernel_blocks_per_kv_block = std::max(max_kernel_blocks_per_kv_block, owned.group.kernelBlocksPerKvBlock());
    }
    config.kernel_blocks_per_kv_block = max_kernel_blocks_per_kv_block;

    if (!groups.empty()) {
        // Scalar layout metadata is meaningful only for a single group: the
        // model-local scalars describe that group, multi-group consumers read
        // their tagged specs and block tables instead.
        config.kernel_seq_size_per_block = 0;
        if (groups.size() == 1) {
            config.block_stride_bytes        = groups.front().group.kvBlockStrideBytes();
            config.scale_stride_bytes        = groups.front().group.kvScaleStrideBytes();
            config.kernel_seq_size_per_block = groups.front().group.kernelSeqSizePerBlock();
        }
    }
}

NormalModelInputGathererConfig
makeModelInputGathererConfig(const ModelConfig&                 model_config,
                             const PDSepConfig&                 pd_sep_config,
                             const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                             const CacheConfig&                 cache_config,
                             size_t                             module_id,
                             bool                               warm_up) {
    const auto&                    module_config = cache_config.moduleConfig(module_id);
    NormalModelInputGathererConfig config;
    config.num_layers              = model_config.num_layers;
    config.vocab_size              = model_config.vocab_size;
    config.input_vocab_size        = model_config.input_vocab_size;
    config.has_positional_encoding = model_config.has_positional_encoding;
    config.is_multimodal           = model_config.mm_model_config.is_multimodal;
    config.has_engram              = !model_config.attn_config.v41_kv_source_layer_ids.empty();
    config.mm_position_ids_style   = static_cast<PositionIdsStyle>(model_config.mm_model_config.mm_position_ids_style);
    config.position_id_len_factor  = model_config.attn_config.rope_config.index_factor;
    config.role_type               = pd_sep_config.role_type;
    config.decode_entrance         = pd_sep_config.decode_entrance;
    config.use_opaque_kv_cache_store = module_config.use_opaque_kv_cache_store;

    std::vector<OwnedGroup> groups;
    for (size_t owner_id = 0; owner_id < cache_config.moduleCount(); ++owner_id) {
        const auto& owner = cache_config.moduleConfig(owner_id);
        if (owner.groupNums() == 0) {
            continue;
        }
        for (const auto& group : owner.groups()) {
            groups.push_back({group.tag, group});
        }
    }
    fillModelInputGathererMetadata(config, groups, module_config.seq_size_per_block);

    config.warm_up                 = warm_up;
    config.enable_detail_log       = profiling_debug_logging_config.enable_detail_log;
    config.enable_model_inputs_log = profiling_debug_logging_config.enable_model_inputs_log;
    return config;
}

NormalModelInputGathererConfig
makeModelInputGathererConfig(const ModelConfig&                 model_config,
                             const PDSepConfig&                 pd_sep_config,
                             const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                             const CacheModuleConfig&           module_config,
                             size_t                             seq_size_per_block,
                             bool                               warm_up) {
    NormalModelInputGathererConfig config;
    config.num_layers              = model_config.num_layers;
    config.vocab_size              = model_config.vocab_size;
    config.input_vocab_size        = model_config.input_vocab_size;
    config.has_positional_encoding = model_config.has_positional_encoding;
    config.is_multimodal           = model_config.mm_model_config.is_multimodal;
    config.has_engram              = !model_config.attn_config.v41_kv_source_layer_ids.empty();
    config.mm_position_ids_style   = static_cast<PositionIdsStyle>(model_config.mm_model_config.mm_position_ids_style);
    config.position_id_len_factor  = model_config.attn_config.rope_config.index_factor;
    config.role_type               = pd_sep_config.role_type;
    config.decode_entrance         = pd_sep_config.decode_entrance;
    config.use_opaque_kv_cache_store = module_config.use_opaque_kv_cache_store;

    std::vector<OwnedGroup> groups;
    if (module_config.groupNums() > 0) {
        groups.reserve(module_config.groupNums());
        for (const auto& group : module_config.groups()) {
            groups.push_back({group.tag, group});
        }
    }
    fillModelInputGathererMetadata(config, groups, seq_size_per_block);

    config.warm_up                 = warm_up;
    config.enable_detail_log       = profiling_debug_logging_config.enable_detail_log;
    config.enable_model_inputs_log = profiling_debug_logging_config.enable_model_inputs_log;
    return config;
}

}  // namespace

NormalBatchStreamProcessor::NormalBatchStreamProcessor(
    const ModelConfig&                 model_config,
    const PDSepConfig&                 pd_sep_config,
    const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
    const CacheConfig&                 cache_config,
    size_t                             module_id,
    bool                               warm_up,
    int                                async_worker_count):
    NormalBatchStreamProcessor(
        model_config,
        pd_sep_config,
        profiling_debug_logging_config,
        makeModelInputGathererConfig(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, module_id, warm_up),
        warm_up,
        async_worker_count) {}

NormalBatchStreamProcessor::NormalBatchStreamProcessor(
    const ModelConfig&                 model_config,
    const PDSepConfig&                 pd_sep_config,
    const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
    const CacheModuleConfig&           cache_config,
    size_t                             seq_size_per_block,
    bool                               warm_up,
    int                                async_worker_count):
    NormalBatchStreamProcessor(
        model_config,
        pd_sep_config,
        profiling_debug_logging_config,
        makeModelInputGathererConfig(
            model_config, pd_sep_config, profiling_debug_logging_config, cache_config, seq_size_per_block, warm_up),
        warm_up,
        async_worker_count) {}

NormalBatchStreamProcessor::NormalBatchStreamProcessor(
    const ModelConfig&                 model_config,
    const PDSepConfig&                 pd_sep_config,
    const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
    NormalModelInputGathererConfig     model_input_gatherer_config,
    bool                               warm_up,
    int                                async_worker_count) {
    model_input_gatherer_config_ = std::move(model_input_gatherer_config);

    model_input_gatherer_   = std::make_unique<NormalModelInputGatherer>(model_input_gatherer_config_);
    sampler_input_gatherer_ = std::make_unique<NormalSamplerInputGatherer>();
    output_dispatcher_ = std::make_unique<NormalOutputDispatcher>(model_config.output_vocab_ids, async_worker_count);
}

absl::Status NormalBatchStreamProcessor::dispatch(const StreamGroups& stream_groups,
                                                  const MergedOutput& merge_outputs) const {
    return output_dispatcher_->dispatch(stream_groups, merge_outputs);
}

absl::Status NormalBatchStreamProcessor::dispatchPrefillOnly(
    const StreamGroups& stream_groups, GenerationPrefillCudaGraphStatus generation_prefill_cuda_graph_status) const {
    for (const auto& stream : stream_groups.allStreams()) {
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
                        generation_prefill_cuda_graph_status});
    }
    return absl::OkStatus();
}

absl::StatusOr<GptModelInputs> NormalBatchStreamProcessor::gatherModelInput(const StreamGroups& stream_groups,
                                                                            TensorHolder&       host_holder,
                                                                            bool                record_lengths) const {
    return model_input_gatherer_->gather(stream_groups, host_holder, record_lengths);
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
