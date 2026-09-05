#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "torch/all.h"
#include "rtp_llm/cpp/cache/CacheGroupTagOrder.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalInputUtils.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

namespace {

bool asyncDebugEnabled() {
    const char* env = std::getenv("RTP_LLM_ASYNC_DEBUG");
    return env != nullptr && std::string(env) == "1";
}

bool deviceInputEnabled() {
    const char* env = std::getenv("RTP_LLM_DEVICE_INPUT");
    return env != nullptr && std::string(env) == "1";
}

struct GatherModelInputContext {
    int               input_vocab_size;
    bool              need_cal_position_id;
    size_t            max_blocks_num;
    int*              merged_tokens;
    int*              input_lengths;
    int*              lm_output_indexes;
    int*              combo_position_ids;
    GroupBlockIdPair* kv_cache_update_mapping;
    int               batch_idx;
    int*              sequence_lengths;
    bool              has_multimodal_input;
    bool              has_mm_extra_input;
    size_t            total_decode_batch_size;
    int*              prefix_lengths;
    int*              prefix_lengths_host;
    int*              merged_text_mask;
    int*              mm_features_locs;
    int               token_idx;
    int               cum_output_seq_len;
    int               mm_feature_index;
};

enum class GatherContextMode {
    DECODE,
    CONTEXT
};

GatherModelInputContext createGatherContext(const NormalModelInputGathererConfig& config,
                                            GptModelInputs&                       model_input,
                                            const StreamGroups&                   stream_groups,
                                            GatherContextMode                     mode) {
    GatherModelInputContext ctx{};
    ctx.input_vocab_size =
        config.input_vocab_size ? static_cast<int>(config.input_vocab_size) : static_cast<int>(config.vocab_size);
    ctx.need_cal_position_id =
        (config.mm_position_ids_style != PositionIdsStyle::DEFAULT) || config.has_positional_encoding;
    ctx.max_blocks_num       = stream_groups.curBlocksNum();
    ctx.merged_tokens        = model_input.combo_tokens.data_ptr<int32_t>();
    ctx.input_lengths        = model_input.input_lengths.data_ptr<int32_t>();
    ctx.sequence_lengths     = model_input.sequence_lengths.data_ptr<int32_t>();
    ctx.combo_position_ids   = ctx.need_cal_position_id ? model_input.combo_position_ids.data_ptr<int32_t>() : nullptr;
    ctx.has_multimodal_input = config.is_multimodal && stream_groups.has_multimodal_input();
    ctx.has_mm_extra_input   = config.is_multimodal && stream_groups.hasMMExtraInput();
    ctx.prefix_lengths       = model_input.prefix_lengths.data_ptr<int32_t>();
    ctx.prefix_lengths_host  = nullptr;
    ctx.merged_text_mask     = ctx.has_multimodal_input ? model_input.text_tokens_mask.data_ptr<int32_t>() : nullptr;
    ctx.mm_features_locs     = ctx.has_multimodal_input ? model_input.mm_features_locs.data_ptr<int32_t>() : nullptr;

    size_t kv_cache_mapping_offset = 0;
    if (mode == GatherContextMode::DECODE) {
        ctx.batch_idx = 0;
    } else {
        ctx.total_decode_batch_size = stream_groups.totalDecodeBatchSize();
        ctx.batch_idx               = static_cast<int>(ctx.total_decode_batch_size);
        ctx.token_idx               = ctx.batch_idx;
        ctx.mm_feature_index        = 0;
        kv_cache_mapping_offset     = stream_groups.decodeBlockUpdateCopyNum();
    }
    ctx.kv_cache_update_mapping =
        model_input.kv_cache_update_mapping.defined() ?
            reinterpret_cast<GroupBlockIdPair*>(model_input.kv_cache_update_mapping.data_ptr())
                + kv_cache_mapping_offset :
            nullptr;

    if (ctx.merged_text_mask) {
        size_t current_tokens_size = stream_groups.modelExecuteTokenSize();
        std::fill(ctx.merged_text_mask, ctx.merged_text_mask + current_tokens_size, 1);
    }

    return ctx;
}

void copyKvCacheBlocksToModelInput(GptModelInputs&                                    model_input,
                                   const BatchKVCacheResource&                        kv_cache,
                                   int                                                stream_batch_idx,
                                   int                                                model_batch_idx,
                                   size_t                                             max_blocks_num,
                                   size_t                                             kernel_blocks_per_kv_block,
                                   const std::vector<std::string>&                    cache_group_tags,
                                   const std::unordered_map<std::string, CacheGroup>& cache_groups) {
    (void)cache_groups;
    if (!model_input.kv_cache_kernel_block_id.defined() || max_blocks_num == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_kernel_block_id.dim() == 3,
                            "hybrid kv_cache_kernel_block_id must be 3-D");
    RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_block_id.dim() == 3, "hybrid kv_cache_block_id must be 3-D");
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(kv_cache.groupNums()) == cache_group_tags.size(),
                            "request cache resource group count=%d does not match cache tag count=%zu",
                            kv_cache.groupNums(),
                            cache_group_tags.size());

    const size_t batch           = model_input.kv_cache_kernel_block_id.size(1);
    int32_t*     kernel_dst_base = model_input.kv_cache_kernel_block_id.data_ptr<int32_t>();
    int32_t*     store_dst_base  = model_input.kv_cache_block_id.data_ptr<int32_t>();

    const size_t kernel_row_capacity = max_blocks_num * kernel_blocks_per_kv_block;
    // Row group_index of both block tables belongs to cache_group_tags[group_index].
    // Fetching by tag here prevents request-resource record order from leaking out.
    for (size_t group_index = 0; group_index < cache_group_tags.size(); ++group_index) {
        const auto&    block_ids = kv_cache.cacheResource(stream_batch_idx).blockIds(cache_group_tags[group_index]);
        const auto&    physical_blocks = block_ids.blocks();
        const size_t   row_offset      = group_index * batch + static_cast<size_t>(model_batch_idx);
        int32_t* const kernel_dst      = kernel_dst_base + row_offset * kernel_row_capacity;
        int32_t* const store_dst       = store_dst_base + row_offset * max_blocks_num;
        std::fill(kernel_dst, kernel_dst + kernel_row_capacity, 0);
        std::fill(store_dst, store_dst + max_blocks_num, 0);
        RTP_LLM_CHECK_WITH_INFO(physical_blocks.size() <= max_blocks_num,
                                "physical block table overflow for tag=%s: blocks=%zu capacity=%zu",
                                cache_group_tags[group_index].c_str(),
                                physical_blocks.size(),
                                max_blocks_num);
        block_ids.writeKernelBlocks(kernel_dst, kernel_row_capacity);
        std::transform(physical_blocks.begin(), physical_blocks.end(), store_dst, toLegacyBlockIdx);
    }
}

void gatherMultimodalInputsForContextBatch(const GenerateStreamPtr&    stream,
                                           GatherModelInputContext&    ctx,
                                           std::vector<torch::Tensor>& gathered_mm_features,
                                           std::vector<torch::Tensor>& gathered_mm_extra_input,
                                           TensorHolder&               host_holder) {
    if (!ctx.has_multimodal_input) {
        return;
    }
    std::vector<torch::Tensor> mm_features = stream->multimodalFeatures();
    torch::Tensor              mm_locs     = stream->multimodalLocations();
    if (!mm_locs.defined()) {
        return;
    }
    auto mm_extra_input = stream->multimodalExtraInput();
    RTP_LLM_CHECK_WITH_INFO(mm_locs.numel() == static_cast<int64_t>(mm_features.size()),
                            "mm_locs count %ld != mm_features count %zu for stream %ld",
                            mm_locs.numel(),
                            mm_features.size(),
                            stream->streamId());
    RTP_LLM_CHECK_WITH_INFO(mm_extra_input.empty() || mm_extra_input.size() == mm_features.size(),
                            "mm_extra_input count %zu != mm_features count %zu for stream %ld",
                            mm_extra_input.size(),
                            mm_features.size(),
                            stream->streamId());

    auto*     mm_locs_data = mm_locs.data_ptr<int32_t>();
    const int reuse_length = stream->reuseLength();
    if (mm_locs.numel() > 1) {
        RTP_LLM_CHECK_WITH_INFO(std::is_sorted(mm_locs_data, mm_locs_data + mm_locs.numel()),
                                "mm_locs must be sorted in ascending order for reuse handling");
    }
    for (int i = 0; i < static_cast<int>(mm_features.size()); ++i) {
        const auto&   mm_feature  = mm_features[i];
        const int64_t feature_len = mm_feature.size(0);
        const int64_t feature_loc = mm_locs_data[i];
        const int64_t feature_end = feature_loc + feature_len;
        if (reuse_length >= feature_end) {
            continue;
        }

        // ViT still runs on and caches the complete image. Only the rows already
        // represented by the reused KV prefix are omitted from this model input.
        // This gatherer owns prefix slicing; downstream injectors receive sliced
        // features with non-negative local locations only.
        const int64_t token_offset    = std::max<int64_t>(reuse_length - feature_loc, 0);
        auto          current_feature = mm_feature.slice(0, token_offset, feature_len).contiguous();
        if (!current_feature.is_cuda()) {
            host_holder.hold_host(current_feature);
            gathered_mm_features.emplace_back(current_feature.to(torch::kCUDA, /*non_blocking=*/true));
        } else {
            gathered_mm_features.emplace_back(std::move(current_feature));
        }

        ctx.mm_features_locs[ctx.mm_feature_index] =
            ctx.token_idx + static_cast<int>(std::max<int64_t>(feature_loc - reuse_length, 0));
        ctx.mm_feature_index++;

        if (!mm_extra_input.empty()) {
            auto current_extra_input =
                sliceMultimodalExtraInput(mm_extra_input[i], mm_feature, token_offset, feature_len);
            if (!current_extra_input.is_cuda()) {
                host_holder.hold_host(current_extra_input);
                gathered_mm_extra_input.emplace_back(current_extra_input.to(torch::kCUDA, /*non_blocking=*/true));
            } else {
                gathered_mm_extra_input.emplace_back(std::move(current_extra_input));
            }
        }
    }
    auto text_token_mask = stream->textTokensMask();
    memcpy(ctx.merged_text_mask + ctx.token_idx, text_token_mask.data(), text_token_mask.size() * sizeof(int));
}

// The first kv_cache_update_mapping column is a group_index in canonical
// sorted-tag order. NormalExecutor resolves it back to a tag before touching
// the cache layer.
void addCacheUpdateCopy(GatherModelInputContext&              ctx,
                        const std::vector<TaggedBlockIdPair>& update_mapping,
                        const std::vector<std::string>&       cache_group_tags) {
    if (!ctx.kv_cache_update_mapping) {
        return;
    }
    for (const auto& mapping : update_mapping) {
        const auto group_index         = groupIndexForTag(cache_group_tags, mapping.tag, "cache update mapping");
        *ctx.kv_cache_update_mapping++ = GroupBlockIdPair{static_cast<int32_t>(group_index), mapping.src, mapping.dst};
    }
}

torch::Tensor buildLmOutputIndexesOnCuda(const GptModelInputs& model_input, const StreamGroups& stream_groups) {
    const auto total_batch_size         = static_cast<int64_t>(stream_groups.totalModelBatchSize());
    const auto total_decode_batch_size  = static_cast<int64_t>(stream_groups.totalDecodeBatchSize());
    const auto total_context_batch_size = total_batch_size - total_decode_batch_size;
    auto       cuda_i32                 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    if (total_batch_size == 0) {
        return torch::empty({0}, cuda_i32);
    }

    std::vector<torch::Tensor> parts;
    parts.reserve(2);

    if (total_decode_batch_size > 0) {
        parts.push_back(torch::arange(0, total_decode_batch_size, cuda_i32));
    }

    if (total_context_batch_size > 0) {
        auto context_input_lengths =
            model_input.input_lengths
                .narrow(/*dim=*/0, /*start=*/total_decode_batch_size, /*length=*/total_context_batch_size)
                .to(cuda_i32);
        auto context_indexes = context_input_lengths.cumsum(/*dim=*/0).to(torch::kInt32)
                               + static_cast<int64_t>(total_decode_batch_size - 1);
        parts.push_back(context_indexes);
    }

    if (parts.size() == 1) {
        return parts.front().contiguous();
    }
    return torch::cat(parts, /*dim=*/0).contiguous();
}

torch::Tensor buildLmOutputIndexesOnHost(const GptModelInputs& model_input, const StreamGroups& stream_groups) {
    const auto total_batch_size         = static_cast<int64_t>(stream_groups.totalModelBatchSize());
    const auto total_decode_batch_size  = static_cast<int64_t>(stream_groups.totalDecodeBatchSize());
    const auto total_context_batch_size = total_batch_size - total_decode_batch_size;
    auto       indexes = torch::empty({total_batch_size}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto*      dst     = indexes.data_ptr<int32_t>();
    for (int64_t i = 0; i < total_decode_batch_size; ++i) {
        dst[i] = static_cast<int32_t>(i);
    }
    if (total_context_batch_size > 0) {
        auto        input_lengths = model_input.input_lengths.is_cuda() ? model_input.input_lengths.cpu().contiguous() :
                                                                          model_input.input_lengths.contiguous();
        const auto* lengths       = input_lengths.data_ptr<int32_t>();
        int32_t     offset        = static_cast<int32_t>(total_decode_batch_size);
        for (int64_t i = 0; i < total_context_batch_size; ++i) {
            offset += lengths[total_decode_batch_size + i];
            dst[total_decode_batch_size + i] = offset - 1;
        }
    }
    return indexes;
}

torch::Tensor publishInt32ToCuda(const torch::Tensor& tensor, TensorHolder& host_holder) {
    if (!tensor.defined()) {
        return tensor;
    }
    auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    if (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) {
        return tensor;
    }
    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), cuda_i32);
    }
    host_holder.hold_host(tensor);
    return tensor.to(cuda_i32, /*non_blocking=*/true);
}

void publishModelInputCoreTensorsToCuda(GptModelInputs& model_input, TensorHolder& host_holder) {
    // TODO(async): stream state is still gathered through CPU pointers above.
    // Publish only device tensors at the model boundary.
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.publish_core_tensors_to_cuda");
    model_input.combo_tokens     = publishInt32ToCuda(model_input.combo_tokens, host_holder);
    model_input.input_lengths    = publishInt32ToCuda(model_input.input_lengths, host_holder);
    model_input.sequence_lengths = publishInt32ToCuda(model_input.sequence_lengths, host_holder);
    model_input.prefix_lengths   = publishInt32ToCuda(model_input.prefix_lengths, host_holder);
    // Migrate the 3-D KV kernel block id tensor with one H2D, replacing the
    // former per-group tensorHoldHostAndToCuda copies in PyWrappedModel.
    model_input.kv_cache_kernel_block_id = publishInt32ToCuda(model_input.kv_cache_kernel_block_id, host_holder);
}

}  // anonymous namespace

namespace {

std::vector<std::string> buildCacheGroupTags(const std::unordered_map<std::string, CacheGroup>& groups) {
    std::vector<std::string> tags;
    tags.reserve(groups.size());
    for (const auto& [tag, group] : groups) {
        (void)group;
        tags.push_back(tag);
    }
    return sortedCacheGroupTags(tags, "model input cache");
}

}  // namespace

NormalModelInputGatherer::NormalModelInputGatherer(const NormalModelInputGathererConfig& config):
    config_(config), cache_group_tags_(buildCacheGroupTags(config.kv_cache_groups)) {}

GptModelInputs NormalModelInputGatherer::allocateModelInputBuffers(const StreamGroups& stream_groups) const {
    const size_t current_tokens_size      = stream_groups.modelExecuteTokenSize();
    const size_t total_batch_size         = stream_groups.totalModelBatchSize();
    const size_t total_decode_batch_size  = stream_groups.totalDecodeBatchSize();
    const size_t total_context_batch_size = stream_groups.totalContextBatchSize();
    const size_t total_block_copy_num     = stream_groups.totalBlockUpdateCopyNum();
    const size_t max_blocks_num           = stream_groups.curBlocksNum();
    const size_t max_cache_keys_num       = std::max(max_blocks_num, stream_groups.maxCacheKeysNum());
    const size_t multimodal_features_len  = stream_groups.mmFeaturesLen();
    const bool   has_multimodal_input     = config_.is_multimodal && stream_groups.has_multimodal_input();
    const bool   need_cal_position_id =
        (config_.mm_position_ids_style != PositionIdsStyle::DEFAULT) || config_.has_positional_encoding;

    static const auto pinned_i32  = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    static const auto pinned_i64  = torch::TensorOptions(torch::kInt64).pinned_memory(true);
    static const auto pinned_bool = torch::TensorOptions(torch::kBool).pinned_memory(true);

    GptModelInputs model_input;
    model_input.combo_tokens          = torch::empty({(int64_t)current_tokens_size}, pinned_i32);
    model_input.input_lengths         = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.sequence_lengths      = torch::empty({(int64_t)total_decode_batch_size}, pinned_i32);
    model_input.prefix_lengths        = torch::empty({(int64_t)total_context_batch_size}, pinned_i32);
    model_input.request_id            = torch::empty({(int64_t)total_context_batch_size}, pinned_i64);
    model_input.request_pd_separation = torch::empty({(int64_t)total_context_batch_size}, pinned_bool);

    if (max_blocks_num) {
        const int64_t group_num = (int64_t)cache_group_tags_.size();
        // Every group dimension below is ordered by cache_group_tags_; the tag
        // list travels with the input so the log and the consuming model can name
        // each row.
        model_input.kv_cache_group_tags      = cache_group_tags_;
        model_input.kv_cache_kernel_block_id = torch::zeros(
            {group_num, (int64_t)total_batch_size, (int64_t)(max_blocks_num * config_.max_kernel_blocks_per_kv_block)},
            pinned_i32);
        model_input.kv_cache_block_id =
            torch::zeros({group_num, (int64_t)total_batch_size, (int64_t)max_blocks_num}, pinned_i32);
        model_input.kv_cache_group_types    = torch::empty({group_num}, pinned_i32);
        model_input.kv_cache_update_mapping = torch::empty({(int64_t)total_block_copy_num, 3}, pinned_i32);
        // CP-sharded group block tables can be narrower than the global cache-key
        // namespace. Keep cache_keys independently sized so PD writer and reader
        // derive identical keys from the complete token sequence.
        model_input.cache_keys =
            torch::zeros({(int64_t)total_context_batch_size, (int64_t)max_cache_keys_num}, pinned_i64);
    }

    if (need_cal_position_id) {
        model_input.combo_position_ids =
            torch::empty({(int64_t)(current_tokens_size * config_.position_id_len_factor)}, pinned_i32);
    }
    if (has_multimodal_input) {
        model_input.text_tokens_mask = torch::empty({(int64_t)current_tokens_size}, pinned_i32);
        model_input.mm_features_locs = torch::empty({(int64_t)multimodal_features_len}, pinned_i32);
    }

    model_input.pd_separation             = config_.role_type == RoleType::PREFILL;
    model_input.warmup                    = config_.warm_up;
    model_input.decode_entrance           = config_.decode_entrance;
    model_input.use_opaque_kv_cache_store = config_.use_opaque_kv_cache_store;
    model_input.is_fake_stream            = stream_groups.isFakeStream();

    return model_input;
}

void NormalModelInputGatherer::initializeKvCacheMetadata(GptModelInputs& model_input) const {
    if (!model_input.kv_cache_group_types.defined()) {
        return;
    }
    // kv_cache_group_types is parallel to the block tables, so it uses the same
    // canonical sorted-tag order.
    auto* dst = model_input.kv_cache_group_types.data_ptr<int32_t>();
    for (size_t idx = 0; idx < cache_group_tags_.size(); ++idx) {
        const auto& group = config_.kv_cache_groups.at(cache_group_tags_[idx]);
        dst[idx]          = static_cast<int32_t>(group.policy.group_type);
    }
}

absl::Status NormalModelInputGatherer::processDecodeStreams(GptModelInputs&     model_input,
                                                            const StreamGroups& stream_groups) const {
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.process_decode_streams");
    auto ctx = createGatherContext(config_, model_input, stream_groups, GatherContextMode::DECODE);

    const char* device_input_env        = std::getenv("RTP_LLM_DEVICE_INPUT");
    bool        use_normal_device_state = device_input_env != nullptr && std::string(device_input_env) == "1"
                                   && stream_groups.totalContextBatchSize() == 0
                                   && stream_groups.totalDecodeBatchSize() > 0 && !ctx.need_cal_position_id;
    if (use_normal_device_state) {
        for (const auto& stream : stream_groups.decodeStreams()) {
            const auto& state = stream->getNormalAsyncDeviceState();
            if (stream->currentBatchSize() != 1 || !state.last_sample_token_gpu.defined()
                || !state.last_sample_token_gpu.is_cuda() || !state.next_seq_len_gpu.defined()
                || !state.next_seq_len_gpu.is_cuda()) {
                use_normal_device_state = false;
                break;
            }
        }
    }
    std::vector<torch::Tensor> normal_combo_tokens_gpu;
    std::vector<torch::Tensor> normal_sequence_lengths_gpu;
    if (use_normal_device_state) {
        normal_combo_tokens_gpu.reserve(stream_groups.totalDecodeBatchSize());
        normal_sequence_lengths_gpu.reserve(stream_groups.totalDecodeBatchSize());
    }

    for (const auto& stream : stream_groups.decodeStreams()) {
        model_input.need_all_logits        = model_input.need_all_logits || stream->calculateLoss();
        model_input.need_all_hidden_states = model_input.need_all_hidden_states || stream->needReturnHiddenStates();
        auto  current_batch_size           = stream->currentBatchSize();
        auto& kv_cache                     = *stream->kvCachePtr();
        RTP_LLM_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        RTP_LLM_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());

        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());
            if (use_normal_device_state) {
                const auto&             state = stream->getNormalAsyncDeviceState();
                static std::atomic<int> debug_log_budget{200};
                if (asyncDebugEnabled() && stream->hasPendingAsyncBookkeeping()
                    && debug_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
                    RTP_LLM_LOG_WARNING("[async-debug] gather decode with pending bookkeeping: stream=%ld pd_sep=%d "
                                        "status=%s cpu_seq=%d state_next_real=%d cur_blocks=%zu batch_idx=%d",
                                        stream->streamId(),
                                        stream->queryPdSep(),
                                        StreamStateToString(stream->getStatus()).c_str(),
                                        stream->seqLength(),
                                        state.next_real_seq_len,
                                        stream->curBlocksNum(),
                                        ctx.batch_idx);
                }
                normal_combo_tokens_gpu.push_back(state.last_sample_token_gpu.reshape({1}));
                normal_sequence_lengths_gpu.push_back((state.next_seq_len_gpu - 1).to(torch::kInt32).reshape({1}));
                ctx.input_lengths[ctx.batch_idx] = stream->inputLength();
            } else {
                auto currentTokens = stream->currentExecuteTokens(i);
                if (currentTokens[0] >= ctx.input_vocab_size) {
                    std::ostringstream error_msg;
                    error_msg << "stream [" << stream->streamId() << "] token_id " << currentTokens[0]
                              << " exceed vocab_size " << ctx.input_vocab_size;
                    return absl::InvalidArgumentError(error_msg.str());
                }
                ctx.merged_tokens[ctx.batch_idx]    = currentTokens[0];
                ctx.input_lengths[ctx.batch_idx]    = stream->inputLength();
                ctx.sequence_lengths[ctx.batch_idx] = stream->seqLength() - 1;
                if (ctx.need_cal_position_id) {
                    stream->generateNextPositionId(ctx.combo_position_ids
                                                   + ctx.batch_idx * config_.position_id_len_factor);
                }
            }
            copyKvCacheBlocksToModelInput(model_input,
                                          kv_cache,
                                          i,
                                          ctx.batch_idx,
                                          ctx.max_blocks_num,
                                          config_.max_kernel_blocks_per_kv_block,
                                          cache_group_tags_,
                                          config_.kv_cache_groups);
            ctx.batch_idx += 1;
        }
        addCacheUpdateCopy(ctx, stream->streamCacheResource().getKVBlockUpdateMapping(), cache_group_tags_);
        stream->step();
    }

    if (use_normal_device_state) {
        model_input.combo_tokens     = torch::cat(normal_combo_tokens_gpu, 0).to(torch::kInt32);
        model_input.sequence_lengths = torch::cat(normal_sequence_lengths_gpu, 0).to(torch::kInt32);
    }
    return absl::OkStatus();
}

absl::Status NormalModelInputGatherer::processContextStreams(GptModelInputs&     model_input,
                                                             const StreamGroups& stream_groups,
                                                             TensorHolder&       host_holder) const {
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.process_context_streams");
    std::vector<torch::Tensor> gathered_mm_features;
    std::vector<torch::Tensor> gathered_mm_extra_input;
    const auto                 context_batch_size = static_cast<int64_t>(stream_groups.totalContextBatchSize());
    auto                       prefix_lengths_host =
        torch::empty({context_batch_size}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto ctx                = createGatherContext(config_, model_input, stream_groups, GatherContextMode::CONTEXT);
    ctx.prefix_lengths_host = prefix_lengths_host.data_ptr<int32_t>();

    for (const auto& stream : stream_groups.contextStreams()) {
        model_input.need_all_logits =
            model_input.need_all_logits || stream->calculateLoss() || stream->returnPromptLogits();
        model_input.need_all_hidden_states = model_input.need_all_hidden_states || stream->needReturnHiddenStates();
        auto  current_batch_size           = stream->currentBatchSize();
        auto& kv_cache                     = *stream->kvCachePtr();
        if (config_.enable_detail_log) {
            RTP_LLM_LOG_DEBUG("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_DEBUG("context stream: %s", stream->debugString().c_str());
        } else {
            RTP_LLM_LOG_TRACE("context kv_cache: %s", kv_cache.debugString().c_str());
            RTP_LLM_LOG_TRACE("context stream: %s", stream->debugString().c_str());
        }

        for (auto i = 0; i < current_batch_size; ++i) {
            const auto prefill_batch_idx = ctx.batch_idx - ctx.total_decode_batch_size;
            model_input.trace_ids.push_back(stream->traceId());
            auto input_tokens = stream->currentExecuteTokens(i);
            auto input_masks  = stream->textTokensMask();
            memcpy(ctx.merged_tokens + ctx.token_idx, input_tokens.data(), input_tokens.size() * sizeof(int));

            for (int index = 0; index < (int)input_tokens.size(); ++index) {
                if (input_tokens[index] >= ctx.input_vocab_size
                    && (index >= (int)input_masks.size() || input_masks[index])) {
                    std::ostringstream error_msg;
                    error_msg << "stream [" << stream->streamId() << "] token_id " << input_tokens[index]
                              << " exceed vocab_size " << ctx.input_vocab_size;
                    return absl::InvalidArgumentError(error_msg.str());
                }
            }

            ctx.input_lengths[ctx.batch_idx]           = input_tokens.size();
            ctx.prefix_lengths_host[prefill_batch_idx] = stream->prefixLength();
            gatherMultimodalInputsForContextBatch(
                stream, ctx, gathered_mm_features, gathered_mm_extra_input, host_holder);

            if (ctx.need_cal_position_id) {
                auto context_pos_ids = stream->generateContextPositionIds();
                int  reuse_offset    = stream->reuseLength() * config_.position_id_len_factor;
                memcpy(ctx.combo_position_ids + ctx.token_idx * config_.position_id_len_factor,
                       context_pos_ids.data_ptr<int>() + reuse_offset,
                       (context_pos_ids.numel() - reuse_offset) * sizeof(int));
            }

            copyKvCacheBlocksToModelInput(model_input,
                                          kv_cache,
                                          i,
                                          ctx.batch_idx,
                                          ctx.max_blocks_num,
                                          config_.max_kernel_blocks_per_kv_block,
                                          cache_group_tags_,
                                          config_.kv_cache_groups);

            if (ctx.max_blocks_num && config_.role_type == RoleType::PREFILL && stream->hasCacheKeys()) {
                RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(stream->cacheKeys(i).size())
                                            <= model_input.cache_keys.size(1),
                                        "cache_keys overflow: stream keys=%zu tensor width=%ld",
                                        stream->cacheKeys(i).size(),
                                        model_input.cache_keys.size(1));
                std::memcpy(model_input.cache_keys.data_ptr<int64_t>()
                                + prefill_batch_idx * model_input.cache_keys.size(1),
                            stream->cacheKeys(i).data(),
                            stream->cacheKeys(i).size() * sizeof(int64_t));
            }

            *(model_input.request_id.data_ptr<int64_t>() + prefill_batch_idx) = stream->streamId();
            *(reinterpret_cast<bool*>(model_input.request_pd_separation.data_ptr()) + prefill_batch_idx) =
                stream->queryPdSep();

            ctx.batch_idx += 1;
            ctx.token_idx += input_tokens.size();
        }

        addCacheUpdateCopy(ctx, stream->streamCacheResource().getKVBlockUpdateMapping(), cache_group_tags_);
        stream->step();
    }

    if (config_.is_multimodal && !gathered_mm_features.empty()) {
        model_input.multimodal_features = std::move(gathered_mm_features);
    }
    if (ctx.has_mm_extra_input && gathered_mm_extra_input.size() > 0) {
        model_input.mm_extra_input = std::move(gathered_mm_extra_input);
    }
    // mm_features_locs was over-allocated using raw stream->multimodalFeaturesLength();
    // slice down to the actual count written (post-reuse) so Python consumers see the
    // correct tensor size.
    if (ctx.has_multimodal_input && model_input.mm_features_locs.defined()
        && ctx.mm_feature_index < model_input.mm_features_locs.numel()) {
        model_input.mm_features_locs = model_input.mm_features_locs.slice(0, 0, ctx.mm_feature_index);
    }
    model_input.prefix_lengths =
        deviceInputEnabled() ? publishInt32ToCuda(prefix_lengths_host, host_holder) : prefix_lengths_host;
    return absl::OkStatus();
}

void NormalModelInputGatherer::gatherKvCacheKernelBlockIdToHost(const StreamGroups& stream_groups,
                                                                torch::Tensor&      host_tensor) const {
    const size_t total_batch_size = stream_groups.totalModelBatchSize();
    RTP_LLM_CHECK_WITH_INFO(host_tensor.device().is_cpu() && host_tensor.scalar_type() == torch::kInt32
                                && host_tensor.dim() == 3,
                            "kernel block staging tensor must be a 3-D CPU int32 tensor");
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(host_tensor.size(0)) == cache_group_tags_.size()
                                && static_cast<size_t>(host_tensor.size(1)) == total_batch_size,
                            "kernel block staging tensor shape does not match tags/batch");
    const size_t per_batch_stride = static_cast<size_t>(host_tensor.size(2));

    size_t     validated_batch_size = 0;
    const auto validate_one_stream  = [&](const GenerateStreamPtr& stream) {
        const auto& kv_cache           = *stream->kvCachePtr();
        const auto  current_batch_size = stream->currentBatchSize();
        RTP_LLM_CHECK_WITH_INFO(current_batch_size >= 0 && current_batch_size == kv_cache.batchSize(),
                                "stream batch size=%d does not match cache batch size=%d",
                                current_batch_size,
                                kv_cache.batchSize());
        for (int i = 0; i < current_batch_size; ++i) {
            const auto& rows_by_group = kv_cache.blocksByGroup(i);
            RTP_LLM_CHECK_WITH_INFO(rows_by_group.size() == cache_group_tags_.size(),
                                    "request cache resource tag count=%zu does not match expected tag count=%zu",
                                    rows_by_group.size(),
                                    cache_group_tags_.size());
            for (const auto& tag : cache_group_tags_) {
                const auto row_it = rows_by_group.find(tag);
                RTP_LLM_CHECK_WITH_INFO(
                    row_it != rows_by_group.end(), "request cache resource missing expected tag=%s", tag.c_str());
                RTP_LLM_CHECK_WITH_INFO(row_it->second.kernelBlocksNum() <= per_batch_stride,
                                        "kernel block refresh row overflow for tag=%s: blocks=%zu capacity=%zu",
                                        tag.c_str(),
                                        row_it->second.kernelBlocksNum(),
                                        per_batch_stride);
            }
            ++validated_batch_size;
        }
    };
    for (const auto& stream : stream_groups.decodeStreams())
        validate_one_stream(stream);
    for (const auto& stream : stream_groups.contextStreams())
        validate_one_stream(stream);
    RTP_LLM_CHECK_WITH_INFO(validated_batch_size == total_batch_size,
                            "validated kernel block batch=%zu does not match expected batch=%zu",
                            validated_batch_size,
                            total_batch_size);
    host_tensor.zero_();
    size_t     staged_batch_size = 0;
    const auto stage_one_stream  = [&](const GenerateStreamPtr& stream) {
        const auto& kv_cache           = *stream->kvCachePtr();
        const auto  current_batch_size = stream->currentBatchSize();
        RTP_LLM_CHECK_WITH_INFO(current_batch_size >= 0 && current_batch_size == kv_cache.batchSize(),
                                "stream batch size=%d does not match cache batch size=%d",
                                current_batch_size,
                                kv_cache.batchSize());
        for (int i = 0; i < current_batch_size; ++i) {
            const auto& rows_by_group = kv_cache.blocksByGroup(i);
            RTP_LLM_CHECK_WITH_INFO(rows_by_group.size() == cache_group_tags_.size(),
                                    "request cache resource tag count=%zu does not match expected tag count=%zu",
                                    rows_by_group.size(),
                                    cache_group_tags_.size());
            for (size_t group_index = 0; group_index < cache_group_tags_.size(); ++group_index) {
                const auto& tag = cache_group_tags_[group_index];
                RTP_LLM_CHECK_WITH_INFO(rows_by_group.find(tag) != rows_by_group.end(),
                                        "request cache resource missing expected tag=%s",
                                        tag.c_str());
                int32_t* dst = host_tensor.data_ptr<int32_t>()
                               + (group_index * total_batch_size + staged_batch_size) * per_batch_stride;
                std::fill(dst, dst + per_batch_stride, 0);
                const size_t kernel_blocks_num = rows_by_group.at(tag).writeKernelBlocks(dst, per_batch_stride);
                RTP_LLM_CHECK_WITH_INFO(kernel_blocks_num <= per_batch_stride,
                                        "kernel block refresh row overflow for tag=%s: blocks=%zu capacity=%zu",
                                        tag.c_str(),
                                        kernel_blocks_num,
                                        per_batch_stride);
            }
            ++staged_batch_size;
        }
    };
    for (const auto& stream : stream_groups.decodeStreams()) {
        stage_one_stream(stream);
    }
    for (const auto& stream : stream_groups.contextStreams()) {
        stage_one_stream(stream);
    }
    RTP_LLM_CHECK_WITH_INFO(staged_batch_size == total_batch_size,
                            "staged kernel block batch=%zu does not match expected batch=%zu",
                            staged_batch_size,
                            total_batch_size);
}

absl::StatusOr<torch::Tensor> NormalModelInputGatherer::gatherKvCacheKernelBlockId(const StreamGroups& stream_groups,
                                                                                   TensorHolder& host_holder) const {
    const size_t total_batch_size = stream_groups.totalModelBatchSize();
    const size_t max_blocks_num   = stream_groups.curBlocksNum();
    if (max_blocks_num == 0 || total_batch_size == 0) {
        return torch::Tensor{};
    }

    static const auto pinned_i32  = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    auto              host_tensor = torch::zeros({(int64_t)cache_group_tags_.size(),
                                                  (int64_t)total_batch_size,
                                                  (int64_t)(max_blocks_num * config_.max_kernel_blocks_per_kv_block)},
                                    pinned_i32);
    gatherKvCacheKernelBlockIdToHost(stream_groups, host_tensor);
    return publishInt32ToCuda(host_tensor, host_holder);
}

absl::StatusOr<GptModelInputs> NormalModelInputGatherer::gather(const StreamGroups& stream_groups,
                                                                TensorHolder&       host_holder) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_LOG_DEBUG("context_streams size = %d, decode_streams size = %d",
                      stream_groups.contextStreams().size(),
                      stream_groups.decodeStreams().size());
    auto model_input = allocateModelInputBuffers(stream_groups);
    initializeKvCacheMetadata(model_input);
    RETURN_IF_STATUS_ERROR(processDecodeStreams(model_input, stream_groups));
    RETURN_IF_STATUS_ERROR(processContextStreams(model_input, stream_groups, host_holder));
    // No host mirrors are kept for ModelInputsLogger: it snapshots every tensor
    // in place (device-side clone + per-device c10::Event) and only pays the D2H
    // on its own worker thread, so it reads post-publish CUDA members directly.
    if (deviceInputEnabled()) {
        publishModelInputCoreTensorsToCuda(model_input, host_holder);
        model_input.lm_output_indexes = buildLmOutputIndexesOnCuda(model_input, stream_groups);
    } else {
        model_input.lm_output_indexes = buildLmOutputIndexesOnHost(model_input, stream_groups);
    }
    return model_input;
}

}  // namespace rtp_llm
