#include <algorithm>
#include <cstring>
#include <sstream>
#include "torch/all.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

namespace {

struct GatherModelInputContext {
    int          input_vocab_size;
    bool         need_cal_position_id;
    size_t       max_blocks_num;
    int*         merged_tokens;
    int*         input_lengths;
    int*         lm_output_indexes;
    int*         lm_output_lengths;
    int*         combo_position_ids;
    BlockIdPair* kv_cache_update_mapping;
    int          batch_idx;
    int*         sequence_lengths;
    bool         has_multimodal_input;
    bool         has_mm_extra_input;
    size_t       total_decode_batch_size;
    int*         prefix_lengths;
    int*         merged_text_mask;
    int*         mm_features_locs;
    int          token_idx;
    int          cum_output_seq_len;
    int          mm_feature_index;
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
    ctx.lm_output_indexes    = model_input.lm_output_indexes.data_ptr<int32_t>();
    ctx.lm_output_lengths    = model_input.lm_output_lengths.data_ptr<int32_t>();
    ctx.combo_position_ids   = ctx.need_cal_position_id ? model_input.combo_position_ids.data_ptr<int32_t>() : nullptr;
    ctx.has_multimodal_input = config.is_multimodal && stream_groups.has_multimodal_input();
    ctx.has_mm_extra_input   = config.is_multimodal && stream_groups.hasMMExtraInput();
    ctx.prefix_lengths       = model_input.prefix_lengths.data_ptr<int32_t>();
    ctx.merged_text_mask     = ctx.has_multimodal_input ? model_input.text_tokens_mask.data_ptr<int32_t>() : nullptr;
    ctx.mm_features_locs     = ctx.has_multimodal_input ? model_input.mm_features_locs.data_ptr<int32_t>() : nullptr;

    size_t kv_cache_mapping_offset = 0;
    if (mode == GatherContextMode::DECODE) {
        ctx.batch_idx = 0;
    } else {
        ctx.total_decode_batch_size = stream_groups.totalDecodeBatchSize();
        ctx.batch_idx               = static_cast<int>(ctx.total_decode_batch_size);
        ctx.token_idx               = ctx.batch_idx;
        ctx.cum_output_seq_len      = ctx.batch_idx;
        ctx.mm_feature_index        = 0;
        kv_cache_mapping_offset     = stream_groups.decodeBlockUpdateCopyNum();
    }
    ctx.kv_cache_update_mapping =
        model_input.kv_cache_update_mapping.defined() ?
            reinterpret_cast<BlockIdPair*>(model_input.kv_cache_update_mapping.data_ptr()) + kv_cache_mapping_offset :
            nullptr;

    if (ctx.merged_text_mask) {
        size_t current_tokens_size = stream_groups.modelExecuteTokenSize();
        std::fill(ctx.merged_text_mask, ctx.merged_text_mask + current_tokens_size, 1);
    }

    return ctx;
}

void copyKvCacheBlocksToModelInput(GptModelInputs&             model_input,
                                   const BatchKVCacheResource& kv_cache,
                                   int                         stream_batch_idx,
                                   int                         model_batch_idx,
                                   size_t                      max_blocks_num,
                                   size_t                      kernel_blocks_per_kv_block) {
    if (!model_input.kv_cache_kernel_block_id.defined() || max_blocks_num == 0) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_kernel_block_id.dim() == 3,
                            "hybrid kv_cache_kernel_block_id must be 3-D");
    RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_block_id.dim() == 3, "hybrid kv_cache_block_id must be 3-D");

    const size_t batch           = model_input.kv_cache_kernel_block_id.size(1);
    int32_t*     kernel_dst_base = model_input.kv_cache_kernel_block_id.data_ptr<int32_t>();
    int32_t*     store_dst_base  = model_input.kv_cache_block_id.data_ptr<int32_t>();

    for (int gid = 0; gid < kv_cache.groupNums(); ++gid) {
        auto&    kernel_blocks = kv_cache.kernelBlocks(stream_batch_idx, gid);
        int32_t* kernel_dst    = kernel_dst_base
                              + (static_cast<size_t>(gid) * batch + static_cast<size_t>(model_batch_idx))
                                    * max_blocks_num * kernel_blocks_per_kv_block;
        std::memcpy(kernel_dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));

        auto&    physical_blocks = kv_cache.blocks(stream_batch_idx, gid);
        int32_t* store_dst =
            store_dst_base + (static_cast<size_t>(gid) * batch + static_cast<size_t>(model_batch_idx)) * max_blocks_num;
        std::memcpy(store_dst, physical_blocks.data(), physical_blocks.size() * sizeof(int32_t));
    }
}

// Count of leading multimodal images whose token spans [loc, loc + feature_len) are
// fully covered by reuse_length. Partially-cached images do NOT count (they must be
// recomputed). The rule lives here (not on GenerateStream) so the stream stays a pure
// data holder.
int computeReusedMultimodalCount(const GenerateStreamPtr& stream) {
    auto mm_features = stream->multimodalFeatures();
    auto mm_locs     = stream->multimodalLocations();
    if (!mm_locs.defined() || mm_features.empty()) {
        return 0;
    }
    const int reuse_length = stream->reuseLength();
    auto*     locs_data    = mm_locs.data_ptr<int32_t>();
    const int n            = std::min<int>(mm_locs.numel(), static_cast<int>(mm_features.size()));
    // Backward scan assumes mm_locs are in ascending document order; if they
    // aren't, finding the last fully-reused image doesn't imply all earlier
    // ones are reused too, silently producing wrong reuse counts.
    RTP_LLM_CHECK_WITH_INFO(std::is_sorted(locs_data, locs_data + n),
                            "mm_locs must be sorted in ascending order for reuse count logic");
    for (int i = n - 1; i >= 0; --i) {
        const int mm_end = locs_data[i] + static_cast<int>(mm_features[i].size(0));
        if (reuse_length >= mm_end) {
            return i + 1;
        }
    }
    return 0;
}

void gatherMultimodalFeaturesForContextBatch(const GenerateStreamPtr&    stream,
                                             GatherModelInputContext&    ctx,
                                             std::vector<torch::Tensor>& gathered_mm_features) {
    if (!ctx.has_multimodal_input) {
        return;
    }
    std::vector<torch::Tensor> mm_features = stream->multimodalFeatures();
    torch::Tensor              mm_locs     = stream->multimodalLocations();
    if (!mm_locs.defined()) {
        return;
    }
    // Stream getters return RAW (unfiltered) data; the gatherer skips leading images
    // whose entire token span is already covered by reuse_length.
    const int reuse_mm_count = computeReusedMultimodalCount(stream);
    auto*     mm_locs_data   = mm_locs.data_ptr<int>();
    // The two loops below iterate mm_locs and mm_features independently; if
    // their counts disagree the per-image alignment is wrong and downstream
    // expandTokenIds reads garbage. Enforce equality up front.
    RTP_LLM_CHECK_WITH_INFO(mm_locs.numel() == static_cast<int64_t>(mm_features.size()),
                            "mm_locs count %ld != mm_features count %zu for stream %ld",
                            mm_locs.numel(),
                            mm_features.size(),
                            stream->streamId());
    for (int i = reuse_mm_count; i < mm_locs.numel(); ++i) {
        ctx.mm_features_locs[ctx.mm_feature_index] = mm_locs_data[i] + ctx.token_idx - stream->reuseLength();
        ctx.mm_feature_index++;
    }
    for (int i = reuse_mm_count; i < static_cast<int>(mm_features.size()); ++i) {
        auto& mm_feature = mm_features[i];
        if (!mm_feature.is_cuda()) {
            gathered_mm_features.emplace_back(mm_feature.to(torch::kCUDA));
        } else {
            gathered_mm_features.emplace_back(mm_feature);
        }
    }
    auto text_token_mask = stream->textTokensMask();
    memcpy(ctx.merged_text_mask + ctx.token_idx, text_token_mask.data(), text_token_mask.size() * sizeof(int));
}

// Sole H2D owner for input_embeddings. Downstream consumers assert CUDA
// placement and forward without copying — keep that invariant if you change this.
void gatherInputEmbeddingsForContextBatch(const GenerateStreamPtr&       stream,
                                          const GatherModelInputContext& ctx,
                                          std::vector<torch::Tensor>&    gathered_input_embeddings,
                                          std::vector<int32_t>&          gathered_input_embedding_locs) {
    if (!stream->hasInputEmbeddings()) {
        return;
    }
    for (const auto& embedding : stream->inputEmbeddings()) {
        if (embedding.is_cuda()) {
            gathered_input_embeddings.emplace_back(embedding);
        } else {
            gathered_input_embeddings.emplace_back(embedding.to(torch::kCUDA));
        }
    }
    for (int32_t loc : stream->inputEmbeddingsLocs()) {
        gathered_input_embedding_locs.push_back(loc - stream->reuseLength() + ctx.token_idx);
    }
}

void addCacheUpdateCopy(GatherModelInputContext& ctx, const std::vector<BlockIdPair>& update_mapping) {
    if (!ctx.kv_cache_update_mapping) {
        return;
    }
    size_t update_copy_num = update_mapping.size();
    std::memcpy(ctx.kv_cache_update_mapping, update_mapping.data(), update_copy_num * sizeof(BlockIdPair));
    ctx.kv_cache_update_mapping += update_copy_num;
}

}  // anonymous namespace

NormalModelInputGatherer::NormalModelInputGatherer(const NormalModelInputGathererConfig& config): config_(config) {}

GptModelInputs NormalModelInputGatherer::allocateModelInputBuffers(const StreamGroups& stream_groups) const {
    const size_t current_tokens_size      = stream_groups.modelExecuteTokenSize();
    const size_t total_batch_size         = stream_groups.totalModelBatchSize();
    const size_t total_decode_batch_size  = stream_groups.totalDecodeBatchSize();
    const size_t total_context_batch_size = stream_groups.totalContextBatchSize();
    const size_t total_block_copy_num     = stream_groups.totalBlockUpdateCopyNum();
    const size_t max_blocks_num           = stream_groups.curBlocksNum();
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
    model_input.lm_output_indexes     = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.lm_output_lengths     = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.prefix_lengths        = torch::empty({(int64_t)total_context_batch_size}, pinned_i32);
    model_input.request_id            = torch::empty({(int64_t)total_context_batch_size}, pinned_i64);
    model_input.request_pd_separation = torch::empty({(int64_t)total_context_batch_size}, pinned_bool);

    if (max_blocks_num) {
        model_input.kv_cache_kernel_block_id =
            torch::zeros({(int64_t)config_.kv_cache_group_nums,
                          (int64_t)total_batch_size,
                          (int64_t)(max_blocks_num * config_.kernel_blocks_per_kv_block)},
                         pinned_i32);
        model_input.kv_cache_block_id = torch::zeros(
            {(int64_t)config_.kv_cache_group_nums, (int64_t)total_batch_size, (int64_t)max_blocks_num}, pinned_i32);
        model_input.kv_cache_layer_to_group = torch::empty({(int64_t)config_.num_layers}, pinned_i32);
        model_input.kv_cache_group_types    = torch::empty({(int64_t)config_.kv_cache_group_nums}, pinned_i32);
        model_input.kv_cache_update_mapping = torch::empty({(int64_t)total_block_copy_num, 2}, pinned_i32);
        model_input.cache_keys = torch::empty({(int64_t)total_context_batch_size, (int64_t)max_blocks_num}, pinned_i64);
    }

    if (need_cal_position_id) {
        model_input.combo_position_ids =
            torch::empty({(int64_t)(current_tokens_size * config_.position_id_len_factor)}, pinned_i32);
    }
    if (has_multimodal_input) {
        model_input.text_tokens_mask = torch::empty({(int64_t)current_tokens_size}, pinned_i32);
        model_input.mm_features_locs = torch::empty({(int64_t)multimodal_features_len}, pinned_i32);
    }

    model_input.kv_block_stride_bytes     = config_.block_stride_bytes;
    model_input.kv_scale_stride_bytes     = config_.scale_stride_bytes;
    model_input.seq_size_per_block        = config_.seq_size_per_block;
    model_input.kernel_seq_size_per_block = config_.kernel_seq_size_per_block;
    model_input.pd_separation             = config_.role_type == RoleType::PREFILL;
    model_input.warmup                    = config_.warm_up;
    model_input.decode_entrance           = config_.decode_entrance;
    model_input.is_fake_stream            = stream_groups.isFakeStream();

    return model_input;
}

void NormalModelInputGatherer::initializeKvCacheMetadata(GptModelInputs& model_input) const {
    if (model_input.kv_cache_layer_to_group.defined()) {
        size_t num_layers = config_.layer_to_kv_cache_group_id.size();
        std::memcpy(model_input.kv_cache_layer_to_group.data_ptr(),
                    config_.layer_to_kv_cache_group_id.data(),
                    num_layers * sizeof(int32_t));
    }
    if (model_input.kv_cache_group_types.defined()) {
        auto* dst = model_input.kv_cache_group_types.data_ptr<int32_t>();
        for (size_t g = 0; g < config_.kv_cache_group_nums; ++g) {
            dst[g] = static_cast<int32_t>(config_.kv_cache_group_types[g]);
        }
    }
}

absl::Status NormalModelInputGatherer::processDecodeStreams(GptModelInputs&     model_input,
                                                            const StreamGroups& stream_groups) const {
    auto ctx = createGatherContext(config_, model_input, stream_groups, GatherContextMode::DECODE);

    for (const auto& stream : stream_groups.decodeStreams()) {
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto  current_batch_size    = stream->currentBatchSize();
        auto& kv_cache              = *stream->kvCachePtr();
        RTP_LLM_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        RTP_LLM_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());

        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());
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
                stream->generateNextPositionId(ctx.combo_position_ids + ctx.batch_idx * config_.position_id_len_factor);
            }
            ctx.lm_output_indexes[ctx.batch_idx] = ctx.batch_idx;
            ctx.lm_output_lengths[ctx.batch_idx] = 1;
            copyKvCacheBlocksToModelInput(
                model_input, kv_cache, i, ctx.batch_idx, ctx.max_blocks_num, config_.kernel_blocks_per_kv_block);
            ctx.batch_idx += 1;
        }
        addCacheUpdateCopy(ctx, stream->streamCacheResource().getKVBlockUpdateMapping());
        stream->step();
    }
    return absl::OkStatus();
}

absl::Status NormalModelInputGatherer::processContextStreams(GptModelInputs&     model_input,
                                                             const StreamGroups& stream_groups) const {
    std::vector<torch::Tensor> gathered_mm_features;
    std::vector<torch::Tensor> gathered_mm_extra_input;
    std::vector<torch::Tensor> gathered_input_embeddings;
    std::vector<int32_t>       gathered_input_embedding_locs;
    auto ctx = createGatherContext(config_, model_input, stream_groups, GatherContextMode::CONTEXT);

    for (const auto& stream : stream_groups.contextStreams()) {
        model_input.need_all_logits =
            model_input.need_all_logits || stream->calculateLoss() || stream->returnPromptLogits();
        auto  current_batch_size = stream->currentBatchSize();
        auto& kv_cache           = *stream->kvCachePtr();
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
            ctx.cum_output_seq_len += input_tokens.size();

            for (int index = 0; index < (int)input_tokens.size(); ++index) {
                if (input_tokens[index] >= ctx.input_vocab_size
                    && (index >= (int)input_masks.size() || input_masks[index])) {
                    std::ostringstream error_msg;
                    error_msg << "stream [" << stream->streamId() << "] token_id " << input_tokens[index]
                              << " exceed vocab_size " << ctx.input_vocab_size;
                    return absl::InvalidArgumentError(error_msg.str());
                }
            }

            ctx.input_lengths[ctx.batch_idx]      = input_tokens.size();
            ctx.prefix_lengths[prefill_batch_idx] = stream->prefixLength();
            ctx.lm_output_indexes[ctx.batch_idx]  = ctx.cum_output_seq_len - 1;
            ctx.lm_output_lengths[ctx.batch_idx]  = 1;

            gatherMultimodalFeaturesForContextBatch(stream, ctx, gathered_mm_features);
            gatherInputEmbeddingsForContextBatch(stream, ctx, gathered_input_embeddings, gathered_input_embedding_locs);

            if (ctx.need_cal_position_id) {
                auto context_pos_ids = stream->generateContextPositionIds();
                int  reuse_offset    = stream->reuseLength() * config_.position_id_len_factor;
                memcpy(ctx.combo_position_ids + ctx.token_idx * config_.position_id_len_factor,
                       context_pos_ids.data_ptr<int>() + reuse_offset,
                       (context_pos_ids.numel() - reuse_offset) * sizeof(int));
            }

            if (ctx.has_mm_extra_input) {
                auto      mm_extra_input = stream->multimodalExtraInput();
                const int reuse_mm_count = computeReusedMultimodalCount(stream);
                RTP_LLM_CHECK_WITH_INFO(mm_extra_input.size() == stream->multimodalFeatures().size()
                                            || mm_extra_input.empty(),
                                        "mm_extra_input count %zu != mm_features count %zu for stream %ld",
                                        mm_extra_input.size(),
                                        stream->multimodalFeatures().size(),
                                        stream->streamId());
                for (int j = reuse_mm_count; j < static_cast<int>(mm_extra_input.size()); ++j) {
                    gathered_mm_extra_input.emplace_back(mm_extra_input[j].to(torch::kCUDA));
                }
            }

            copyKvCacheBlocksToModelInput(
                model_input, kv_cache, i, ctx.batch_idx, ctx.max_blocks_num, config_.kernel_blocks_per_kv_block);

            if (ctx.max_blocks_num && config_.role_type == RoleType::PREFILL && stream->hasCacheKeys()) {
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

        addCacheUpdateCopy(ctx, stream->streamCacheResource().getKVBlockUpdateMapping());
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
    if (!gathered_input_embeddings.empty()) {
        model_input.input_embeddings      = std::move(gathered_input_embeddings);
        model_input.input_embeddings_locs = torch::tensor(gathered_input_embedding_locs, torch::kInt32);
    }
    return absl::OkStatus();
}

absl::StatusOr<GptModelInputs> NormalModelInputGatherer::gather(const StreamGroups& stream_groups) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_LOG_DEBUG("context_streams size = %d, decode_streams size = %d",
                      stream_groups.contextStreams().size(),
                      stream_groups.decodeStreams().size());
    auto model_input = allocateModelInputBuffers(stream_groups);
    initializeKvCacheMetadata(model_input);
    RETURN_IF_STATUS_ERROR(processDecodeStreams(model_input, stream_groups));
    RETURN_IF_STATUS_ERROR(processContextStreams(model_input, stream_groups));
    return model_input;
}

}  // namespace rtp_llm
