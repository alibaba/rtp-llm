#include <algorithm>
#include <cstring>
#include <sstream>
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "torch/all.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

namespace {

bool hasStandaloneNormalDeviceState(const GenerateStream::NormalAsyncDeviceState& state) {
    return state.last_sample_token_gpu.defined() && state.last_sample_token_gpu.is_cuda()
           && state.next_seq_len_gpu.defined() && state.next_seq_len_gpu.is_cuda();
}

bool hasBatchedNormalDeviceState(const GenerateStream::NormalAsyncDeviceState& state) {
    return state.device_batch_index >= 0 && state.batched_last_sample_tokens_gpu.defined()
           && state.batched_last_sample_tokens_gpu.is_cuda() && state.batched_last_sample_tokens_gpu.dim() == 1
           && state.device_batch_index < state.batched_last_sample_tokens_gpu.size(0)
           && state.batched_next_seq_lens_gpu.defined() && state.batched_next_seq_lens_gpu.is_cuda()
           && state.batched_next_seq_lens_gpu.dim() == 1
           && state.device_batch_index < state.batched_next_seq_lens_gpu.size(0);
}

torch::Tensor normalDeviceTokenView(const GenerateStream::NormalAsyncDeviceState& state) {
    if (state.last_sample_token_gpu.defined() && state.last_sample_token_gpu.is_cuda()) {
        return state.last_sample_token_gpu;
    }
    return state.batched_last_sample_tokens_gpu.narrow(0, state.device_batch_index, 1);
}

torch::Tensor normalDeviceNextSeqLenView(const GenerateStream::NormalAsyncDeviceState& state) {
    if (state.next_seq_len_gpu.defined() && state.next_seq_len_gpu.is_cuda()) {
        return state.next_seq_len_gpu;
    }
    return state.batched_next_seq_lens_gpu.narrow(0, state.device_batch_index, 1);
}

struct GatherModelInputContext {
    int          input_vocab_size;
    bool         need_cal_position_id;
    size_t       max_blocks_num;
    int*         merged_tokens;
    int*         input_lengths;
    int*         combo_position_ids;
    BlockIdPair* kv_cache_update_mapping;
    int          batch_idx;
    int*         sequence_lengths;
    bool         has_multimodal_input;
    bool         has_mm_extra_input;
    size_t       total_decode_batch_size;
    int*         prefix_lengths_host;
    int*         merged_text_mask;
    int*         mm_features_locs;
    int          token_idx;
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
    ctx.combo_position_ids   = ctx.need_cal_position_id ? model_input.combo_position_ids.data_ptr<int32_t>() : nullptr;
    ctx.has_multimodal_input = config.is_multimodal && stream_groups.has_multimodal_input();
    ctx.prefix_lengths_host  = nullptr;
    ctx.has_mm_extra_input   = config.is_multimodal && stream_groups.hasMMExtraInput();
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
            reinterpret_cast<BlockIdPair*>(model_input.kv_cache_update_mapping.data_ptr()) + kv_cache_mapping_offset :
            nullptr;

    if (ctx.merged_text_mask) {
        size_t current_tokens_size = stream_groups.modelExecuteTokenSize();
        std::fill(ctx.merged_text_mask, ctx.merged_text_mask + current_tokens_size, 1);
    }

    return ctx;
}

void copyKvCacheBlocksToModelInput(GptModelInputs&                    model_input,
                                   const BatchKVCacheResource&        kv_cache,
                                   int                                stream_batch_idx,
                                   int                                model_batch_idx,
                                   size_t                             max_blocks_num,
                                   size_t                             kernel_blocks_per_kv_block,
                                   const std::vector<CacheGroupType>& group_types,
                                   bool                               skip_linear_cache_groups) {
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
        if (skip_linear_cache_groups && static_cast<size_t>(gid) < group_types.size()
            && group_types[gid] == CacheGroupType::LINEAR) {
            // Decode-only MTP async mode repairs the LINEAR kernel table on
            // device. Neither host table is safe to read while specUpdate may
            // swap the shared BlockIds; host consumers must stay synchronous.
            continue;
        }

        const auto& kernel_blocks = kv_cache.kernelBlocks(stream_batch_idx, gid);
        int32_t*    kernel_dst    = kernel_dst_base
                              + (static_cast<size_t>(gid) * batch + static_cast<size_t>(model_batch_idx))
                                    * max_blocks_num * kernel_blocks_per_kv_block;
        std::memcpy(kernel_dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));

        const auto& physical_blocks = kv_cache.blocks(stream_batch_idx, gid);
        int32_t*    store_dst =
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
                                             std::vector<torch::Tensor>& gathered_mm_features,
                                             TensorHolder&               host_holder) {
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
            host_holder.hold_host(mm_feature);
            gathered_mm_features.emplace_back(mm_feature.to(torch::kCUDA, /*non_blocking=*/true));
        } else {
            gathered_mm_features.emplace_back(mm_feature);
        }
    }
    auto text_token_mask = stream->textTokensMask();
    memcpy(ctx.merged_text_mask + ctx.token_idx, text_token_mask.data(), text_token_mask.size() * sizeof(int));
}

void addCacheUpdateCopy(GatherModelInputContext& ctx, const std::vector<BlockIdPair>& update_mapping) {
    if (!ctx.kv_cache_update_mapping) {
        return;
    }
    size_t update_copy_num = update_mapping.size();
    std::memcpy(ctx.kv_cache_update_mapping, update_mapping.data(), update_copy_num * sizeof(BlockIdPair));
    ctx.kv_cache_update_mapping += update_copy_num;
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

NormalModelInputGatherer::NormalModelInputGatherer(const NormalModelInputGathererConfig& config): config_(config) {}

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
    static const auto cuda_i32    = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);

    GptModelInputs model_input;
    model_input.combo_tokens          = torch::empty({(int64_t)current_tokens_size}, pinned_i32);
    model_input.input_lengths         = torch::empty({(int64_t)total_batch_size}, pinned_i32);
    model_input.sequence_lengths      = torch::empty({(int64_t)total_decode_batch_size}, pinned_i32);
    model_input.prefix_lengths        = torch::empty({(int64_t)total_context_batch_size}, cuda_i32);
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
        const size_t layer_to_group_len =
            config_.layer_to_kv_cache_group_id.empty() ? config_.num_layers : config_.layer_to_kv_cache_group_id.size();
        model_input.kv_cache_layer_to_group = torch::empty({(int64_t)layer_to_group_len}, pinned_i32);
        model_input.kv_cache_group_types    = torch::empty({(int64_t)config_.kv_cache_group_nums}, pinned_i32);
        model_input.kv_cache_update_mapping = torch::empty({(int64_t)total_block_copy_num, 2}, pinned_i32);
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

    model_input.kv_block_stride_bytes     = config_.block_stride_bytes;
    model_input.kv_scale_stride_bytes     = config_.scale_stride_bytes;
    model_input.seq_size_per_block        = config_.seq_size_per_block;
    model_input.kernel_seq_size_per_block = config_.kernel_seq_size_per_block;
    model_input.pd_separation             = config_.role_type == RoleType::PREFILL;
    model_input.warmup                    = config_.warm_up;
    model_input.decode_entrance           = config_.decode_entrance;
    model_input.use_opaque_kv_cache_store = config_.use_opaque_kv_cache_store;
    model_input.is_fake_stream            = stream_groups.isFakeStream();

    return model_input;
}

void NormalModelInputGatherer::initializeKvCacheMetadata(GptModelInputs& model_input) const {
    if (model_input.kv_cache_layer_to_group.defined()) {
        const size_t dst_numel = static_cast<size_t>(model_input.kv_cache_layer_to_group.numel());
        const size_t src_numel = config_.layer_to_kv_cache_group_id.size();
        RTP_LLM_CHECK_WITH_INFO(src_numel <= dst_numel,
                                "kv_cache_layer_to_group overflow: dst_numel=%zu, src_numel=%zu, config_num_layers=%zu",
                                dst_numel,
                                src_numel,
                                config_.num_layers);
        std::memcpy(model_input.kv_cache_layer_to_group.data_ptr(),
                    config_.layer_to_kv_cache_group_id.data(),
                    src_numel * sizeof(int32_t));
    }
    if (model_input.kv_cache_group_types.defined()) {
        auto* dst = model_input.kv_cache_group_types.data_ptr<int32_t>();
        for (size_t g = 0; g < config_.kv_cache_group_nums; ++g) {
            dst[g] = static_cast<int32_t>(config_.kv_cache_group_types[g]);
        }
    }
}

absl::Status NormalModelInputGatherer::processDecodeStreams(GptModelInputs&     model_input,
                                                            const StreamGroups& stream_groups,
                                                            bool                skip_linear_cache_groups) const {
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.process_decode_streams");
    auto ctx = createGatherContext(config_, model_input, stream_groups, GatherContextMode::DECODE);

    bool use_normal_device_state = stream_groups.totalContextBatchSize() == 0
                                   && stream_groups.totalDecodeBatchSize() > 0 && !ctx.need_cal_position_id;
    if (use_normal_device_state) {
        for (const auto& stream : stream_groups.decodeStreams()) {
            const auto& state = stream->getNormalAsyncDeviceState();
            if (stream->currentBatchSize() != 1
                || (!hasStandaloneNormalDeviceState(state) && !hasBatchedNormalDeviceState(state))) {
                use_normal_device_state = false;
                break;
            }
        }
    }
    std::vector<torch::Tensor> normal_combo_tokens_gpu;
    std::vector<torch::Tensor> normal_sequence_lengths_gpu;
    torch::Tensor              normal_sequence_lengths_host;
    torch::Tensor              shared_batched_tokens_gpu;
    torch::Tensor              shared_batched_next_seq_lens_gpu;
    bool                       can_reuse_batched_state = use_normal_device_state;
    if (use_normal_device_state) {
        normal_combo_tokens_gpu.reserve(stream_groups.totalDecodeBatchSize());
        normal_sequence_lengths_gpu.reserve(stream_groups.totalDecodeBatchSize());
        normal_sequence_lengths_host = torch::empty({(int64_t)stream_groups.totalDecodeBatchSize()},
                                                    torch::TensorOptions(torch::kInt32).pinned_memory(true));

        int64_t device_batch_index = 0;
        for (const auto& stream : stream_groups.decodeStreams()) {
            const auto& state = stream->getNormalAsyncDeviceState();
            if (!hasBatchedNormalDeviceState(state) || state.device_batch_index != device_batch_index
                || state.batched_last_sample_tokens_gpu.size(0)
                       != static_cast<int64_t>(stream_groups.totalDecodeBatchSize())
                || state.batched_next_seq_lens_gpu.size(0)
                       != static_cast<int64_t>(stream_groups.totalDecodeBatchSize())) {
                can_reuse_batched_state = false;
                break;
            }
            if (!shared_batched_tokens_gpu.defined()) {
                shared_batched_tokens_gpu        = state.batched_last_sample_tokens_gpu;
                shared_batched_next_seq_lens_gpu = state.batched_next_seq_lens_gpu;
            } else if (shared_batched_tokens_gpu.unsafeGetTensorImpl()
                           != state.batched_last_sample_tokens_gpu.unsafeGetTensorImpl()
                       || shared_batched_next_seq_lens_gpu.unsafeGetTensorImpl()
                              != state.batched_next_seq_lens_gpu.unsafeGetTensorImpl()) {
                can_reuse_batched_state = false;
                break;
            }
            device_batch_index += 1;
        }
    }

    for (const auto& stream : stream_groups.decodeStreams()) {
        model_input.need_all_logits        = model_input.need_all_logits || stream->calculateLoss();
        model_input.need_all_hidden_states = model_input.need_all_hidden_states || stream->needReturnHiddenStates();
        auto  current_batch_size           = stream->currentBatchSize();
        auto& kv_cache                     = *stream->kvCachePtr();
        if (!skip_linear_cache_groups) {
            RTP_LLM_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        }
        RTP_LLM_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());

        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());
            if (use_normal_device_state) {
                const auto&             state = stream->getNormalAsyncDeviceState();
                // Standalone states are already [1]; batched fallback returns
                // a [1] view. Avoid another reshape per stream: at BS128 each
                // reshape creates an additional ATen view in the gather trace.
                if (!can_reuse_batched_state) {
                    normal_combo_tokens_gpu.push_back(normalDeviceTokenView(state));
                }
                // Preserve the device state as the source of truth, but defer
                // the arithmetic until the whole decode batch is assembled.
                // Doing `state.next_seq_len_gpu - 1` here launches one CUDA
                // kernel per request (128 launches at BS128).
                if (!can_reuse_batched_state) {
                    normal_sequence_lengths_gpu.push_back(normalDeviceNextSeqLenView(state));
                }
                normal_sequence_lengths_host.data_ptr<int32_t>()[ctx.batch_idx] = state.next_real_seq_len - 1;
                ctx.input_lengths[ctx.batch_idx]                                = stream->inputLength();
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
                                          config_.kernel_blocks_per_kv_block,
                                          config_.kv_cache_group_types,
                                          skip_linear_cache_groups);
            ctx.batch_idx += 1;
        }
        addCacheUpdateCopy(ctx, stream->streamCacheResource().getKVBlockUpdateMapping());
        stream->step();
    }

    if (use_normal_device_state) {
        auto combo_tokens_gpu = can_reuse_batched_state ?
                                    shared_batched_tokens_gpu :
                                    (normal_combo_tokens_gpu.size() == 1 ? normal_combo_tokens_gpu.front() :
                                                                           torch::cat(normal_combo_tokens_gpu, 0));
        auto next_sequence_lengths_gpu = can_reuse_batched_state ? shared_batched_next_seq_lens_gpu :
                                                                   (normal_sequence_lengths_gpu.size() == 1 ?
                                                                        normal_sequence_lengths_gpu.front() :
                                                                        torch::cat(normal_sequence_lengths_gpu, 0));
        model_input.combo_tokens       = combo_tokens_gpu.to(torch::kInt32);
        // Preserve the device state's already-computed next length so Python
        // attention preparation does not launch another sequence_lengths + 1 kernel.
        model_input.sequence_lengths_plus_1       = next_sequence_lengths_gpu.to(torch::kInt32);
        model_input.sequence_lengths              = model_input.sequence_lengths_plus_1 - 1;
        model_input.sequence_lengths_host_for_log = normal_sequence_lengths_host;
    }
    return absl::OkStatus();
}

absl::Status NormalModelInputGatherer::processContextStreams(GptModelInputs&     model_input,
                                                             const StreamGroups& stream_groups,
                                                             TensorHolder&       host_holder,
                                                             bool                skip_linear_cache_groups) const {
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.process_context_streams");
    std::vector<torch::Tensor> gathered_mm_features;
    std::vector<torch::Tensor> gathered_mm_extra_input;
    const auto                 context_batch_size = static_cast<int64_t>(stream_groups.totalContextBatchSize());
    // TODO(async): prefixLength() is still stream CPU state. Stage it explicitly
    // on host here, then publish only a CUDA tensor in GptModelInputs.
    auto prefix_lengths_host =
        torch::empty({context_batch_size}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto ctx                = createGatherContext(config_, model_input, stream_groups, GatherContextMode::CONTEXT);
    ctx.prefix_lengths_host = prefix_lengths_host.data_ptr<int32_t>();

    for (const auto& stream : stream_groups.contextStreams()) {
        model_input.need_all_logits        = model_input.need_all_logits || stream->calculateLoss();
        model_input.need_all_hidden_states = model_input.need_all_hidden_states || stream->needReturnHiddenStates();
        // The FIFO scheduler keeps SP-enabled and target-only Prefill streams
        // in separate batches, so this scalar describes the entire context
        // batch and can be consumed directly by Python model implementations.
        model_input.force_disable_sp_run = model_input.force_disable_sp_run || stream->forceDisableSpRun();
        auto  current_batch_size           = stream->currentBatchSize();
        auto& kv_cache                     = *stream->kvCachePtr();
        if (config_.enable_detail_log) {
            if (!skip_linear_cache_groups) {
                RTP_LLM_LOG_DEBUG("context kv_cache: %s", kv_cache.debugString().c_str());
            }
            RTP_LLM_LOG_DEBUG("context stream: %s", stream->debugString().c_str());
        } else {
            if (!skip_linear_cache_groups) {
                RTP_LLM_LOG_TRACE("context kv_cache: %s", kv_cache.debugString().c_str());
            }
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
            gatherMultimodalFeaturesForContextBatch(stream, ctx, gathered_mm_features, host_holder);

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

            copyKvCacheBlocksToModelInput(model_input,
                                          kv_cache,
                                          i,
                                          ctx.batch_idx,
                                          ctx.max_blocks_num,
                                          config_.kernel_blocks_per_kv_block,
                                          config_.kv_cache_group_types,
                                          skip_linear_cache_groups);

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

        addCacheUpdateCopy(ctx, stream->streamCacheResource().getKVBlockUpdateMapping());
        stream->step();
    }

    if (config_.is_multimodal && !gathered_mm_features.empty()) {
        model_input.multimodal_features = std::move(gathered_mm_features);
    }
    model_input.prefix_lengths_host_for_log = prefix_lengths_host;
    model_input.prefix_lengths              = publishInt32ToCuda(prefix_lengths_host, host_holder);
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
    return absl::OkStatus();
}

absl::StatusOr<torch::Tensor> NormalModelInputGatherer::gatherKvCacheKernelBlockId(const StreamGroups& stream_groups,
                                                                                   TensorHolder& host_holder) const {
    const size_t total_batch_size = stream_groups.totalModelBatchSize();
    const size_t max_blocks_num   = stream_groups.curBlocksNum();
    if (max_blocks_num == 0 || total_batch_size == 0) {
        return torch::Tensor{};
    }

    static const auto pinned_i32  = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    auto              host_tensor = torch::zeros({(int64_t)config_.kv_cache_group_nums,
                                                  (int64_t)total_batch_size,
                                                  (int64_t)(max_blocks_num * config_.kernel_blocks_per_kv_block)},
                                    pinned_i32);

    const size_t per_batch_stride = max_blocks_num * config_.kernel_blocks_per_kv_block;
    int32_t*     dst_base         = host_tensor.data_ptr<int32_t>();

    auto fill_one_stream = [&](const GenerateStreamPtr& stream, int& batch_idx) {
        auto& kv_cache           = *stream->kvCachePtr();
        auto  current_batch_size = stream->currentBatchSize();
        for (int i = 0; i < current_batch_size; ++i) {
            for (int gid = 0; gid < kv_cache.groupNums(); ++gid) {
                const auto& kernel_blocks = kv_cache.kernelBlocks(i, gid);
                int32_t*    dst =
                    dst_base
                    + (static_cast<size_t>(gid) * total_batch_size + static_cast<size_t>(batch_idx)) * per_batch_stride;
                std::memcpy(dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));
            }
            batch_idx += 1;
        }
    };

    int batch_idx = 0;
    for (const auto& stream : stream_groups.decodeStreams()) {
        fill_one_stream(stream, batch_idx);
    }
    for (const auto& stream : stream_groups.contextStreams()) {
        fill_one_stream(stream, batch_idx);
    }

    return publishInt32ToCuda(host_tensor, host_holder);
}

absl::StatusOr<MtpLinearKvCacheGatherResult>
NormalModelInputGatherer::gatherMtpLinearKvCacheKernelBlockId(const StreamGroups& stream_groups,
                                                              TensorHolder&       host_holder) const {
    MtpLinearKvCacheGatherResult result;
    const size_t                 total_batch_size = stream_groups.totalModelBatchSize();
    const size_t                 max_blocks_num   = stream_groups.curBlocksNum();
    if (max_blocks_num == 0 || total_batch_size == 0) {
        return result;
    }

    static const auto pinned_i32              = torch::TensorOptions(torch::kInt32).pinned_memory(true);
    const auto        cuda_i32                = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto              host_block_ids          = torch::zeros({static_cast<int64_t>(config_.kv_cache_group_nums),
                                                              static_cast<int64_t>(total_batch_size),
                                                              static_cast<int64_t>(max_blocks_num * config_.kernel_blocks_per_kv_block)},
                                       pinned_i32);
    auto              valid_block_counts_host = torch::zeros(
        {static_cast<int64_t>(config_.kv_cache_group_nums), static_cast<int64_t>(total_batch_size)}, pinned_i32);
    auto pending_patches_host = torch::zeros({static_cast<int64_t>(total_batch_size)}, pinned_i32);
    auto group_types_host     = torch::empty({static_cast<int64_t>(config_.kv_cache_group_nums)}, pinned_i32);
    for (size_t group_id = 0; group_id < config_.kv_cache_group_nums; ++group_id) {
        const auto group_type                          = group_id < config_.kv_cache_group_types.size() ?
                                                             config_.kv_cache_group_types[group_id] :
                                                             CacheGroupType::FULL;
        group_types_host.data_ptr<int32_t>()[group_id] = static_cast<int32_t>(group_type);
    }

    constexpr int64_t          patch_width = 4;
    const int64_t              group_num   = static_cast<int64_t>(config_.kv_cache_group_nums);
    const size_t               row_width   = max_blocks_num * config_.kernel_blocks_per_kv_block;
    int32_t*                   dst_base    = host_block_ids.data_ptr<int32_t>();
    std::vector<torch::Tensor> patch_position_slices;
    std::vector<torch::Tensor> patch_source_slot_slices;
    std::vector<torch::Tensor> patch_before_slices;
    std::vector<torch::Tensor> patch_after_slices;
    std::vector<torch::Tensor> patch_valid_slices;
    patch_position_slices.reserve(total_batch_size);
    patch_source_slot_slices.reserve(total_batch_size);
    patch_before_slices.reserve(total_batch_size);
    patch_after_slices.reserve(total_batch_size);
    patch_valid_slices.reserve(total_batch_size);
    const auto dummy_positions    = torch::full({1, patch_width}, -1, cuda_i32);
    const auto dummy_source_slots = torch::full({1, patch_width}, -1, cuda_i32);
    const auto dummy_values       = torch::full({1, group_num, patch_width}, -1, cuda_i32);
    const auto dummy_valid        = torch::zeros({1, group_num}, cuda_i32);

    auto append_dummy_patch = [&]() {
        patch_position_slices.push_back(dummy_positions);
        patch_source_slot_slices.push_back(dummy_source_slots);
        patch_before_slices.push_back(dummy_values);
        patch_after_slices.push_back(dummy_values);
        patch_valid_slices.push_back(dummy_valid);
    };

    auto fill_one_stream = [&](const GenerateStreamPtr& stream, int& batch_idx) {
        auto snapshot = stream->snapshotKVCacheBlocks();
        for (int stream_batch_idx = 0; stream_batch_idx < snapshot.batch_size; ++stream_batch_idx) {
            RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(stream_batch_idx) < snapshot.kernel_blocks.size(),
                                    "stream batch index %d exceeds snapshot batch size %zu",
                                    stream_batch_idx,
                                    snapshot.kernel_blocks.size());
            for (size_t group_id = 0; group_id < snapshot.kernel_blocks[stream_batch_idx].size(); ++group_id) {
                const auto& kernel_blocks = snapshot.kernel_blocks[stream_batch_idx][group_id];
                RTP_LLM_CHECK_WITH_INFO(kernel_blocks.size() <= row_width,
                                        "kernel block snapshot overflow: %zu > %zu",
                                        kernel_blocks.size(),
                                        row_width);
                int32_t* dst = dst_base + (group_id * total_batch_size + static_cast<size_t>(batch_idx)) * row_width;
                std::memcpy(dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));
                valid_block_counts_host.data_ptr<int32_t>()[group_id * total_batch_size + batch_idx] =
                    static_cast<int32_t>(kernel_blocks.size());
            }

            if (snapshot.needs_mtp_linear_patch) {
                const auto& patch = snapshot.linear_patch;
                const bool  valid_device_state =
                    snapshot.batch_size == 1 && patch.positions_gpu.defined() && patch.positions_gpu.is_cuda()
                    && patch.positions_gpu.scalar_type() == torch::kInt32 && patch.positions_gpu.is_contiguous()
                    && patch.positions_gpu.dim() == 2 && patch.positions_gpu.size(0) == 1
                    && patch.positions_gpu.size(1) == patch_width && patch.source_slots_gpu.defined()
                    && patch.source_slots_gpu.is_cuda() && patch.source_slots_gpu.scalar_type() == torch::kInt32
                    && patch.source_slots_gpu.is_contiguous()
                    && patch.source_slots_gpu.sizes() == patch.positions_gpu.sizes()
                    && patch.before_values_gpu.defined() && patch.before_values_gpu.is_cuda()
                    && patch.before_values_gpu.scalar_type() == torch::kInt32 && patch.before_values_gpu.is_contiguous()
                    && patch.before_values_gpu.dim() == 3 && patch.before_values_gpu.size(0) == 1
                    && patch.before_values_gpu.size(1) == group_num && patch.before_values_gpu.size(2) == patch_width
                    && patch.after_values_gpu.defined() && patch.after_values_gpu.is_cuda()
                    && patch.after_values_gpu.scalar_type() == torch::kInt32 && patch.after_values_gpu.is_contiguous()
                    && patch.after_values_gpu.sizes() == patch.before_values_gpu.sizes() && patch.valid_gpu.defined()
                    && patch.valid_gpu.is_cuda() && patch.valid_gpu.scalar_type() == torch::kInt32
                    && patch.valid_gpu.is_contiguous() && patch.valid_gpu.dim() == 2 && patch.valid_gpu.size(0) == 1
                    && patch.valid_gpu.size(1) == group_num && patch.ready_event != nullptr;
                result.device_patch_ready &= valid_device_state;
                if (valid_device_state) {
#if USING_CUDA
                    auto ready_event = std::static_pointer_cast<torch::Event>(patch.ready_event);
                    ready_event->block(cuda_graph::graphGetCurrentStream());
#endif
                    pending_patches_host.data_ptr<int32_t>()[batch_idx] = 1;
                    patch_position_slices.push_back(patch.positions_gpu);
                    patch_source_slot_slices.push_back(patch.source_slots_gpu);
                    patch_before_slices.push_back(patch.before_values_gpu);
                    patch_after_slices.push_back(patch.after_values_gpu);
                    patch_valid_slices.push_back(patch.valid_gpu);
                } else {
                    append_dummy_patch();
                }
            } else {
                append_dummy_patch();
            }
            batch_idx += 1;
        }
    };

    int batch_idx = 0;
    for (const auto& stream : stream_groups.decodeStreams()) {
        fill_one_stream(stream, batch_idx);
    }
    for (const auto& stream : stream_groups.contextStreams()) {
        fill_one_stream(stream, batch_idx);
    }
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(batch_idx) == total_batch_size,
                            "gathered batch size %d does not match expected %zu",
                            batch_idx,
                            total_batch_size);

    result.block_ids           = publishInt32ToCuda(host_block_ids, host_holder);
    result.group_types         = publishInt32ToCuda(group_types_host, host_holder);
    result.valid_block_counts  = publishInt32ToCuda(valid_block_counts_host, host_holder);
    result.pending_patches     = publishInt32ToCuda(pending_patches_host, host_holder);
    result.patch_positions     = torch::cat(patch_position_slices, 0).contiguous();
    result.patch_source_slots  = torch::cat(patch_source_slot_slices, 0).contiguous();
    result.patch_before_values = torch::cat(patch_before_slices, 0).contiguous();
    result.patch_after_values  = torch::cat(patch_after_slices, 0).contiguous();
    result.patch_valid         = torch::cat(patch_valid_slices, 0).contiguous();
    return result;
}

absl::StatusOr<GptModelInputs> NormalModelInputGatherer::gather(const StreamGroups& stream_groups,
                                                                TensorHolder&       host_holder,
                                                                bool                skip_linear_cache_groups) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    RTP_LLM_LOG_DEBUG("context_streams size = %d, decode_streams size = %d",
                      stream_groups.contextStreams().size(),
                      stream_groups.decodeStreams().size());
    auto model_input = allocateModelInputBuffers(stream_groups);
    initializeKvCacheMetadata(model_input);
    RETURN_IF_STATUS_ERROR(processDecodeStreams(model_input, stream_groups, skip_linear_cache_groups));
    RETURN_IF_STATUS_ERROR(processContextStreams(model_input, stream_groups, host_holder, skip_linear_cache_groups));
    if (model_input.combo_tokens.defined() && !model_input.combo_tokens.is_cuda()) {
        model_input.combo_tokens_host_for_log = model_input.combo_tokens;
    }
    if (model_input.input_lengths.defined() && !model_input.input_lengths.is_cuda()) {
        model_input.input_lengths_host_for_log = model_input.input_lengths;
    }
    if (model_input.sequence_lengths.defined() && !model_input.sequence_lengths.is_cuda()) {
        model_input.sequence_lengths_host_for_log = model_input.sequence_lengths;
    }
    if (model_input.kv_cache_block_id.defined() && !model_input.kv_cache_block_id.is_cuda()) {
        model_input.kv_cache_block_id_host = model_input.kv_cache_block_id;
    }
    if (model_input.kv_cache_kernel_block_id.defined() && !model_input.kv_cache_kernel_block_id.is_cuda()) {
        model_input.kv_cache_kernel_block_id_host = model_input.kv_cache_kernel_block_id;
    }
    if (model_input.kv_cache_layer_to_group.defined() && !model_input.kv_cache_layer_to_group.is_cuda()) {
        model_input.kv_cache_layer_to_group_host = model_input.kv_cache_layer_to_group;
    }
    if (model_input.kv_cache_group_types.defined() && !model_input.kv_cache_group_types.is_cuda()) {
        model_input.kv_cache_group_types_host = model_input.kv_cache_group_types;
    }
    publishModelInputCoreTensorsToCuda(model_input, host_holder);
    model_input.lm_output_indexes = buildLmOutputIndexesOnCuda(model_input, stream_groups);
    return model_input;
}

}  // namespace rtp_llm
