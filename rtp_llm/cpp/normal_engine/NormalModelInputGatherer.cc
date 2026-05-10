#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "torch/all.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"

namespace rtp_llm {

namespace {

bool asyncDebugEnabled() {
    const char* env = std::getenv("RTP_LLM_ASYNC_DEBUG");
    return env != nullptr && std::string(env) == "1";
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
    auto* mm_locs_data = mm_locs.data_ptr<int>();
    for (int i = 0; i < mm_locs.numel(); ++i) {
        ctx.mm_features_locs[ctx.mm_feature_index] = mm_locs_data[i] + ctx.token_idx - stream->reuseLength();
        ctx.mm_feature_index++;
    }
    for (auto& mm_feature : mm_features) {
        if (!mm_feature.is_cuda()) {
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

torch::Tensor publishInt32ToCuda(const torch::Tensor& tensor) {
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
    return tensor.to(cuda_i32, /*non_blocking=*/true);
}

void publishModelInputCoreTensorsToCuda(GptModelInputs& model_input) {
    // TODO(async): stream state is still gathered through CPU pointers above.
    // Publish only device tensors at the model boundary.
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.publish_core_tensors_to_cuda");
    model_input.combo_tokens     = publishInt32ToCuda(model_input.combo_tokens);
    model_input.input_lengths    = publishInt32ToCuda(model_input.input_lengths);
    model_input.sequence_lengths = publishInt32ToCuda(model_input.sequence_lengths);
    model_input.prefix_lengths   = publishInt32ToCuda(model_input.prefix_lengths);
    // Migrate the 3-D KV kernel block id tensor to CUDA. Filled host-side above via
    // copyKvCacheBlocksToModelInput; this is the single H2D that replaces the per-group
    // tensorHoldHostAndToCuda copies that PyWrappedModel::setupKVCacheForAttentionInputs
    // used to launch.
    model_input.kv_cache_kernel_block_id = publishInt32ToCuda(model_input.kv_cache_kernel_block_id);
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
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.process_decode_streams");
    auto ctx = createGatherContext(config_, model_input, stream_groups, GatherContextMode::DECODE);

    bool use_normal_device_state = stream_groups.totalContextBatchSize() == 0
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
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto  current_batch_size    = stream->currentBatchSize();
        auto& kv_cache              = *stream->kvCachePtr();
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
            copyKvCacheBlocksToModelInput(
                model_input, kv_cache, i, ctx.batch_idx, ctx.max_blocks_num, config_.kernel_blocks_per_kv_block);
            ctx.batch_idx += 1;
        }
        addCacheUpdateCopy(ctx, stream->streamCacheResource().getKVBlockUpdateMapping());
        stream->step();
    }

    if (use_normal_device_state) {
        model_input.combo_tokens     = torch::cat(normal_combo_tokens_gpu, 0).to(torch::kInt32);
        model_input.sequence_lengths = torch::cat(normal_sequence_lengths_gpu, 0).to(torch::kInt32);
    }
    return absl::OkStatus();
}

absl::Status NormalModelInputGatherer::processContextStreams(GptModelInputs&     model_input,
                                                             const StreamGroups& stream_groups) const {
    RTP_LLM_PROFILE_SCOPE("normal_engine.model_input_gatherer.process_context_streams");
    std::vector<torch::Tensor> gathered_mm_features;
    const auto                 context_batch_size = static_cast<int64_t>(stream_groups.totalContextBatchSize());
    // TODO(async): prefixLength() is still stream CPU state. Stage it explicitly
    // on host here, then publish only a CUDA tensor in GptModelInputs.
    auto prefix_lengths_host =
        torch::empty({context_batch_size}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto ctx                = createGatherContext(config_, model_input, stream_groups, GatherContextMode::CONTEXT);
    ctx.prefix_lengths_host = prefix_lengths_host.data_ptr<int32_t>();

    for (const auto& stream : stream_groups.contextStreams()) {
        model_input.need_all_logits = model_input.need_all_logits || stream->calculateLoss();
        auto  current_batch_size    = stream->currentBatchSize();
        auto& kv_cache              = *stream->kvCachePtr();
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
            gatherMultimodalFeaturesForContextBatch(stream, ctx, gathered_mm_features);

            if (ctx.need_cal_position_id) {
                auto context_pos_ids = stream->generateContextPositionIds();
                int  reuse_offset    = stream->reuseLength() * config_.position_id_len_factor;
                memcpy(ctx.combo_position_ids + ctx.token_idx * config_.position_id_len_factor,
                       context_pos_ids.data_ptr<int>() + reuse_offset,
                       (context_pos_ids.numel() - reuse_offset) * sizeof(int));
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
    model_input.prefix_lengths = publishInt32ToCuda(prefix_lengths_host);
    return absl::OkStatus();
}

absl::StatusOr<torch::Tensor>
NormalModelInputGatherer::gatherKvCacheKernelBlockId(const StreamGroups& stream_groups) const {
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

    return publishInt32ToCuda(host_tensor);
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
    publishModelInputCoreTensorsToCuda(model_input);
    model_input.lm_output_indexes = buildLmOutputIndexesOnCuda(model_input, stream_groups);
    return model_input;
}

}  // namespace rtp_llm
