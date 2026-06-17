#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "torch/all.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/normal_engine/NormalModelInputGatherer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

const bool kAsyncDebugEnabled = []() {
    const char* env = std::getenv("RTP_LLM_ASYNC_DEBUG");
    return env != nullptr && std::strcmp(env, "1") == 0;
}();

const bool kPdDebugEnabled = []() {
    const char* env = std::getenv("RTP_LLM_PD_DEBUG");
    return env != nullptr && std::strcmp(env, "1") == 0;
}();

bool asyncDebugEnabled() {
    return kAsyncDebugEnabled;
}

bool pdDebugEnabled() {
    return kPdDebugEnabled;
}

torch::TensorOptions runtimeCudaI32Options() {
    return torch::TensorOptions().dtype(torch::kInt32).device(getTorchCudaDevice());
}

void checkRuntimeCudaDevice(const torch::Tensor& tensor, const char* name) {
    if (!tensor.defined() || !tensor.is_cuda()) {
        return;
    }
    const auto expected_device = static_cast<int>(getDeviceId());
    RTP_LLM_CHECK_WITH_INFO(tensor.get_device() == expected_device,
                            "%s is on cuda:%d, expected runtime cuda:%d",
                            name,
                            tensor.get_device(),
                            expected_device);
}

std::string tensorSummary(const torch::Tensor& tensor, int64_t limit = 4) {
    if (!tensor.defined()) {
        return "None";
    }
    std::ostringstream oss;
    oss << "shape=[";
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << tensor.size(i);
    }
    oss << "] device=" << tensor.device() << " dtype=" << tensor.dtype() << " numel=" << tensor.numel();
    if (tensor.numel() == 0) {
        return oss.str();
    }
    auto flat       = tensor.reshape({-1});
    auto head_count = std::min<int64_t>(limit, flat.numel());
    auto tail_count = std::min<int64_t>(limit, flat.numel());
    auto head       = flat.slice(0, 0, head_count);
    auto tail       = flat.slice(0, flat.numel() - tail_count, flat.numel());
    if (head.device().is_cuda()) {
        head = head.cpu();
        tail = tail.cpu();
    }
    oss << " head=" << head << " tail=" << tail;
    return oss.str();
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
    BlockIdPair* kv_cache_update_mapping_end;
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
    if (model_input.kv_cache_update_mapping.defined()) {
        RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_update_mapping.dim() == 2
                                    && model_input.kv_cache_update_mapping.size(1) == 2,
                                "kv_cache_update_mapping must be [N,2], got dim=%ld",
                                model_input.kv_cache_update_mapping.dim());
        const size_t total_pairs = static_cast<size_t>(model_input.kv_cache_update_mapping.size(0));
        RTP_LLM_CHECK_WITH_INFO(kv_cache_mapping_offset <= total_pairs,
                                "kv_cache_update_mapping offset overflow: offset=%zu total_pairs=%zu",
                                kv_cache_mapping_offset,
                                total_pairs);
        auto* base                  = reinterpret_cast<BlockIdPair*>(model_input.kv_cache_update_mapping.data_ptr());
        ctx.kv_cache_update_mapping = base + kv_cache_mapping_offset;
        ctx.kv_cache_update_mapping_end = base + total_pairs;
    } else {
        ctx.kv_cache_update_mapping     = nullptr;
        ctx.kv_cache_update_mapping_end = nullptr;
    }

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

    const size_t group           = model_input.kv_cache_kernel_block_id.size(0);
    const size_t batch           = model_input.kv_cache_kernel_block_id.size(1);
    const size_t kernel_capacity = model_input.kv_cache_kernel_block_id.size(2);
    const size_t store_capacity  = model_input.kv_cache_block_id.size(2);
    int32_t*     kernel_dst_base = model_input.kv_cache_kernel_block_id.data_ptr<int32_t>();
    int32_t*     store_dst_base  = model_input.kv_cache_block_id.data_ptr<int32_t>();

    RTP_LLM_CHECK_WITH_INFO(model_input.kv_cache_block_id.size(0) == static_cast<int64_t>(group)
                                && model_input.kv_cache_block_id.size(1) == static_cast<int64_t>(batch),
                            "kv cache block tensors shape mismatch: kernel=[%ld,%ld,%ld] physical=[%ld,%ld,%ld]",
                            model_input.kv_cache_kernel_block_id.size(0),
                            model_input.kv_cache_kernel_block_id.size(1),
                            model_input.kv_cache_kernel_block_id.size(2),
                            model_input.kv_cache_block_id.size(0),
                            model_input.kv_cache_block_id.size(1),
                            model_input.kv_cache_block_id.size(2));
    RTP_LLM_CHECK_WITH_INFO(model_batch_idx >= 0 && static_cast<size_t>(model_batch_idx) < batch,
                            "model_batch_idx overflow: model_batch_idx=%d batch=%zu",
                            model_batch_idx,
                            batch);
    RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(kv_cache.groupNums()) <= group,
                            "kv_cache group overflow: kv_cache.groupNums=%d dst_group=%zu",
                            kv_cache.groupNums(),
                            group);
    RTP_LLM_CHECK_WITH_INFO(kernel_capacity >= max_blocks_num * kernel_blocks_per_kv_block,
                            "kv_cache_kernel_block_id capacity too small: capacity=%zu required=%zu",
                            kernel_capacity,
                            max_blocks_num * kernel_blocks_per_kv_block);
    RTP_LLM_CHECK_WITH_INFO(store_capacity >= max_blocks_num,
                            "kv_cache_block_id capacity too small: capacity=%zu required=%zu",
                            store_capacity,
                            max_blocks_num);

    for (int gid = 0; gid < kv_cache.groupNums(); ++gid) {
        auto& kernel_blocks = kv_cache.kernelBlocks(stream_batch_idx, gid);
        RTP_LLM_CHECK_WITH_INFO(kernel_blocks.size() <= kernel_capacity,
                                "kernel block copy overflow: gid=%d stream_batch_idx=%d size=%zu capacity=%zu",
                                gid,
                                stream_batch_idx,
                                kernel_blocks.size(),
                                kernel_capacity);
        int32_t* kernel_dst =
            kernel_dst_base
            + (static_cast<size_t>(gid) * batch + static_cast<size_t>(model_batch_idx)) * kernel_capacity;
        std::memcpy(kernel_dst, kernel_blocks.data(), kernel_blocks.size() * sizeof(int32_t));

        auto& physical_blocks = kv_cache.blocks(stream_batch_idx, gid);
        RTP_LLM_CHECK_WITH_INFO(physical_blocks.size() <= store_capacity,
                                "physical block copy overflow: gid=%d stream_batch_idx=%d size=%zu capacity=%zu",
                                gid,
                                stream_batch_idx,
                                physical_blocks.size(),
                                store_capacity);
        int32_t* store_dst =
            store_dst_base + (static_cast<size_t>(gid) * batch + static_cast<size_t>(model_batch_idx)) * store_capacity;
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
            gathered_mm_features.emplace_back(mm_feature.to(getTorchCudaDevice(), /*non_blocking=*/true));
        } else {
            checkRuntimeCudaDevice(mm_feature, "multimodal feature");
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
    RTP_LLM_CHECK_WITH_INFO(ctx.kv_cache_update_mapping + update_copy_num <= ctx.kv_cache_update_mapping_end,
                            "kv_cache_update_mapping overflow: update_copy_num=%zu remaining=%zu",
                            update_copy_num,
                            static_cast<size_t>(ctx.kv_cache_update_mapping_end - ctx.kv_cache_update_mapping));
    std::memcpy(ctx.kv_cache_update_mapping, update_mapping.data(), update_copy_num * sizeof(BlockIdPair));
    ctx.kv_cache_update_mapping += update_copy_num;
}

torch::Tensor buildLmOutputIndexesOnCuda(const GptModelInputs& model_input, const StreamGroups& stream_groups) {
    const auto total_batch_size         = static_cast<int64_t>(stream_groups.totalModelBatchSize());
    const auto total_decode_batch_size  = static_cast<int64_t>(stream_groups.totalDecodeBatchSize());
    const auto total_context_batch_size = total_batch_size - total_decode_batch_size;
    auto       cuda_i32                 = runtimeCudaI32Options();

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
    auto cuda_i32 = runtimeCudaI32Options();
    if (tensor.is_cuda()) {
        checkRuntimeCudaDevice(tensor, "publishInt32ToCuda input");
    }
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
    const auto        cuda_i32    = runtimeCudaI32Options();

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
        auto* dst = model_input.kv_cache_layer_to_group.data_ptr<int32_t>();
        std::fill(dst, dst + dst_numel, 0);
        if (src_numel > 0) {
            std::memcpy(dst, config_.layer_to_kv_cache_group_id.data(), src_numel * sizeof(int32_t));
        }
    }
    if (model_input.kv_cache_group_types.defined()) {
        auto* dst = model_input.kv_cache_group_types.data_ptr<int32_t>();
        if (config_.kv_cache_group_types.empty()) {
            std::fill(dst, dst + config_.kv_cache_group_nums, static_cast<int32_t>(CacheGroupType::FULL));
        } else {
            RTP_LLM_CHECK_WITH_INFO(config_.kv_cache_group_types.size() >= config_.kv_cache_group_nums,
                                    "kv_cache_group_types overflow: group_types=%zu kv_cache_group_nums=%zu",
                                    config_.kv_cache_group_types.size(),
                                    config_.kv_cache_group_nums);
            for (size_t g = 0; g < config_.kv_cache_group_nums; ++g) {
                dst[g] = static_cast<int32_t>(config_.kv_cache_group_types[g]);
            }
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
    const bool pd_debug_enabled     = pdDebugEnabled();
    bool       pd_debug_long_decode = false;

    for (const auto& stream : stream_groups.decodeStreams()) {
        model_input.need_all_logits        = model_input.need_all_logits || stream->calculateLoss();
        model_input.need_all_hidden_states = model_input.need_all_hidden_states || stream->needReturnHiddenStates();
        auto  current_batch_size           = stream->currentBatchSize();
        auto& kv_cache                     = *stream->kvCachePtr();
        RTP_LLM_LOG_DEBUG("decode kv_cache: %s", kv_cache.debugString().c_str());
        RTP_LLM_LOG_DEBUG("decode stream: %s", stream->debugString().c_str());
        if (pd_debug_enabled) {
            pd_debug_long_decode = pd_debug_long_decode || stream->inputLength() > 1024 || stream->seqLength() > 1024;
        }

        for (auto i = 0; i < current_batch_size; ++i) {
            model_input.trace_ids.push_back(stream->traceId());
            if (use_normal_device_state) {
                const auto& state = stream->getNormalAsyncDeviceState();
                checkRuntimeCudaDevice(state.last_sample_token_gpu, "normal async last_sample_token_gpu");
                checkRuntimeCudaDevice(state.next_seq_len_gpu, "normal async next_seq_len_gpu");
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
        model_input.combo_tokens     = torch::cat(normal_combo_tokens_gpu, 0).to(runtimeCudaI32Options());
        model_input.sequence_lengths = torch::cat(normal_sequence_lengths_gpu, 0).to(runtimeCudaI32Options());
    }
    if (pd_debug_enabled && pd_debug_long_decode) {
        static std::atomic<int> debug_log_budget{256};
        if (debug_log_budget.fetch_sub(1, std::memory_order_relaxed) > 0) {
            RTP_LLM_LOG_INFO("[PD_DEBUG][MODEL_INPUT_DECODE] use_normal_device_state=%d total_decode_bs=%zu "
                             "max_blocks=%zu combo_tokens=%s input_lengths=%s sequence_lengths=%s "
                             "kv_kernel_blocks=%s kv_blocks=%s",
                             static_cast<int>(use_normal_device_state),
                             stream_groups.totalDecodeBatchSize(),
                             ctx.max_blocks_num,
                             tensorSummary(model_input.combo_tokens).c_str(),
                             tensorSummary(model_input.input_lengths).c_str(),
                             tensorSummary(model_input.sequence_lengths).c_str(),
                             tensorSummary(model_input.kv_cache_kernel_block_id).c_str(),
                             tensorSummary(model_input.kv_cache_block_id).c_str());
        }
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
    // TODO(async): prefixLength() is still stream CPU state. Stage it explicitly
    // on host here, then publish only a CUDA tensor in GptModelInputs.
    auto prefix_lengths_host =
        torch::empty({context_batch_size}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    auto ctx                = createGatherContext(config_, model_input, stream_groups, GatherContextMode::CONTEXT);
    ctx.prefix_lengths_host = prefix_lengths_host.data_ptr<int32_t>();

    for (const auto& stream : stream_groups.contextStreams()) {
        model_input.need_all_logits        = model_input.need_all_logits || stream->calculateLoss();
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

            copyKvCacheBlocksToModelInput(
                model_input, kv_cache, i, ctx.batch_idx, ctx.max_blocks_num, config_.kernel_blocks_per_kv_block);

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
    model_input.prefix_lengths = publishInt32ToCuda(prefix_lengths_host, host_holder);
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
    publishModelInputCoreTensorsToCuda(model_input, host_holder);
    model_input.lm_output_indexes = buildLmOutputIndexesOnCuda(model_input, stream_groups);
    return model_input;
}

}  // namespace rtp_llm
