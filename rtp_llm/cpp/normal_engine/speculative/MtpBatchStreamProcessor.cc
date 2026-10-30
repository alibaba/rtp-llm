#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalInputUtils.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#endif
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <map>
#include <limits>
#include <numeric>
#include <string>
#include <vector>
#include <cstring>

namespace rtp_llm {
namespace {
int32_t mtpTailTextMask(const GenerateStreamPtr& stream) {
    const auto& mask     = stream->generateInput()->text_tokens_mask;
    const int   boundary = stream->prefixLength() + stream->contextLength();
    return stream->isMiddleChunk() && mask && boundary < mask->numel() ? mask->data_ptr<int32_t>()[boundary] : 1;
}

struct MtpPrefillRow {
    GenerateStream* stream;
    int64_t         begin;
    int64_t         end;
    int32_t         tail_text_mask;
};

torch::Tensor shiftedMtpTokenField(const torch::Tensor&              field,
                                    const std::vector<MtpPrefillRow>& rows,
                                    bool                              is_text_mask,
                                    TensorHolder&                     host_holder) {
    if (!field.defined() || field.numel() == 0) {
        return field;
    }
    RTP_LLM_CHECK_WITH_INFO(field.scalar_type() == torch::kInt32,
                            "MTP multimodal token field must be int32, got %s",
                            c10::toString(field.scalar_type()));
    const int64_t total_tokens = rows.empty() ? 0 : rows.back().end;
    RTP_LLM_CHECK_WITH_INFO(field.numel() == total_tokens,
                            "MTP token field length mismatch: field=%ld input_lengths=%ld",
                            field.numel(),
                            total_tokens);
    const auto source = field.cpu().contiguous();
    auto output =
        torch::empty({total_tokens}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    const auto* src    = source.data_ptr<int32_t>();
    auto*       dst    = output.data_ptr<int32_t>();
    for (const auto& row : rows) {
        const int64_t length = row.end - row.begin;
        if (length > 1) {
            std::memcpy(dst + row.begin, src + row.begin + 1, static_cast<size_t>(length - 1) * sizeof(int32_t));
        }
        dst[row.end - 1] = is_text_mask ? row.tail_text_mask : 0;
    }
    host_holder.hold_host(output);
    return field.is_cuda() ? output.to(field.device(), /*non_blocking=*/true) : output;
}

void shiftMtpMultimodalLocations(GptModelInputs&                   model_input,
                                 const std::vector<MtpPrefillRow>& rows,
                                 TensorHolder&                    host_holder) {
    const bool has_features = model_input.multimodal_features.has_value() && !model_input.multimodal_features->empty();
    const bool has_locs     = model_input.mm_features_locs.defined() && model_input.mm_features_locs.numel() > 0;
    RTP_LLM_CHECK_WITH_INFO(has_features == has_locs,
                            "MTP multimodal features and mm_features_locs must be provided together");
    auto locs = has_locs ? model_input.mm_features_locs.cpu() : torch::empty({0}, torch::kInt32);
    locs = locs.contiguous();
    RTP_LLM_CHECK_WITH_INFO(locs.scalar_type() == torch::kInt32 && locs.dim() == 1,
                            "MTP mm_features_locs must be a 1-D int32 tensor");
    const auto* features = has_features ? &model_input.multimodal_features.value() : nullptr;
    const auto  feature_count = features ? features->size() : 0;
    RTP_LLM_CHECK_WITH_INFO(locs.numel() == static_cast<int64_t>(feature_count),
                            "MTP multimodal feature/location count mismatch: features=%zu locs=%ld",
                            feature_count,
                            locs.numel());
    const bool has_extra_input =
        (model_input.mm_extra_input.has_value() && !model_input.mm_extra_input->empty())
        || std::any_of(rows.begin(), rows.end(), [](const auto& row) {
               return row.tail_text_mask == 0 && row.stream->hasMultimodalExtraInput();
           });
    const auto* extra_input = model_input.mm_extra_input ? &model_input.mm_extra_input.value() : nullptr;
    RTP_LLM_CHECK_WITH_INFO(!has_extra_input || feature_count == 0
                                || (extra_input && extra_input->size() == feature_count),
                            "MTP mm_extra_input count mismatch: extra_input=%zu features=%zu",
                            extra_input ? extra_input->size() : 0,
                            feature_count);

    std::vector<torch::Tensor> shifted_features;
    std::vector<torch::Tensor> shifted_extra_input;
    std::vector<int32_t>       shifted_locs;
    std::vector<int64_t>       shifted_spans;
    const size_t capacity = feature_count + rows.size();
    shifted_features.reserve(capacity);
    shifted_locs.reserve(capacity);
    shifted_spans.reserve(capacity * 3);
    if (has_extra_input) {
        shifted_extra_input.reserve(capacity);
    }

    const auto* src_locs = locs.data_ptr<int32_t>();
    size_t feature_idx = 0;
    GenerateStream* tail_stream = nullptr;
    torch::Tensor tail_feature, tail_extra;
    // Gather produces features in packed request order. Consume each feature
    // once, then append that request's boundary image row (if any).
    for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
        const auto& row = rows[row_idx];
        for (; feature_idx < feature_count && src_locs[feature_idx] < row.end; ++feature_idx) {
            const auto& feature = (*features)[feature_idx];
            RTP_LLM_CHECK_WITH_INFO(feature.defined() && feature.dim() == 2,
                                    "MTP multimodal feature %zu must be a defined 2-D tensor", feature_idx);
            const int64_t start  = src_locs[feature_idx];
            const int64_t length = feature.size(0);
            RTP_LLM_CHECK_WITH_INFO(start >= row.begin && length > 0 && start + length <= row.end,
                                    "MTP multimodal feature %zu [%ld,%ld) is outside request %zu [%ld,%ld)",
                                    feature_idx, start, start + length, row_idx, row.begin, row.end);
            // Removing the first token also removes the first feature row
            // only when the image starts at the request boundary.
            const int64_t trim = start == row.begin ? 1 : 0;
            if (length == trim) {
                continue;
            }
            shifted_features.emplace_back(feature.slice(0, trim, length).contiguous());
            shifted_locs.emplace_back(static_cast<int32_t>(start + trim - 1));
            const int64_t prompt_start = row.stream->prefixLength() + start - row.begin + trim;
            shifted_spans.insert(shifted_spans.end(),
                                 {static_cast<int64_t>(row_idx), prompt_start, prompt_start + length - trim});
            if (has_extra_input) {
                shifted_extra_input.emplace_back(
                    sliceMultimodalExtraInput((*extra_input)[feature_idx], feature, trim, length));
            }
        }
        if (row.tail_text_mask != 0) {
            continue;
        }
        const int64_t boundary = row.stream->prefixLength() + row.stream->contextLength();
        // Multiple batch rows from one stream share the same immutable tail.
        if (tail_stream != row.stream) {
            const auto raw_features = row.stream->multimodalFeatures();
            const auto raw_extra    = row.stream->multimodalExtraInput();
            const auto raw_locs     = row.stream->multimodalLocations();
            RTP_LLM_CHECK_WITH_INFO(raw_locs.defined() && raw_locs.numel() == static_cast<int64_t>(raw_features.size())
                                        && (!has_extra_input || raw_extra.size() == raw_features.size()),
                                    "MTP boundary image metadata is incomplete");
            tail_feature = torch::Tensor();
            for (size_t i = 0; i < raw_features.size(); ++i) {
                const int64_t offset = boundary - raw_locs.data_ptr<int32_t>()[i];
                if (offset < 0 || offset >= raw_features[i].size(0)) {
                    continue;
                }
                auto feature = raw_features[i].narrow(0, offset, 1).contiguous();
                host_holder.hold_host(feature);
                tail_feature = feature.to(torch::kCUDA, /*non_blocking=*/true);
                if (has_extra_input) {
                    auto extra = sliceMultimodalExtraInput(raw_extra[i], raw_features[i], offset, offset + 1);
                    host_holder.hold_host(extra);
                    tail_extra = extra.to(torch::kCUDA, /*non_blocking=*/true);
                }
                break;
            }
            RTP_LLM_CHECK_WITH_INFO(tail_feature.defined(), "MTP image boundary at %ld has no feature row", boundary);
            tail_stream = row.stream;
        }
        shifted_features.emplace_back(tail_feature);
        shifted_locs.emplace_back(static_cast<int32_t>(row.end - 1));
        shifted_spans.insert(shifted_spans.end(), {static_cast<int64_t>(row_idx), boundary, boundary + 1});
        if (has_extra_input) {
            shifted_extra_input.emplace_back(tail_extra);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(feature_idx == feature_count, "MTP multimodal feature lies beyond the packed input");

    auto shifted_locs_cpu = torch::empty({static_cast<int64_t>(shifted_locs.size())},
                                         torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    if (!shifted_locs.empty()) {
        std::memcpy(shifted_locs_cpu.data_ptr<int32_t>(), shifted_locs.data(), shifted_locs.size() * sizeof(int32_t));
    }
    model_input.multimodal_features = std::move(shifted_features);
    if (has_extra_input) {
        model_input.mm_extra_input = std::move(shifted_extra_input);
    }
    auto spans_cpu = torch::tensor(shifted_spans, torch::kInt64).reshape({-1, 3}).pin_memory();
    host_holder.hold_host(spans_cpu);
    host_holder.hold_host(shifted_locs_cpu);
    model_input.mm_features_spans = model_input.mm_features_spans.defined() && model_input.mm_features_spans.is_cuda() ?
        spans_cpu.to(model_input.mm_features_spans.device(), /*non_blocking=*/true) : spans_cpu;
    model_input.mm_features_locs =
        model_input.mm_features_locs.defined() && model_input.mm_features_locs.is_cuda() ?
            shifted_locs_cpu.to(model_input.mm_features_locs.device(), /*non_blocking=*/true) :
            shifted_locs_cpu;
}

}  // namespace

namespace {

torch::Tensor cloneHiddenSlice(const torch::Tensor& hidden_states, int64_t start, int64_t length) {
    if (!hidden_states.defined() || length <= 0) {
        return torch::Tensor();
    }
    RTP_LLM_CHECK_WITH_INFO(
        hidden_states.dim() == 2, "MTP hidden states must be 2-D, got dim=%ld", hidden_states.dim());
    RTP_LLM_CHECK_WITH_INFO(start >= 0 && start + length <= hidden_states.size(0),
                            "MTP hidden slice out of range: start=%ld, length=%ld, rows=%ld",
                            start,
                            length,
                            hidden_states.size(0));
    return hidden_states.narrow(0, start, length).clone();
}

}  // namespace

torch::Tensor MtpBatchStreamProcessor::makeEngramVerifyWindows(const torch::Tensor& anchor,
                                                               const torch::Tensor& tokens) {
    RTP_LLM_CHECK_WITH_INFO(anchor.dim() == 2 && anchor.size(1) == 4 && tokens.dim() == 2
                                && anchor.size(0) == tokens.size(0),
                            "invalid Engram verify history");
    auto chronological = torch::cat({anchor.narrow(1, 1, 3).flip({1}), tokens}, 1);
    return chronological.unfold(1, 4, 1).flip({2}).contiguous().reshape({tokens.numel(), 4});
}

torch::Tensor MtpBatchStreamProcessor::advanceEngramTokenWindows(const torch::Tensor& anchor,
                                                                 const torch::Tensor& tokens,
                                                                 const torch::Tensor& lengths) {
    auto chronological = torch::cat({anchor.flip({1}), tokens}, 1);
    auto indexes       = lengths.reshape({-1, 1}).to(torch::kLong)
                   + torch::arange(3, -1, -1, lengths.options().dtype(torch::kLong)).reshape({1, 4});
    return chronological.gather(1, indexes);
}

GptModelOutputs MtpBatchStreamProcessor::gatherAcceptedTargetDiagnostics(const GptModelOutputs& target_output,
                                                                         const torch::Tensor&   accept_lengths,
                                                                         int64_t                verify_width) {
    GptModelOutputs result;
    if (!target_output.logits.defined() && !target_output.hidden_states.defined()) {
        return result;
    }
    TORCH_CHECK(verify_width > 0 && accept_lengths.dim() == 1 && accept_lengths.numel() > 0,
                "Target diagnostics require nonempty per-request accepted lengths and a positive verify width");
    TORCH_CHECK(accept_lengths.scalar_type() == torch::kInt32 || accept_lengths.scalar_type() == torch::kInt64,
                "Target diagnostic accepted lengths must be integers");
    // Diagnostic requests already return device tensors to the host. Keep this
    // validation off the default path, and reject invalid rows rather than clamp.
    TORCH_CHECK(accept_lengths.min().item<int64_t>() >= 1 && accept_lengths.max().item<int64_t>() <= verify_width,
                "Target diagnostic accepted length is outside the verify window");
    auto gather = [&](const torch::Tensor& values) {
        if (!values.defined()) {
            return torch::Tensor();
        }
        const auto batch = accept_lengths.numel();
        TORCH_CHECK(values.dim() == 2 && values.size(0) >= batch * verify_width,
                    "Target diagnostic rows do not cover the logical verify batch");
        auto rows = torch::arange(batch, values.options().dtype(torch::kInt64)) * verify_width
                    + accept_lengths.to(values.device(), torch::kInt64) - 1;
        // index_select allocates: these outputs cannot alias graph buffers.
        return values.index_select(0, rows);
    };
    result.logits        = gather(target_output.logits);
    result.hidden_states = gather(target_output.hidden_states);
    return result;
}

torch::Tensor MtpBatchStreamProcessor::advanceLinearCacheBlockTable(const torch::Tensor& current_table,
                                                                    const torch::Tensor& previous_seq_lengths,
                                                                    const torch::Tensor& accept_lengths,
                                                                    const std::vector<CacheGroupType>& group_types,
                                                                    const std::vector<int64_t>&        group_spans) {
    RTP_LLM_PROFILE_SCOPE("executor.mtp.advance_linear_cache_block_table");
    RTP_LLM_CHECK_WITH_INFO(current_table.defined() && current_table.is_cuda()
                                && current_table.scalar_type() == torch::kInt32 && current_table.dim() == 3,
                            "MTP cache snapshot source must be CUDA int32 [group,batch,blocks]");
    RTP_LLM_CHECK_WITH_INFO(previous_seq_lengths.defined() && previous_seq_lengths.is_cuda()
                                && previous_seq_lengths.scalar_type() == torch::kInt32
                                && previous_seq_lengths.dim() == 1,
                            "MTP cache snapshot previous_seq_lengths must be CUDA int32 [batch]");
    RTP_LLM_CHECK_WITH_INFO(accept_lengths.defined() && accept_lengths.is_cuda()
                                && accept_lengths.scalar_type() == torch::kInt32 && accept_lengths.dim() == 1,
                            "MTP cache snapshot accept_lengths must be CUDA int32 [batch]");
    const int64_t group_count = current_table.size(0);
    const int64_t batch_size  = current_table.size(1);
    const int64_t block_count = current_table.size(2);
    RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(group_types.size()) == group_count,
                            "MTP cache snapshot group count mismatch: table=%ld config=%zu",
                            group_count,
                            group_types.size());
    RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(group_spans.size()) == group_count,
                            "MTP cache snapshot span count mismatch: table=%ld spans=%zu",
                            group_count,
                            group_spans.size());
    for (int64_t group_id = 0; group_id < group_count; ++group_id) {
        if (group_types[static_cast<size_t>(group_id)] == CacheGroupType::LINEAR) {
            RTP_LLM_CHECK_WITH_INFO(group_spans[static_cast<size_t>(group_id)] > 0,
                                    "MTP LINEAR cache snapshot span must be positive: group=%ld span=%ld",
                                    group_id,
                                    group_spans[static_cast<size_t>(group_id)]);
        }
    }
    RTP_LLM_CHECK_WITH_INFO(previous_seq_lengths.numel() == batch_size && accept_lengths.numel() == batch_size,
                            "MTP cache snapshot batch mismatch: table=%ld previous=%ld accept=%ld",
                            batch_size,
                            previous_seq_lengths.numel(),
                            accept_lengths.numel());
    if (batch_size == 0 || block_count == 0
        || std::find(group_types.begin(), group_types.end(), CacheGroupType::LINEAR) == group_types.end()) {
        return current_table.clone();
    }

    // Vectorize LinearBlocksUtil.h's two sequential swaps over requests, in
    // each group's own block-table coordinates. The caller checks effective
    // per-request row capacities; block_count also includes padding from
    // other requests/groups and cannot establish that contract itself.
    const auto long_options = current_table.options().dtype(torch::kInt64);
    const auto identity =
        torch::arange(block_count, long_options).reshape({1, block_count}).expand({batch_size, block_count});
    auto accept         = accept_lengths.to(torch::kInt64);
    auto current_cached = previous_seq_lengths.to(torch::kInt64) - 1;
    auto next_cached    = current_cached + accept;
    auto active         = accept > 1;
    auto zeros          = torch::zeros_like(current_cached);

    auto next_table = current_table.clone();
    for (int64_t group_id = 0; group_id < group_count; ++group_id) {
        if (group_types[static_cast<size_t>(group_id)] != CacheGroupType::LINEAR) {
            continue;
        }
        const auto seq_size_per_block = group_spans[static_cast<size_t>(group_id)];
        auto       permutation        = identity.clone();
        auto       apply_swap = [&](torch::Tensor source, torch::Tensor destination, const torch::Tensor& enabled) {
            auto safe_source        = torch::where(enabled, source, zeros);
            auto safe_destination   = torch::where(enabled, destination, zeros);
            auto source_values      = permutation.gather(1, safe_source.unsqueeze(1)).squeeze(1);
            auto destination_values = permutation.gather(1, safe_destination.unsqueeze(1)).squeeze(1);
            permutation.scatter_(
                1, safe_source.unsqueeze(1), torch::where(enabled, destination_values, source_values).unsqueeze(1));
            permutation.scatter_(1,
                                 safe_destination.unsqueeze(1),
                                 torch::where(enabled, source_values, destination_values).unsqueeze(1));
        };

        auto cached_swap_needed = torch::remainder(current_cached + 1, seq_size_per_block)
                                  > torch::remainder(next_cached + seq_size_per_block - 1, seq_size_per_block);
        auto cached_source = torch::floor_divide(current_cached, seq_size_per_block)
                             + (seq_size_per_block - torch::remainder(current_cached, seq_size_per_block) - 1);
        auto cached_destination = torch::floor_divide(next_cached, seq_size_per_block) - 1;
        apply_swap(cached_source, cached_destination, active & cached_swap_needed);

        auto final_source      = torch::floor_divide(current_cached, seq_size_per_block) + accept - 1;
        auto final_destination = torch::floor_divide(next_cached - 1, seq_size_per_block);
        apply_swap(final_source, final_destination, active);

        next_table.select(0, group_id).copy_(current_table.select(0, group_id).gather(1, permutation));
    }
    return next_table;
}

void MtpBatchStreamProcessor::expandTargetVerifyPositionIdsFromDecode(GptModelInputs& model_input) const {
    if (!model_input.combo_position_ids.defined() || model_input.combo_position_ids.numel() == 0)
        return;
    const int64_t batch  = model_input.input_lengths.numel();
    const int64_t factor = model_input_gatherer_config_.position_id_len_factor;
    const int64_t width  = propose_step_ + 1;
    if (propose_step_ == 1 && model_input.combo_position_ids.numel() == batch * width * factor)
        return;
    RTP_LLM_CHECK_WITH_INFO(factor > 0 && model_input.combo_position_ids.numel() == batch * factor,
                            "target prepare position bases must have one row per request");
    // The bases have already been TP-broadcast. Do not consult mutable stream
    // state, or fabricate zero positions on non-driver ranks.
    auto bases   = model_input.combo_position_ids.reshape({batch, 1, factor});
    auto offsets = torch::arange(width, bases.options()).reshape({1, width, 1});
    // CPU arithmetic does not inherit the input's pinned allocator. Write
    // directly into pinned storage for PyWrappedModel's deferred fused H2D copy.
    auto options            = bases.is_cuda() ? bases.options() : bases.options().pinned_memory(true);
    auto expanded_positions = torch::empty({batch, width, factor}, options);
    torch::add_out(expanded_positions, bases, offsets);
    model_input.combo_position_ids = expanded_positions.reshape({-1});
}

void MtpBatchStreamProcessor::expandTargetVerifyPositionIds(const StreamGroups& stream_groups,
                                                            GptModelInputs&     model_input) const {
    if (!model_input.combo_position_ids.defined()) {
        return;
    }

    const size_t position_id_len_factor = model_input_gatherer_config_.position_id_len_factor;
    const size_t batch_size             = model_input.combo_position_ids.numel() / position_id_len_factor;
    auto         target_combo_position_ids =
        torch::empty({(int64_t)(batch_size * (propose_step_ + 1) * position_id_len_factor)}, torch::kInt32)
            .pin_memory();

    int* dst_position_ids = target_combo_position_ids.data_ptr<int>();
    if (stream_groups.empty()) {
        // Non-driver TP ranks only need a correctly shaped placeholder here; draftModelDecode broadcasts rank0's
        // stream-derived combo_position_ids into this storage before target verify.
        target_combo_position_ids.zero_();
        model_input.combo_position_ids = std::move(target_combo_position_ids);
        return;
    }

    size_t batch_idx = 0;
    // Speculative decoding rejects num_return_sequences > 1 and beam search before this path.
    for (const auto& stream : stream_groups.allStreams()) {
        int* base_position_ids = dst_position_ids + batch_idx * (propose_step_ + 1) * position_id_len_factor;
        stream->generateNextPositionId(base_position_ids);
        for (int step = 0; step < propose_step_ + 1; ++step) {
            int* step_position_ids = base_position_ids + step * position_id_len_factor;
            for (size_t dim = 0; dim < position_id_len_factor; ++dim) {
                step_position_ids[dim] = base_position_ids[dim] + step;
            }
        }
        batch_idx++;
    }

    model_input.combo_position_ids = std::move(target_combo_position_ids);
}

namespace {

// Fallback-hit counters. Each counter is incremented when a hot-path
// gather/prepare function falls through to the legacy CPU/GPU mixed path.
// The counters are read by the rate-limited log emitter below; production
// monitoring should prefer the kmonitor metric `executor.mtp.async_fallback.reason` once it is wired, but this
// in-process counter provides immediate visibility today without requiring a metrics-schema change.
std::atomic<uint64_t> g_mtp_device_state_fallback_count{0};
std::atomic<uint64_t> g_mtp_device_state_success_count{0};

// Rate-limited log emitter — first 5 fallbacks logged in full, then every
// 1000th hit. Avoids drowning the log when a config keeps hitting the
// fallback path. Returns true when the caller should emit the log line.
bool shouldLogFallback(uint64_t count) {
    if (count <= 5) {
        return true;
    }
    return count % 1000 == 0;
}

bool useMtpDeviceInput() {
    static const bool enabled = []() {
        const char* env = std::getenv("RTP_LLM_DEVICE_INPUT");
        bool        on  = (env != nullptr && std::string(env) == "1");
        RTP_LLM_LOG_INFO("[mtp-device-input] RTP_LLM_DEVICE_INPUT=%s -> processor enabled=%d",
                         env ? env : "(unset)",
                         static_cast<int>(on));
        return on;
    }();
    return enabled;
}

bool useMtpDeviceState() {
    const char* stream_async = std::getenv("RTP_LLM_STREAM_ASYNC");
    const char* device_state = std::getenv("RTP_LLM_MTP_ASYNC_DEVICE_STATE");
    return (stream_async != nullptr && std::string(stream_async) == "1")
           || (device_state != nullptr && std::string(device_state) == "1");
}

torch::Tensor emptyInt32OnPreferredDevice(std::initializer_list<int64_t> shape) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(useMtpDeviceInput() ? torch::kCUDA : torch::kCPU);
    if (!useMtpDeviceInput()) {
        options = options.pinned_memory(true);
    }
    return torch::empty(shape, options);
}

torch::TensorOptions cudaInt32Options() {
    return torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
}

torch::Tensor emptyInt32OnCuda(std::initializer_list<int64_t> shape) {
    return torch::empty(shape, cudaInt32Options());
}

torch::Tensor fullInt32OnCuda(std::initializer_list<int64_t> shape, int64_t value) {
    return torch::full(shape, value, cudaInt32Options());
}

torch::Tensor toCudaInt32(const torch::Tensor& tensor, TensorHolder& host_holder) {
    if (!tensor.defined()) {
        return tensor;
    }
    if (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) {
        return tensor;
    }
    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), cudaInt32Options());
    }
    host_holder.hold_host(tensor);
    return tensor.to(cudaInt32Options(), /*non_blocking=*/true);
}

torch::Tensor lastColumnAsFlat(const torch::Tensor& tensor) {
    const int64_t last_col = tensor.size(-1) - 1;
    return tensor.select(-1, last_col).reshape({-1});
}

torch::Tensor columnAsFlat(const torch::Tensor& tensor, int64_t col) {
    if (!tensor.defined() || tensor.numel() == 0 || tensor.dim() == 0) {
        return torch::Tensor();
    }
    const int64_t dim = tensor.dim() - 1;
    if (col < 0) {
        col += tensor.size(dim);
    }
    if (col < 0 || col >= tensor.size(dim)) {
        return torch::Tensor();
    }
    return tensor.select(dim, col).reshape({-1});
}

// DSpark anchor: the stream's newest committed token. Steady streams read the
// per-round GPU accept state; a fresh stream (first round after PD handoff)
// falls back to host token ids, where the handoff wrote the first generated
// token. The SPOutputBuffer is deliberately not consulted: prepareStreams
// zero-initializes it for dspark streams, so its content is noise here.
torch::Tensor dsparkNewestToken(const GenerateStreamPtr& stream, const GenerateStream::MtpAsyncDeviceState& state) {
    const auto& accept_tokens = state.accept_tokens_gpu;
    const auto& accept_len    = state.accept_len_gpu;
    if (accept_tokens.defined() && accept_tokens.is_cuda() && accept_len.defined() && accept_len.is_cuda()) {
        auto idx_t = (accept_len - 1).to(torch::kLong);
        return accept_tokens.squeeze(0).index_select(/*dim=*/0, idx_t);
    }
    if (stream->isFakeStream()) {
        return torch::zeros({1}, cudaInt32Options());
    }
    return stream->completeTokenIds()
        .index({0, static_cast<int64_t>(stream->seqLength()) - 1})
        .reshape({1})
        .to(torch::kInt32);
}

// Round-head per-stream state: the anchor (newest committed token) and the
// committed end (committed length - 1). The propose and verify builders read
// exactly the same stream state, so both consume this one derivation.
std::pair<torch::Tensor, torch::Tensor>
dsparkRoundHeadState(const StreamGroups& stream_groups, const GptModelInputs& model_input, TensorHolder& host_holder) {
    const int64_t              batch_size = static_cast<int64_t>(stream_groups.size());
    std::vector<torch::Tensor> anchors;
    std::vector<torch::Tensor> committed_end_parts;
    anchors.reserve(batch_size);
    committed_end_parts.reserve(batch_size);
    // Both arms are per-stream, mirroring the anchors: a fresh stream
    // (proposal-less PD handoff, no device state yet) has no bookkeeping
    // worker in flight, so its host sequence length is safe; steady streams
    // must use the device value, which under stream-async can be ahead of
    // the host copy. A batch-wide host fallback would shift the steady
    // streams' geometry backwards whenever one fresh stream joins the batch.
    torch::Tensor host_seq_lens;
    int64_t       idx = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        const auto state = stream->getMtpAsyncDeviceState();
        anchors.push_back(toCudaInt32(dsparkNewestToken(stream, state), host_holder).reshape({1}));
        const auto& next_seq_len = state.next_seq_len_gpu;
        if (next_seq_len.defined() && next_seq_len.is_cuda()) {
            committed_end_parts.push_back((next_seq_len.reshape({1}) - 1).to(torch::kInt32));
        } else {
            if (!host_seq_lens.defined()) {
                host_seq_lens = toCudaInt32(model_input.sequence_lengths, host_holder);
            }
            committed_end_parts.push_back(host_seq_lens.narrow(0, idx, 1));
        }
        ++idx;
    }
    auto anchors_cat    = torch::cat(anchors, 0);
    auto committed_ends = torch::cat(committed_end_parts, 0);
    return {std::move(anchors_cat), std::move(committed_ends)};
}

torch::Tensor pickOneStepTargetLastToken(const GenerateStreamPtr& stream) {
    const auto  state         = stream->getMtpAsyncDeviceState();
    const auto& accept_tokens = state.accept_tokens_gpu;
    const auto& accept_len    = state.accept_len_gpu;
    if (useMtpDeviceState() && accept_tokens.defined() && accept_tokens.is_cuda() && accept_len.defined()
        && accept_len.is_cuda()) {
        auto idx_t = (accept_len - 1).to(torch::kLong);
        return accept_tokens.squeeze(0).index_select(/*dim=*/0, idx_t);
    }

    auto sp_output_buffer = stream->getSPOutputBuffer();
    if (!sp_output_buffer) {
        return torch::Tensor();
    }
    return columnAsFlat(sp_output_buffer->tokens, 0);
}

torch::Tensor pickOneStepDraftToken(const GenerateStreamPtr& stream) {
    const auto& state_propose = stream->getProposeTokensGpu();
    if (useMtpDeviceState() && state_propose.defined()) {
        return lastColumnAsFlat(state_propose);
    }

    auto sp_output_buffer = stream->getSPOutputBuffer();
    if (!sp_output_buffer) {
        return torch::Tensor();
    }
    if (sp_output_buffer->propose_tokens_gpu.defined()) {
        return lastColumnAsFlat(sp_output_buffer->propose_tokens_gpu);
    }
    return columnAsFlat(sp_output_buffer->tokens, 1);
}

torch::Tensor makeCudaInt32Range(int64_t end) {
    return torch::arange(0, end, cudaInt32Options());
}

torch::Tensor committedLenToDraftDecodePosition(const torch::Tensor& committed_len, TensorHolder& host_holder) {
    // The carried target token is counted in committed_len, but shifted
    // draft KV has only committed_len - 1 entries before the next proposal.
    return (toCudaInt32(committed_len, host_holder) - 1).to(torch::kInt32);
}

torch::Tensor normalDecodePositionToDraftDecodePosition(const torch::Tensor& normal_decode_position,
                                                        TensorHolder&        host_holder) {
    auto position = toCudaInt32(normal_decode_position, host_holder);
    if (!position.defined() || position.numel() == 0) {
        return position;
    }
    return position;
}

void setVerifyPairInputs(GptModelInputs& model_input,
                         torch::Tensor   combo_tokens,
                         size_t          batch_size,
                         size_t          score_len,
                         TensorHolder&   host_holder) {
    model_input.combo_tokens       = std::move(combo_tokens);
    model_input.sequence_lengths   = emptyInt32OnCuda({0});
    model_input.last_hidden_states = torch::Tensor();
    model_input.prefix_lengths     = toCudaInt32(model_input.prefix_lengths, host_holder).contiguous();
    model_input.input_lengths      = fullInt32OnCuda({static_cast<int64_t>(batch_size)}, score_len);
    model_input.lm_output_indexes  = makeCudaInt32Range(static_cast<int64_t>(batch_size * score_len));
}

torch::Tensor interleaveTokenPairs(const torch::Tensor& first, const torch::Tensor& second) {
    return torch::stack({first, second}, /*dim=*/1).reshape({-1});
}

void copyScoreSamplerTokenIds(torch::Tensor&       token_ids,
                              const torch::Tensor& complete_token_ids,
                              int64_t              batch_idx,
                              int64_t              score_len,
                              int64_t              seq_len) {
    if (score_len <= 0 || seq_len <= 0) {
        return;
    }
    auto dst = token_ids.narrow(0, batch_idx, score_len).narrow(1, 0, seq_len);
    auto src = complete_token_ids.narrow(0, 0, 1).narrow(1, 0, seq_len).expand({score_len, seq_len});
    dst.copy_(src);
}

const char* missingMtpStateReason(const GenerateStream::MtpAsyncDeviceState& state) {
    if (!state.accept_tokens_gpu.defined()) {
        return "accept_tokens_gpu_missing";
    }
    if (!state.accept_len_gpu.defined()) {
        return "accept_len_gpu_missing";
    }
    if (!state.propose_tokens_gpu.defined()) {
        return "propose_tokens_gpu_missing";
    }
    if (!state.next_seq_len_gpu.defined()) {
        return "next_seq_len_gpu_missing";
    }
    return nullptr;
}

void logMtpStateFallback(const GenerateStreamPtr& stream, const char* reason) {
    const uint64_t count = g_mtp_device_state_fallback_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!shouldLogFallback(count)) {
        return;
    }
    const auto mtp_state        = stream->getMtpAsyncDeviceState();
    auto       sp_output_buffer = stream->getSPOutputBuffer();
    RTP_LLM_LOG_INFO("[mtp-async-fallback] reason=%s stream=%ld epoch=%lu fallback_count=%lu success_count=%lu "
                     "tensors_holder_size=%zu seq_len=%d",
                     reason,
                     stream->streamId(),
                     mtp_state.epoch,
                     count,
                     g_mtp_device_state_success_count.load(std::memory_order_relaxed),
                     sp_output_buffer ? sp_output_buffer->tensors_holder.size() : 0,
                     stream->seqLength());
}

bool collectMtpStateProposeSlices(const std::list<GenerateStreamPtr>& streams,
                                  std::vector<torch::Tensor>&         propose_slices,
                                  std::vector<torch::Tensor>*         next_seq_lengths = nullptr) {
    propose_slices.clear();
    if (next_seq_lengths) {
        next_seq_lengths->clear();
    }
    for (const auto& stream : streams) {
        const auto    state = stream->getMtpAsyncDeviceState();
        torch::Tensor gpu_t = state.propose_tokens_gpu;
        if (!gpu_t.defined() || !gpu_t.is_cuda()) {
            logMtpStateFallback(stream, "propose_tokens_gpu_missing");
            return false;
        }
        propose_slices.push_back(lastColumnAsFlat(gpu_t));
        if (next_seq_lengths) {
            torch::Tensor next_seq_len = state.next_seq_len_gpu;
            if (next_seq_len.defined() && next_seq_len.is_cuda()) {
                next_seq_lengths->push_back(std::move(next_seq_len));
            }
        }
    }
    return true;
}

bool collectLegacyProposeSlices(const std::list<GenerateStreamPtr>& streams,
                                std::vector<torch::Tensor>&         propose_slices) {
    propose_slices.clear();
    for (const auto& stream : streams) {
        auto sp_output_buffer = stream->getSPOutputBuffer();
        if (!sp_output_buffer) {
            return false;
        }
        const auto& gpu_t = sp_output_buffer->propose_tokens_gpu;
        if (!gpu_t.defined() || !gpu_t.is_cuda()) {
            return false;
        }
        propose_slices.push_back(lastColumnAsFlat(gpu_t));
    }
    return true;
}

// Negative keeps the legacy GPU path forced on. Set this to a positive batch
// threshold if small-batch launch overhead needs to be avoided again.
static constexpr int64_t kMinBatchForLegacyGpuProposeTokens = -1;

bool legacyGpuProposePathEnabled(size_t batch_size) {
    return kMinBatchForLegacyGpuProposeTokens < 0
           || static_cast<int64_t>(batch_size) >= kMinBatchForLegacyGpuProposeTokens;
}

}  // namespace

torch::Tensor MtpBatchStreamProcessor::collectDSparkPositionBases(const StreamGroups& stream_groups,
                                                                  TensorHolder&       host_holder) const {
    const int64_t position_factor    = static_cast<int64_t>(model_input_gatherer_config_.position_id_len_factor);
    const int64_t batch_size         = static_cast<int64_t>(stream_groups.size());
    const bool    needs_position_ids = model_input_gatherer_config_.has_positional_encoding
                                    || model_input_gatherer_config_.mm_position_ids_style != PositionIdsStyle::DEFAULT;
    if (!needs_position_ids || position_factor <= 0 || batch_size == 0) {
        return {};
    }

    std::vector<torch::Tensor> base_positions;
    base_positions.reserve(batch_size);
    for (const auto& stream : stream_groups.allStreams()) {
        const auto& device_positions = stream->getNextPositionIdsGpu();
        if (device_positions.defined() && device_positions.is_cuda() && device_positions.scalar_type() == torch::kInt32
            && device_positions.numel() == position_factor) {
            base_positions.push_back(device_positions.reshape({1, position_factor}));
            continue;
        }

        auto host_positions =
            torch::zeros({position_factor}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
        stream->generateNextPositionId(host_positions.data_ptr<int32_t>());
        base_positions.push_back(toCudaInt32(host_positions, host_holder).reshape({1, position_factor}));
    }

    return torch::cat(base_positions, 0).reshape({batch_size, position_factor}).contiguous();
}

torch::Tensor MtpBatchStreamProcessor::expandDSparkPositionIds(const torch::Tensor& position_bases,
                                                               int64_t              width) const {
    if (!position_bases.defined() || position_bases.numel() == 0) {
        return {};
    }
    RTP_LLM_CHECK_WITH_INFO(width > 0, "DSpARK position width must be positive, got %ld", width);
    RTP_LLM_CHECK_WITH_INFO(
        position_bases.dim() == 2, "DSpARK position bases must be [batch, factor], got dim=%ld", position_bases.dim());
    auto bases = position_bases.to(torch::kInt32).unsqueeze(1);
    auto steps = torch::arange(width, cudaInt32Options()).reshape({1, width, 1});
    return (bases + steps).reshape({-1}).contiguous();
}

absl::Status MtpBatchStreamProcessor::dispatchPrefill(const StreamGroups& stream_groups,
                                                      const MergedOutput& prefill_output,
                                                      const MergedOutput& propose_output) const {
    return dispatchPrefill(stream_groups, prefill_output, propose_output, torch::Tensor());
}

absl::Status MtpBatchStreamProcessor::dispatchPrefill(const StreamGroups&  stream_groups,
                                                      const MergedOutput&  prefill_output,
                                                      const MergedOutput&  propose_output,
                                                      const torch::Tensor& draft_last_hidden_states) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t                      total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    auto                              new_tokens_all = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);
    std::vector<StreamSpecUpdateInfo> spec_update_infos;

    preparePrefillSpecUpdateInfo(
        stream_groups, prefill_output, propose_output, draft_last_hidden_states, new_tokens_all, spec_update_infos);

    // we set propose token in extra loop to avoid cuda sync
    updateProposeTokens(stream_groups, propose_output, spec_update_infos);

    // update streams
    stream_groups.updateStreams(spec_update_infos);

    RTP_LLM_LOG_DEBUG("dispatch prefill done");
    return absl::OkStatus();
}

absl::Status MtpBatchStreamProcessor::dispatchDecode(const StreamGroups&                          stream_groups,
                                                     const speculative::SpeculativeSamplerOutput& spec_decode_output,
                                                     const MergedOutput& draft_prefill_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    std::vector<StreamSpecUpdateInfo> spec_update_infos;

    prepareDecodeSpecUpdateInfo(stream_groups, spec_decode_output, draft_prefill_output, spec_update_infos);

    // to avoid cuda sync, we need to set propose token in extra loop
    updateProposeTokens(stream_groups, draft_prefill_output, spec_update_infos);

    stream_groups.updateStreams(spec_update_infos);

    RTP_LLM_LOG_DEBUG("dispatch decode done");
    return absl::OkStatus();
}

absl::StatusOr<GptModelInputs> MtpBatchStreamProcessor::gatherDecodeModelInput(const StreamGroups& stream_groups,
                                                                               TensorHolder&       host_holder) const {
    const bool needs_host_positions =
        !is_dspark_ && (model_input_gatherer_config_.has_positional_encoding
                       || model_input_gatherer_config_.mm_position_ids_style != PositionIdsStyle::DEFAULT);
    if (needs_host_positions) {
        for (const auto& stream : stream_groups.decodeStreams()) {
            stream->waitPendingAsyncBookkeeping();
        }
    }
    auto model_input = NormalBatchStreamProcessor::gatherModelInput(stream_groups, host_holder);

    RTP_LLM_CHECK(model_input.ok());

    overlayMtpCacheSnapshots(stream_groups, model_input.value(), host_holder);
    // Empty decode rounds are valid after prefill and on non-leader TP ranks.
    // Their allocated [0, 4] history is defined, but torch::cat rejects an
    // empty tensor list; leave that empty history intact until TP sync/skip.
    if (model_input.value().engram_token_windows.defined() && model_input.value().engram_token_windows.numel() > 0) {
        std::vector<torch::Tensor> history;
        int64_t                    row = 0;
        for (const auto& stream : stream_groups.allStreams()) {
            auto state  = stream->getMtpAsyncDeviceState();
            auto window = state.engram_token_window_gpu;
            if (!window.defined()) {
                RTP_LLM_CHECK_WITH_INFO(!stream->hasMtpCacheSnapshot(), "MTP snapshot missing Engram history");
                window = model_input.value().engram_token_windows.narrow(0, row, stream->currentBatchSize());
            }
            history.push_back(toCudaInt32(window, host_holder));
            row += window.size(0);
        }
        model_input.value().engram_token_windows = torch::cat(history, 0);
    }
    // TP followers have no local rows before rank 0 broadcasts the inputs.
    // Preserve their allocated empty position tensor instead of reshaping an undefined result.
    if (!stream_groups.empty() && model_input.value().combo_position_ids.defined()) {
        if (is_dspark_) {
            model_input.value().combo_position_ids =
                collectDSparkPositionBases(stream_groups, host_holder).reshape({-1});
        } else {
            // Snapshot gathering leaves position placeholders. Refill them from
            // host state after the preceding bookkeeping has completed.
            const auto factor = model_input_gatherer_config_.position_id_len_factor;
            auto*      dst    = model_input.value().combo_position_ids.data_ptr<int32_t>();
            size_t     row    = 0;
            for (const auto& stream : stream_groups.decodeStreams()) {
                const auto rows = stream->hasMtpCacheSnapshot() ? 1 : stream->currentBatchSize();
                for (int i = 0; i < rows; ++i, ++row) {
                    auto* positions = dst + row * factor;
                    std::fill_n(positions, factor, 0);
                    stream->generateNextPositionId(positions);
                }
            }
        }
    }

    if (propose_step_ == 1 || is_dspark_) {
        return model_input;
    }

    gatherHiddenStates(stream_groups, model_input.value());

    return model_input;
}

absl::Status MtpBatchStreamProcessor::gatherLinearReplayInputs(
    const StreamGroups& stream_groups,
    const CacheConfig& cache_config,
    GptModelInputs& model_input,
    TensorHolder& host_holder,
    std::vector<GenerateStream::LinearReplayRound>& rounds) const {
    if (!cache_config.linear_replay.enabled()) {
        return absl::OkStatus();
    }
    const auto streams = stream_groups.allStreams();
    if (streams.empty()) {
        return absl::OkStatus();
    }
    const int64_t batch = static_cast<int64_t>(streams.size());
    const int64_t groups = cache_config.groupNums();
    auto          host   = LinearReplayInputs::allocate(batch, groups, torch::kCPU, /*pin_memory=*/true);
    for (auto* tensor : host.tensors()) {
        tensor->zero_();
    }
    host.slot_ids.fill_(-1);
    host.active_block_ids.fill_(-1);
    host.state_read_block_ids.fill_(-1);
    std::vector<int64_t> history_rows;
    std::vector<torch::Tensor> accepts;
    std::vector<torch::Tensor> anchors;
    std::vector<torch::Tensor> processed_lengths;
    rounds.clear();
    rounds.reserve(static_cast<size_t>(batch));
    int64_t row = 0;
    for (const auto& stream : streams) {
        if (stream->isFakeStream()) {
            rounds.emplace_back();
            ++row;
            continue;
        }
        if (stream->nextBatchSize() != 1) {
            return absl::InvalidArgumentError("MTP LINEAR replay requires one sequence per stream");
        }
        auto round_status = stream->prepareLinearReplayRound();
        if (!round_status.ok()) {
            return round_status.status();
        }
        auto round = std::move(round_status.value());
        host.slot_ids.data_ptr<int32_t>()[row] = round.lease->slot_id;
        host.slot_generations.data_ptr<int64_t>()[row] = round.lease->generation;
        host.verify_epochs.data_ptr<int64_t>()[row] = round.verify_epoch;
        if (round.previous_window) {
            const auto& previous = *round.previous_window;
            if (previous.device_ready) {
                previous.device_ready->block(cuda_graph::graphGetCurrentStream());
            }
            host.history_valid_lengths.data_ptr<int32_t>()[row] = previous.produced_steps;
            host.history_epochs.data_ptr<int64_t>()[row] = previous.verify_epoch;
            history_rows.push_back(row);
            accepts.push_back(previous.accept_len_gpu.reshape({1}));
            anchors.push_back(previous.anchor_block_ids_gpu);
            processed_lengths.push_back(previous.anchor_processed_len_gpu.reshape({1}));
        } else {
            const int32_t processed = std::max(0, stream->seqLength() - 1);
            host.anchor_processed_lengths.data_ptr<int32_t>()[row] = processed;
            host.init_kinds.data_ptr<int32_t>()[row] = processed == 0 ? 2 : 1;
            if (processed > 0) {
                for (const int group_id : cache_config.linear_replay.group_ids) {
                    host.state_read_block_ids.data_ptr<int32_t>()[group_id * batch + row] =
                        round.initial_block_ids.at(static_cast<size_t>(group_id));
                }
            }
        }
        rounds.push_back(std::move(round));
        ++row;
    }
    auto replay = host;
    for (auto* tensor : replay.tensors()) {
        host_holder.hold_host(*tensor);
        *tensor = tensor->to(torch::kCUDA, /*non_blocking=*/true);
    }
    if (!history_rows.empty()) {
        auto rows_host = torch::empty({static_cast<int64_t>(history_rows.size())},
                                      torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
        std::copy(history_rows.begin(), history_rows.end(), rows_host.data_ptr<int64_t>());
        host_holder.hold_host(rows_host);
        auto rows = rows_host.to(torch::kCUDA, /*non_blocking=*/true);
        auto accepted = torch::cat(accepts).to(torch::kInt32);
        replay.prev_accept_lengths.index_copy_(0, rows, accepted);
        replay.anchor_processed_lengths.index_copy_(0, rows, torch::cat(processed_lengths) + accepted);
        replay.state_read_block_ids.index_copy_(1, rows, torch::cat(anchors, 1));
    }
    auto block_ids = toCudaInt32(model_input.kv_cache_kernel_block_id, host_holder);
    if (block_ids.dim() == 2) {
        block_ids = block_ids.unsqueeze(0);
    }
    RTP_LLM_CHECK_WITH_INFO(block_ids.dim() == 3 && block_ids.size(0) == groups && block_ids.size(1) == batch,
                            "LINEAR replay requires a grouped [group,batch,block] table");
    const auto active_rows = replay.slot_ids >= 0;
    for (const int group_id : cache_config.linear_replay.group_ids) {
        const auto& group = cache_config.groups().at(static_cast<size_t>(group_id));
        RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == CacheGroupType::LINEAR && group.seqSizePerBlock() > 0,
                                "LINEAR replay tag=%s has invalid geometry", group.tag.c_str());
        const auto tail = torch::floor_divide(replay.anchor_processed_lengths,
                                             static_cast<int64_t>(group.seqSizePerBlock())).to(torch::kInt64);
        const auto destination = block_ids.select(0, group_id).gather(1, tail.reshape({batch, 1})).squeeze(1);
        replay.active_block_ids.select(0, group_id).copy_(
            torch::where(active_rows, destination, torch::full_like(destination, -1)));
    }
    model_input.linear_replay = std::move(replay);
    return absl::OkStatus();
}

void MtpBatchStreamProcessor::overlayMtpCacheSnapshots(const StreamGroups& stream_groups,
                                                       GptModelInputs&     model_input,
                                                       TensorHolder&       host_holder) const {
    struct SnapshotBucket {
        std::vector<torch::Tensor> physical_tables;
        std::vector<torch::Tensor> kernel_tables;
        std::vector<int64_t>       rows;
    };

    std::map<std::pair<int64_t, int64_t>, SnapshotBucket> buckets;
    int64_t                                               row = 0;
    for (const auto& stream : stream_groups.decodeStreams()) {
        const auto state        = stream->getMtpAsyncDeviceState();
        const bool use_snapshot = GenerateStream::hasMtpCacheSnapshot(state);
        if (use_snapshot) {
            RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1 && !stream->hasNumBeams(),
                                    "MTP cache snapshots require one non-beam sequence per stream, stream=%ld",
                                    stream->streamId());
            const auto& physical = state.next_kv_cache_block_id_gpu;
            const auto& kernel   = state.next_kv_cache_kernel_block_id_gpu;
            RTP_LLM_CHECK_WITH_INFO(physical.defined() && physical.is_cuda() && physical.scalar_type() == torch::kInt32
                                        && physical.dim() == 3 && physical.size(1) == 1 && kernel.defined()
                                        && kernel.is_cuda() && kernel.scalar_type() == torch::kInt32
                                        && kernel.dim() == 3 && kernel.size(1) == 1,
                                    "MTP cache snapshots require CUDA int32 [group,1,blocks] physical/kernel tensors");
            auto& bucket = buckets[{physical.size(2), kernel.size(2)}];
            // Keep the immutable device snapshots alive through the asynchronous
            // overlay copies below, without reading their mutable host rows.
            host_holder.hold(physical);
            host_holder.hold(kernel);
            bucket.physical_tables.push_back(physical);
            bucket.kernel_tables.push_back(kernel);
            bucket.rows.push_back(row);
        }
        row += use_snapshot ? 1 : stream->currentBatchSize();
    }

    if (buckets.empty()) {
        return;
    }
    // Generic gathering produces pinned host tables. Move the complete mixed
    // batch once, preserving freshly admitted host rows, then replace only the
    // steady rows with their device snapshots. TensorHolder keeps the pinned
    // sources alive until the non-blocking copies complete.
    model_input.kv_cache_block_id        = toCudaInt32(model_input.kv_cache_block_id, host_holder);
    model_input.kv_cache_kernel_block_id = toCudaInt32(model_input.kv_cache_kernel_block_id, host_holder);
    RTP_LLM_CHECK_WITH_INFO(
        model_input.kv_cache_block_id.defined() && model_input.kv_cache_block_id.is_cuda()
            && model_input.kv_cache_block_id.scalar_type() == torch::kInt32 && model_input.kv_cache_block_id.dim() == 3
            && model_input.kv_cache_kernel_block_id.defined() && model_input.kv_cache_kernel_block_id.is_cuda()
            && model_input.kv_cache_kernel_block_id.scalar_type() == torch::kInt32
            && model_input.kv_cache_kernel_block_id.dim() == 3,
        "MTP cache snapshot overlay requires physical and kernel CUDA int32 destinations "
        "[group,batch,blocks]");

    auto overlay =
        [](torch::Tensor& destination, const std::vector<torch::Tensor>& sources, const std::vector<int64_t>& rows) {
            auto source = torch::cat(sources, 1);
            RTP_LLM_CHECK_WITH_INFO(source.size(0) == destination.size(0),
                                    "MTP cache snapshot group mismatch: source=%ld destination=%ld",
                                    source.size(0),
                                    destination.size(0));
            RTP_LLM_CHECK_WITH_INFO(source.size(2) <= destination.size(2),
                                    "MTP cache snapshot exceeds destination width: source=%ld destination=%ld",
                                    source.size(2),
                                    destination.size(2));
            const int64_t copy_width  = source.size(2);
            auto          row_indices = torch::tensor(rows, torch::TensorOptions().dtype(torch::kInt64))
                                   .to(destination.device(), /*non_blocking=*/true);
            for (int64_t group_id = 0; group_id < source.size(0); ++group_id) {
                // Generic gathering leaves every steady row empty. Replay
                // groups also need their canonical pages from this snapshot;
                // dispatch excludes them from the legacy LINEAR permutation.
                destination.select(0, group_id).narrow(1, 0, copy_width).index_copy_(
                    0, row_indices, source.select(0, group_id));
            }
        };

    for (auto& [widths, bucket] : buckets) {
        (void)widths;
        overlay(model_input.kv_cache_block_id, bucket.physical_tables, bucket.rows);
        overlay(model_input.kv_cache_kernel_block_id, bucket.kernel_tables, bucket.rows);
    }
}

SamplerInputs
MtpBatchStreamProcessor::prepareSpecSamplerInput(const StreamGroups&                  stream_groups,
                                                 const torch::Tensor&                 draft_token_ids,
                                                 const std::shared_ptr<torch::Event>& draft_tokens_ready) const {
    RTP_LLM_PROFILE_SCOPE("mtp_batch_stream_processor.prepare_spec_sampler_input");
    RTP_LLM_CHECK(!stream_groups.empty());
    auto all_streams = stream_groups.allStreams();

    for (auto& stream : all_streams) {
        RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1, "stream tile num must be 1 in ScoreExecutor");
    }

    size_t score_len        = propose_step_ + 1;
    size_t total_batch_size = stream_groups.size() * score_len;

    SamplerInputs sampler_inputs =
        allocateSamplerInputs(stream_groups, total_batch_size, total_batch_size, propose_step_);
    fillSamplerCommonInputs(sampler_inputs, all_streams, true, propose_step_);
    sampler_inputs.capture_original_probs = std::any_of(all_streams.begin(), all_streams.end(), [](const auto& stream) {
        return stream->generateConfig()->return_all_probs == ReturnAllProbsMode::ORIGINAL;
    });

    int64_t batch_idx = 0;
    for (auto& stream : all_streams) {
        auto complete_token_ids = stream->completeTokenIds();
        auto seq_len            = static_cast<int64_t>(stream->seqLength());

        copyScoreSamplerTokenIds(
            sampler_inputs.token_ids, complete_token_ids, batch_idx, static_cast<int64_t>(score_len), seq_len);
        RTP_LLM_CHECK_WITH_INFO(
            seq_len + static_cast<int64_t>(propose_step_) <= static_cast<int64_t>(sampler_inputs.step + 1),
            "MTP verify sampler history exceeds its allocated width: seq_len=%ld gamma=%zu step=%zu",
            seq_len,
            propose_step_,
            sampler_inputs.step);
        batch_idx += static_cast<int64_t>(score_len);
        RTP_LLM_LOG_DEBUG("stream [%s], sampler inputs token ids = [%s]",
                          stream->streamLogTag().c_str(),
                          tensorDebugStringWithData<int32_t>(sampler_inputs.token_ids).c_str());
    }

    if (draft_token_ids.defined()) {
        prepareSpecSamplerInputDevice(stream_groups, sampler_inputs, draft_token_ids, draft_tokens_ready);
    }
    return sampler_inputs;
}

void MtpBatchStreamProcessor::prepareSpecSamplerInputDevice(
    const StreamGroups&                  stream_groups,
    SamplerInputs&                       sampler_inputs,
    const torch::Tensor&                 draft_token_ids,
    const std::shared_ptr<torch::Event>& draft_tokens_ready) const {
    RTP_LLM_PROFILE_SCOPE("mtp_batch_stream_processor.prepare_spec_sampler_input_device");
    const size_t score_len        = propose_step_ + 1;
    const size_t total_batch_size = stream_groups.size() * score_len;
    const auto   return_all_probs = stream_groups.needReturnAllProbs();

    sampler_inputs.vocab_size = vocab_size_;
    if (return_all_probs != ReturnAllProbsMode::NONE) {
        sampler_inputs.all_probs = torch::zeros({(int64_t)total_batch_size, (int64_t)vocab_size_},
                                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        if (return_all_probs == ReturnAllProbsMode::ORIGINAL) {
            sampler_inputs.return_original_all_probs = true;
        }
    }

    const auto batch_size = static_cast<int64_t>(stream_groups.size());
    const auto gamma      = static_cast<int64_t>(propose_step_);
    const auto rows       = static_cast<int64_t>(score_len);
    RTP_LLM_CHECK_WITH_INFO(draft_token_ids.defined() && draft_token_ids.scalar_type() == torch::kInt32
                                && draft_token_ids.dim() == 2 && draft_token_ids.size(0) == batch_size
                                && draft_token_ids.size(1) == gamma,
                            "MTP verify draft tokens must be int32 [B,gamma]");

    // Fill every proposal slot once for all score rows, then let each row's
    // sequence length expose exactly proposals [0, score_position).
    // Tensor::to records the pinned history with PyTorch's host allocator.
    auto token_ids_gpu = sampler_inputs.token_ids.to(torch::kCUDA, /*non_blocking=*/true);
    auto histories     = token_ids_gpu.view({batch_size, rows, static_cast<int64_t>(sampler_inputs.step + 1)});
    // History H2D is independent of proposal production. Join only before
    // reading proposal values on the prepare stream.
    if (draft_tokens_ready) {
        draft_tokens_ready->block(cuda_graph::graphGetCurrentStream());
    }
    // CPU proposals also use allocator-managed pinned storage for async H2D.
    auto proposals = draft_token_ids.is_cuda() ? draft_token_ids : draft_token_ids.pin_memory();
#if USING_CUDA
    if (proposals.is_cuda()) {
        proposals.record_stream(cuda_graph::graphGetCurrentStream());
    }
#endif
    proposals = proposals.to(torch::kCUDA, /*non_blocking=*/true)
                    .contiguous()
                    .view({batch_size, 1, gamma})
                    .expand({batch_size, rows, gamma});
    auto base_lengths = sampler_inputs.sequence_lengths.view({batch_size, rows})
                            .select(1, 0)
                            .to(torch::Device(torch::kCUDA), torch::kLong, /*non_blocking=*/true)
                            .view({batch_size, 1, 1});
    auto proposal_offsets =
        torch::arange(gamma, torch::TensorOptions().dtype(torch::kLong).device(torch::Device(torch::kCUDA)))
            .view({1, 1, gamma});
    auto proposal_columns = (base_lengths + proposal_offsets).expand({batch_size, rows, gamma});
    histories.scatter_(2, proposal_columns, proposals);
    sampler_inputs.token_ids = std::move(token_ids_gpu);
}

SamplerInputs MtpBatchStreamProcessor::finalizeSpecSamplerInput(
    SamplerInputs                               sampler_inputs,
    const GptModelOutputs&                      model_output,
    const SpecLogitsVerifyRunner::LaunchResult& spec_logits_result) const {
    RTP_LLM_PROFILE_SCOPE("mtp_batch_stream_processor.finalize_spec_sampler_input");
    RTP_LLM_CHECK_WITH_INFO(model_output.logits.dim() == 2
                                && model_output.logits.size(1) >= static_cast<int64_t>(vocab_size_),
                            "MTP target logits width %ld is smaller than vocab size %zu",
                            model_output.logits.dim() == 2 ? model_output.logits.size(1) : -1,
                            vocab_size_);

    // lm_head shards may contain TP-alignment padding. Synthetic columns must
    // not participate in target sampling or the rejection distribution.
    sampler_inputs.logits = model_output.logits.narrow(1, 0, static_cast<int64_t>(vocab_size_)).clone();
    if (spec_logits_result.has_active_processor) {
        // The runner uploads packed masks on its own (worker) stream; order this
        // stream's mask application after the uploads, then apply on GPU.
        if (spec_logits_result.ready_event) {
            spec_logits_result.ready_event->block(cuda_graph::graphGetCurrentStream());
        }
        SpecLogitsVerifyRunner::applyMaskToLogits(sampler_inputs.logits, spec_logits_result, sampler_inputs.vocab_size);
    }

    // TODO(async): debug formatting is CPU-only. Keep the .cpu() explicit
    // and do not route this through the model-input fast path.
    RTP_LLM_LOG_DEBUG("sampler inputs logits [%s]",
                      tensorDebugStringWithData<float>(sampler_inputs.logits.cpu(), 10).c_str());

    RTP_LLM_LOG_DEBUG("gatherSamplerInput done");
    return sampler_inputs;
}

void MtpBatchStreamProcessor::updateProposeTokens(const StreamGroups&                stream_groups,
                                                  const MergedOutput&                draft_prefill_output,
                                                  std::vector<StreamSpecUpdateInfo>& spec_update_infos) const {
    // Prefer per-stream GPU slices and avoid D2H/CPU loops.
    // The legacy draft_token int stays -1 unless CPU/PD-disagg still needs it.
    const auto& propose_token_ids = draft_prefill_output.sampler_output.token_ids;
    if (!propose_token_ids.defined()) {
        // DSpARK commit-only dispatches explicitly invalidate proposal state.
        // Traditional MTP PD partial updates omit this tensor to preserve the
        // proposal received at handoff.
        if (is_dspark_) {
            for (auto& update_info : spec_update_infos) {
                update_info.draft_token_gpu = torch::Tensor();
            }
        }
        return;
    }

    const bool         on_gpu       = propose_token_ids.is_cuda();
    const int          token_stride = propose_token_ids.size(1);
    const torch::Dtype dtype        = propose_token_ids.scalar_type();

    // TODO(async): lazy CPU mirror is only built when at least one stream
    // needs the legacy int draft_token. Remove after downstream paths consume
    // draft_token_gpu exclusively.
    torch::Tensor propose_token_ids_h;
    auto          ensure_cpu_mirror = [&]() -> const torch::Tensor& {
        if (!propose_token_ids_h.defined()) {
            propose_token_ids_h = on_gpu ? propose_token_ids.cpu() : propose_token_ids;
        }
        return propose_token_ids_h;
    };

    int batch_idx_in  = 0;
    int batch_idx_out = 0;
    int stream_idx    = 0;

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();

        // GPU slice for next-step propose tokens: [next_batch_size, token_stride].
        // Readers select the last column for one-step decode, or the full row
        // when propose_step > 1.
        if (on_gpu && next_batch_size > 0) {
            spec_update_infos[stream_idx].draft_token_gpu = propose_token_ids.narrow(0, batch_idx_out, next_batch_size);
        } else {
            // A CPU proposal supersedes any previous GPU mirror. Store an
            // explicit undefined Tensor so specUpdate clears the mirror and
            // readers fall back to the freshly updated CPU draft token.
            spec_update_infos[stream_idx].draft_token_gpu = torch::Tensor();
        }

        // Fill the legacy int only when the tensor is CPU or PD-disagg needs
        // the gRPC-visible one-step MTP proposal. PDFUSION consumes
        // draft_token_gpu and keeps this at -1, so ensure_cpu_mirror() stays
        // lazy. DSpARK never reaches this path for commit-only dispatches.
        const bool need_cpu_int = !on_gpu || stream->queryPdSep();
        if (need_cpu_int) {
            const auto& cpu_ids = ensure_cpu_mirror();
            int         propose_token =
                (dtype == torch::kLong) ?
                            static_cast<int>(cpu_ids.data_ptr<int64_t>()[batch_idx_out * token_stride + token_stride - 1]) :
                            cpu_ids.data_ptr<int32_t>()[batch_idx_out * token_stride + token_stride - 1];
            spec_update_infos[stream_idx].draft_token = propose_token;
        } else {
            spec_update_infos[stream_idx].draft_token = -1;
        }

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        stream_idx++;
    }
}

void MtpBatchStreamProcessor::prepareDecodeDraftModelInput(const StreamGroups& stream_groups,
                                                           GptModelInputs&     model_input,
                                                           TensorHolder&       host_holder) {
    const size_t batch_size = stream_groups.size();
    if (batch_size == 0) {
        model_input.combo_tokens      = emptyInt32OnCuda({0});
        model_input.input_lengths     = emptyInt32OnCuda({0});
        model_input.sequence_lengths  = emptyInt32OnCuda({0});
        model_input.prefix_lengths    = emptyInt32OnCuda({0});
        model_input.lm_output_indexes = emptyInt32OnCuda({0});
        return;
    }

    // Fast path: consume next-step propose tokens published by dispatchDecodeAsync.
    {
        const auto                 all_streams = stream_groups.allStreams();
        std::vector<torch::Tensor> propose_slices_gpu;
        std::vector<torch::Tensor> sequence_lengths_gpu;
        propose_slices_gpu.reserve(batch_size);
        sequence_lengths_gpu.reserve(batch_size);
        if (!all_streams.empty()
            && collectMtpStateProposeSlices(all_streams, propose_slices_gpu, &sequence_lengths_gpu)) {
            auto combo_tokens_gpu         = torch::cat(propose_slices_gpu, 0);
            model_input.combo_tokens      = std::move(combo_tokens_gpu);
            model_input.lm_output_indexes = makeCudaInt32Range(model_input.combo_tokens.numel());
            if (sequence_lengths_gpu.size() == batch_size) {
                // next_seq_len includes the carried target token. Shifted
                // draft KV and target verification both resume at length - 1;
                // explicit rotary position IDs are managed separately.
                auto committed_len           = torch::cat(sequence_lengths_gpu, 0);
                model_input.sequence_lengths = committedLenToDraftDecodePosition(committed_len, host_holder);
                model_input.prefix_lengths   = model_input.sequence_lengths;
            } else if (model_input.sequence_lengths.defined()) {
                auto target_prefix_lengths = toCudaInt32(model_input.sequence_lengths, host_holder);
                model_input.prefix_lengths = target_prefix_lengths;
                model_input.sequence_lengths =
                    normalDecodePositionToDraftDecodePosition(target_prefix_lengths, host_holder);
            }
            model_input.input_lengths = toCudaInt32(model_input.input_lengths, host_holder);
            return;
        }
    }

    std::vector<torch::Tensor> propose_slices_gpu;
    if (legacyGpuProposePathEnabled(batch_size)
        && collectLegacyProposeSlices(stream_groups.allStreams(), propose_slices_gpu)) {
        auto combo_tokens_gpu         = torch::cat(propose_slices_gpu, 0);
        model_input.combo_tokens      = std::move(combo_tokens_gpu);
        model_input.lm_output_indexes = makeCudaInt32Range(model_input.combo_tokens.numel());
        model_input.input_lengths     = toCudaInt32(model_input.input_lengths, host_holder);
        auto target_prefix_lengths    = toCudaInt32(model_input.sequence_lengths, host_holder);
        model_input.prefix_lengths    = target_prefix_lengths;
        model_input.sequence_lengths  = normalDecodePositionToDraftDecodePosition(target_prefix_lengths, host_holder);
        return;
    }

    int  batch_idx    = 0;
    auto combo_tokens = torch::empty({(int64_t)batch_size}, torch::kInt32).pin_memory();

    for (const auto& stream : stream_groups.allStreams()) {
        int propose_token                       = stream->getSPOutputBuffer()->tokens.data_ptr<int>()[1];
        combo_tokens.data_ptr<int>()[batch_idx] = propose_token;
        batch_idx++;
    }

    model_input.combo_tokens      = toCudaInt32(combo_tokens, host_holder);
    model_input.input_lengths     = toCudaInt32(model_input.input_lengths, host_holder);
    auto target_prefix_lengths    = toCudaInt32(model_input.sequence_lengths, host_holder);
    model_input.prefix_lengths    = target_prefix_lengths;
    model_input.sequence_lengths  = normalDecodePositionToDraftDecodePosition(target_prefix_lengths, host_holder);
    model_input.lm_output_indexes = makeCudaInt32Range(static_cast<int64_t>(batch_size));
}

bool MtpBatchStreamProcessor::gatherMtpDecodeModelInputFromDeviceState(const StreamGroups& stream_groups,
                                                                       GptModelInputs&     model_input,
                                                                       TensorHolder&       host_holder) const {
    if (!useMtpDeviceState()) {
        return false;
    }
    const size_t batch_size = stream_groups.size();
    if (batch_size == 0) {
        return false;
    }
    const auto                                       all_streams = stream_groups.allStreams();
    std::vector<GenerateStream::MtpAsyncDeviceState> states;
    states.reserve(batch_size);
    for (const auto& stream : all_streams) {
        const auto state = stream->getMtpAsyncDeviceState();
        if (const char* reason = missingMtpStateReason(state)) {
            logMtpStateFallback(stream, reason);
            return false;
        }
        states.push_back(state);
    }
    g_mtp_device_state_success_count.fetch_add(1, std::memory_order_relaxed);

    std::vector<torch::Tensor> target_last_slices_gpu;
    std::vector<torch::Tensor> propose_slices_gpu;
    std::vector<torch::Tensor> next_seq_len_slices_gpu;
    target_last_slices_gpu.reserve(batch_size);
    propose_slices_gpu.reserve(batch_size);
    next_seq_len_slices_gpu.reserve(batch_size);

    for (const auto& state : states) {
        const auto& accept_tokens  = state.accept_tokens_gpu;   // [1, propose+1]
        const auto& accept_len     = state.accept_len_gpu;      // [1]
        const auto& propose_tokens = state.propose_tokens_gpu;  // [1, token_stride]
        const auto& next_seq_len   = state.next_seq_len_gpu;    // [1]

        auto idx_t       = (accept_len - 1).to(torch::kLong);
        auto target_last = accept_tokens.squeeze(0).index_select(/*dim=*/0, idx_t);

        target_last_slices_gpu.push_back(target_last);
        propose_slices_gpu.push_back(lastColumnAsFlat(propose_tokens));
        next_seq_len_slices_gpu.push_back(next_seq_len);
    }

    auto target_last_gpu         = torch::cat(target_last_slices_gpu, 0).to(torch::kInt32);
    auto propose_gpu             = torch::cat(propose_slices_gpu, 0);
    auto pair_gpu                = interleaveTokenPairs(target_last_gpu, propose_gpu);
    auto next_seq_len_gpu_concat = torch::cat(next_seq_len_slices_gpu, 0);

    model_input.prefix_lengths = (next_seq_len_gpu_concat - 1).to(torch::kInt32);
    setVerifyPairInputs(model_input, std::move(pair_gpu), batch_size, propose_step_ + 1, host_holder);
    return true;
}

void MtpBatchStreamProcessor::prepareOneStepSpecDecodeModelInput(const StreamGroups& stream_groups,
                                                                 GptModelInputs&     model_input,
                                                                 TensorHolder&       host_holder) {
    const size_t batch_size = stream_groups.size();
    if (batch_size == 0) {
        return;
    }

    if (gatherMtpDecodeModelInputFromDeviceState(stream_groups, model_input, host_holder)) {
        expandTargetVerifyPositionIdsFromDecode(model_input);
        return;
    }

    std::vector<torch::Tensor> target_last_slices;
    std::vector<torch::Tensor> propose_slices;
    target_last_slices.reserve(batch_size);
    propose_slices.reserve(batch_size);

    for (const auto& stream : stream_groups.allStreams()) {
        auto target_last = pickOneStepTargetLastToken(stream);
        auto propose     = pickOneStepDraftToken(stream);
        RTP_LLM_CHECK_WITH_INFO(
            target_last.defined(), "one-step MTP target token missing for stream %ld", stream->streamId());
        RTP_LLM_CHECK_WITH_INFO(
            propose.defined(), "one-step MTP draft token missing for stream %ld", stream->streamId());
        target_last_slices.push_back(toCudaInt32(target_last, host_holder));
        propose_slices.push_back(toCudaInt32(propose, host_holder));
    }

    expandTargetVerifyPositionIds(stream_groups, model_input);
    auto target_last_gpu = torch::cat(target_last_slices, 0).to(torch::kInt32);
    auto propose_gpu     = torch::cat(propose_slices, 0).to(torch::kInt32);

    model_input.prefix_lengths = toCudaInt32(model_input.sequence_lengths, host_holder).clone();
    setVerifyPairInputs(
        model_input, interleaveTokenPairs(target_last_gpu, propose_gpu), batch_size, propose_step_ + 1, host_holder);
}

void MtpBatchStreamProcessor::updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                                          const GptModelOutputs& model_output,
                                                          const torch::Tensor&   draft_token_ids,
                                                          TensorHolder&          host_holder) {
    int batch_size                 = model_input.combo_tokens.size(0);
    model_input.last_hidden_states = model_output.all_hidden_states;

    // here combo_tokens is a device buffer
    model_input.combo_tokens = draft_token_ids.reshape({batch_size});

    if (model_input.combo_position_ids.defined()) {
        if (model_input.combo_position_ids.is_cuda()) {
            model_input.combo_position_ids = model_input.combo_position_ids + 1;
        } else {
            const size_t position_id_len_factor = model_input_gatherer_config_.position_id_len_factor;
            auto         next_position_ids =
                torch::empty({(int64_t)(batch_size * position_id_len_factor)}, torch::kInt32).pin_memory();
            int*       dst_position_ids = next_position_ids.data_ptr<int>();
            const auto src_position_ids = model_input.combo_position_ids.cpu().contiguous();
            const int* src              = src_position_ids.data_ptr<int>();
            for (int64_t i = 0; i < next_position_ids.numel(); ++i) {
                dst_position_ids[i] = src[i] + 1;
            }
            model_input.combo_position_ids = std::move(next_position_ids);
        }
    }

    if (useMtpDeviceInput() || model_input.sequence_lengths.is_cuda()) {
        auto seq_lengths_d           = model_input.sequence_lengths.is_cuda() ? model_input.sequence_lengths :
                                                                                model_input.sequence_lengths.to(torch::kCUDA);
        model_input.sequence_lengths = (seq_lengths_d + 1).to(torch::kInt32);
    } else {
        // Legacy CPU fallback when device input is disabled and the caller has
        // not already published sequence_lengths on CUDA.
        auto sequence_lengths_cpu = model_input.sequence_lengths.cpu().clone().pin_memory();
        for (int i = 0; i < batch_size; i++) {
            sequence_lengths_cpu.data_ptr<int>()[i]++;
        }
        model_input.sequence_lengths = toCudaInt32(sequence_lengths_cpu, host_holder);
    }
}

void MtpBatchStreamProcessor::updatePrefillPostDraftModelInput(const StreamGroups&    stream_groups,
                                                               GptModelInputs&        model_input,
                                                               const GptModelOutputs& model_output,
                                                               const SamplerOutput&   sampler_output,
                                                               TensorHolder&          host_holder) {
    model_input.last_hidden_states = model_output.all_hidden_states;
    const auto& new_all_token_ids  = sampler_output.token_ids;

    const auto all_streams = stream_groups.allStreams();
    const bool has_multimodal =
        (model_input.multimodal_features.has_value() && !model_input.multimodal_features->empty())
        || std::any_of(all_streams.begin(), all_streams.end(), [](const auto& s) { return mtpTailTextMask(s) == 0; });
    // The default one-dimensional position policy reuses the final input
    // position for the sampled token. Other styles keep the stream policy below.
    const bool shift_default_positions =
        model_input.combo_position_ids.defined()
        && model_input_gatherer_config_.mm_position_ids_style == PositionIdsStyle::DEFAULT
        && model_input_gatherer_config_.position_id_len_factor == 1;
#if USING_CUDA
    if (!has_multimodal && !model_input.combo_tokens_type_ids.defined()
        && (!model_input.combo_position_ids.defined() || shift_default_positions)) {
        const int64_t token_stride = new_all_token_ids.size(1);
        RTP_LLM_CHECK_WITH_INFO(token_stride > 0 && token_stride <= std::numeric_limits<int32_t>::max(),
                                "invalid MTP Prefill sample stride: %ld",
                                token_stride);
        auto input_lengths_d = toCudaInt32(model_input.input_lengths, host_holder).contiguous();
        auto combo_tokens_d  = toCudaInt32(model_input.combo_tokens, host_holder).contiguous();
        auto sampled_d = new_all_token_ids.is_cuda() ? new_all_token_ids : toCudaInt32(new_all_token_ids, host_holder);
        if (sampled_d.scalar_type() != torch::kInt32) {
            sampled_d = sampled_d.to(torch::kInt32);
        }
        sampled_d      = sampled_d.contiguous();

        std::vector<int64_t> middle_rows;
        std::vector<int32_t> middle_tokens;
        std::vector<int32_t> middle_positions;
        int64_t row_idx = 0;
        for (const auto& stream : all_streams) {
            const int rows = stream->nextBatchSize();
            if (stream->isMiddleChunk()) {
                const int32_t token = stream->nextChunkBoundaryToken();
                int32_t position = 0;
                if (shift_default_positions) {
                    const int boundary = stream->prefixLength() + stream->contextLength();
                    position = stream->generateContextPositionIds().data_ptr<int32_t>()[boundary];
                }
                for (int row = 0; row < rows; ++row) {
                    middle_rows.push_back(row_idx + row);
                    middle_tokens.push_back(token);
                    middle_positions.push_back(position);
                }
            }
            row_idx += rows;
        }
        RTP_LLM_CHECK_WITH_INFO(row_idx == sampled_d.size(0),
                                "prefill draft input stream rows %ld != sampler rows %ld",
                                row_idx, sampled_d.size(0));

        torch::Tensor middle_rows_d;
        if (!middle_rows.empty()) {
            auto rows_h = torch::tensor(middle_rows, torch::kInt64).pin_memory();
            host_holder.hold_host(rows_h);
            middle_rows_d = rows_h.to(torch::kCUDA, /*non_blocking=*/true);
            auto tokens_d = toCudaInt32(torch::tensor(middle_tokens, torch::kInt32).pin_memory(), host_holder);
            // The CUDA shift consumes the last column as its tail. Preserve the
            // sampler output while replacing only middle-chunk rows with real prompt tokens.
            sampled_d = sampled_d.clone();
            sampled_d.select(1, token_stride - 1).index_copy_(0, middle_rows_d, tokens_d);
        }
        auto offsets_d = input_lengths_d.cumsum(0).to(torch::kInt32);
        auto shifted_d = torch::empty_like(combo_tokens_d);
        if (shift_default_positions) {
            auto positions_d         = toCudaInt32(model_input.combo_position_ids, host_holder).contiguous();
            auto shifted_positions_d = torch::empty_like(positions_d);
            invokeMtpPrefillShiftAppend(combo_tokens_d,
                                        input_lengths_d,
                                        offsets_d,
                                        sampled_d,
                                        shifted_d,
                                        positions_d,
                                        shifted_positions_d,
                                        static_cast<int32_t>(token_stride),
                                        cuda_graph::graphGetCurrentStream().stream());
            if (middle_rows_d.defined()) {
                auto positions_d =
                    toCudaInt32(torch::tensor(middle_positions, torch::kInt32).pin_memory(), host_holder);
                auto tail_offsets = offsets_d.index_select(0, middle_rows_d).to(torch::kInt64) - 1;
                // A prompt continuation uses its exact next position, while the
                // final chunk retains the existing sampled-token position policy.
                shifted_positions_d.index_copy_(0, tail_offsets, positions_d);
            }
            model_input.combo_position_ids = shifted_positions_d;
        } else {
            invokeMtpPrefillShiftAppend(combo_tokens_d,
                                        input_lengths_d,
                                        offsets_d,
                                        sampled_d,
                                        shifted_d,
                                        static_cast<int32_t>(token_stride),
                                        cuda_graph::graphGetCurrentStream().stream());
        }
        model_input.input_lengths = input_lengths_d;
        model_input.combo_tokens  = shifted_d;
        return;
    }
#endif

    const size_t batch_size   = new_all_token_ids.size(0);
    const size_t token_stride = new_all_token_ids.size(1);
    RTP_LLM_CHECK_WITH_INFO(token_stride > 0, "MTP prefill samples must have a nonempty token dimension");
    const auto new_all_token_ids_cpu = new_all_token_ids.cpu().contiguous();
    const auto input_lengths_cpu =
        model_input.input_lengths.is_cuda() ? model_input.input_lengths.cpu().pin_memory() : model_input.input_lengths;
    RTP_LLM_CHECK_WITH_INFO(input_lengths_cpu.dim() == 1 && input_lengths_cpu.scalar_type() == torch::kInt32
                                && input_lengths_cpu.is_contiguous()
                                && input_lengths_cpu.numel() == static_cast<int64_t>(batch_size),
                            "MTP input_lengths must contain one contiguous int32 entry per sampler row");
    // Read source tensors without pinning/copying them again. Only the shifted
    // outputs need pinned storage for asynchronous H2D; never mutate shared inputs.
    const auto tokens_source = model_input.combo_tokens.cpu().contiguous();
    const auto host_options  = torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    auto       combo_tokens_cpu = torch::empty(tokens_source.sizes(), host_options);
    const int* sampled_tokens   = new_all_token_ids_cpu.data_ptr<int>();
    const int* input_lengths    = input_lengths_cpu.data_ptr<int>();
    const int* source_tokens    = tokens_source.data_ptr<int>();
    int*       combo_tokens     = combo_tokens_cpu.data_ptr<int>();
    const bool shift_positions = model_input.combo_position_ids.defined();
    const auto positions_source = shift_positions ? model_input.combo_position_ids.cpu().contiguous() : torch::Tensor();
    auto positions_cpu = shift_positions ? torch::empty(positions_source.sizes(), host_options) : torch::Tensor();
    int* combo_position_ids   = shift_positions ? positions_cpu.data_ptr<int>() : nullptr;
    const size_t position_id_len_factor = model_input_gatherer_config_.position_id_len_factor;
    RTP_LLM_CHECK_WITH_INFO(!shift_positions
                                || (position_id_len_factor > 0
                                    && positions_source.numel() == tokens_source.numel() * position_id_len_factor),
                            "MTP position ids must match the packed token count and position factor");

    std::vector<MtpPrefillRow> rows;
    rows.reserve(batch_size);
    size_t row_idx = 0;
    int64_t offset = 0;
    for (const auto& stream : all_streams) {
        const int  stream_rows  = stream->nextBatchSize();
        const bool middle_chunk = stream->isMiddleChunk();
        const int32_t tail_mask = mtpTailTextMask(stream);
        for (int row = 0; row < stream_rows; ++row, ++row_idx) {
            RTP_LLM_CHECK_WITH_INFO(row_idx < batch_size, "MTP stream rows exceed sampler rows");
            const int input_length = input_lengths[row_idx];
            RTP_LLM_CHECK_WITH_INFO(input_length > 0 && offset + input_length <= tokens_source.numel(),
                                    "MTP input length %d at row %zu exceeds the packed token window",
                                    input_length, row_idx);
            rows.push_back({stream.get(), offset, offset + input_length, tail_mask});
            std::memcpy(combo_tokens + offset, source_tokens + offset + 1, (input_length - 1) * sizeof(int));

            // Middle chunk tail is the next prompt token; other prefill tails use sampled token.
            // Using sampled token at a chunk boundary would build draft KV from the wrong input.
            const int tail_token = middle_chunk ? stream->nextChunkBoundaryToken() :
                                                  sampled_tokens[row_idx * token_stride + token_stride - 1];
            combo_tokens[offset + input_length - 1] = tail_token;

            if (combo_position_ids != nullptr) {
                int* stream_position_ids = combo_position_ids + offset * position_id_len_factor;
                const int* source_position_ids = positions_source.data_ptr<int>() + offset * position_id_len_factor;
                std::memcpy(stream_position_ids,
                             source_position_ids + position_id_len_factor,
                             (input_length - 1) * position_id_len_factor * sizeof(int));
                int* new_position_ids = stream_position_ids + (input_length - 1) * position_id_len_factor;
                if (middle_chunk) {
                    // The appended token is the next real prompt token, so reuse its exact position id.
                    const int  boundary_pos        = stream->prefixLength() + stream->contextLength();
                    const auto prompt_position_ids = stream->generateContextPositionIds();
                    std::memcpy(new_position_ids,
                                prompt_position_ids.data_ptr<int>() + boundary_pos * position_id_len_factor,
                                position_id_len_factor * sizeof(int));
                } else {
                    // DEFAULT keeps the original last position. Other policies
                    // may also leave it unchanged when the stream has no cached positions.
                    std::memcpy(new_position_ids,
                                source_position_ids + (input_length - 1) * position_id_len_factor,
                                position_id_len_factor * sizeof(int));
                    if (!shift_default_positions) {
                        stream->generateNextPositionId(new_position_ids);
                    }
                }
            }
            offset += input_length;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(row_idx == batch_size && offset == tokens_source.numel(),
                            "MTP stream rows/input lengths do not cover the packed input");

    if (has_multimodal) {
        RTP_LLM_CHECK_WITH_INFO(model_input.text_tokens_mask.defined(), "MTP multimodal prefill requires a text mask");
        model_input.text_tokens_mask =
            shiftedMtpTokenField(model_input.text_tokens_mask, rows, /*is_text_mask=*/true, host_holder);
        shiftMtpMultimodalLocations(model_input, rows, host_holder);
    }
    model_input.combo_tokens_type_ids =
        shiftedMtpTokenField(model_input.combo_tokens_type_ids, rows, /*is_text_mask=*/false, host_holder);

    model_input.input_lengths = toCudaInt32(input_lengths_cpu, host_holder);
    model_input.combo_tokens  = toCudaInt32(combo_tokens_cpu, host_holder);
    if (shift_positions) {
        host_holder.hold_host(positions_cpu);
        model_input.combo_position_ids = model_input.combo_position_ids.is_cuda() ?
            positions_cpu.to(model_input.combo_position_ids.device(), /*non_blocking=*/true) : positions_cpu;
    }
}

torch::Tensor MtpBatchStreamProcessor::dsparkComboTokens(int64_t batch_size, const torch::Tensor& anchors) {
    const int64_t draft_width = dsparkQueryWidth();
    if (!dspark_combo_cache_.defined() || dspark_combo_cache_.size(0) < batch_size
        || dspark_combo_cache_.size(1) != draft_width) {
        dspark_combo_cache_ = fullInt32OnCuda({batch_size, draft_width}, dspark_mask_token_id_);
    }
    auto combo = dspark_combo_cache_.narrow(0, 0, batch_size);
    combo.select(1, 0).copy_(anchors);
    return combo.reshape({-1});
}

torch::Tensor MtpBatchStreamProcessor::dsparkDraftInputLengths(int64_t batch_size) {
    if (!dspark_input_lengths_cache_.defined() || dspark_input_lengths_cache_.size(0) < batch_size) {
        dspark_input_lengths_cache_ = fullInt32OnCuda({batch_size}, dsparkQueryWidth());
    }
    return dspark_input_lengths_cache_.narrow(0, 0, batch_size);
}

torch::Tensor MtpBatchStreamProcessor::dsparkDraftLmIndexes(int64_t batch_size) {
    const int64_t token_count = batch_size * propose_step_;
    if (!dspark_lm_indexes_cache_.defined() || dspark_lm_indexes_cache_.size(0) < token_count) {
        if (!dspark_sample_from_anchor_) {
            dspark_lm_indexes_cache_ = torch::arange(batch_size * dsparkQueryWidth(), cudaInt32Options())
                                           .view({batch_size, dsparkQueryWidth()})
                                           .narrow(1, 1, propose_step_)
                                           .contiguous()
                                           .view({-1});
        } else {
            dspark_lm_indexes_cache_ = torch::arange(token_count, cudaInt32Options());
        }
    }
    return dspark_lm_indexes_cache_.narrow(0, 0, token_count);
}

void MtpBatchStreamProcessor::validatePrefillDSparkCommitInput(const GptModelInputs& model_input) const {
    // The commit call keeps the target's own incremental-prefill geometry:
    // combo = prompt suffix tokens, input_lengths = suffix rows,
    // prefix_lengths = reused prefix, feature rows handed off from the target's
    // normalized all_hidden_states into last_hidden_states (rank-local rows
    // under CP; the draft commit gathers only the projected KV). CacheStore
    // keys derive from these standard fields, which describe exactly the
    // committed prompt — no override channel needed.
    RTP_LLM_CHECK_WITH_INFO(propose_step_ > 0, "dspark draft width must be positive");
    RTP_LLM_CHECK_WITH_INFO(model_input.last_hidden_states.defined(),
                            "dspark prefill: target MTP hidden buffer did not provide aux features");
}

void MtpBatchStreamProcessor::buildDSparkProposeInput(GptModelInputs&      model_input,
                                                      const torch::Tensor& anchors,
                                                      const torch::Tensor& committed_ends,
                                                      TensorHolder&        host_holder) {
    const int64_t batch_size = anchors.numel();
    RTP_LLM_CHECK_WITH_INFO(propose_step_ > 0, "dspark draft width must be positive");
    RTP_LLM_CHECK_WITH_INFO(
        dspark_mask_token_id_ >= 0, "dspark requires a non-negative noise token id, got %d", dspark_mask_token_id_);
    // Fixed-width proposal block: no feature input (the block reads the
    // committed feature KV written by the commit call).
    model_input.combo_tokens       = dsparkComboTokens(batch_size, toCudaInt32(anchors, host_holder));
    model_input.last_hidden_states = torch::Tensor();
    model_input.prefix_lengths     = toCudaInt32(committed_ends, host_holder).contiguous();
    model_input.input_lengths      = dsparkDraftInputLengths(batch_size);
    model_input.sequence_lengths   = emptyInt32OnCuda({0});
    model_input.lm_output_indexes  = dsparkDraftLmIndexes(batch_size);
}

MtpBatchStreamProcessor::DSparkRoundState MtpBatchStreamProcessor::buildDSparkRoundState(
    const StreamGroups& stream_groups, const GptModelInputs& model_input, TensorHolder& host_holder) const {
    if (stream_groups.size() == 0) {
        return {};
    }
    // A freshly handed-over PD stream carries its first generated token and
    // prompt length in the same fields, so new and old streams take one
    // identical path here.
    auto [anchors, committed_ends] = dsparkRoundHeadState(stream_groups, model_input, host_holder);
    auto position_bases            = collectDSparkPositionBases(stream_groups, host_holder);
    return {std::move(anchors), std::move(committed_ends), std::move(position_bases), model_input.engram_token_windows};
}

void MtpBatchStreamProcessor::prepareDSparkProposeModelInput(const DSparkRoundState& round_state,
                                                             GptModelInputs&         model_input,
                                                             TensorHolder&           host_holder) {
    if (!round_state.anchors.defined() || round_state.anchors.numel() == 0) {
        return;
    }
    buildDSparkProposeInput(model_input, round_state.anchors, round_state.committed_ends, host_holder);
    model_input.engram_token_windows = torch::Tensor();
    model_input.combo_position_ids   = expandDSparkPositionIds(round_state.position_bases, dsparkQueryWidth());
    // The fixed-width proposal CUDA graph uses the framework's multi-token
    // graph geometry. This flag classifies that geometry; forward_propose is
    // still the model-semantic phase selector.
    model_input.is_target_verify = true;
}

void MtpBatchStreamProcessor::prepareDSparkTargetVerifyModelInput(const DSparkRoundState& round_state,
                                                                  GptModelInputs&         model_input,
                                                                  const torch::Tensor&    proposals,
                                                                  TensorHolder&           host_holder) {
    if (!round_state.anchors.defined() || round_state.anchors.numel() == 0) {
        return;
    }
    const int64_t batch_size = round_state.anchors.numel();

    RTP_LLM_CHECK_WITH_INFO(proposals.defined() && proposals.dim() == 2 && proposals.size(0) == batch_size
                                && proposals.size(1) == propose_step_,
                            "dspark verify: round-head proposals must be [%ld, %d]",
                            batch_size,
                            propose_step_);

    auto anchor_col = round_state.anchors.reshape({batch_size, 1});
    auto verify     = torch::cat({anchor_col, proposals.to(torch::kInt32)}, 1).reshape({-1});
    if (round_state.engram_windows.defined()) {
        model_input.engram_token_windows = makeEngramVerifyWindows(toCudaInt32(round_state.engram_windows, host_holder),
                                                                   verify.reshape({batch_size, propose_step_ + 1}));
    }
    model_input.prefix_lengths = round_state.committed_ends;
    setVerifyPairInputs(model_input, std::move(verify), batch_size, propose_step_ + 1, host_holder);
    model_input.combo_position_ids = expandDSparkPositionIds(round_state.position_bases, propose_step_ + 1);
    model_input.is_target_verify   = true;
}

void MtpBatchStreamProcessor::expandDSparkTargetVerifyPositionIdsFromProposal(GptModelInputs& model_input) const {
    if (!model_input.combo_position_ids.defined() || model_input.combo_position_ids.numel() == 0) {
        return;
    }
    const int64_t batch_size      = model_input.input_lengths.numel();
    const int64_t proposal_width  = dsparkQueryWidth();
    const int64_t position_factor = static_cast<int64_t>(model_input_gatherer_config_.position_id_len_factor);
    const int64_t expected        = batch_size * proposal_width * position_factor;
    RTP_LLM_CHECK_WITH_INFO(position_factor > 0 && model_input.combo_position_ids.numel() == expected,
                            "DSpARK proposal position shape mismatch: got %ld, expected %ld "
                            "(batch=%ld, width=%ld, factor=%ld)",
                            model_input.combo_position_ids.numel(),
                            expected,
                            batch_size,
                            proposal_width,
                            position_factor);
    auto bases = model_input.combo_position_ids.reshape({batch_size, proposal_width, position_factor}).select(1, 0);
    model_input.combo_position_ids = expandDSparkPositionIds(bases, propose_step_ + 1);
}

void MtpBatchStreamProcessor::updateDecodePostDSparkCommitInput(GptModelInputs&      model_input,
                                                                const torch::Tensor& target_features,
                                                                size_t               batch_size) {
    const int64_t verify_width = propose_step_ + 1;
    RTP_LLM_CHECK_WITH_INFO(target_features.defined(),
                            "dspark decode tail: target MTP hidden buffer did not provide aux features");
    RTP_LLM_CHECK_WITH_INFO(target_features.size(0) == static_cast<int64_t>(batch_size) * verify_width,
                            "dspark decode tail: aux rows %ld != batch*verify_width %ld",
                            target_features.size(0),
                            static_cast<int64_t>(batch_size) * verify_width);

    // Dense commit reuses the target verify geometry verbatim. In particular,
    // combo_tokens remains [anchor, p1, ..., p_gamma], independent of the
    // rejection result; later rounds overwrite rows beyond the accepted
    // prefix. Only the rank-local target feature tensor is rebound here.
    model_input.is_target_verify   = true;
    model_input.last_hidden_states = target_features;
}

void MtpBatchStreamProcessor::updateDecodePostDraftModelInput(
    GptModelInputs&                              model_input,
    const GptModelOutputs&                       model_output,
    const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
    const size_t                                 batch_size,
    torch::Tensor&                               hidden_states_d_t,
    TensorHolder&                                host_holder) {
    // Keep the target's B x Q geometry for every MTP update. The sampler
    // publishes all Q slots in both synchronous and device-state scheduling.
    // Candidate rows beyond accept_len are overwritten next round; only
    // accepted positions advance the stream's logical cache.
    model_input.is_target_verify = false;
    model_input.is_mtp_draft_update = true;
    const int64_t width             = propose_step_ + 1;
    const int64_t total_tokens      = width * static_cast<int64_t>(batch_size);
    RTP_LLM_CHECK_WITH_INFO(speculative_sampler_output.accept_tokens.numel() == total_tokens
                                && speculative_sampler_output.accept_len.numel() == static_cast<int64_t>(batch_size),
                            "MTP update requires B x Q accepted-token slots");
    RTP_LLM_CHECK_WITH_INFO(model_output.all_hidden_states.defined()
                                && model_output.all_hidden_states.size(0) == total_tokens,
                            "MTP update requires one target hidden row per physical token");
    // Input lengths, position IDs and cache tables retain the target-verify layout.
    model_input.combo_tokens =
        toCudaInt32(speculative_sampler_output.accept_tokens.reshape({(int64_t)total_tokens}), host_holder);
    auto accept_len_d = toCudaInt32(speculative_sampler_output.accept_len, host_holder);
    model_input.lm_output_indexes =
        torch::arange(
            0, total_tokens, propose_step_ + 1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))
        + (accept_len_d - 1);
    model_input.last_hidden_states = model_output.all_hidden_states;
    hidden_states_d_t              = model_input.last_hidden_states;
}

void MtpBatchStreamProcessor::updateOneStepDraftSamplerOutput(const StreamGroups& stream_groups,
                                                              SamplerOutput&      draft_sampler_output,
                                                              torch::Tensor&      draft_token_probs_d_t,
                                                              TensorHolder&       host_holder) {
    const size_t batch_size      = stream_groups.size();
    auto         draft_token_ids = emptyInt32OnPreferredDevice({(int64_t)batch_size, (int64_t)propose_step_});

    std::vector<torch::Tensor> draft_token_probs_list;
    std::vector<torch::Tensor> draft_token_id_slices;
    draft_token_id_slices.reserve(batch_size);

    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer = stream->getSPOutputBuffer();
        auto draft_token      = pickOneStepDraftToken(stream);
        RTP_LLM_CHECK_WITH_INFO(
            draft_token.defined(), "one-step MTP draft sampler token missing for stream %ld", stream->streamId());
        draft_token_id_slices.push_back(draft_token);

        // Prefer main-thread device-state all_probs; fallback is safe after
        // worker clear because clear runs after specUpdate writes all_probs.
        const auto& dev_probs     = stream->getDraftAllProbsGpu();
        const bool  use_dev_probs = useMtpDeviceState() && dev_probs.defined();
        RTP_LLM_CHECK_WITH_INFO(use_dev_probs || (sp_output_buffer && sp_output_buffer->all_probs.defined()),
                                "one-step MTP draft all_probs missing for stream %ld",
                                stream->streamId());
        draft_token_probs_list.push_back(use_dev_probs ? dev_probs : sp_output_buffer->all_probs);
    }

    if (!draft_token_id_slices.empty()) {
        draft_token_ids = torch::cat(draft_token_id_slices, 0)
                              .to(torch::kInt32)
                              .reshape({(int64_t)batch_size, (int64_t)propose_step_});
        if (useMtpDeviceInput() && !draft_token_ids.is_cuda()) {
            host_holder.hold_host(draft_token_ids);
            draft_token_ids = draft_token_ids.to(torch::kCUDA, /*non_blocking=*/true);
        }
    }

    draft_token_probs_d_t          = torch::stack(draft_token_probs_list, 0).contiguous();
    draft_sampler_output.all_probs = draft_token_probs_d_t;
    draft_sampler_output.token_ids = std::move(draft_token_ids);
}

void MtpBatchStreamProcessor::updateMultiStepDraftSamplerOutput(const StreamGroups&         stream_groups,
                                                                SamplerOutput&              draft_sampler_output,
                                                                torch::Tensor&              draft_token_ids_d_t,
                                                                torch::Tensor&              spec_token_ids_d_t,
                                                                torch::Tensor&              draft_token_probs_d_t,
                                                                std::vector<torch::Tensor>& draft_token_probs_list) {
    std::vector<torch::Tensor> prev_draft_token_probs_list;
    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer = stream->getSPOutputBuffer();
        // Prefer device-state draft_all_probs (see comment in
        // updateOneStepDraftSamplerOutput for the same fallback contract).
        const auto& dev_probs = stream->getDraftAllProbsGpu();
        prev_draft_token_probs_list.push_back(useMtpDeviceState() && dev_probs.defined() ? dev_probs :
                                                                                           sp_output_buffer->all_probs);
    }

    auto pre_draft_token_probs = torch::stack(prev_draft_token_probs_list, 0).contiguous();
    draft_token_probs_list.insert(draft_token_probs_list.begin(), pre_draft_token_probs);

    draft_token_probs_d_t          = torch::cat(draft_token_probs_list, 1).contiguous();
    draft_sampler_output.all_probs = draft_token_probs_d_t;

    // draft_token_ids_d_t = draft_token_ids_d_t[:, 1:]
    spec_token_ids_d_t             = draft_token_ids_d_t.slice(1, 1).contiguous();
    draft_sampler_output.token_ids = spec_token_ids_d_t;
}

void MtpBatchStreamProcessor::preparePrefillSpecUpdateInfo(const StreamGroups&                stream_groups,
                                                           const MergedOutput&                prefill_output,
                                                           const MergedOutput&                propose_output,
                                                           const torch::Tensor&               draft_last_hidden_states,
                                                           const torch::Tensor&               new_tokens_all,
                                                           std::vector<StreamSpecUpdateInfo>& spec_update_infos) const {
    const auto& sampler_output       = prefill_output.sampler_output;
    const auto& draft_sampler_output = propose_output.sampler_output;
    const auto& draft_model_output   = propose_output.model_output;

    const auto& new_all_token_ids         = sampler_output.token_ids;
    const auto& propose_new_all_token_ids = draft_sampler_output.token_ids;

    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", tensorDebugStringWithData<int32_t>(new_all_token_ids).c_str());
    RTP_LLM_LOG_DEBUG("propose_new_all_token_ids = [%s]",
                      tensorDebugStringWithData<int32_t>(propose_new_all_token_ids).c_str());

    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == (size_t)new_all_token_ids.size(0));
    const size_t token_stride = new_all_token_ids.size(1);

    // TODO(async): stream bookkeeping below still iterates token_ids/success
    // on CPU. Keep the .cpu() explicit until spec-update assembly is
    // device-native.
    const torch::Tensor new_all_token_ids_cpu =
        new_all_token_ids.is_cuda() ? new_all_token_ids.cpu() : new_all_token_ids;
    const torch::Tensor success_cpu = sampler_output.success.defined() ? sampler_output.success.cpu() : torch::Tensor();

    int batch_idx_in  = 0;
    int batch_idx_out = 0;
    int token_offset  = 0;

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();
        auto token_size      = stream->currentExecuteTokenSize();

        // normal stream info
        auto new_tokens = new_tokens_all.narrow(0, batch_idx_out, next_batch_size);
        for (size_t i = 0; i < next_batch_size; ++i) {
            new_tokens.data_ptr<int32_t>()[i] =
                new_all_token_ids_cpu.data_ptr<int32_t>()[(batch_idx_out + i) * token_stride + token_stride - 1];
        }

        for (int i = 0; i < cur_batch_size; ++i) {
            if (success_cpu.defined() && !(success_cpu.data_ptr<bool>()[batch_idx_in + i])) {
                stream->reportError(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
            }
        }

        // speculative decoding info
        torch::Tensor propose_all_probs;
        if (draft_sampler_output.all_probs.defined()) {
            propose_all_probs =
                draft_sampler_output.all_probs.narrow(0, batch_idx_out, next_batch_size).to(torch::kCUDA).clone();
        }

        torch::Tensor last_hidden_states;
        if (propose_step_ > 1 && !is_dspark_) {
            if (draft_last_hidden_states.defined() && draft_last_hidden_states.numel() > 0) {
                // CP: the draft forward output is rank-local, so the per-request
                // rows come from the draft model's MTP last-hidden buffer.
                last_hidden_states = cloneHiddenSlice(draft_last_hidden_states, batch_idx_out, 1);
            } else {
                last_hidden_states =
                    cloneHiddenSlice(draft_model_output.all_hidden_states, token_offset + token_size - 1, 1);
            }
        }

        StreamSpecUpdateInfo update_info{
            new_tokens, 1, -1, std::move(last_hidden_states), std::move(propose_all_probs)};
        // Match normal prefill output rows. These are target outputs, not the
        // draft recurrence feature used to construct the next proposal.
        if (stream->generateConfig()->return_hidden_states) {
            update_info.target_hidden_states =
                prefill_output.model_output.hidden_states.narrow(0, batch_idx_in, cur_batch_size).clone();
        }
        if (stream->returnLogits()) {
            update_info.target_logits = prefill_output.model_output.logits.narrow(0, batch_idx_in, cur_batch_size);
        }
        if (stream->generateConfig()->return_all_probs != ReturnAllProbsMode::NONE) {
            const auto& response_probs = stream->generateConfig()->return_all_probs == ReturnAllProbsMode::ORIGINAL ?
                                             sampler_output.original_all_probs :
                                             sampler_output.all_probs;
            RTP_LLM_CHECK_WITH_INFO(response_probs.defined() && response_probs.dim() == 2,
                                    "MTP prefill target probabilities must be [batch, vocab]");
            update_info.all_probs =
                response_probs.narrow(0, batch_idx_out, next_batch_size).unsqueeze(1).cpu();
        }
        spec_update_infos.push_back(std::move(update_info));

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += token_size;
    }
}

void MtpBatchStreamProcessor::prepareDecodeSpecUpdateInfo(
    const StreamGroups&                          stream_groups,
    const speculative::SpeculativeSamplerOutput& spec_decode_output,
    const MergedOutput&                          draft_prefill_output,
    std::vector<StreamSpecUpdateInfo>&           spec_update_infos) const {
    // wait for the transfer to complete
    spec_decode_output.transfer_done_event->synchronize();
    const auto& accept_len    = spec_decode_output.accept_len_cpu;
    const auto& accept_tokens = spec_decode_output.accept_tokens_cpu;

    const auto& draft_model_output   = draft_prefill_output.model_output;
    const auto& draft_sampler_output = draft_prefill_output.sampler_output;

    int batch_idx_in  = 0;
    int batch_idx_out = 0;
    int token_offset  = 0;

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();

        // speculative decoding info
        torch::Tensor propose_all_probs;
        if (draft_sampler_output.all_probs.defined()) {
            propose_all_probs =
                draft_sampler_output.all_probs.narrow(0, batch_idx_out, next_batch_size).to(torch::kCUDA).clone();
        }

        // This scalar read runs on the bookkeeping worker after accept_len is
        // ready, so it does not sync the main thread. Move to main thread only
        // after replacing .item() with a device-side index.
        int cur_accept_len = accept_len[batch_idx_out].item<int>();

        torch::Tensor last_hidden_states;
        if (propose_step_ > 1 && !is_dspark_) {
            last_hidden_states =
                cloneHiddenSlice(draft_model_output.all_hidden_states, token_offset + cur_accept_len - 1, 1);
        }

        torch::Tensor accept_tokens_tensor =
            accept_tokens.narrow(0, batch_idx_out, next_batch_size).narrow(1, 0, cur_accept_len).contiguous();
        StreamSpecUpdateInfo spec_update_info{
            accept_tokens_tensor, cur_accept_len, -1, std::move(last_hidden_states), std::move(propose_all_probs)};
        if (stream->generateConfig()->return_all_probs != ReturnAllProbsMode::NONE) {
            RTP_LLM_CHECK_WITH_INFO(spec_decode_output.target_probs_cpu.defined()
                                        && spec_decode_output.target_probs_cpu.dim() == 3
                                        && spec_decode_output.target_probs_cpu.size(1) >= cur_accept_len,
                                    "MTP decode target probabilities must cover every accepted token");
            spec_update_info.all_probs = spec_decode_output.target_probs_cpu.narrow(0, batch_idx_out, next_batch_size)
                                             .narrow(1, 0, cur_accept_len);
        }
        spec_update_info.speculative_propose_step = propose_step_;
        spec_update_info.accepted_draft_tokens    = std::max(0, cur_accept_len - 1);
        if (stream->generateConfig()->return_hidden_states && spec_decode_output.target_hidden_states.defined()) {
            spec_update_info.target_hidden_states =
                spec_decode_output.target_hidden_states.narrow(0, batch_idx_out, next_batch_size);
        }
        if (stream->returnLogits() && spec_decode_output.target_logits.defined()) {
            spec_update_info.target_logits = spec_decode_output.target_logits.narrow(0, batch_idx_out, next_batch_size);
        }
        // Per-stream verify errors from SpecLogitsVerifyRunner ride the update
        // path so grammar/think mask failures reach the stream (main #1006).
        const size_t stream_idx = spec_update_infos.size();
        if (stream_idx < spec_decode_output.processor_errors.size()) {
            spec_update_info.error_info = spec_decode_output.processor_errors[stream_idx];
        }
        spec_update_infos.push_back(std::move(spec_update_info));

        // Draft update retains Q physical rows regardless of accepted length.
        token_offset += propose_step_ + 1;
        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
    }
}

void MtpBatchStreamProcessor::gatherHiddenStates(const StreamGroups& stream_groups, GptModelInputs& model_input) const {
    RTP_LLM_PROFILE_SCOPE("normal_engine.mtp_batch_stream_processor.gather_hidden_states");
    auto            all_streams = stream_groups.allStreams();
    c10::ScalarType dtype       = c10::ScalarType::Undefined;
    size_t          hidden_size = 0;

    // Prefer main-thread device-state hidden_states to avoid racing worker
    // writes to sp_output_buffer when DROP_BROAD_SYNC=1. Fallback covers older
    // or first-step streams without published device state.
    auto pick_hidden_states = [](const GenerateStreamPtr& stream) -> torch::Tensor {
        auto dev = stream->getLastHiddenStatesGpu();
        if (dev.defined()) {
            return dev;
        }
        return stream->getSPOutputBuffer()->hidden_states;
    };

    size_t all_hidden_tokens_num = 0;
    for (auto& stream : all_streams) {
        const auto& hidden_states = pick_hidden_states(stream);
        RTP_LLM_CHECK(hidden_states.defined());
        RTP_LLM_CHECK(hidden_states.dim() == 2);
        if (dtype == c10::ScalarType::Undefined) {
            dtype = hidden_states.scalar_type();
        } else {
            RTP_LLM_CHECK(dtype == hidden_states.scalar_type());
        }
        if (hidden_size == 0) {
            hidden_size = hidden_states.size(1);
        } else {
            RTP_LLM_CHECK(hidden_size == (size_t)hidden_states.size(1));
        }
        all_hidden_tokens_num += hidden_states.size(0);
    }

    // copy hidden
    torch::Tensor all_hidden_states;
    if (all_streams.size() == 0) {
        model_input.last_hidden_states = torch::Tensor();
        return;
    } else if (all_streams.size() == 1) {
        all_hidden_states = pick_hidden_states(all_streams.front());
    } else {
        RTP_LLM_PROFILE_SCOPE("normal_engine.mtp_batch_stream_processor.gather_hidden_states.fused_copy");
        all_hidden_states = torch::empty({(int64_t)all_hidden_tokens_num, (int64_t)hidden_size},
                                         torch::TensorOptions().dtype(dtype).device(torch::kCUDA));

        bool all_sources_fused_copy_ready = true;
        for (auto& stream : all_streams) {
            const auto& hidden_states = pick_hidden_states(stream);
            if (!hidden_states.is_cuda() || !hidden_states.is_contiguous()) {
                all_sources_fused_copy_ready = false;
                break;
            }
        }

        size_t accu_dst_offset = 0;
        if (all_sources_fused_copy_ready) {
            auto               dst_base = static_cast<char*>(all_hidden_states.data_ptr());
            FusedD2DCopyParams params;
            auto               flush_fused_copy = [&]() {
                if (params.num_copies > 0) {
                    fusedCopy(params);
                    params.clear();
                }
            };

            // Do not use execMultiMergeCopy here: its thrust::device_vector
            // metadata staging creates H2D work on this hot path. fusedCopy
            // passes copy metadata as kernel params and can be chunked.
            for (auto& stream : all_streams) {
                const auto& hidden_states    = pick_hidden_states(stream);
                size_t      hidden_copy_size = hidden_states.nbytes();
                if (params.num_copies == MAX_FUSED_D2D_COPIES) {
                    flush_fused_copy();
                }
                params.add(hidden_states.data_ptr(), dst_base + accu_dst_offset, hidden_copy_size);
                accu_dst_offset += hidden_copy_size;
            }
            flush_fused_copy();
        } else {
            size_t index = 0;
            for (auto& stream : all_streams) {
                const auto& hidden_states = pick_hidden_states(stream);
                auto        hidden_num    = hidden_states.size(0);
                all_hidden_states.narrow(0, index, hidden_num).copy_(hidden_states);
                index += hidden_num;
            }
        }
    }

    model_input.last_hidden_states = all_hidden_states;
}

}  // namespace rtp_llm
