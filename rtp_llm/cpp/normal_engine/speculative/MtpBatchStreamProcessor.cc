#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalInputUtils.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
// REBASE CONFLICT CONTEXT(518707c73): keep new base logits-processor state
// support and add the source branch fused prefill shift/append CUDA helper.
#include "rtp_llm/models_py/bindings/cuda/kernels/mtp_target_verify_prepare.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>
#include <cstring>

namespace rtp_llm {
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

torch::Tensor clonePrefillLastHiddenSlice(const torch::Tensor& hidden_states,
                                          int64_t              batch_idx_out,
                                          int64_t              token_offset,
                                          int64_t              token_size,
                                          int64_t              total_batch_size_out) {
    if (!hidden_states.defined() || hidden_states.numel() == 0) {
        return torch::Tensor();
    }
    const bool compact_last_hidden = hidden_states.dim() == 2 && hidden_states.size(0) == total_batch_size_out;
    const auto start               = compact_last_hidden ? batch_idx_out : token_offset + token_size - 1;
    return cloneHiddenSlice(hidden_states, start, 1);
}

torch::Tensor copyToPinnedCpuAsync(const torch::Tensor& tensor, bool& need_sync) {
    if (!tensor.defined() || !tensor.is_cuda()) {
        return tensor;
    }

    auto cpu_tensor = torch::empty(
        tensor.sizes(), torch::TensorOptions().dtype(tensor.scalar_type()).device(torch::kCPU).pinned_memory(true));
    cpu_tensor.copy_(tensor, /*non_blocking=*/true);
    need_sync = true;
    return cpu_tensor;
}

void syncPinnedCpuCopies(bool need_sync) {
    if (need_sync) {
        // Sampler kernels may run on the graph/current stream. Make the D2H
        // completion explicit before CPU stream bookkeeping consumes it.
        cuda_graph::graphGetCurrentStream().synchronize();
    }
}

torch::Tensor shiftedMtpTokenField(const torch::Tensor& field,
                                   const torch::Tensor& input_lengths_cpu,
                                   int32_t              append_value,
                                   bool                 append_last_value) {
    if (!field.defined() || field.numel() == 0) {
        return field;
    }
    RTP_LLM_CHECK_WITH_INFO(field.scalar_type() == torch::kInt32,
                            "MTP multimodal token field must be int32, got %s",
                            c10::toString(field.scalar_type()));
    RTP_LLM_CHECK_WITH_INFO(input_lengths_cpu.dim() == 1 && input_lengths_cpu.scalar_type() == torch::kInt32,
                            "MTP input_lengths must be a 1-D int32 tensor");
    auto source                = field.is_cuda() ? field.cpu() : field;
    source                     = source.contiguous();
    const int64_t total_tokens = source.numel();
    int64_t       length_sum   = 0;
    const auto*   lengths      = input_lengths_cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < input_lengths_cpu.numel(); ++i) {
        RTP_LLM_CHECK_WITH_INFO(
            lengths[i] > 0, "MTP input length must be positive, got %d at request %ld", lengths[i], i);
        length_sum += lengths[i];
    }
    RTP_LLM_CHECK_WITH_INFO(length_sum == total_tokens,
                            "MTP token field length mismatch: field=%ld input_lengths=%ld",
                            total_tokens,
                            length_sum);

    auto output =
        torch::empty({total_tokens}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    const auto* src    = source.data_ptr<int32_t>();
    auto*       dst    = output.data_ptr<int32_t>();
    int64_t     offset = 0;
    for (int64_t request = 0; request < input_lengths_cpu.numel(); ++request) {
        const int64_t length = lengths[request];
        if (length > 1) {
            std::memcpy(dst + offset, src + offset + 1, static_cast<size_t>(length - 1) * sizeof(int32_t));
        }
        dst[offset + length - 1] = append_last_value ? src[offset + length - 1] : append_value;
        offset += length;
    }
    return field.is_cuda() ? output.to(field.device(), /*non_blocking=*/true) : output;
}

torch::Tensor shiftedMtpPositionIds(const torch::Tensor& field, const torch::Tensor& input_lengths_cpu) {
    if (!field.defined() || field.numel() == 0) {
        return field;
    }
    RTP_LLM_CHECK_WITH_INFO(field.scalar_type() == torch::kInt32,
                            "MTP position ids must be int32, got %s",
                            c10::toString(field.scalar_type()));
    auto source            = field.is_cuda() ? field.cpu() : field;
    source                 = source.contiguous();
    int64_t     length_sum = 0;
    const auto* lengths    = input_lengths_cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < input_lengths_cpu.numel(); ++i) {
        RTP_LLM_CHECK_WITH_INFO(
            lengths[i] > 0, "MTP input length must be positive, got %d at request %ld", lengths[i], i);
        length_sum += lengths[i];
    }
    RTP_LLM_CHECK_WITH_INFO(length_sum > 0 && source.numel() % length_sum == 0,
                            "MTP position ids numel=%ld is not divisible by token count=%ld",
                            source.numel(),
                            length_sum);
    const int64_t factor = source.numel() / length_sum;
    auto          output =
        torch::empty({source.numel()}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    const auto* src    = source.data_ptr<int32_t>();
    auto*       dst    = output.data_ptr<int32_t>();
    int64_t     offset = 0;
    for (int64_t request = 0; request < input_lengths_cpu.numel(); ++request) {
        const int64_t length = lengths[request];
        if (length > 1) {
            std::memcpy(dst + offset * factor,
                        src + (offset + 1) * factor,
                        static_cast<size_t>(length - 1) * static_cast<size_t>(factor) * sizeof(int32_t));
        }
        int32_t next_position = src[(offset + length - 1) * factor];
        for (int64_t component = 1; component < factor; ++component) {
            next_position = std::max(next_position, src[(offset + length - 1) * factor + component]);
        }
        ++next_position;
        for (int64_t component = 0; component < factor; ++component) {
            dst[(offset + length - 1) * factor + component] = next_position;
        }
        offset += length;
    }
    return field.is_cuda() ? output.to(field.device(), /*non_blocking=*/true) : output;
}

void shiftMtpMultimodalLocations(GptModelInputs& model_input, const torch::Tensor& input_lengths_cpu) {
    const bool has_features = model_input.multimodal_features.has_value() && !model_input.multimodal_features->empty();
    const bool has_locs     = model_input.mm_features_locs.defined() && model_input.mm_features_locs.numel() > 0;
    if (!has_features && !has_locs) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(has_features && has_locs,
                            "MTP multimodal features and mm_features_locs must be provided together");
    RTP_LLM_CHECK_WITH_INFO(input_lengths_cpu.scalar_type() == torch::kInt32 && input_lengths_cpu.dim() == 1,
                            "MTP input_lengths must be a 1-D int32 tensor");

    auto locs =
        model_input.mm_features_locs.is_cuda() ? model_input.mm_features_locs.cpu() : model_input.mm_features_locs;
    locs = locs.contiguous();
    RTP_LLM_CHECK_WITH_INFO(locs.scalar_type() == torch::kInt32 && locs.dim() == 1,
                            "MTP mm_features_locs must be a 1-D int32 tensor");
    const auto& features = model_input.multimodal_features.value();
    RTP_LLM_CHECK_WITH_INFO(locs.numel() == static_cast<int64_t>(features.size()),
                            "MTP multimodal feature/location count mismatch: features=%zu locs=%ld",
                            features.size(),
                            locs.numel());
    const bool  has_extra_input = model_input.mm_extra_input.has_value() && !model_input.mm_extra_input->empty();
    const auto* extra_input     = has_extra_input ? &model_input.mm_extra_input.value() : nullptr;
    RTP_LLM_CHECK_WITH_INFO(!has_extra_input || extra_input->size() == features.size(),
                            "MTP mm_extra_input count mismatch: extra_input=%zu features=%zu",
                            extra_input ? extra_input->size() : 0,
                            features.size());

    std::vector<int64_t> request_starts(input_lengths_cpu.numel() + 1, 0);
    const auto*          lengths = input_lengths_cpu.data_ptr<int32_t>();
    for (int64_t request = 0; request < input_lengths_cpu.numel(); ++request) {
        RTP_LLM_CHECK_WITH_INFO(lengths[request] > 0,
                                "MTP input length must be positive, got %d at request %ld",
                                lengths[request],
                                request);
        request_starts[request + 1] = request_starts[request] + lengths[request];
    }

    std::vector<torch::Tensor> shifted_features;
    std::vector<torch::Tensor> shifted_extra_input;
    std::vector<int32_t>       shifted_locs;
    shifted_features.reserve(features.size());
    shifted_locs.reserve(features.size());
    if (has_extra_input) {
        shifted_extra_input.reserve(extra_input->size());
    }

    const auto* src_locs = locs.data_ptr<int32_t>();
    for (int64_t feature_idx = 0; feature_idx < locs.numel(); ++feature_idx) {
        const auto& feature = features[feature_idx];
        RTP_LLM_CHECK_WITH_INFO(feature.defined() && feature.dim() == 2,
                                "MTP multimodal feature %ld must be a defined 2-D tensor",
                                feature_idx);
        const int64_t feature_start = src_locs[feature_idx];
        const int64_t feature_len   = feature.size(0);
        const int64_t feature_end   = feature_start + feature_len;
        RTP_LLM_CHECK_WITH_INFO(feature_start >= 0 && feature_len > 0,
                                "MTP multimodal feature %ld has invalid range [%ld,%ld)",
                                feature_idx,
                                feature_start,
                                feature_end);
        int64_t owner = -1;
        for (int64_t request = 0; request < input_lengths_cpu.numel(); ++request) {
            if (feature_start >= request_starts[request] && feature_start < request_starts[request + 1]) {
                owner = request;
                break;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(owner >= 0,
                                "MTP multimodal feature %ld [%ld,%ld) does not overlap any request",
                                feature_idx,
                                feature_start,
                                feature_end);
        RTP_LLM_CHECK_WITH_INFO(feature_end <= request_starts[owner + 1],
                                "MTP multimodal feature %ld [%ld,%ld) crosses request %ld boundary [%ld,%ld)",
                                feature_idx,
                                feature_start,
                                feature_end,
                                owner,
                                request_starts[owner],
                                request_starts[owner + 1]);

        // MTP removes the first token of every request and writes the sampled
        // token into that request's last slot. A feature that starts at the
        // request boundary therefore loses its first row as well; all other
        // features simply move left by one packed position. The request owner
        // is used only to detect this boundary and must not be subtracted from
        // the packed global location.
        const bool    starts_at_request = feature_start == request_starts[owner];
        const int64_t feature_offset    = starts_at_request ? 1 : 0;
        const int64_t shifted_len       = feature_len - feature_offset;
        if (shifted_len <= 0) {
            continue;
        }

        shifted_features.emplace_back(feature.slice(0, feature_offset, feature_len).contiguous());
        shifted_locs.emplace_back(static_cast<int32_t>(starts_at_request ? feature_start : feature_start - 1));
        if (has_extra_input) {
            shifted_extra_input.emplace_back(
                sliceMultimodalExtraInput((*extra_input)[feature_idx], feature, feature_offset, feature_len));
        }
    }

    auto shifted_locs_cpu = torch::empty({static_cast<int64_t>(shifted_locs.size())},
                                         torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    if (!shifted_locs.empty()) {
        std::memcpy(shifted_locs_cpu.data_ptr<int32_t>(), shifted_locs.data(), shifted_locs.size() * sizeof(int32_t));
    }
    model_input.multimodal_features = std::move(shifted_features);
    if (has_extra_input) {
        model_input.mm_extra_input = std::move(shifted_extra_input);
    }
    model_input.mm_features_locs =
        model_input.mm_features_locs.is_cuda() ?
            shifted_locs_cpu.to(model_input.mm_features_locs.device(), /*non_blocking=*/true) :
            shifted_locs_cpu;
}

void shiftMtpMultimodalMetadata(GptModelInputs& model_input, const torch::Tensor& input_lengths) {
    if (!hasMultimodalModelInputs(model_input)) {
        return;
    }
    auto input_lengths_cpu = input_lengths.is_cuda() ? input_lengths.cpu() : input_lengths;
    input_lengths_cpu      = input_lengths_cpu.contiguous();
    if (model_input.text_tokens_mask.defined() && model_input.text_tokens_mask.numel() > 0) {
        model_input.text_tokens_mask = shiftedMtpTokenField(model_input.text_tokens_mask, input_lengths_cpu, 1, false);
    }
    if (model_input.combo_tokens_type_ids.defined() && model_input.combo_tokens_type_ids.numel() > 0) {
        model_input.combo_tokens_type_ids =
            shiftedMtpTokenField(model_input.combo_tokens_type_ids, input_lengths_cpu, 0, true);
    }
    if (model_input.combo_position_ids.defined() && model_input.combo_position_ids.numel() > 0) {
        model_input.combo_position_ids = shiftedMtpPositionIds(model_input.combo_position_ids, input_lengths_cpu);
    }
    shiftMtpMultimodalLocations(model_input, input_lengths_cpu);
}

}  // namespace

namespace {

// Fallback-hit counters. Each counter is incremented when a hot-path
// gather/prepare function falls through to the legacy CPU/GPU mixed path.
// The counters are read by the rate-limited log emitter below; production
// monitoring should prefer the kmonitor metric `executor.mtp.async_fallback.reason` once it is wired, but this
// in-process counter provides immediate visibility today without requiring a metrics-schema change.
std::atomic<uint64_t> g_mtp_device_state_fallback_count{0};
std::atomic<uint64_t> g_mtp_device_state_success_count{0};

struct CachedMtpDeviceInputFlag {
    bool        on;
    std::string value;
};

const CachedMtpDeviceInputFlag kUseMtpDeviceInput = []() {
    const char* env = std::getenv("RTP_LLM_DEVICE_INPUT");
    bool        on  = env != nullptr && std::string(env) == "1";
    return CachedMtpDeviceInputFlag{on, env ? env : "(unset)"};
}();

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
    static const bool logged = []() {
        RTP_LLM_LOG_INFO("[mtp-device-input] RTP_LLM_DEVICE_INPUT=%s -> processor enabled=%d",
                         kUseMtpDeviceInput.value.c_str(),
                         static_cast<int>(kUseMtpDeviceInput.on));
        return true;
    }();
    (void)logged;
    return kUseMtpDeviceInput.on;
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

torch::Tensor pickOneStepTargetLastToken(const GenerateStreamPtr& stream) {
    const auto& accept_tokens = stream->getAcceptTokensGpu();
    const auto& accept_len    = stream->getAcceptLenGpu();
    if (accept_tokens.defined() && accept_tokens.is_cuda() && accept_len.defined() && accept_len.is_cuda()) {
        auto idx_t = (accept_len - 1).to(torch::kLong);
        return accept_tokens.squeeze(0).index_select(/*dim=*/0, idx_t);
    }

    auto sp_output_buffer = stream->getSPOutputBuffer();
    if (!sp_output_buffer) {
        return torch::Tensor();
    }
    if (sp_output_buffer->target_token_gpu.defined() && sp_output_buffer->target_token_gpu.is_cuda()) {
        return lastColumnAsFlat(sp_output_buffer->target_token_gpu);
    }
    return columnAsFlat(sp_output_buffer->tokens, 0);
}

torch::Tensor pickOneStepDraftToken(const GenerateStreamPtr& stream) {
    const auto& state_propose = stream->getProposeTokensGpu();
    if (state_propose.defined()) {
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
    return toCudaInt32(committed_len, host_holder);
}

torch::Tensor normalDecodePositionToDraftDecodePosition(const torch::Tensor& normal_decode_position,
                                                        TensorHolder&        host_holder) {
    auto position = toCudaInt32(normal_decode_position, host_holder);
    if (!position.defined() || position.numel() == 0) {
        return position;
    }
    return (position + 1).to(torch::kInt32);
}

void setVerifyPairInputs(GptModelInputs& model_input,
                         torch::Tensor   combo_tokens,
                         size_t          batch_size,
                         size_t          score_len,
                         TensorHolder&   host_holder) {
    model_input.combo_tokens     = std::move(combo_tokens);
    model_input.sequence_lengths = emptyInt32OnCuda({0});
    model_input.clearLastHiddenStates();
    model_input.prefix_lengths    = toCudaInt32(model_input.prefix_lengths, host_holder).contiguous();
    model_input.input_lengths     = fullInt32OnCuda({static_cast<int64_t>(batch_size)}, score_len);
    model_input.lm_output_indexes = makeCudaInt32Range(static_cast<int64_t>(batch_size * score_len));
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

const char* missingMtpStateReason(const GenerateStreamPtr& stream) {
    if (!stream->getAcceptTokensGpu().defined()) {
        return "accept_tokens_gpu_missing";
    }
    if (!stream->getAcceptLenGpu().defined()) {
        return "accept_len_gpu_missing";
    }
    if (!stream->getProposeTokensGpu().defined()) {
        return "propose_tokens_gpu_missing";
    }
    if (!stream->getNextSeqLenGpu().defined()) {
        return "next_seq_len_gpu_missing";
    }
    return nullptr;
}

void logMtpStateFallback(const GenerateStreamPtr& stream, const char* reason) {
    const uint64_t count = g_mtp_device_state_fallback_count.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!shouldLogFallback(count)) {
        return;
    }
    const auto& mtp_state        = stream->getMtpAsyncDeviceState();
    auto        sp_output_buffer = stream->getSPOutputBuffer();
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
        torch::Tensor gpu_t = stream->getProposeTokensGpu();
        if (!gpu_t.defined() || !gpu_t.is_cuda()) {
            auto sp_output_buffer = stream->getSPOutputBuffer();
            if (sp_output_buffer && sp_output_buffer->propose_tokens_gpu.defined()
                && sp_output_buffer->propose_tokens_gpu.is_cuda()) {
                gpu_t = sp_output_buffer->propose_tokens_gpu;
            } else {
                logMtpStateFallback(stream, "propose_tokens_gpu_missing");
                return false;
            }
        }
        propose_slices.push_back(lastColumnAsFlat(gpu_t));
        if (next_seq_lengths) {
            torch::Tensor next_seq_len = stream->getNextSeqLenGpu();
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

bool MtpBatchStreamProcessor::needsPrefillTargetOutputs(const StreamGroups& stream_groups) const {
    if (!is_dspark_) {
        return false;
    }
    const auto requested = [](const auto& streams) {
        return std::any_of(streams.begin(), streams.end(), [](const auto& stream) {
            return stream->returnLogits() || stream->generateConfig()->return_hidden_states;
        });
    };
    return requested(stream_groups.decodeStreams()) || requested(stream_groups.contextStreams());
}

absl::StatusOr<std::vector<MtpBatchStreamProcessor::PrefillTargetOutput>>
MtpBatchStreamProcessor::capturePrefillTargetOutputs(const StreamGroups&    stream_groups,
                                                     const GptModelOutputs& target,
                                                     bool                   target_need_all_logits,
                                                     const torch::Tensor&   original_lm_output_indexes) const {
    // No list/vector allocation or CUDA work for ordinary production requests.
    if (!needsPrefillTargetOutputs(stream_groups)) {
        return std::vector<PrefillTargetOutput>{};
    }
    const auto streams    = stream_groups.allStreams();
    int64_t    total_rows = 0;
    for (const auto& stream : streams) {
        total_rows += stream->currentBatchSize();
    }
    std::vector<PrefillTargetOutput> outputs(streams.size());
    int64_t                          row   = 0;
    size_t                           index = 0;
    for (const auto& stream : streams) {
        auto&      output       = outputs[index++];
        const bool wants_logits = stream->returnLogits();
        const bool wants_hidden = stream->generateConfig()->return_hidden_states;
        if (wants_logits || wants_hidden) {
            if (stream->currentBatchSize() != 1 || stream->nextBatchSize() != 1 || stream->maxBatchSize() != 1
                || stream->hasNumBeams()) {
                return absl::InvalidArgumentError("DSpARK Prefill target outputs require one untiled row per stream");
            }
            if (wants_logits) {
                if (!target.logits.defined() || target.logits.dim() != 2 || target.logits.size(0) != total_rows) {
                    return absl::InvalidArgumentError("DSpARK Prefill target logits must be LM-selected rows");
                }
                output.logits = target.logits.narrow(0, row, 1).clone();
            }
            if (wants_hidden) {
                if (!target.hidden_states.defined() || target.hidden_states.dim() != 2) {
                    return absl::InvalidArgumentError("DSpARK Prefill target hidden states must be two-dimensional");
                }
                if (target_need_all_logits) {
                    if (!original_lm_output_indexes.defined() || original_lm_output_indexes.dim() != 1
                        || original_lm_output_indexes.numel() != total_rows
                        || (original_lm_output_indexes.scalar_type() != torch::kInt32
                            && original_lm_output_indexes.scalar_type() != torch::kInt64)) {
                        return absl::InvalidArgumentError(
                            "DSpARK full-output hidden states require original LM indices");
                    }
                    auto selected =
                        original_lm_output_indexes.narrow(0, row, 1).to(target.hidden_states.device()).to(torch::kLong);
                    output.hidden_states = target.hidden_states.index_select(0, selected);
                } else {
                    if (target.hidden_states.size(0) != total_rows) {
                        return absl::InvalidArgumentError(
                            "DSpARK compact Prefill hidden states must be LM-selected rows");
                    }
                    output.hidden_states = target.hidden_states.narrow(0, row, 1).clone();
                }
            }
        }
        row += stream->currentBatchSize();
    }
    return outputs;
}

absl::Status MtpBatchStreamProcessor::dispatchPrefill(const StreamGroups& stream_groups,
                                                      const MergedOutput& prefill_output,
                                                      const MergedOutput& propose_output) const {
    return dispatchPrefill(stream_groups, prefill_output, propose_output, torch::Tensor());
}

absl::Status MtpBatchStreamProcessor::dispatchPrefill(const StreamGroups&                     stream_groups,
                                                      const MergedOutput&                     prefill_output,
                                                      const MergedOutput&                     propose_output,
                                                      const torch::Tensor&                    draft_last_hidden_states,
                                                      const std::vector<PrefillTargetOutput>& target_outputs) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t                      total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    auto                              new_tokens_all = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);
    std::vector<StreamSpecUpdateInfo> spec_update_infos;

    preparePrefillSpecUpdateInfo(
        stream_groups, prefill_output, propose_output, draft_last_hidden_states, new_tokens_all, spec_update_infos);

    if (!target_outputs.empty()) {
        if (!is_dspark_ || target_outputs.size() != spec_update_infos.size()) {
            return absl::InvalidArgumentError("DSpARK Prefill target-output stream count mismatch");
        }
        for (size_t i = 0; i < target_outputs.size(); ++i) {
            spec_update_infos[i].target_logits        = target_outputs[i].logits;
            spec_update_infos[i].target_hidden_states = target_outputs[i].hidden_states;
        }
    }

    // we set propose token in extra loop to avoid cuda sync
    if (!is_dspark_) {
        updateProposeTokens(stream_groups, propose_output, spec_update_infos);
    }

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
    if (!is_dspark_) {
        updateProposeTokens(stream_groups, draft_prefill_output, spec_update_infos);
    }

    stream_groups.updateStreams(spec_update_infos);

    RTP_LLM_LOG_DEBUG("dispatch decode done");
    return absl::OkStatus();
}

absl::StatusOr<GptModelInputs> MtpBatchStreamProcessor::gatherDecodeModelInput(const StreamGroups& stream_groups,
                                                                               TensorHolder&       host_holder) const {
    auto model_input = NormalBatchStreamProcessor::gatherModelInput(stream_groups, host_holder);

    RTP_LLM_CHECK(model_input.ok());

    if (propose_step_ == 1 || is_dspark_) {
        return model_input;
    }

    gatherHiddenStates(stream_groups, model_input.value());

    return model_input;
}

absl::StatusOr<SamplerInputs>
MtpBatchStreamProcessor::gatherSpecSamplerInput(const StreamGroups&                         stream_groups,
                                                const GptModelInputs&                       model_inputs,
                                                const GptModelOutputs&                      model_output,
                                                const SpecLogitsVerifyRunner::LaunchResult& spec_logits_result,
                                                const torch::Tensor&                        verify_token_ids) const {
    RTP_LLM_PROFILE_SCOPE("mtp_batch_stream_processor.gather_spec_sampler_input");
    RTP_LLM_CHECK(!stream_groups.empty());
    auto               all_streams      = stream_groups.allStreams();
    ReturnAllProbsMode return_all_probs = stream_groups.needReturnAllProbs();

    for (auto& stream : all_streams) {
        RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1, "stream tile num must be 1 in ScoreExecutor");
    }

    size_t score_len        = verify_step_ + 1;
    size_t total_batch_size = stream_groups.size() * score_len;

    // The verify consumer reads only the sampled last column. Avoid expanding
    // long histories when no processor or penalty can inspect them. Keep all
    // legacy MTP and history-dependent cases on the existing layout.
    const bool compact_token_ids =
        is_dspark_
        && (spec_logits_result.has_active_processor
            || (!spec_logits_result.spec_vocab_mask_gpu.defined() && !spec_logits_result.spec_cap_gpu.defined()
                && spec_logits_result.applied_processors.empty()))
        && std::all_of(all_streams.begin(), all_streams.end(), [](const auto& stream) {
               const auto& config     = *stream->generateConfig();
               const auto  processors = stream->getAllLogitsProcessorPtr();
               return !stream->hasNumBeams() && stream->maxBatchSize() == 1 && config.repetition_penalty == 1.0f
                      && config.presence_penalty == 0.0f && config.frequency_penalty == 0.0f
                      && config.no_repeat_ngram_size.value_or(0) == 0
                      && std::all_of(processors.begin(), processors.end(), [](const auto& processor) {
                             return processor && !processor->requiresTokenHistory();
                         });
           });
    SamplerInputs sampler_inputs =
        allocateSamplerInputs(stream_groups, total_batch_size, total_batch_size, verify_step_, compact_token_ids);
    fillSamplerCommonInputs(sampler_inputs, all_streams, true, verify_step_);
    if (!compact_token_ids || spec_logits_result.has_active_processor
        || std::any_of(all_streams.begin(), all_streams.end(), [](const auto& stream) {
               return !stream->getAllLogitsProcessorPtr().empty();
           })) {
        setLogitsProcessorInputs(sampler_inputs, all_streams, true);
    }
    sampler_inputs.phase = LogitsProcessorPhase::MTP_VERIFY;
    if (spec_logits_result.has_active_processor) {
        sampler_inputs.spec_vocab_mask_gpu      = spec_logits_result.spec_vocab_mask_gpu;
        sampler_inputs.spec_cap_gpu             = spec_logits_result.spec_cap_gpu;
        sampler_inputs.spec_mask_ready_event    = spec_logits_result.ready_event;
        sampler_inputs.spec_mask_consumed_event = spec_logits_result.consumed_event;
        sampler_inputs.spec_applied_processors  = spec_logits_result.applied_processors;
        sampler_inputs.spec_propose_step        = verify_step_;
    }

    if (!compact_token_ids) {
        torch::Tensor dspark_verify_cpu;
        if (is_dspark_) {
            // A row at position i predicts after committed history plus i
            // proposals. Preserve these prefixes for penalties/ngram/custom
            // processors; never let them read an uninitialized gamma tail.
            // The executor passes the original global [anchor, proposals]
            // tensor because CP forward may have replaced combo_tokens.
            const auto& verify = verify_token_ids.defined() ? verify_token_ids : model_inputs.combo_tokens;
            if (!verify.defined() || verify.numel() != static_cast<int64_t>(total_batch_size)) {
                return absl::InternalError("DSpARK history sampling requires global [batch,verify_steps+1] tokens");
            }
            dspark_verify_cpu =
                verify.to(torch::kCPU)
                    .to(torch::kInt32)
                    .reshape({static_cast<int64_t>(all_streams.size()), static_cast<int64_t>(score_len)});
            sampler_inputs.token_ids.zero_();
            sampler_inputs.token_history_lengths_are_counts = true;
        }
        int64_t batch_idx = 0;
        for (auto& stream : all_streams) {
            auto complete_token_ids = stream->completeTokenIds();
            auto seq_len            = static_cast<int64_t>(stream->seqLength());

            copyScoreSamplerTokenIds(
                sampler_inputs.token_ids, complete_token_ids, batch_idx, static_cast<int64_t>(score_len), seq_len);
            if (is_dspark_) {
                const auto proposals = dspark_verify_cpu[batch_idx / score_len].narrow(0, 1, verify_step_);
                for (int64_t position = 0; position < static_cast<int64_t>(score_len); ++position) {
                    sampler_inputs.sequence_lengths.data_ptr<int32_t>()[batch_idx + position] = seq_len + position;
                    if (position > 0) {
                        sampler_inputs.token_ids[batch_idx + position]
                            .narrow(0, seq_len, position)
                            .copy_(proposals.narrow(0, 0, position));
                    }
                }
            }
            batch_idx += static_cast<int64_t>(score_len);
            RTP_LLM_LOG_DEBUG("stream [%s], sampler inputs token ids = [%s]",
                              stream->streamLogTag().c_str(),
                              tensorDebugStringWithData<int32_t>(sampler_inputs.token_ids).c_str());
        }
    }

    if (!model_output.logits.defined()) {
        return absl::InternalError("target verify output logits must be defined for speculative sampling");
    }
    if (model_output.logits.dim() != 2) {
        return absl::InternalError(fmtstr("target verify logits must be 2-D, got dim=%ld", model_output.logits.dim()));
    }
    if (model_output.logits.size(0) != static_cast<int64_t>(total_batch_size)) {
        return absl::InternalError(fmtstr("target verify logits row mismatch: rows=%ld expected=%zu "
                                          "(stream_count=%zu score_len=%zu verify_step=%d)",
                                          model_output.logits.size(0),
                                          total_batch_size,
                                          stream_groups.size(),
                                          score_len,
                                          verify_step_));
    }
    auto vocab_size           = (size_t)model_output.logits.size(1);
    sampler_inputs.vocab_size = vocab_size;
    if (return_all_probs != ReturnAllProbsMode::NONE) {
        sampler_inputs.all_probs = torch::zeros({(int64_t)total_batch_size, (int64_t)vocab_size},
                                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        if (return_all_probs == ReturnAllProbsMode::ORIGINAL) {
            sampler_inputs.return_original_all_probs = true;
        }
    }

    sampler_inputs.logits = model_output.logits.clone();

    // TODO(async): debug formatting is CPU-only. Keep the .cpu() explicit
    // and do not route this through the model-input fast path.
    RTP_LLM_LOG_DEBUG("sampler inputs logits [%s]",
                      tensorDebugStringWithData<float>(sampler_inputs.logits.cpu(), 10).c_str());

    RTP_LLM_LOG_DEBUG("gatherSamplerInput done");
    return std::move(sampler_inputs);
}

void MtpBatchStreamProcessor::updateProposeTokens(const StreamGroups&                stream_groups,
                                                  const MergedOutput&                draft_prefill_output,
                                                  std::vector<StreamSpecUpdateInfo>& spec_update_infos) const {
    // Prefer per-stream GPU slices and avoid D2H/CPU loops.
    // The legacy draft_token int stays -1 unless CPU/PD-disagg still needs it.
    const auto& propose_token_ids = draft_prefill_output.sampler_output.token_ids;
    if (!propose_token_ids.defined()) {
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
        }

        // Fill legacy int only when the tensor is CPU or PD-disagg needs the
        // gRPC-visible vector. PDFUSION consumes draft_token_gpu and keeps this
        // at -1, so ensure_cpu_mirror() stays lazy.
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
            auto combo_tokens_gpu         = torch::cat(propose_slices_gpu, 0).to(torch::kInt32);
            model_input.combo_tokens      = std::move(combo_tokens_gpu);
            model_input.lm_output_indexes = makeCudaInt32Range(model_input.combo_tokens.numel());
            model_input.prefix_lengths    = emptyInt32OnCuda({0});
            if (sequence_lengths_gpu.size() == batch_size) {
                // The propose token is for the next uncommitted position. The
                // published next_seq_len_gpu is the committed length after the
                // previous accept step, which is exactly that decode position.
                model_input.sequence_lengths =
                    committedLenToDraftDecodePosition(torch::cat(sequence_lengths_gpu, 0), host_holder);
            } else if (model_input.sequence_lengths.defined()) {
                model_input.sequence_lengths =
                    normalDecodePositionToDraftDecodePosition(model_input.sequence_lengths, host_holder);
            }
            model_input.input_lengths = toCudaInt32(model_input.input_lengths, host_holder);
            return;
        }
    }

    // Legacy GPU fallback (older sp_output_buffer mirrors). With pre-allocated
    // propose_tokens_gpu (see MtpExecutor::prepareStreams), the primary fast
    // path above should always succeed. Keep this as a safety net for streams
    // missing the new device state APIs but having sp_output_buffer mirrors.
    std::vector<torch::Tensor> propose_slices_gpu;
    if (legacyGpuProposePathEnabled(batch_size)
        && collectLegacyProposeSlices(stream_groups.allStreams(), propose_slices_gpu)) {
        auto combo_tokens_gpu         = torch::cat(propose_slices_gpu, 0).to(torch::kInt32);
        model_input.combo_tokens      = std::move(combo_tokens_gpu);
        model_input.lm_output_indexes = makeCudaInt32Range(model_input.combo_tokens.numel());
        model_input.input_lengths     = toCudaInt32(model_input.input_lengths, host_holder);
        model_input.sequence_lengths =
            normalDecodePositionToDraftDecodePosition(model_input.sequence_lengths, host_holder);
        model_input.prefix_lengths = toCudaInt32(model_input.prefix_lengths, host_holder);
        return;
    }

    // Host fallback removed: pre-allocating propose_tokens_gpu in
    // MtpExecutor::prepareStreams (and makeFakeSPOutputBuffer) guarantees the
    // device fast paths above succeed. Reaching here means a stream lacks both
    // MTP device-state APIs and sp_output_buffer mirrors — a programming error.
    RTP_LLM_CHECK_WITH_INFO(false,
                            "prepareDecodeDraftModelInput: no device-side propose tokens available "
                            "(batch_size=%zu). All sp_output_buffer mirrors must be pre-allocated on CUDA.",
                            batch_size);
}

bool MtpBatchStreamProcessor::gatherMtpDecodeModelInputFromDeviceState(const StreamGroups& stream_groups,
                                                                       GptModelInputs&     model_input,
                                                                       TensorHolder&       host_holder) const {
    const size_t batch_size = stream_groups.size();
    if (batch_size == 0) {
        return false;
    }
    const auto all_streams = stream_groups.allStreams();
    for (const auto& stream : all_streams) {
        if (const char* reason = missingMtpStateReason(stream)) {
            logMtpStateFallback(stream, reason);
            return false;
        }
    }
    g_mtp_device_state_success_count.fetch_add(1, std::memory_order_relaxed);

    std::vector<torch::Tensor> target_last_slices_gpu;
    std::vector<torch::Tensor> propose_slices_gpu;
    std::vector<torch::Tensor> next_seq_len_slices_gpu;
    target_last_slices_gpu.reserve(batch_size);
    propose_slices_gpu.reserve(batch_size);
    next_seq_len_slices_gpu.reserve(batch_size);

    for (const auto& stream : all_streams) {
        const auto& accept_tokens  = stream->getAcceptTokensGpu();   // [1, propose+1]
        const auto& accept_len     = stream->getAcceptLenGpu();      // [1]
        const auto& propose_tokens = stream->getProposeTokensGpu();  // [1, token_stride]
        const auto& next_seq_len   = stream->getNextSeqLenGpu();     // [1]

        auto idx_t       = (accept_len - 1).to(torch::kLong);
        auto target_last = accept_tokens.squeeze(0).index_select(/*dim=*/0, idx_t);

        target_last_slices_gpu.push_back(target_last);
        propose_slices_gpu.push_back(lastColumnAsFlat(propose_tokens));
        next_seq_len_slices_gpu.push_back(next_seq_len);
    }

    auto target_last_gpu         = torch::cat(target_last_slices_gpu, 0).to(torch::kInt32);
    auto propose_gpu             = torch::cat(propose_slices_gpu, 0).to(torch::kInt32);
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
    RTP_LLM_CHECK_WITH_INFO(
        propose_step_ == 1, "prepareOneStepSpecDecodeModelInput requires propose_step=1, got %zu", propose_step_);

    if (gatherMtpDecodeModelInputFromDeviceState(stream_groups, model_input, host_holder)) {
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

    auto target_last_gpu = torch::cat(target_last_slices, 0).to(torch::kInt32);
    auto propose_gpu     = torch::cat(propose_slices, 0).to(torch::kInt32);
    auto verify_pairs    = interleaveTokenPairs(target_last_gpu, propose_gpu);
    RTP_LLM_CHECK_WITH_INFO(verify_pairs.numel() == static_cast<int64_t>(batch_size * (propose_step_ + 1)),
                            "one-step target verify token shape mismatch: tokens=%ld, batch=%zu, propose_step=%zu",
                            verify_pairs.numel(),
                            batch_size,
                            propose_step_);

    // Normal decode gatherer stores sequence_lengths as the current decode
    // position (seqLength - 1), which is also the first verify token position.
    model_input.prefix_lengths = toCudaInt32(model_input.sequence_lengths, host_holder).to(torch::kInt32);
    setVerifyPairInputs(model_input, std::move(verify_pairs), batch_size, propose_step_ + 1, host_holder);
}

void MtpBatchStreamProcessor::updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                                          const GptModelOutputs& model_output,
                                                          const torch::Tensor&   draft_token_ids,
                                                          TensorHolder&          host_holder) {
    int batch_size = model_input.combo_tokens.size(0);
    model_input.setLastHiddenStates(model_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);

    // here combo_tokens is a device buffer
    model_input.combo_tokens = draft_token_ids.reshape({batch_size});

    // Device-input is the only supported path: sequence_lengths must already be a
    // CUDA int32 tensor (published by gatherDecodeModelInput / prepareStreams).
    // Legacy CPU fallback removed — its .cpu()+clone+pin_memory triggered a sync.
    RTP_LLM_CHECK_WITH_INFO(
        model_input.sequence_lengths.defined() && model_input.sequence_lengths.is_cuda(),
        "updateDecodeDraftModelInput requires CUDA sequence_lengths "
        "(useMtpDeviceInput=%d, defined=%d, is_cuda=%d)",
        static_cast<int>(useMtpDeviceInput()),
        static_cast<int>(model_input.sequence_lengths.defined()),
        static_cast<int>(model_input.sequence_lengths.defined() && model_input.sequence_lengths.is_cuda()));
    model_input.sequence_lengths = (model_input.sequence_lengths + 1).to(torch::kInt32);
}

void MtpBatchStreamProcessor::updatePrefillPostDraftModelInput(GptModelInputs&        model_input,
                                                               const GptModelOutputs& model_output,
                                                               const SamplerOutput&   sampler_output,
                                                               TensorHolder&          host_holder) {
    const auto& new_all_token_ids = sampler_output.token_ids;

    // set model_input.combo_tokens
    const int64_t batch_size   = new_all_token_ids.size(0);
    const int64_t token_stride = new_all_token_ids.size(1);

    // Device path: do the shift+append entirely on GPU via invokeMtpPrefillShiftAppend.
    // The legacy CPU path (3x .cpu() + pin_memory + for-loop) is gone — it forced
    // pageable D2H syncs every prefill+draft cycle.
    const auto    cuda_i32        = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor input_lengths_d = toCudaInt32(model_input.input_lengths, host_holder);
    torch::Tensor combo_tokens_d  = toCudaInt32(model_input.combo_tokens, host_holder);
    torch::Tensor new_all_tokens_d =
        new_all_token_ids.is_cuda() ? new_all_token_ids : toCudaInt32(new_all_token_ids, host_holder);
    if (new_all_tokens_d.scalar_type() != torch::kInt32) {
        new_all_tokens_d = new_all_tokens_d.to(torch::kInt32);
    }
    if (!new_all_tokens_d.is_contiguous()) {
        new_all_tokens_d = new_all_tokens_d.contiguous();
    }

    // batch_offsets[b] = cumulative input_lengths through batch b (inclusive end offset).
    auto batch_offsets_d  = input_lengths_d.cumsum(0).to(torch::kInt32);
    auto combo_tokens_out = torch::empty({combo_tokens_d.numel()}, cuda_i32);

#if USING_CUDA
    invokeMtpPrefillShiftAppend(combo_tokens_d,
                                input_lengths_d,
                                batch_offsets_d,
                                new_all_tokens_d,
                                combo_tokens_out,
                                static_cast<int32_t>(token_stride),
                                cuda_graph::graphGetCurrentStream().stream());
#else
    RTP_LLM_CHECK_WITH_INFO(false, "updatePrefillPostDraftModelInput requires CUDA");
#endif

    // The target hidden rows intentionally keep their original alignment for MTP.
    // Only multimodal requests need their token-aligned side inputs shifted with
    // combo_tokens; text-only requests stay on the existing zero-copy path.
    shiftMtpMultimodalMetadata(model_input, input_lengths_d);

    model_input.input_lengths = input_lengths_d;
    model_input.combo_tokens  = combo_tokens_out;
    (void)batch_size;  // batch_size verified inside kernel via input_lengths.numel()
}

torch::Tensor MtpBatchStreamProcessor::dsparkComboTokens(int64_t batch_size, const torch::Tensor& anchors) {
    const int64_t query_width = dsparkQueryWidth();
    if (!dspark_combo_cache_.defined() || dspark_combo_cache_.size(0) < batch_size
        || dspark_combo_cache_.size(1) != query_width) {
        dspark_combo_cache_ = fullInt32OnCuda({batch_size, query_width}, dspark_mask_token_id_);
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
        if (dspark_sample_from_anchor_) {
            dspark_lm_indexes_cache_ = torch::arange(token_count, cudaInt32Options());
        } else {
            dspark_lm_indexes_cache_ = torch::arange(batch_size * dsparkQueryWidth(), cudaInt32Options())
                                           .view({batch_size, dsparkQueryWidth()})
                                           .narrow(1, 1, propose_step_)
                                           .contiguous()
                                           .view({-1});
        }
    }
    return dspark_lm_indexes_cache_.narrow(0, 0, token_count);
}

void MtpBatchStreamProcessor::validatePrefillDSparkCommitInput(const GptModelInputs& model_input) const {
    RTP_LLM_CHECK_WITH_INFO(is_dspark_, "DSpARK commit validation requires SP_TYPE_DSPARK");
    RTP_LLM_CHECK_WITH_INFO(propose_step_ > 0, "DSpARK draft width must be positive");
    RTP_LLM_CHECK_WITH_INFO(model_input.last_hidden_states.defined(),
                            "DSpARK prefill commit requires target auxiliary features");
}

void MtpBatchStreamProcessor::buildDSparkProposeInput(GptModelInputs&      model_input,
                                                      const torch::Tensor& anchors,
                                                      const torch::Tensor& committed_ends,
                                                      TensorHolder&        host_holder) {
    RTP_LLM_CHECK_WITH_INFO(is_dspark_, "DSpARK proposal input requires SP_TYPE_DSPARK");
    RTP_LLM_CHECK_WITH_INFO(propose_step_ > 0, "DSpARK draft width must be positive");
    RTP_LLM_CHECK_WITH_INFO(
        dspark_mask_token_id_ >= 0, "DSpARK requires a non-negative noise token id, got %d", dspark_mask_token_id_);
    RTP_LLM_CHECK_WITH_INFO(anchors.defined() && anchors.dim() == 1, "DSpARK anchors must be a one-dimensional tensor");
    RTP_LLM_CHECK_WITH_INFO(committed_ends.defined() && committed_ends.numel() == anchors.numel(),
                            "DSpARK committed ends must contain one value per anchor");

    const int64_t batch_size = anchors.numel();
    model_input.combo_tokens = dsparkComboTokens(batch_size, toCudaInt32(anchors, host_holder));
    model_input.clearLastHiddenStates();
    model_input.prefix_lengths    = toCudaInt32(committed_ends, host_holder).contiguous();
    model_input.input_lengths     = dsparkDraftInputLengths(batch_size);
    model_input.sequence_lengths  = emptyInt32OnCuda({0});
    model_input.lm_output_indexes = dsparkDraftLmIndexes(batch_size);
    model_input.is_target_verify  = true;
}

MtpBatchStreamProcessor::DSparkRoundState MtpBatchStreamProcessor::buildDSparkRoundState(
    const StreamGroups& stream_groups, const GptModelInputs& model_input, TensorHolder& host_holder) const {
    const int64_t batch_size = static_cast<int64_t>(stream_groups.size());
    if (batch_size == 0) {
        return {};
    }

    std::vector<torch::Tensor> anchors;
    std::vector<torch::Tensor> committed_ends;
    anchors.reserve(batch_size);
    committed_ends.reserve(batch_size);
    auto    host_seq_lens = toCudaInt32(model_input.sequence_lengths, host_holder);
    int64_t row           = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        const auto state = stream->getMtpAsyncDeviceState();
        if (state.dspark_anchor_gpu.defined()) {
            RTP_LLM_CHECK_WITH_INFO(state.dspark_anchor_gpu.is_cuda()
                                        && state.dspark_anchor_gpu.scalar_type() == torch::kInt32
                                        && state.dspark_anchor_gpu.numel() == 1,
                                    "DSpARK published anchor must be one CUDA int32 value");
            anchors.push_back(state.dspark_anchor_gpu.reshape({1}));
        } else if (state.accept_tokens_gpu.defined() && state.accept_tokens_gpu.is_cuda()
                   && state.accept_len_gpu.defined() && state.accept_len_gpu.is_cuda()) {
            auto last_index = (state.accept_len_gpu - 1).to(torch::kLong);
            anchors.push_back(state.accept_tokens_gpu.reshape({-1}).index_select(0, last_index).to(torch::kInt32));
        } else if (stream->isFakeStream()) {
            anchors.push_back(torch::zeros({1}, cudaInt32Options()));
        } else if (stream->isPerfTest() && stream->outputTokenLen() > 0) {
            // Perf output history is zero-filled; the SP buffer preserves the
            // sampled prefill token. Direct-decode streams without output keep
            // their existing prompt-token fallback below.
            auto anchor = pickOneStepTargetLastToken(stream);
            RTP_LLM_CHECK_WITH_INFO(anchor.defined() && anchor.numel() == 1,
                                    "DSpARK initialized perf stream requires one real target token");
            anchors.push_back(toCudaInt32(anchor, host_holder));
        } else {
            anchors.push_back(stream->completeTokenIds()
                                  .index({0, static_cast<int64_t>(stream->seqLength()) - 1})
                                  .reshape({1})
                                  .to(cudaInt32Options(), /*non_blocking=*/true));
        }

        if (state.dspark_committed_end_gpu.defined()) {
            RTP_LLM_CHECK_WITH_INFO(state.dspark_committed_end_gpu.is_cuda()
                                        && state.dspark_committed_end_gpu.scalar_type() == torch::kInt32
                                        && state.dspark_committed_end_gpu.numel() == 1,
                                    "DSpARK published committed end must be one CUDA int32 value");
            committed_ends.push_back(state.dspark_committed_end_gpu.reshape({1}));
        } else if (state.next_seq_len_gpu.defined() && state.next_seq_len_gpu.is_cuda()) {
            committed_ends.push_back((state.next_seq_len_gpu.reshape({1}) - 1).to(torch::kInt32));
        } else {
            committed_ends.push_back(host_seq_lens.narrow(0, row, 1));
        }
        ++row;
    }
    return {torch::cat(anchors, 0), torch::cat(committed_ends, 0), collectDSparkPositionBases(stream_groups)};
}

torch::Tensor MtpBatchStreamProcessor::collectDSparkPositionBases(const StreamGroups& stream_groups) const {
    const int64_t factor          = static_cast<int64_t>(model_input_gatherer_config_.position_id_len_factor);
    const bool    needs_positions = model_input_gatherer_config_.has_positional_encoding
                                 || model_input_gatherer_config_.mm_position_ids_style != PositionIdsStyle::DEFAULT;
    if (!needs_positions || factor <= 0 || stream_groups.size() == 0) {
        return {};
    }
    auto    host = torch::empty({static_cast<int64_t>(stream_groups.size()), factor},
                             torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
    int64_t row  = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        stream->generateNextPositionId(host[row++].data_ptr<int32_t>());
    }
    return host.to(cudaInt32Options(), /*non_blocking=*/false);
}

torch::Tensor MtpBatchStreamProcessor::expandDSparkPositionIds(const torch::Tensor& position_bases,
                                                               int64_t              width) const {
    if (!position_bases.defined() || position_bases.numel() == 0) {
        return {};
    }
    RTP_LLM_CHECK_WITH_INFO(width > 0 && position_bases.dim() == 2,
                            "DSpARK position bases require positive width and [batch,factor] shape");
    auto steps = torch::arange(width, cudaInt32Options()).reshape({1, width, 1});
    return (position_bases.to(torch::kInt32).unsqueeze(1) + steps).reshape({-1}).contiguous();
}

void MtpBatchStreamProcessor::prepareDSparkProposeModelInput(const DSparkRoundState& round_state,
                                                             GptModelInputs&         model_input,
                                                             TensorHolder&           host_holder) {
    buildDSparkProposeInput(model_input, round_state.anchors, round_state.committed_ends, host_holder);
    model_input.combo_position_ids = expandDSparkPositionIds(round_state.position_bases, dsparkQueryWidth());
}

void MtpBatchStreamProcessor::prepareDSparkTargetVerifyModelInput(const DSparkRoundState& round_state,
                                                                  GptModelInputs&         model_input,
                                                                  const torch::Tensor&    proposals,
                                                                  TensorHolder&           host_holder) {
    prepareDSparkTargetVerifyModelInput(
        model_input, round_state.anchors, round_state.committed_ends, proposals, host_holder);
    model_input.combo_position_ids = expandDSparkPositionIds(round_state.position_bases, verify_step_ + 1);
}

void MtpBatchStreamProcessor::prepareCompactDSparkTargetVerifyModelInput(const DSparkRoundState& round_state,
                                                                         GptModelInputs&         model_input,
                                                                         const torch::Tensor&    proposals,
                                                                         const torch::Tensor&    verify_lengths,
                                                                         const torch::Tensor&    compact_to_dense,
                                                                         TensorHolder&           host_holder) {
    RTP_LLM_CHECK_WITH_INFO(is_dspark_, "compact DSpARK target verify requires SP_TYPE_DSPARK");
    const int64_t batch_size = round_state.anchors.numel();
    RTP_LLM_CHECK_WITH_INFO(round_state.anchors.defined() && round_state.anchors.dim() == 1,
                            "DSpARK anchors must be a one-dimensional tensor");
    RTP_LLM_CHECK_WITH_INFO(proposals.defined() && proposals.dim() == 2 && proposals.size(0) == batch_size
                                && proposals.size(1) == verify_step_,
                            "compact DSpARK proposals must be [batch, verify_steps]");
    RTP_LLM_CHECK_WITH_INFO(verify_lengths.defined() && verify_lengths.is_cuda()
                                && verify_lengths.scalar_type() == torch::kInt32
                                && verify_lengths.numel() == batch_size,
                            "compact DSpARK verify lengths must be CUDA int32 [batch]");
    RTP_LLM_CHECK_WITH_INFO(compact_to_dense.defined() && compact_to_dense.is_cuda()
                                && compact_to_dense.scalar_type() == torch::kInt32,
                            "compact DSpARK row mapping must be CUDA int32");

    auto dense_tokens        = torch::cat({toCudaInt32(round_state.anchors, host_holder).reshape({batch_size, 1}),
                                           toCudaInt32(proposals, host_holder).reshape({batch_size, verify_step_})},
                                   1);
    auto dense_positions     = expandDSparkPositionIds(round_state.position_bases, verify_step_ + 1);
    auto gather_rows         = compact_to_dense.to(torch::kLong);
    model_input.combo_tokens = dense_tokens.reshape({-1}).index_select(0, gather_rows).contiguous();
    model_input.combo_position_ids =
        dense_positions.defined() ? dense_positions.index_select(0, gather_rows).contiguous() : torch::Tensor{};
    model_input.sequence_lengths = emptyInt32OnCuda({0});
    model_input.clearLastHiddenStates();
    model_input.prefix_lengths          = toCudaInt32(round_state.committed_ends, host_holder).contiguous();
    model_input.input_lengths           = verify_lengths.contiguous();
    model_input.lm_output_indexes       = makeCudaInt32Range(compact_to_dense.numel());
    model_input.is_target_verify        = true;
    model_input.is_ragged_target_verify = true;
}

void MtpBatchStreamProcessor::prepareDSparkTargetVerifyModelInput(GptModelInputs&      model_input,
                                                                  const torch::Tensor& anchors,
                                                                  const torch::Tensor& committed_ends,
                                                                  const torch::Tensor& proposals,
                                                                  TensorHolder&        host_holder) {
    RTP_LLM_CHECK_WITH_INFO(is_dspark_, "DSpARK target verify input requires SP_TYPE_DSPARK");
    const int64_t batch_size = anchors.numel();
    RTP_LLM_CHECK_WITH_INFO(anchors.defined() && anchors.dim() == 1, "DSpARK anchors must be a one-dimensional tensor");
    RTP_LLM_CHECK_WITH_INFO(proposals.defined() && proposals.dim() == 2 && proposals.size(0) == batch_size
                                && proposals.size(1) == verify_step_,
                            "DSpARK verify proposals must be [batch, verify_steps]");
    RTP_LLM_CHECK_WITH_INFO(committed_ends.defined() && committed_ends.numel() == batch_size,
                            "DSpARK committed ends must contain one value per request");

    auto verify = torch::cat({toCudaInt32(anchors, host_holder).reshape({batch_size, 1}),
                              toCudaInt32(proposals, host_holder).reshape({batch_size, verify_step_})},
                             1)
                      .reshape({-1});
    model_input.prefix_lengths = toCudaInt32(committed_ends, host_holder).contiguous();
    setVerifyPairInputs(model_input, std::move(verify), batch_size, verify_step_ + 1, host_holder);
    model_input.is_target_verify        = true;
    model_input.is_ragged_target_verify = false;
}

void MtpBatchStreamProcessor::updateDecodePostDSparkCommitInput(GptModelInputs&      model_input,
                                                                const torch::Tensor& target_features,
                                                                size_t               batch_size) {
    RTP_LLM_CHECK_WITH_INFO(is_dspark_, "DSpARK decode commit requires SP_TYPE_DSPARK");
    RTP_LLM_CHECK_WITH_INFO(target_features.defined() && target_features.dim() == 2,
                            "DSpARK decode commit requires two-dimensional target auxiliary features");
    RTP_LLM_CHECK_WITH_INFO(model_input.input_lengths.defined()
                                && model_input.input_lengths.numel() == static_cast<int64_t>(batch_size),
                            "DSpARK decode commit requires one input length per request");
    RTP_LLM_CHECK_WITH_INFO(model_input.combo_tokens.defined()
                                && target_features.size(0) == model_input.combo_tokens.numel(),
                            "DSpARK decode commit feature rows must match packed target tokens");
    model_input.is_target_verify = true;
    model_input.setLastHiddenStates(target_features, MtpHiddenStatesLayout::GLOBAL);
}

void MtpBatchStreamProcessor::updateDecodePostDraftModelInput(
    GptModelInputs&                              model_input,
    const GptModelOutputs&                       model_output,
    const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
    const size_t                                 batch_size,
    torch::Tensor&                               hidden_states_d_t,
    TensorHolder&                                host_holder,
    bool                                         use_model_output_hidden_states) {
    // Keep dense accept_tokens for CUDA graph reuse; lm_output_indexes selects
    // only the last accepted position. All outputs stay on CUDA so the next
    // stream-async step can prepare without waiting for worker D2H.
    const int total_tokens = (propose_step_ + 1) * batch_size;
    RTP_LLM_CHECK_WITH_INFO(speculative_sampler_output.accept_len.defined(),
                            "decode post-draft update requires accept_len");
    RTP_LLM_CHECK_WITH_INFO(speculative_sampler_output.accept_tokens.defined(),
                            "decode post-draft update requires accept_tokens");
    RTP_LLM_CHECK_WITH_INFO(speculative_sampler_output.accept_len.numel() == static_cast<int64_t>(batch_size),
                            "accept_len shape mismatch: numel=%ld, batch=%zu",
                            speculative_sampler_output.accept_len.numel(),
                            batch_size);
    RTP_LLM_CHECK_WITH_INFO(speculative_sampler_output.accept_tokens.numel() == total_tokens,
                            "accept_tokens shape mismatch: numel=%ld, expected=%d, batch=%zu, propose_step=%zu",
                            speculative_sampler_output.accept_tokens.numel(),
                            total_tokens,
                            batch_size,
                            propose_step_);
    if (use_model_output_hidden_states) {
        RTP_LLM_CHECK_WITH_INFO(model_output.all_hidden_states.defined(),
                                "target verify output must carry all_hidden_states for draft prefill");
        RTP_LLM_CHECK_WITH_INFO(model_output.all_hidden_states.dim() == 2
                                    && model_output.all_hidden_states.size(0) >= total_tokens,
                                "target verify hidden shape mismatch: dim=%ld, rows=%ld, required_rows=%d",
                                model_output.all_hidden_states.dim(),
                                model_output.all_hidden_states.defined() && model_output.all_hidden_states.dim() > 0 ?
                                    model_output.all_hidden_states.size(0) :
                                    0,
                                total_tokens);
    }
    model_input.combo_tokens =
        toCudaInt32(speculative_sampler_output.accept_tokens.reshape({(int64_t)total_tokens}), host_holder);
    model_input.input_lengths =
        fullInt32OnCuda({static_cast<int64_t>(batch_size)}, static_cast<int64_t>(propose_step_ + 1));
    auto accept_len_d = toCudaInt32(speculative_sampler_output.accept_len, host_holder);
    model_input.lm_output_indexes =
        torch::arange(
            0, total_tokens, propose_step_ + 1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))
        + (accept_len_d - 1);
    if (use_model_output_hidden_states) {
        model_input.setLastHiddenStates(model_output.all_hidden_states, MtpHiddenStatesLayout::GLOBAL);
        hidden_states_d_t = model_input.last_hidden_states;
    } else {
        model_input.clearLastHiddenStates();
        hidden_states_d_t = torch::Tensor();
    }
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
        const auto& dev_probs = stream->getDraftAllProbsGpu();
        RTP_LLM_CHECK_WITH_INFO(dev_probs.defined() || (sp_output_buffer && sp_output_buffer->all_probs.defined()),
                                "one-step MTP draft all_probs missing for stream %ld",
                                stream->streamId());
        draft_token_probs_list.push_back(dev_probs.defined() ? dev_probs : sp_output_buffer->all_probs);
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
        prev_draft_token_probs_list.push_back(dev_probs.defined() ? dev_probs : sp_output_buffer->all_probs);
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
                      tensorDebugStringWithData<int64_t>(propose_new_all_token_ids).c_str());

    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == (size_t)new_all_token_ids.size(0));
    // Only the sampled last column is consumed below. Stage that narrow slice
    // through pinned memory and explicitly wait for the sampler/current stream;
    // Tensor::cpu() alone does not express the custom-stream dependency.
    const int64_t last_col              = new_all_token_ids.size(1) - 1;
    const auto    token_ids_for_copy    = new_all_token_ids.narrow(1, last_col, 1).contiguous();
    bool          need_d2h_sync         = false;
    const auto    new_all_token_ids_cpu = copyToPinnedCpuAsync(token_ids_for_copy, need_d2h_sync);
    const auto    success_cpu           = copyToPinnedCpuAsync(sampler_output.success, need_d2h_sync);
    syncPinnedCpuCopies(need_d2h_sync);

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
            new_tokens.data_ptr<int32_t>()[i] = new_all_token_ids_cpu.data_ptr<int32_t>()[batch_idx_out + i];
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
                last_hidden_states = cloneHiddenSlice(draft_last_hidden_states, batch_idx_out, 1);
            } else {
                // CP prefill may use PyWrappedModel's last-hidden-only exit for
                // the draft model, yielding compact [batch, hidden] rows. The
                // non-CP/full-hidden path still yields [tokens, hidden].
                last_hidden_states = clonePrefillLastHiddenSlice(draft_model_output.all_hidden_states,
                                                                 batch_idx_out,
                                                                 token_offset,
                                                                 token_size,
                                                                 total_batch_size_out);
            }
        }

        spec_update_infos.push_back({new_tokens, 1, -1, std::move(last_hidden_states), std::move(propose_all_probs)});

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
        if (spec_decode_output.success_cpu.defined()
            && !spec_decode_output.success_cpu.data_ptr<bool>()[batch_idx_out]) {
            stream->reportError(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
            // Maintain one update entry per stream. specUpdate sees the error
            // and skips output/length/anchor publication.
            spec_update_infos.push_back({accept_tokens.narrow(0, batch_idx_out, next_batch_size).narrow(1, 0, 1),
                                         0,
                                         -1,
                                         torch::Tensor(),
                                         torch::Tensor()});
            token_offset += verify_step_ + 1;
            batch_idx_in += cur_batch_size;
            batch_idx_out += next_batch_size;
            continue;
        }
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
        torch::Tensor target_token_gpu;
        if (spec_decode_output.accept_tokens.defined() && spec_decode_output.accept_tokens.is_cuda()) {
            target_token_gpu = spec_decode_output.accept_tokens.narrow(0, batch_idx_out, next_batch_size)
                                   .narrow(1, cur_accept_len - 1, 1)
                                   .reshape({static_cast<int64_t>(next_batch_size)})
                                   .to(torch::kInt32);
        }
        spec_update_infos.push_back({accept_tokens_tensor,
                                     cur_accept_len,
                                     -1,
                                     std::move(last_hidden_states),
                                     std::move(propose_all_probs),
                                     torch::Tensor(),
                                     std::move(target_token_gpu)});

        token_offset += verify_step_ + 1;
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
    auto pick_hidden_states = [](const GenerateStreamPtr& stream) -> const torch::Tensor& {
        const auto& dev = stream->getLastHiddenStatesGpu();
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
        model_input.clearLastHiddenStates();
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

    model_input.setLastHiddenStates(all_hidden_states, MtpHiddenStatesLayout::GLOBAL);
}

}  // namespace rtp_llm
