#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <atomic>
#include <cstdlib>
#include <numeric>
#include <string>
#include <cstring>

namespace rtp_llm {

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

torch::Tensor toCudaInt32(const torch::Tensor& tensor) {
    if (!tensor.defined()) {
        return tensor;
    }
    if (tensor.is_cuda() && tensor.scalar_type() == torch::kInt32) {
        return tensor;
    }
    if (tensor.numel() == 0) {
        return torch::empty(tensor.sizes(), cudaInt32Options());
    }
    return tensor.to(cudaInt32Options());
}

torch::Tensor lastColumnAsFlat(const torch::Tensor& tensor) {
    const int64_t last_col = tensor.size(-1) - 1;
    return tensor.select(-1, last_col).reshape({-1});
}

torch::Tensor makeCudaInt32Range(int64_t end) {
    return torch::arange(0, end, cudaInt32Options());
}

void setVerifyPairInputs(GptModelInputs& model_input, torch::Tensor combo_tokens, size_t batch_size, size_t score_len) {
    model_input.combo_tokens       = std::move(combo_tokens);
    model_input.sequence_lengths   = emptyInt32OnCuda({0});
    model_input.last_hidden_states = torch::Tensor();
    model_input.input_lengths      = fullInt32OnCuda({static_cast<int64_t>(batch_size)}, score_len);
    model_input.lm_output_indexes  = makeCudaInt32Range(static_cast<int64_t>(batch_size * score_len));
}

torch::Tensor interleaveTokenPairs(const torch::Tensor& first, const torch::Tensor& second) {
    return torch::stack({first, second}, /*dim=*/1).reshape({-1});
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
            logMtpStateFallback(stream, "propose_tokens_gpu_missing");
            return false;
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

absl::Status MtpBatchStreamProcessor::dispatchPrefill(const StreamGroups& stream_groups,
                                                      const MergedOutput& prefill_output,
                                                      const MergedOutput& propose_output) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    const size_t                      total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    auto                              new_tokens_all = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);
    std::vector<StreamSpecUpdateInfo> spec_update_infos;

    preparePrefillSpecUpdateInfo(stream_groups, prefill_output, propose_output, new_tokens_all, spec_update_infos);

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

absl::StatusOr<GptModelInputs>
MtpBatchStreamProcessor::gatherDecodeModelInput(const StreamGroups& stream_groups) const {
    auto model_input = NormalBatchStreamProcessor::gatherModelInput(stream_groups);

    RTP_LLM_CHECK(model_input.ok());

    if (propose_step_ == 1) {
        return model_input;
    }

    gatherHiddenStates(stream_groups, model_input.value());

    return model_input;
}

absl::StatusOr<SamplerInputs> MtpBatchStreamProcessor::gatherSpecSamplerInput(
    const StreamGroups& stream_groups, const GptModelInputs& model_inputs, const GptModelOutputs& model_output) const {
    (void)model_inputs;
    RTP_LLM_CHECK(!stream_groups.empty());
    auto all_streams      = stream_groups.allStreams();
    bool return_all_probs = stream_groups.needReturnAllProbs();

    for (auto& stream : all_streams) {
        RTP_LLM_CHECK_WITH_INFO(stream->maxBatchSize() == 1, "stream tile num must be 1 in ScoreExecutor");
    }

    size_t score_len        = propose_step_ + 1;
    size_t total_batch_size = stream_groups.size() * score_len;

    SamplerInputs sampler_inputs =
        allocateSamplerInputs(stream_groups, total_batch_size, total_batch_size, propose_step_);
    fillSamplerCommonInputs(sampler_inputs, all_streams, true, propose_step_);

    int batch_idx = 0;
    for (auto& stream : all_streams) {
        auto complete_token_ids = stream->completeTokenIds();
        auto seq_len            = stream->seqLength();
        auto current_batch_size = score_len;

        for (int i = 0; i < current_batch_size; ++i) {
            memcpy(sampler_inputs.token_ids.data_ptr<int32_t>() + ((batch_idx) * (sampler_inputs.step + 1)),
                   complete_token_ids.data_ptr<int32_t>(),
                   seq_len * sizeof(int));
            batch_idx += 1;
        }
        RTP_LLM_LOG_DEBUG("stream [%ld], sampler inputs token ids = [%s]",
                          stream->streamId(),
                          tensorDebugStringWithData<int32_t>(sampler_inputs.token_ids).c_str());
    }

    auto vocab_size = (size_t)model_output.logits.size(1);
    if (return_all_probs) {
        sampler_inputs.all_probs = torch::zeros({(int64_t)total_batch_size, (int64_t)vocab_size},
                                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
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
    // avoid the D2H + per-stream CPU loop. Keep the GPU tensor and
    // hand each stream a per-stream GPU slice. The stale draft_token int field
    // is kept (initialised to -1) for compatibility with downstream paths that
    // do not understand draft_token_gpu yet (e.g., PD-disaggregate, debug logs).
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

        // GPU slice for the next decode step's propose tokens.
        // Shape: [next_batch_size, token_stride]. Downstream readers select
        // column [..., token_stride - 1] for next-token, or take the full
        // row when propose_step > 1 (multi-step path).
        if (on_gpu && next_batch_size > 0) {
            spec_update_infos[stream_idx].draft_token_gpu = propose_token_ids.narrow(0, batch_idx_out, next_batch_size);
        }

        // Legacy int field. Required when the GPU tensor isn't usable downstream:
        //   - on_gpu == false: propose tensor is already CPU, just read it.
        //   - stream->queryPdSep(): PD-disagg prefill ships propose_token_ over
        //     gRPC (PrefillRpcServer.cc), and the receiver only sees the int
        //     vector — leaving -1 here would propagate -1 as the draft token
        //     to the decode side and corrupt the first verify step.
        // Pure PDFUSION decode path keeps -1 because consumers prefer
        // draft_token_gpu when defined; ensure_cpu_mirror() stays lazy so we
        // only pay the D2H when at least one stream actually needs it.
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
                                                           GptModelInputs&     model_input) {
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
                model_input.sequence_lengths = (torch::cat(sequence_lengths_gpu, 0) - 1).to(torch::kInt32);
            } else if (model_input.sequence_lengths.defined() && !model_input.sequence_lengths.is_cuda()) {
                model_input.sequence_lengths = toCudaInt32(model_input.sequence_lengths);
            }
            model_input.input_lengths = toCudaInt32(model_input.input_lengths);
            return;
        }
    }

    std::vector<torch::Tensor> propose_slices_gpu;
    if (legacyGpuProposePathEnabled(batch_size)
        && collectLegacyProposeSlices(stream_groups.allStreams(), propose_slices_gpu)) {
        auto combo_tokens_gpu         = torch::cat(propose_slices_gpu, 0).to(torch::kInt32);
        model_input.combo_tokens      = std::move(combo_tokens_gpu);
        model_input.lm_output_indexes = makeCudaInt32Range(model_input.combo_tokens.numel());
        model_input.input_lengths     = toCudaInt32(model_input.input_lengths);
        model_input.sequence_lengths  = toCudaInt32(model_input.sequence_lengths);
        model_input.prefix_lengths    = toCudaInt32(model_input.prefix_lengths);
        return;
    }

    int  batch_idx    = 0;
    auto combo_tokens = torch::empty({(int64_t)batch_size}, torch::kInt32).pin_memory();

    for (const auto& stream : stream_groups.allStreams()) {
        int propose_token                       = stream->getSPOutputBuffer()->tokens.data_ptr<int>()[1];
        combo_tokens.data_ptr<int>()[batch_idx] = propose_token;
        batch_idx++;
    }

    model_input.combo_tokens      = toCudaInt32(combo_tokens);
    model_input.input_lengths     = toCudaInt32(model_input.input_lengths);
    model_input.sequence_lengths  = toCudaInt32(model_input.sequence_lengths);
    model_input.prefix_lengths    = toCudaInt32(model_input.prefix_lengths);
    model_input.lm_output_indexes = makeCudaInt32Range(static_cast<int64_t>(batch_size));
}

bool MtpBatchStreamProcessor::gatherMtpDecodeModelInputFromDeviceState(const StreamGroups& stream_groups,
                                                                       GptModelInputs&     model_input) const {
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
    setVerifyPairInputs(model_input, std::move(pair_gpu), batch_size, propose_step_ + 1);
    return true;
}

void MtpBatchStreamProcessor::prepareOneStepSpecDecodeModelInput(const StreamGroups& stream_groups,
                                                                 GptModelInputs&     model_input) {
    const size_t batch_size = stream_groups.size();
    if (batch_size == 0) {
        return;
    }

    if (gatherMtpDecodeModelInputFromDeviceState(stream_groups, model_input)) {
        return;
    }

    std::vector<torch::Tensor> target_last_cpu;
    std::vector<torch::Tensor> propose_slices_gpu;
    bool                       legacy_gpu_ready = legacyGpuProposePathEnabled(batch_size);
    if (legacy_gpu_ready) {
        target_last_cpu.reserve(batch_size);
        propose_slices_gpu.reserve(batch_size);
        for (const auto& stream : stream_groups.allStreams()) {
            auto        sp_output_buffer = stream->getSPOutputBuffer();
            const auto& gpu_t            = sp_output_buffer->propose_tokens_gpu;
            if (!gpu_t.defined() || !gpu_t.is_cuda() || !sp_output_buffer->tokens.defined()) {
                legacy_gpu_ready = false;
                break;
            }
            const int target_last = sp_output_buffer->tokens.data_ptr<int>()[0];
            target_last_cpu.push_back(torch::tensor({target_last}, torch::kInt32));
            propose_slices_gpu.push_back(lastColumnAsFlat(gpu_t));
        }
    }

    if (legacy_gpu_ready) {
        auto target_last_concat = torch::cat(target_last_cpu, 0);
        auto target_last_gpu    = target_last_concat.to(torch::kCUDA, /*non_blocking=*/true);
        auto propose_concat_gpu = torch::cat(propose_slices_gpu, 0).to(torch::kInt32);

        model_input.prefix_lengths = toCudaInt32(model_input.sequence_lengths).clone();
        setVerifyPairInputs(
            model_input, interleaveTokenPairs(target_last_gpu, propose_concat_gpu), batch_size, propose_step_ + 1);
        return;
    }

    auto target_combo_tokens =
        torch::empty({(int64_t)(stream_groups.size() * (propose_step_ + 1))}, torch::kInt32).pin_memory();
    int batch_idx = 0;
    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer = stream->getSPOutputBuffer();
        int* propose_tokens   = sp_output_buffer->tokens.data_ptr<int>();

        RTP_LLM_LOG_DEBUG("propose_tokens = [%s]", tensorDebugStringWithData<int>(sp_output_buffer->tokens).c_str());

        memcpy(target_combo_tokens.data_ptr<int>() + batch_idx * (propose_step_ + 1), propose_tokens, sizeof(int) * 2);
        batch_idx++;
    }

    model_input.prefix_lengths = toCudaInt32(model_input.sequence_lengths).clone();
    setVerifyPairInputs(model_input, toCudaInt32(target_combo_tokens), batch_size, propose_step_ + 1);
}

void MtpBatchStreamProcessor::updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                                          const GptModelOutputs& model_output,
                                                          const torch::Tensor&   draft_token_ids) {
    int batch_size                 = model_input.combo_tokens.size(0);
    model_input.last_hidden_states = model_output.all_hidden_states;

    // here combo_tokens is a device buffer
    model_input.combo_tokens = draft_token_ids.reshape({batch_size});

    if (useMtpDeviceInput() || model_input.sequence_lengths.is_cuda()) {
        auto seq_lengths_d           = model_input.sequence_lengths.is_cuda() ? model_input.sequence_lengths :
                                                                                model_input.sequence_lengths.to(torch::kCUDA);
        model_input.sequence_lengths = (seq_lengths_d + 1).to(torch::kInt32);
    } else {
        // Legacy CPU fallback. Reachable only when both
        // RTP_LLM_DEVICE_INPUT=0 (the default off path is dead in the
        // stream-async config we ship today) AND the upstream caller did
        // not already publish sequence_lengths on CUDA. The MtpExecutor
        // boot-time log makes the gate decision visible, so a regression
        // landing here will be obvious in the log. When the device-input
        // env gate is retired (after hardening), this branch should be
        // deleted along with `useMtpDeviceInput()` itself.
        auto sequence_lengths_cpu = model_input.sequence_lengths.cpu().clone().pin_memory();
        for (int i = 0; i < batch_size; i++) {
            sequence_lengths_cpu.data_ptr<int>()[i]++;
        }
        model_input.sequence_lengths = toCudaInt32(sequence_lengths_cpu);
    }
}

void MtpBatchStreamProcessor::updatePrefillPostDraftModelInput(GptModelInputs&        model_input,
                                                               const GptModelOutputs& model_output,
                                                               const SamplerOutput&   sampler_output) {
    model_input.last_hidden_states = model_output.all_hidden_states;
    const auto& new_all_token_ids  = sampler_output.token_ids;

    // set model_input.combo_tokens
    const size_t batch_size   = new_all_token_ids.size(0);
    const size_t token_stride = new_all_token_ids.size(1);
    // TODO(async): data_ptr iteration below is CPU-only; keep all .cpu()
    // conversions explicit, then republish model-bound tensors to CUDA.
    const torch::Tensor new_all_token_ids_cpu =
        new_all_token_ids.is_cuda() ? new_all_token_ids.cpu() : new_all_token_ids;
    torch::Tensor input_lengths_cpu =
        model_input.input_lengths.is_cuda() ? model_input.input_lengths.cpu().pin_memory() : model_input.input_lengths;
    torch::Tensor combo_tokens_cpu =
        model_input.combo_tokens.is_cuda() ? model_input.combo_tokens.cpu().pin_memory() : model_input.combo_tokens;

    int* input_lengths = input_lengths_cpu.data_ptr<int>();
    int* combo_tokens  = combo_tokens_cpu.data_ptr<int>();

    int offset = 0;
    for (int i = 0; i < batch_size; i++) {
        // should shift one token for combo_tokens
        int input_length = input_lengths[i];
        memcpy(combo_tokens + offset, combo_tokens + offset + 1, (input_length - 1) * sizeof(int));

        // set new token id
        int new_token_id = new_all_token_ids_cpu.data_ptr<int>()[i * token_stride + token_stride - 1];
        combo_tokens[offset + input_length - 1] = new_token_id;

        offset += input_length;
    }

    model_input.input_lengths = toCudaInt32(input_lengths_cpu);
    model_input.combo_tokens  = toCudaInt32(combo_tokens_cpu);
}

void MtpBatchStreamProcessor::updateDecodePostDraftModelInput(
    GptModelInputs&                              model_input,
    const GptModelOutputs&                       model_output,
    const speculative::SpeculativeSamplerOutput& speculative_sampler_output,
    const size_t                                 batch_size,
    torch::Tensor&                               hidden_states_d_t) {
    // accept_tokens is (batch_size, propose_step_+1) with -1 padding for rejected positions.
    // batchSample already resets -1 entries to token 0 via index_put_, so the padding slots
    // contain token-0 ids. We intentionally keep the dense fixed-width shape and leave
    // input_lengths at propose_step_+1 for CUDA graph reuse; lm_output_indexes below selects
    // only the last accepted position per sequence, so padding positions are not surfaced as
    // next-step last_hidden_states.
    //
    // Device-resident contract for propose_step > 1: every output of
    // this function is on CUDA and stays on CUDA across the next decode
    // step's prepare/forward.
    //   - combo_tokens   : toCudaInt32(...) on accept_tokens (already CUDA from sampler)
    //   - lm_output_indexes : torch::arange + accept_len_d (both CUDA i32)
    //   - last_hidden_states / hidden_states_d_t : alias of model_output.all_hidden_states,
    //     which is the target verify forward output and never round-trips to host.
    // This is load-bearing for the stream-async path: the worker thread's
    // specUpdate uses CPU mirrors of these tensors via async D2H, but the
    // main thread launches the next decode step from these device tensors
    // alone. Do not introduce a `.cpu()` here without first auditing the
    // next-step gather path (currently MtpBatchStreamProcessor::
    // gatherMtpDecodeModelInputFromDeviceState).
    int total_tokens         = (propose_step_ + 1) * batch_size;
    model_input.combo_tokens = toCudaInt32(speculative_sampler_output.accept_tokens.reshape({(int64_t)total_tokens}));
    auto accept_len_d        = toCudaInt32(speculative_sampler_output.accept_len);
    model_input.lm_output_indexes =
        torch::arange(
            0, total_tokens, propose_step_ + 1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))
        + (accept_len_d - 1);
    model_input.last_hidden_states = model_output.all_hidden_states;
    hidden_states_d_t              = model_input.last_hidden_states;
}

void MtpBatchStreamProcessor::updateOneStepDraftSamplerOutput(const StreamGroups& stream_groups,
                                                              SamplerOutput&      draft_sampler_output,
                                                              torch::Tensor&      draft_token_probs_d_t) {
    const size_t batch_size      = stream_groups.size();
    auto         draft_token_ids = emptyInt32OnPreferredDevice({(int64_t)batch_size, (int64_t)propose_step_});

    std::vector<torch::Tensor> draft_token_probs_list;
    std::vector<torch::Tensor> draft_token_id_slices_gpu;
    int                        batch_idx      = 0;
    bool                       all_gpu_tokens = useMtpDeviceInput() && batch_size > 0;

    for (const auto& stream : stream_groups.allStreams()) {
        auto        sp_output_buffer   = stream->getSPOutputBuffer();
        auto        propose_tokens     = sp_output_buffer->tokens.data_ptr<int>();
        const auto& propose_tokens_gpu = sp_output_buffer->propose_tokens_gpu;
        if (all_gpu_tokens && propose_tokens_gpu.defined() && propose_tokens_gpu.is_cuda()) {
            const int last_col = static_cast<int>(propose_tokens_gpu.size(-1)) - 1;
            draft_token_id_slices_gpu.push_back(propose_tokens_gpu.select(-1, last_col).reshape({-1}));
        } else {
            all_gpu_tokens = false;
        }
        if (!draft_token_ids.is_cuda()) {
            draft_token_ids.data_ptr<int>()[batch_idx * propose_step_] = propose_tokens[1];
        }
        // Prefer device-state draft_all_probs published by main thread
        // in dispatchDecodeAsync. Falls back to sp_output_buffer->all_probs
        // (legacy worker-written field) when the device-state was already
        // cleared by the worker — by which time the worker has finished
        // writing sp_output_buffer->all_probs (clear runs after specUpdate).
        const auto& dev_probs = stream->getDraftAllProbsGpu();
        draft_token_probs_list.push_back(dev_probs.defined() ? dev_probs : sp_output_buffer->all_probs);
        batch_idx++;
    }

    if (all_gpu_tokens) {
        draft_token_ids = torch::cat(draft_token_id_slices_gpu, 0)
                              .to(torch::kInt32)
                              .reshape({(int64_t)batch_size, (int64_t)propose_step_});
    } else if (draft_token_ids.is_cuda()) {
        auto draft_token_ids_cpu = torch::empty({(int64_t)batch_size, (int64_t)propose_step_},
                                                torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
        batch_idx                = 0;
        for (const auto& stream : stream_groups.allStreams()) {
            auto sp_output_buffer                                          = stream->getSPOutputBuffer();
            auto propose_tokens                                            = sp_output_buffer->tokens.data_ptr<int>();
            draft_token_ids_cpu.data_ptr<int>()[batch_idx * propose_step_] = propose_tokens[1];
            batch_idx++;
        }
        draft_token_ids = draft_token_ids_cpu.to(torch::kCUDA);
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
        torch::Tensor propose_all_probs =
            draft_sampler_output.all_probs.narrow(0, batch_idx_out, next_batch_size).to(torch::kCUDA).clone();

        torch::Tensor last_hidden_states;
        if (propose_step_ > 1) {
            last_hidden_states = draft_model_output.all_hidden_states.narrow(0, token_offset + token_size - 1, 1);
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
        torch::Tensor propose_all_probs =
            draft_sampler_output.all_probs.narrow(0, batch_idx_out, next_batch_size).to(torch::kCUDA).clone();

        // This `.item<int>()` is the *only* host scalar read in
        // MTP decode bookkeeping that is correct to keep. It runs on the
        // bookkeeping worker thread (called from
        // MtpExecutor::dispatchDecodeAsync's lambda), and `accept_len` is
        // a CPU-pinned tensor whose H2D-mirroring D2H is already drained
        // by spec_decode_output.transfer_done_event->synchronize() at the
        // top of this function. There is no main-thread sync incurred
        // here. Do NOT move this function into the main thread without
        // first replacing the .item() with a device-side index — the
        // intent of the worker fork is precisely to absorb cheap scalar
        // host reads so the main thread can issue the next decode step.
        int cur_accept_len = accept_len[batch_idx_out].item<int>();

        torch::Tensor last_hidden_states;
        if (propose_step_ > 1) {
            auto slice_t       = draft_model_output.all_hidden_states.narrow(0, token_offset + cur_accept_len - 1, 1);
            last_hidden_states = slice_t;
        }

        torch::Tensor accept_tokens_tensor =
            accept_tokens.narrow(0, batch_idx_out, next_batch_size).narrow(1, 0, cur_accept_len).contiguous();
        spec_update_infos.push_back(
            {accept_tokens_tensor, cur_accept_len, -1, std::move(last_hidden_states), std::move(propose_all_probs)});

        token_offset += propose_step_ + 1;
        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
    }
}

void MtpBatchStreamProcessor::gatherHiddenStates(const StreamGroups& stream_groups, GptModelInputs& model_input) const {
    auto            all_streams = stream_groups.allStreams();
    c10::ScalarType dtype       = c10::ScalarType::Undefined;
    size_t          hidden_size = 0;

    // Prefer the per-stream device-state hidden_states (published by
    // MtpExecutor::dispatchDecodeAsync on the main thread, epoch-guarded).
    // This bypasses the worker-thread write to sp_output_buffer->hidden_states
    // which races the next step's main thread when DROP_BROAD_SYNC=1.
    // Device-state is expected to stay alive until the next publish overwrites
    // it; the fallback only protects older streams or first-step streams that
    // have not published device hidden states yet.
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
        model_input.last_hidden_states = torch::Tensor();
        return;
    } else if (all_streams.size() == 1) {
        all_hidden_states = pick_hidden_states(all_streams.front());
    } else if (all_streams.size() < 8) {
        all_hidden_states = torch::empty({(int64_t)all_hidden_tokens_num, (int64_t)hidden_size},
                                         torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
        size_t index      = 0;
        for (auto& stream : all_streams) {
            const auto& hidden_states = pick_hidden_states(stream);
            auto        hidden_num    = hidden_states.size(0);
            all_hidden_states.narrow(0, index, hidden_num).copy_(hidden_states);
            index += hidden_num;
        }
    } else {
        all_hidden_states = torch::empty({(int64_t)all_hidden_tokens_num, (int64_t)hidden_size},
                                         torch::TensorOptions().dtype(dtype).device(torch::kCUDA));

        MultiMergeCopyParams params;
        params.dst_ptr         = all_hidden_states.data_ptr();
        size_t accu_dst_offset = 0;
        for (auto& stream : all_streams) {
            const auto& hidden_states    = pick_hidden_states(stream);
            size_t      hidden_copy_size = hidden_states.nbytes();
            params.src_ptrs.push_back(hidden_states.data_ptr());
            params.copy_size.push_back(hidden_copy_size);
            params.dst_offsets.push_back(accu_dst_offset);
            accu_dst_offset += hidden_copy_size;
        }

        if (accu_dst_offset > 0) {
            execMultiMergeCopy(params);
        }
    }

    model_input.last_hidden_states = all_hidden_states;
}

}  // namespace rtp_llm
