#include "rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/StringUtil.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <numeric>
#include <cstring>

namespace rtp_llm {

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

    // Lazy CPU mirror: only built when at least one stream needs the int draft_token
    // (i.e. when we cannot rely on draft_token_gpu downstream).
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

        // Legacy int field. Filled only when GPU tensor is unavailable (PD
        // disaggregate decode-side already on CPU). PDFUSION path can leave it
        // as -1 because consumers prefer draft_token_gpu when defined.
        if (!on_gpu) {
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

// Minimum batch_size before the GPU-resident propose-tokens path
// kicks in. Below this threshold the GPU path's extra NCCL broadcast (one for
// the GPU-packed buffer on top of the CPU-packed buffer) plus per-op ATen
// launch overhead outweighs the saved CPU loop. Empirically batch=1 sees a
// +126us regression on Qwen3.5-MoE TP=2 H20 (timeline issue 001). Tuning
// threshold higher trades MTP latency for safety; 4 keeps the small-batch
// (interactive) path on the legacy CPU loop while opening the door for big
// batch wins. Negative threshold disables the GPU path entirely; zero forces it.
static constexpr int64_t kPhase31MinBatchForGpu = -1;

void MtpBatchStreamProcessor::prepareDecodeDraftModelInput(const StreamGroups& stream_groups,
                                                           GptModelInputs&     model_input) {
    // Prefer the GPU propose-tokens path when batch is large enough
    // to amortise the extra NCCL broadcast and ATen launch overhead. PDFUSION
    // small-batch and PD-disaggregate handoff fall back to the legacy CPU loop.
    const size_t batch_size = stream_groups.size();
    if (batch_size == 0) {
        model_input.combo_tokens = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
        model_input.lm_output_indexes =
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32).pinned_memory(true));
        return;
    }

    // Stream-async: when MtpExecutor::dispatchDecodeAsync has
    // attached device-resident propose tokens to the streams ,
    // the next step's prepare can read them directly without depending on the
    // worker's specUpdate having populated sp_output_buffer->propose_tokens_gpu.
    // Activated per-stream by getProposeTokensGpu().defined(); first decode
    // step (no prior dispatchDecodeAsync) falls through to the existing path.
    {
        const auto all_streams           = stream_groups.allStreams();
        bool       stream_async_eligible = !all_streams.empty();
        for (const auto& stream : all_streams) {
            const auto& gpu_t = stream->getProposeTokensGpu();
            if (!gpu_t.defined() || !gpu_t.is_cuda()) {
                stream_async_eligible = false;
                break;
            }
        }
        if (stream_async_eligible) {
            std::vector<torch::Tensor> propose_slices_gpu;
            propose_slices_gpu.reserve(batch_size);
            for (const auto& stream : all_streams) {
                const auto& gpu_t    = stream->getProposeTokensGpu();
                const int   last_col = static_cast<int>(gpu_t.size(-1)) - 1;
                propose_slices_gpu.push_back(gpu_t.select(-1, last_col).reshape({-1}));
            }
            auto combo_tokens_gpu = torch::cat(propose_slices_gpu, 0).to(torch::kInt32);
            auto lm_output_indexes_gpu =
                torch::arange(0,
                              static_cast<int64_t>(combo_tokens_gpu.numel()),
                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            model_input.combo_tokens      = std::move(combo_tokens_gpu);
            model_input.lm_output_indexes = std::move(lm_output_indexes_gpu);
            return;
        }
    }

    bool                       all_gpu = (static_cast<int64_t>(batch_size) >= kPhase31MinBatchForGpu);
    std::vector<torch::Tensor> propose_slices_gpu;
    if (all_gpu) {
        propose_slices_gpu.reserve(batch_size);
        for (const auto& stream : stream_groups.allStreams()) {
            const auto& gpu_t = stream->getSPOutputBuffer()->propose_tokens_gpu;
            if (!gpu_t.defined() || !gpu_t.is_cuda()) {
                all_gpu = false;
                break;
            }
            // Each per-stream slice has shape [next_batch_size, token_stride].
            // For draft-decode we need the latest column (token_stride - 1) per row.
            const int last_col = static_cast<int>(gpu_t.size(-1)) - 1;
            propose_slices_gpu.push_back(gpu_t.select(-1, last_col).reshape({-1}));
        }
    }

    if (all_gpu) {
        auto combo_tokens_gpu         = torch::cat(propose_slices_gpu, 0).to(torch::kInt32);
        auto lm_output_indexes_gpu    = torch::arange(0,
                                                   static_cast<int64_t>(combo_tokens_gpu.numel()),
                                                   torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        model_input.combo_tokens      = std::move(combo_tokens_gpu);
        model_input.lm_output_indexes = std::move(lm_output_indexes_gpu);
        return;
    }

    // Legacy CPU fallback path (unchanged behaviour).
    int  batch_idx         = 0;
    auto combo_tokens      = torch::empty({(int64_t)batch_size}, torch::kInt32).pin_memory();
    auto lm_output_indexes = torch::empty({(int64_t)batch_size}, torch::kInt32).pin_memory();

    for (const auto& stream : stream_groups.allStreams()) {
        int propose_token                            = stream->getSPOutputBuffer()->tokens.data_ptr<int>()[1];
        combo_tokens.data_ptr<int>()[batch_idx]      = propose_token;
        lm_output_indexes.data_ptr<int>()[batch_idx] = batch_idx;
        batch_idx++;
    }

    model_input.combo_tokens      = std::move(combo_tokens);
    model_input.lm_output_indexes = std::move(lm_output_indexes);
}

void MtpBatchStreamProcessor::prepareOneStepSpecDecodeModelInput(const StreamGroups& stream_groups,
                                                                 GptModelInputs&     model_input) {
    // GPU-resident assembly when every stream has a GPU mirror of
    // its 2-token (target_last + propose) slot. Each sp_output_buffer->tokens
    // CPU tensor has shape [1, 2]; the GPU mirror (propose_tokens_gpu) has
    // shape [next_batch_size, token_stride]. For propose_step==1, the GPU
    // tensor was sourced from fast_topk_sampler with token_stride==1 and only
    // contains the new propose token. The target_last token is still produced
    // on CPU by GenerateStream::specUpdate (from sampler new_tokens), so the
    // [target_last, propose] pair must be reconstructed here.
    const size_t batch_size = stream_groups.size();
    if (batch_size == 0) {
        return;
    }

    // Stream-async: when dispatchDecodeAsync has attached
    // device-resident state, build the [target_last, propose] verify input
    // entirely on GPU. target_last comes from accept_tokens_gpu indexed by
    // accept_len_gpu - 1; propose comes from propose_tokens_gpu's last column;
    // prefix_lengths comes from next_seq_len_gpu - 1 (the seq position of the
    // last committed token before this step's verify). Activated per-stream
    // by getAcceptTokensGpu().defined() so the first decode step falls
    // through to the existing path.
    {
        const auto all_streams           = stream_groups.allStreams();
        bool       stream_async_eligible = !all_streams.empty();
        for (const auto& stream : all_streams) {
            if (!stream->getAcceptTokensGpu().defined() || !stream->getAcceptLenGpu().defined()
                || !stream->getProposeTokensGpu().defined() || !stream->getNextSeqLenGpu().defined()) {
                stream_async_eligible = false;
                break;
            }
        }
        if (stream_async_eligible) {
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

                // target_last = accept_tokens[0, accept_len[0] - 1]. Use
                // index_select on dim 0 of the squeezed [propose+1] view so
                // the indexing happens on GPU without a host round-trip.
                auto idx_t       = (accept_len - 1).to(torch::kLong);
                auto target_last = accept_tokens.squeeze(0).index_select(/*dim=*/0, idx_t);

                const int last_col = static_cast<int>(propose_tokens.size(-1)) - 1;
                auto      propose  = propose_tokens.select(-1, last_col).reshape({-1});

                target_last_slices_gpu.push_back(target_last);
                propose_slices_gpu.push_back(propose);
                next_seq_len_slices_gpu.push_back(next_seq_len);
            }

            auto target_last_gpu = torch::cat(target_last_slices_gpu, 0).to(torch::kInt32);
            auto propose_gpu     = torch::cat(propose_slices_gpu, 0).to(torch::kInt32);
            // Interleave [target_last, propose] per stream.
            auto pair_gpu                = torch::stack({target_last_gpu, propose_gpu}, /*dim=*/1).reshape({-1});
            auto next_seq_len_gpu_concat = torch::cat(next_seq_len_slices_gpu, 0);

            model_input.combo_tokens = std::move(pair_gpu);
            // prefix_lengths = next_seq_len - 1 (last committed position).
            // Kept on GPU; downstream forward must handle device-resident
            // prefix_lengths or sync as needed (+ refines this).
            model_input.prefix_lengths     = (next_seq_len_gpu_concat - 1).to(torch::kInt32);
            model_input.sequence_lengths   = torch::empty({0}, torch::kInt32);
            model_input.last_hidden_states = torch::Tensor();
            model_input.input_lengths      = torch::full({(int64_t)batch_size},
                                                    static_cast<int64_t>(propose_step_ + 1),
                                                    torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            model_input.lm_output_indexes =
                torch::arange(0,
                              static_cast<int64_t>(batch_size * (propose_step_ + 1)),
                              torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
            return;
        }
    }

    bool                       all_gpu = (static_cast<int64_t>(batch_size) >= kPhase31MinBatchForGpu);
    std::vector<torch::Tensor> target_last_cpu;
    std::vector<torch::Tensor> propose_slices_gpu;
    if (all_gpu) {
        target_last_cpu.reserve(batch_size);
        propose_slices_gpu.reserve(batch_size);
        for (const auto& stream : stream_groups.allStreams()) {
            auto        sp_output_buffer = stream->getSPOutputBuffer();
            const auto& gpu_t            = sp_output_buffer->propose_tokens_gpu;
            if (!gpu_t.defined() || !gpu_t.is_cuda() || !sp_output_buffer->tokens.defined()) {
                all_gpu = false;
                break;
            }
            // target_last lives in tokens[0] (CPU) and is required for the verify pair.
            const int target_last = sp_output_buffer->tokens.data_ptr<int>()[0];
            target_last_cpu.push_back(torch::tensor({target_last}, torch::kInt32));
            // propose slice -> last column reshape to [next_batch_size]
            const int last_col = static_cast<int>(gpu_t.size(-1)) - 1;
            propose_slices_gpu.push_back(gpu_t.select(-1, last_col).reshape({-1}));
        }
    }

    if (all_gpu) {
        auto target_last_concat = torch::cat(target_last_cpu, 0);
        auto target_last_gpu    = target_last_concat.to(torch::kCUDA, /*non_blocking=*/true);
        auto propose_concat_gpu = torch::cat(propose_slices_gpu, 0).to(torch::kInt32);
        // Interleave [target_last, propose] per stream by stacking and reshaping.
        // shape: [batch_size, 2] -> reshape to [batch_size * 2]
        auto pair_gpu = torch::stack({target_last_gpu, propose_concat_gpu}, /*dim=*/1).reshape({-1});

        model_input.combo_tokens       = std::move(pair_gpu);
        model_input.prefix_lengths     = model_input.sequence_lengths.is_cuda() ?
                                             model_input.sequence_lengths.clone() :
                                             model_input.sequence_lengths.clone().pin_memory();
        model_input.sequence_lengths   = torch::empty({0}, torch::kInt32);
        model_input.last_hidden_states = torch::Tensor();

        // input_lengths and lm_output_indexes can be assembled fully on GPU.
        model_input.input_lengths     = torch::full({(int64_t)batch_size},
                                                static_cast<int64_t>(propose_step_ + 1),
                                                torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        model_input.lm_output_indexes = torch::arange(0,
                                                      static_cast<int64_t>(batch_size * (propose_step_ + 1)),
                                                      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        return;
    }

    // Legacy CPU fallback (unchanged).
    auto target_prefix_lengths = model_input.sequence_lengths.cpu().clone().pin_memory();
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

    model_input.combo_tokens       = std::move(target_combo_tokens);
    model_input.prefix_lengths     = target_prefix_lengths;
    model_input.sequence_lengths   = torch::empty({0}, torch::kInt32).pin_memory();
    model_input.last_hidden_states = torch::Tensor();

    for (int i = 0; i < model_input.input_lengths.size(0); i++) {
        model_input.input_lengths.data_ptr<int>()[i] = propose_step_ + 1;
    }

    auto lm_output_indexes = torch::empty({(int64_t)(batch_size * (propose_step_ + 1))}, torch::kInt32).pin_memory();
    for (int i = 0; i < batch_size * (propose_step_ + 1); i++) {
        lm_output_indexes.data_ptr<int>()[i] = i;
    }
    model_input.lm_output_indexes = std::move(lm_output_indexes);
}

void MtpBatchStreamProcessor::updateDecodeDraftModelInput(GptModelInputs&        model_input,
                                                          const GptModelOutputs& model_output,
                                                          const torch::Tensor&   draft_token_ids) {
    int batch_size                 = model_input.combo_tokens.size(0);
    model_input.last_hidden_states = model_output.all_hidden_states;

    // here combo_tokens is a device buffer
    model_input.combo_tokens = draft_token_ids.reshape({batch_size});

    model_input.sequence_lengths = model_input.sequence_lengths.cpu().clone().pin_memory();
    for (int i = 0; i < batch_size; i++) {
        model_input.sequence_lengths.data_ptr<int>()[i]++;
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
    // token_ids may be a CUDA tensor; move to CPU for data_ptr access.
    const torch::Tensor new_all_token_ids_cpu =
        new_all_token_ids.is_cuda() ? new_all_token_ids.cpu() : new_all_token_ids;

    int* input_lengths = model_input.input_lengths.data_ptr<int>();
    int* combo_tokens  = model_input.combo_tokens.data_ptr<int>();

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
    int total_tokens         = (propose_step_ + 1) * batch_size;
    model_input.combo_tokens = speculative_sampler_output.accept_tokens.reshape({(int64_t)total_tokens});
    model_input.lm_output_indexes =
        torch::arange(
            0, total_tokens, propose_step_ + 1, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA))
        + (speculative_sampler_output.accept_len - 1);
    model_input.last_hidden_states = model_output.all_hidden_states;
    hidden_states_d_t              = model_input.last_hidden_states;
}

void MtpBatchStreamProcessor::updateOneStepDraftSamplerOutput(const StreamGroups& stream_groups,
                                                              SamplerOutput&      draft_sampler_output,
                                                              torch::Tensor&      draft_token_probs_d_t) {
    const size_t batch_size      = stream_groups.size();
    auto         draft_token_ids = torch::empty({(int64_t)batch_size, (int64_t)propose_step_}, torch::kInt32);

    std::vector<torch::Tensor> draft_token_probs_list;
    int                        batch_idx = 0;

    for (const auto& stream : stream_groups.allStreams()) {
        auto sp_output_buffer                                      = stream->getSPOutputBuffer();
        auto propose_tokens                                        = sp_output_buffer->tokens.data_ptr<int>();
        draft_token_ids.data_ptr<int>()[batch_idx * propose_step_] = propose_tokens[1];
        draft_token_probs_list.push_back(sp_output_buffer->all_probs);
        batch_idx++;
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
        prev_draft_token_probs_list.push_back(sp_output_buffer->all_probs);
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

    // token_ids and success may be CUDA tensors; move to CPU once before iterating.
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
    // Stream-async v2: replace `.cpu()` (allocates a non-pinned host tensor
    // and synchronously D2H-copies via a staging buffer — measured ~8.5ms even
    // for a few-byte payload, dominating the bookkeeping worker timeline) with
    // an explicit pinned destination + non_blocking copy. Both source tensors
    // are produced on the current CUDA stream (worker stream when called from
    // MtpExecutor::dispatchDecodeAsync's worker lambda; main stream on the
    // synchronous fallback path), and we sync that stream once at the end —
    // semantically equivalent to the original `.cpu()`'s end-of-call sync but
    // without the staging-buffer overhead.
    auto accept_len_cpu = torch::empty(
        spec_decode_output.accept_len.sizes(),
        torch::TensorOptions().dtype(spec_decode_output.accept_len.dtype()).device(torch::kCPU).pinned_memory(true));
    accept_len_cpu.copy_(spec_decode_output.accept_len, /*non_blocking=*/true);

    auto accept_tokens_cpu = torch::empty(
        spec_decode_output.accept_tokens.sizes(),
        torch::TensorOptions().dtype(spec_decode_output.accept_tokens.dtype()).device(torch::kCPU).pinned_memory(true));
    accept_tokens_cpu.copy_(spec_decode_output.accept_tokens, /*non_blocking=*/true);

    cuda_graph::graphGetCurrentStream().synchronize();

    const auto& accept_len    = accept_len_cpu;
    const auto& accept_tokens = accept_tokens_cpu;

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

    size_t all_hidden_tokens_num = 0;
    for (auto& stream : all_streams) {
        auto& hidden_states = stream->getSPOutputBuffer()->hidden_states;
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
        all_hidden_states = all_streams.front()->getSPOutputBuffer()->hidden_states;
    } else if (all_streams.size() < 8) {
        all_hidden_states = torch::empty({(int64_t)all_hidden_tokens_num, (int64_t)hidden_size},
                                         torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
        size_t index      = 0;
        for (auto& stream : all_streams) {
            auto  sp_output_buffer = stream->getSPOutputBuffer();
            auto& hidden_states    = sp_output_buffer->hidden_states;
            auto  hidden_num       = hidden_states.size(0);
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
            auto   sp_output_buffer = stream->getSPOutputBuffer();
            auto&  hidden_states    = sp_output_buffer->hidden_states;
            size_t hidden_copy_size = hidden_states.nbytes();
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
