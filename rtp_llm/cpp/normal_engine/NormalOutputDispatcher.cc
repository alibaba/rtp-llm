#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/normal_engine/PromptScoringRows.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <c10/util/Exception.h>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <string>
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "ATen/cuda/CUDAContext.h"
#endif

namespace rtp_llm {
namespace {

bool asyncDebugEnabled() {
    const char* env = std::getenv("RTP_LLM_ASYNC_DEBUG");
    return env != nullptr && std::string(env) == "1";
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
    if (!need_sync) {
        return;
    }
    // Keep D2H waiting explicit here instead of hiding it inside Tensor::cpu().
    // The copy launch returns quickly; only this worker thread blocks on its
    // stream while the main engine thread can continue issuing CUDA work.
    cuda_graph::graphGetCurrentStream().synchronize();
}

// Out-of-memory diagnostics that arrive wrapped as a generic c10::Error or
// std::runtime_error instead of c10::OutOfMemoryError. Only used inside the
// narrow prompt-scoring block, so a false positive cannot shield unrelated code.
bool isOutOfMemoryMessage(const char* what) {
    if (what == nullptr) {
        return false;
    }
    std::string lowered(what);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered.find("out of memory") != std::string::npos;
}

}  // namespace

std::shared_ptr<std::function<void(const GenerateStreamPtr&)>> NormalOutputDispatcher::test_scoring_failure_hook =
    nullptr;

bool NormalOutputDispatcher::failStreamOnScoringOom(const GenerateStreamPtr& stream, const char* what) const {
    if (warm_up_) {
        // Warm-up must surface allocation failures: a swallowed OOM would let startup size
        // the KV cache from an underestimate.
        return false;
    }
    std::string detail = what == nullptr ? std::string() : std::string(what);
    if (detail.size() > 512) {
        detail.resize(512);
        detail += "...";
    }
    RTP_LLM_LOG_ERROR(
        "stream [%ld] prompt-scoring output aborted by out-of-memory: %s", stream->streamId(), detail.c_str());
    // The client gets a stable minimal message; raw allocator diagnostics (device addresses,
    // pool internals) stay in the server log above.
    stream->reportError(ErrorCode::EXECUTION_EXCEPTION, "prompt-scoring request aborted by out-of-memory");
    return true;
}

std::optional<ErrorInfo> collectStreamSamplerError(const SamplerOutput& sampler_output,
                                                   const torch::Tensor& success_cpu,
                                                   int                  batch_idx_in,
                                                   int                  cur_batch_size) {
    std::optional<ErrorInfo> error_info;
    const auto               set_first_error = [&error_info](const ErrorInfo& error) {
        if (!error_info.has_value()) {
            error_info = error;
        }
    };

    // Processor errors and sampling success both use sampler-input coordinates;
    // output coordinates can diverge when beam search changes the batch size.
    for (int i = 0; i < cur_batch_size; ++i) {
        const size_t error_idx = static_cast<size_t>(batch_idx_in + i);
        if (error_idx < sampler_output.processor_errors.size()
            && sampler_output.processor_errors[error_idx].has_value()) {
            set_first_error(sampler_output.processor_errors[error_idx].value());
        }
    }

    if (success_cpu.defined()) {
        const auto* success = success_cpu.data_ptr<bool>();
        for (int i = 0; i < cur_batch_size; ++i) {
            if (!success[batch_idx_in + i]) {
                set_first_error(ErrorInfo(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed"));
            }
        }
    }

    return error_info;
}

bool NormalOutputDispatcher::restoreCurrentTokenIds(const GenerateStreamPtr& stream,
                                                    torch::Tensor&           batch_token_ids,
                                                    torch::Tensor&           current_token_ids,
                                                    size_t                   token_position) const {
    if (output_vocab_ids_.empty()) {
        return true;
    }

    RTP_LLM_CHECK(batch_token_ids.device().is_cpu() && current_token_ids.device().is_cpu());
    RTP_LLM_CHECK(batch_token_ids.scalar_type() == torch::kInt32 && current_token_ids.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(batch_token_ids.dim() == 2 && current_token_ids.dim() == 2);
    RTP_LLM_CHECK(batch_token_ids.is_contiguous() && current_token_ids.is_contiguous());
    RTP_LLM_CHECK(batch_token_ids.size(0) == current_token_ids.size(0));
    RTP_LLM_CHECK(current_token_ids.size(1) == 1);
    RTP_LLM_CHECK(token_position < static_cast<size_t>(batch_token_ids.size(1)));

    const auto token_stride = static_cast<size_t>(batch_token_ids.size(1));
    auto*      batch_data   = batch_token_ids.data_ptr<int32_t>();
    auto*      current_data = current_token_ids.data_ptr<int32_t>();
    for (int64_t row = 0; row < current_token_ids.size(0); ++row) {
        const auto compact_id = current_data[row];
        if (compact_id < 0 || static_cast<size_t>(compact_id) >= output_vocab_ids_.size()) {
            stream->reportError(ErrorCode::OUT_OF_VOCAB_RANGE,
                                "compact output token id " + std::to_string(compact_id)
                                    + " is outside configured output vocabulary size "
                                    + std::to_string(output_vocab_ids_.size()));
            return false;
        }
        const auto canonical_id                         = static_cast<int32_t>(output_vocab_ids_[compact_id]);
        current_data[row]                               = canonical_id;
        batch_data[row * token_stride + token_position] = canonical_id;
    }
    return true;
}

absl::Status NormalOutputDispatcher::dispatch(const StreamGroups& stream_groups,
                                              const MergedOutput& merge_outputs) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto&  sampler_output       = merge_outputs.sampler_output;
    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == (size_t)sampler_output.token_ids.size(0));
    // token_ids and success may be CUDA tensors. Keep the non-beam token copy
    // narrow, then stage D2H through pinned CPU buffers and synchronize once.
    bool any_beam_search = false;
    if (sampler_output.token_ids.defined() && sampler_output.token_ids.size(1) > 1) {
        for (auto& stream : stream_groups.allStreams()) {
            if (stream->currentNumBeams() > 1 || stream->nextNumBeams() > 1) {
                any_beam_search = true;
                break;
            }
        }
    }
    torch::Tensor token_ids_for_copy;
    if (sampler_output.token_ids.defined()) {
        if (any_beam_search) {
            token_ids_for_copy = sampler_output.token_ids;
        } else {
            // Slice the last column on-device so the D2H is only [B, 1] int32.
            const int64_t last_col = sampler_output.token_ids.size(1) - 1;
            token_ids_for_copy     = sampler_output.token_ids.narrow(1, last_col, 1).contiguous();
        }
    }
    bool                need_d2h_sync = false;
    const torch::Tensor token_ids_cpu = copyToPinnedCpuAsync(token_ids_for_copy, need_d2h_sync);
    const torch::Tensor success_cpu   = copyToPinnedCpuAsync(sampler_output.success, need_d2h_sync);
    syncPinnedCpuCopies(need_d2h_sync);
    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", tensorDebugStringWithData<int32_t>(token_ids_cpu).c_str());
    int batch_idx_in  = 0;
    int batch_idx_out = 0;
    int token_offset  = 0;
    // Prefix of this stream's prompt-scoring/loss rows inside a row-selected all_logits
    // (advanced per stream; unused when all_logits keeps the legacy token-aligned layout).
    int  all_logits_row_offset = 0;
    bool return_all_probs      = stream_groups.needReturnAllProbs() != ReturnAllProbsMode::NONE;
    auto new_tokens_all        = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();
        auto token_size      = stream->currentExecuteTokenSize();

        dispatchSingleStream(stream,
                             merge_outputs,
                             batch_idx_in,
                             batch_idx_out,
                             token_offset,
                             all_logits_row_offset,
                             return_all_probs,
                             new_tokens_all,
                             token_ids_cpu,
                             success_cpu);

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += token_size;
    }

    RTP_LLM_LOG_DEBUG("dispatch done");
    return absl::OkStatus();
}

void NormalOutputDispatcher::dispatchSingleStream(GenerateStreamPtr    stream,
                                                  const MergedOutput&  merge_outputs,
                                                  int                  batch_idx_in,
                                                  int                  batch_idx_out,
                                                  int                  token_offset,
                                                  int&                 all_logits_row_offset,
                                                  bool                 return_all_probs,
                                                  const torch::Tensor& new_tokens_all,
                                                  const torch::Tensor& token_ids_cpu,
                                                  const torch::Tensor& success_cpu) const {

    const auto&  model_output      = merge_outputs.model_output;
    const auto&  sampler_output    = merge_outputs.sampler_output;
    const auto&  new_all_token_ids = token_ids_cpu;
    const size_t token_stride      = new_all_token_ids.size(1);

    auto cur_batch_size  = stream->currentBatchSize();
    auto next_batch_size = stream->nextBatchSize();
    auto token_size      = stream->currentExecuteTokenSize();

    // Row origin of this stream's execute tokens inside model_output.all_logits. The legacy layout
    // is token-aligned (all_logits has one row per executed token); the row-selected layout
    // (all_logits_lm_rows >= 0) packs the lm_output_indexes rows first and then each stream's
    // prompt-scoring/loss row range, so this stream's token row r lives at
    // lm_rows + prefix + (r - range_a). The range rule must match NormalModelInputGatherer.
    int all_logits_origin = token_offset;
    // Row origin inside the TP-sharded scoring outputs (all_logits_lse/target_logits/topk_*):
    // those tensors carry the selected scoring rows only — no lm prefix — so this stream's
    // token row r lives at prefix + (r - range_a).
    int scoring_origin = 0;
    if (model_output.all_logits_lm_rows >= 0) {
        const auto config             = stream->generateConfig();
        const auto [range_a, range_b] = promptScoringRowRange(token_size,
                                                              stream->calculateLoss(),
                                                              stream->returnPromptLogits(),
                                                              config->prompt_logits_start,
                                                              config->prompt_logits_end);
        all_logits_origin = static_cast<int>(model_output.all_logits_lm_rows) + all_logits_row_offset - range_a;
        scoring_origin                = all_logits_row_offset - range_a;
        all_logits_row_offset += range_b - range_a;
    }

    auto batch_new_all_token_ids = new_all_token_ids.narrow(0, batch_idx_out, next_batch_size);

    bool has_beam_search = stream->usesBeamSearchTokenLayoutForCurrentStep();
    bool has_var_batch   = stream->currentBatchSize() != stream->nextBatchSize();

    // construct mapping from output batches to input batches
    torch::Tensor src_batch_indices;
    if (has_beam_search) {
        // beam search
        if (!sampler_output.beam_index.defined() || sampler_output.beam_index.dim() != 1
            || batch_idx_out + next_batch_size > sampler_output.beam_index.numel()) {
            stream->reportError(ErrorCode::UNKNOWN_ERROR, "sampler returned an invalid parent index tensor");
            return;
        }
        src_batch_indices = sampler_output.beam_index.narrow(0, batch_idx_out, next_batch_size);
    } else if (has_var_batch) {
        // from context stream to decode straem, there might be other cases in future
        src_batch_indices = torch::zeros({(int64_t)next_batch_size}, torch::kInt32);
    }
    if (src_batch_indices.defined()) {
        src_batch_indices = src_batch_indices.to(torch::kCPU, torch::kInt32).contiguous();
        if (src_batch_indices.dim() != 1 || src_batch_indices.numel() != next_batch_size) {
            stream->reportError(ErrorCode::UNKNOWN_ERROR, "sampler returned an invalid parent index tensor");
            return;
        }
        const auto* parent_data = src_batch_indices.data_ptr<int32_t>();
        for (int32_t row = 0; row < next_batch_size; ++row) {
            if (parent_data[row] < 0 || parent_data[row] >= cur_batch_size) {
                stream->reportError(ErrorCode::UNKNOWN_ERROR,
                                    "sampler parent index " + std::to_string(parent_data[row])
                                        + " is outside the previous batch size " + std::to_string(cur_batch_size));
                return;
            }
        }
    }
    const auto get_src_idx = [&](int32_t dst_idx) {
        return src_batch_indices.defined() ? src_batch_indices.data_ptr<int32_t>()[dst_idx] : dst_idx;
    };

    // construct update info
    torch::Tensor batch_hidden_states;
    if (stream->generateConfig()->return_hidden_states) {
        auto raw_hidden_states = model_output.hidden_states.narrow(0, batch_idx_in, cur_batch_size);
        if (has_beam_search && src_batch_indices.defined()) {
            auto src_indices_device = src_batch_indices.to(raw_hidden_states.device(), torch::kLong);
            batch_hidden_states     = raw_hidden_states.index_select(0, src_indices_device);
        } else {
            batch_hidden_states = raw_hidden_states;
        }
    }

    torch::Tensor raw_logits;
    torch::Tensor batch_logits;
    if (stream->returnLogits() || stream->calculateSoftmaxProbs()) {
        raw_logits = model_output.logits.narrow(0, batch_idx_in, cur_batch_size);
        if (stream->returnLogits()) {
            if (has_beam_search && src_batch_indices.defined()) {
                auto src_indices_device = src_batch_indices.to(raw_logits.device(), torch::kLong);
                batch_logits            = raw_logits.index_select(0, src_indices_device);
            } else {
                batch_logits = raw_logits;
            }
        }
    }

    torch::Tensor all_probs;
    if (return_all_probs) {
        all_probs = sampler_output.all_probs.narrow(0, batch_idx_out, next_batch_size);
    };

    torch::Tensor batch_cum_log_probs;
    if (sampler_output.cum_log_probs.defined()) {
        batch_cum_log_probs = sampler_output.cum_log_probs.narrow(0, batch_idx_out, next_batch_size);
    }

    // Scoring-side temporaries can still allocate [rows, vocab]-scale buffers per stream —
    // the tp_size==1 / non-selected producers keep legacy all_logits and the tp>1 sharded
    // path allocates the contract-sized merged outputs — and can exhaust device memory on
    // very long prompts. Guard ONLY this stream's own scoring computation: on an
    // out-of-memory throw, fail this stream request-locally and let every other stream in
    // the batch finish. Anything that is not an out-of-memory rethrows unchanged, and
    // warm-up never swallows.
    torch::Tensor                     loss;
    std::optional<PromptLogitsOutput> prompt_logits_output;
    try {
        // Sharded scoring outputs replace all_logits on the tp_size>1 row-selected path; the
        // legacy gathered tensor remains for tp_size==1 and non-selected producers.
        const bool has_sharded_scoring = model_output.all_logits_lse.defined();
        const bool consumes_scoring =
            stream->calculateLoss()
            || (stream->returnPromptLogits() && (model_output.all_logits.defined() || has_sharded_scoring));
        if (consumes_scoring) {
            const auto failure_hook = std::atomic_load(&test_scoring_failure_hook);
            if (failure_hook) {
                (*failure_hook)(stream);
            }
        }

        if (stream->calculateLoss()) {
            if (has_sharded_scoring) {
                // CE(label) = logsumexp - target_logit, both already reduced globally per row.
                auto lse_rows    = model_output.all_logits_lse.narrow(0, scoring_origin, token_size - 1);
                auto target_rows = model_output.all_logits_target_logits.narrow(0, scoring_origin, token_size - 1);
                loss             = (lse_rows - target_rows).to(torch::kFloat32);
            } else {
                auto all_logits_tensor = model_output.all_logits.narrow(0, all_logits_origin, token_size - 1);
                auto tokens            = stream->currentExecuteTokens(0);
                auto label_tensor =
                    torch::from_blob(const_cast<int*>(tokens.data() + 1), {(int64_t)(tokens.size() - 1)}, torch::kInt32)
                        .to(torch::kCUDA);
                auto labels_int64 = label_tensor.toType(torch::kInt64);
                loss = torch::cross_entropy_loss(all_logits_tensor, labels_int64, torch::nullopt, at::Reduction::None)
                           .to(torch::kFloat32);
            }
        }

        // Prompt scoring: guarded by scoring outputs being defined, which the model produces
        // whenever need_all_logits is set — prefill prompt-logits/loss requests and decode-phase
        // calculate_loss both reach this path.
        if (stream->returnPromptLogits() && !model_output.all_logits.defined() && !has_sharded_scoring) {
            RTP_LLM_LOG_WARNING("stream [%ld] prompt_logits requested but all_logits not produced", stream->streamId());
        }
        if (stream->returnPromptLogits() && (model_output.all_logits.defined() || has_sharded_scoring)) {
            auto config    = stream->generateConfig();
            int  ts        = (int)token_size;
            int  start_pos = std::clamp(config->prompt_logits_start >= 0 ? config->prompt_logits_start : 0, 0, ts);
            int  end_pos   = std::clamp(config->prompt_logits_end >= 0 ? config->prompt_logits_end : ts, start_pos, ts);
            int  slice_len = end_pos - start_pos;
            if (slice_len > 0) {
                torch::Tensor topk_logprobs, topk_indices, target_logprobs;
                if (has_sharded_scoring) {
                    // Per-row global lse / raw target logit / merged top-k (value desc, then
                    // ascending token id) were produced by the sharded reduction; narrow this
                    // stream's window the same way the gathered path does.
                    const int64_t merged_k = model_output.all_logits_topk_values.size(1);
                    int           top_k    = std::min(config->prompt_logits_top_k, (int)merged_k);
                    auto          lse = model_output.all_logits_lse.narrow(0, scoring_origin + start_pos, slice_len);
                    topk_logprobs = model_output.all_logits_topk_values.narrow(0, scoring_origin + start_pos, slice_len)
                                        .narrow(1, 0, top_k)
                                    - lse.unsqueeze(1);
                    topk_indices = model_output.all_logits_topk_ids.narrow(0, scoring_origin + start_pos, slice_len)
                                       .narrow(1, 0, top_k);
                    if (config->return_target_logprob) {
                        auto tokens      = stream->currentExecuteTokens(0);
                        int  label_start = start_pos + 1;
                        int  label_end   = std::min(end_pos + 1, (int)tokens.size());
                        int  logprob_len = label_end - label_start;
                        if (logprob_len > 0) {
                            auto target_raw = model_output.all_logits_target_logits.narrow(
                                0, scoring_origin + start_pos, logprob_len);
                            target_logprobs = (target_raw - lse.narrow(0, 0, logprob_len)).cpu();
                        }
                    }
                } else {
                    int top_k = std::min(config->prompt_logits_top_k, (int)model_output.all_logits.size(1));

                    auto sliced_logits =
                        model_output.all_logits.narrow(0, all_logits_origin + start_pos, slice_len).to(torch::kFloat32);

                    // topk on raw logits (monotonicity of softmax preserves ranking)
                    auto [topk_values_raw, topk_idx] = sliced_logits.topk(top_k, -1);

                    // single reduce for log-normalizer, avoids materializing [slice_len, vocab_size]
                    auto log_sum_exp = sliced_logits.logsumexp(-1, /*keepdim=*/true);
                    topk_logprobs    = topk_values_raw - log_sum_exp;
                    topk_indices     = topk_idx;

                    // target_logprobs[i] = logprob of token[start_pos+i+1] at position start_pos+i.
                    // Length = min(slice_len, tokens.size() - start_pos - 1): equals slice_len when
                    // end_pos < tokens.size(), or slice_len-1 when end_pos == tokens.size() (last
                    // position has no next token as label).
                    if (config->return_target_logprob) {
                        auto tokens      = stream->currentExecuteTokens(0);
                        int  label_start = start_pos + 1;
                        int  label_end   = std::min(end_pos + 1, (int)tokens.size());
                        int  logprob_len = label_end - label_start;
                        if (logprob_len > 0) {
                            // from_blob + to(kCUDA) is a synchronous copy; token buffer is stable during prefill.
                            auto label_tensor = torch::from_blob(const_cast<int*>(tokens.data() + label_start),
                                                                 {(int64_t)logprob_len},
                                                                 torch::kInt32)
                                                    .to(torch::kCUDA)
                                                    .toType(torch::kInt64)
                                                    .unsqueeze(1);
                            auto target_raw =
                                sliced_logits.narrow(0, 0, logprob_len).gather(1, label_tensor).squeeze(1);
                            target_logprobs = (target_raw - log_sum_exp.narrow(0, 0, logprob_len).squeeze(1)).cpu();
                        }
                    }
                }

                prompt_logits_output = PromptLogitsOutput{
                    topk_logprobs.cpu(), topk_indices.to(torch::kInt32).cpu(), target_logprobs, start_pos, end_pos};
            }
        }
    } catch (const c10::OutOfMemoryError& e) {
        if (!failStreamOnScoringOom(stream, e.what())) {
            throw;
        }
        return;
    } catch (const std::exception& e) {
        if (!isOutOfMemoryMessage(e.what()) || !failStreamOnScoringOom(stream, e.what())) {
            throw;
        }
        return;
    }

    torch::Tensor all_hidden_states;
    if (stream->needReturnHiddenStates()) {
        all_hidden_states = model_output.all_hidden_states.narrow(0, token_offset, token_size);
    }

    auto         new_tokens     = new_tokens_all.narrow(0, batch_idx_out, next_batch_size);
    const size_t token_position = has_beam_search ? stream->seqLength() : token_stride - 1;
    RTP_LLM_CHECK(token_position < token_stride);
    for (size_t i = 0; i < next_batch_size; ++i) {
        new_tokens.data_ptr<int32_t>()[i] =
            new_all_token_ids.data_ptr<int32_t>()[(batch_idx_out + i) * token_stride + token_position];
    }

    if (!output_vocab_ids_.empty()) {
        for (int i = 0; i < cur_batch_size; ++i) {
            if (success_cpu.defined() && !success_cpu.data_ptr<bool>()[batch_idx_in + i]) {
                stream->reportError(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
                return;
            }
        }
        for (int i = 0; i < next_batch_size; ++i) {
            const auto compact_id = new_tokens.data_ptr<int32_t>()[i];
            if (compact_id < 0 || static_cast<size_t>(compact_id) >= output_vocab_ids_.size()) {
                stream->reportError(ErrorCode::OUT_OF_VOCAB_RANGE,
                                    "compact output token id " + std::to_string(compact_id)
                                        + " is outside configured output vocabulary size "
                                        + std::to_string(output_vocab_ids_.size()));
                return;
            }
        }
    }

    torch::Tensor current_softmax_result;
    if (stream->calculateSoftmaxProbs()) {
        auto batch_softmax_input = raw_logits.to(torch::kFloat32).contiguous();
#if USING_CUDA
        cudaSoftmaxInplace(batch_softmax_input, at::cuda::getCurrentCUDAStream().stream());
#else
        batch_softmax_input = torch::softmax(batch_softmax_input, -1);
#endif
        auto batch_softmax_tensor = batch_softmax_input.cpu();
        current_softmax_result    = torch::empty({(int64_t)next_batch_size, 1}, torch::kFloat32);
        for (int i = 0; i < next_batch_size; ++i) {
            current_softmax_result[i][0] = batch_softmax_tensor[get_src_idx(i)][new_tokens.data_ptr<int32_t>()[i]];
        }
    }

    auto error_info = collectStreamSamplerError(sampler_output, success_cpu, batch_idx_in, cur_batch_size);
    if (asyncDebugEnabled() && success_cpu.defined()) {
        for (int i = 0; i < cur_batch_size; ++i) {
            if (!(success_cpu.data_ptr<bool>()[batch_idx_in + i])) {
                const auto& state = stream->getNormalAsyncDeviceState();
                RTP_LLM_LOG_ERROR("[async-debug] sampler success=false: stream=%ld pd_sep=%d status=%s "
                                  "pending=%d seq_len=%d state_next_real=%d batch_idx_in=%d token_stride=%zu",
                                  stream->streamId(),
                                  stream->queryPdSep(),
                                  StreamStateToString(stream->getStatus()).c_str(),
                                  stream->hasPendingAsyncBookkeeping(),
                                  stream->seqLength(),
                                  state.next_real_seq_len,
                                  batch_idx_in + i,
                                  token_stride);
            }
        }
    }

    if (!restoreCurrentTokenIds(stream, batch_new_all_token_ids, new_tokens, token_position)) {
        return;
    }

    RTP_LLM_LOG_DEBUG("stream [%ld], new_tokens size = [%ld]", stream->streamId(), new_tokens.numel());

    StreamUpdateInfo update_info{has_beam_search ? batch_new_all_token_ids : new_tokens,
                                 1,
                                 batch_hidden_states,
                                 batch_logits,
                                 current_softmax_result,
                                 batch_cum_log_probs,
                                 all_probs,
                                 loss,
                                 src_batch_indices,
                                 all_hidden_states,
                                 /*update_remote_generate=*/true,
                                 /*force_update_info=*/false,
                                 std::move(prompt_logits_output),
                                 std::move(error_info)};
    stream->update(update_info);
}

}  // namespace rtp_llm
