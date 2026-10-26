#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
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

}  // namespace

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
    // In the ordinary non-beam case, sampler input and output rows have the
    // same ownership. Normalize once on device and copy only the selected
    // probabilities, rather than a full vocabulary for every stream.
    bool any_softmax_probs = false;
    bool direct_rows       = stream_groups.totalSamplerBatchSizeIn() == total_batch_size_out;
    for (const auto& stream : stream_groups.allStreams()) {
        any_softmax_probs |= stream->calculateSoftmaxProbs();
        direct_rows &=
            !stream->usesBeamSearchTokenLayoutForCurrentStep() && stream->currentBatchSize() == stream->nextBatchSize();
    }
    torch::Tensor selected_probs_cpu;
    torch::Tensor selected_logprobs_cpu;
    const auto&   log_normalizers = merge_outputs.model_output.log_normalizers;
    RTP_LLM_CHECK_WITH_INFO(!log_normalizers.defined() || direct_rows,
                            "batch invariant logprobs do not support beam expansion or row reordering");
    if (any_softmax_probs && direct_rows) {
        const auto& logits = merge_outputs.model_output.logits;
        RTP_LLM_CHECK(logits.defined() && logits.dim() == 2
                      && logits.size(0) >= static_cast<int64_t>(total_batch_size_out) && logits.size(1) > 0);
        // The in-place normalization must not overwrite logits returned to
        // another stream, or logits retained by the model/sampler.
        auto raw_logits   = logits.narrow(0, 0, total_batch_size_out);
        auto selected_ids = token_ids_for_copy.to(logits.device(), torch::kLong).clamp(0, logits.size(1) - 1);
        if (log_normalizers.defined()) {
            RTP_LLM_CHECK(log_normalizers.dim() == 2 && log_normalizers.size(1) == 1
                          && log_normalizers.size(0) >= static_cast<int64_t>(total_batch_size_out)
                          && log_normalizers.scalar_type() == torch::kFloat32
                          && raw_logits.scalar_type() == torch::kFloat32);
            auto selected = raw_logits.gather(1, selected_ids) - log_normalizers.narrow(0, 0, total_batch_size_out);
            selected_logprobs_cpu = copyToPinnedCpuAsync(selected, need_d2h_sync);
            selected_probs_cpu    = copyToPinnedCpuAsync(selected.exp(), need_d2h_sync);
        } else {
            auto probabilities = raw_logits.to(torch::kFloat32, /*non_blocking=*/false, /*copy=*/true).contiguous();
#if USING_CUDA
            if (probabilities.is_cuda()) {
                cudaSoftmaxInplace(probabilities, at::cuda::getCurrentCUDAStream().stream());
            } else {
                probabilities = torch::softmax(probabilities, -1);
            }
#else
            probabilities = torch::softmax(probabilities, -1);
#endif
            // A failed sampler can leave an invalid token. Its stream still gets
            // the original failure below; avoid a device-side gather assertion
            // while collecting successful peers' probabilities.
            selected_probs_cpu = copyToPinnedCpuAsync(probabilities.gather(1, selected_ids), need_d2h_sync);
        }
    }
    syncPinnedCpuCopies(need_d2h_sync);
    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", tensorDebugStringWithData<int32_t>(token_ids_cpu).c_str());
    int  batch_idx_in     = 0;
    int  batch_idx_out    = 0;
    int  token_offset     = 0;
    bool return_all_probs = stream_groups.needReturnAllProbs() != ReturnAllProbsMode::NONE;
    auto new_tokens_all   = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);

    for (auto& stream : stream_groups.allStreams()) {
        auto cur_batch_size  = stream->currentBatchSize();
        auto next_batch_size = stream->nextBatchSize();
        auto token_size      = stream->currentExecuteTokenSize();

        dispatchSingleStream(stream,
                             merge_outputs,
                             batch_idx_in,
                             batch_idx_out,
                             token_offset,
                             return_all_probs,
                             new_tokens_all,
                             token_ids_cpu,
                             success_cpu,
                             selected_probs_cpu,
                             selected_logprobs_cpu);

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
                                                  bool                 return_all_probs,
                                                  const torch::Tensor& new_tokens_all,
                                                  const torch::Tensor& token_ids_cpu,
                                                  const torch::Tensor& success_cpu,
                                                  const torch::Tensor& selected_probs_cpu,
                                                  const torch::Tensor& selected_logprobs_cpu) const {

    const auto&  model_output      = merge_outputs.model_output;
    const auto&  sampler_output    = merge_outputs.sampler_output;
    const auto&  new_all_token_ids = token_ids_cpu;
    const size_t token_stride      = new_all_token_ids.size(1);

    auto cur_batch_size  = stream->currentBatchSize();
    auto next_batch_size = stream->nextBatchSize();
    auto token_size      = stream->currentExecuteTokenSize();

    bool          has_beam_search = stream->usesBeamSearchTokenLayoutForCurrentStep();
    bool          has_var_batch   = stream->currentBatchSize() != stream->nextBatchSize();
    torch::Tensor batch_new_all_token_ids;
    if (has_beam_search || !output_vocab_ids_.empty()) {
        batch_new_all_token_ids = new_all_token_ids.narrow(0, batch_idx_out, next_batch_size);
    }

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
    if (stream->returnLogits() || (stream->calculateSoftmaxProbs() && !selected_probs_cpu.defined())) {
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

    torch::Tensor loss;
    if (stream->calculateLoss()) {
        auto all_logits_tensor = model_output.all_logits.narrow(0, token_offset, token_size - 1);
        auto tokens            = stream->currentExecuteTokens(0);
        auto label_tensor =
            torch::from_blob(const_cast<int*>(tokens.data() + 1), {(int64_t)(tokens.size() - 1)}, torch::kInt32)
                .to(torch::kCUDA);
        auto labels_int64 = label_tensor.toType(torch::kInt64);
        loss = torch::cross_entropy_loss(all_logits_tensor, labels_int64, torch::nullopt, at::Reduction::None)
                   .to(torch::kFloat32);
    }

    // Prompt scoring: guarded by all_logits.defined() which is only produced during prefill
    // (NormalModelInputGatherer sets need_all_logits only in processContextStreams).
    std::optional<PromptLogitsOutput> prompt_logits_output;
    if (stream->returnPromptLogits() && !model_output.all_logits.defined()) {
        RTP_LLM_LOG_WARNING("stream [%ld] prompt_logits requested but all_logits not produced", stream->streamId());
    }
    if (stream->returnPromptLogits() && model_output.all_logits.defined()) {
        auto config    = stream->generateConfig();
        int  ts        = (int)token_size;
        int  start_pos = std::clamp(config->prompt_logits_start >= 0 ? config->prompt_logits_start : 0, 0, ts);
        int  end_pos   = std::clamp(config->prompt_logits_end >= 0 ? config->prompt_logits_end : ts, start_pos, ts);
        int  slice_len = end_pos - start_pos;
        if (slice_len > 0) {
            int top_k = std::min(config->prompt_logits_top_k, (int)model_output.all_logits.size(1));

            auto sliced_logits =
                model_output.all_logits.narrow(0, token_offset + start_pos, slice_len).to(torch::kFloat32);

            // topk on raw logits (monotonicity of softmax preserves ranking)
            auto [topk_values_raw, topk_indices] = sliced_logits.topk(top_k, -1);

            // single reduce for log-normalizer, avoids materializing [slice_len, vocab_size]
            auto log_sum_exp   = model_output.all_log_normalizers.defined() ?
                                     model_output.all_log_normalizers.narrow(0, token_offset + start_pos, slice_len) :
                                     sliced_logits.logsumexp(-1, /*keepdim=*/true);
            auto topk_logprobs = topk_values_raw - log_sum_exp;

            // target_logprobs[i] = logprob of token[start_pos+i+1] at position start_pos+i.
            // Length = min(slice_len, tokens.size() - start_pos - 1): equals slice_len when
            // end_pos < tokens.size(), or slice_len-1 when end_pos == tokens.size() (last
            // position has no next token as label).
            torch::Tensor target_logprobs;
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
                    auto target_raw = sliced_logits.narrow(0, 0, logprob_len).gather(1, label_tensor).squeeze(1);
                    target_logprobs = (target_raw - log_sum_exp.narrow(0, 0, logprob_len).squeeze(1)).cpu();
                }
            }

            prompt_logits_output = PromptLogitsOutput{
                topk_logprobs.cpu(), topk_indices.to(torch::kInt32).cpu(), target_logprobs, start_pos, end_pos};
        }
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
    if (stream->calculateSoftmaxProbs() && selected_probs_cpu.defined()) {
        current_softmax_result = selected_probs_cpu.narrow(0, batch_idx_out, next_batch_size);
    } else if (stream->calculateSoftmaxProbs()) {
        // Keep returned logits intact when normalizing fallback beam/variable-batch rows.
        auto batch_softmax_input = raw_logits.to(torch::kFloat32, /*non_blocking=*/false, /*copy=*/true).contiguous();
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
    if (selected_probs_cpu.defined() && stream->calculateSoftmaxProbs() && !error_info.has_value()) {
        for (int i = 0; i < next_batch_size; ++i) {
            const auto token = new_tokens.data_ptr<int32_t>()[i];
            if (token < 0 || token >= model_output.logits.size(1)) {
                error_info =
                    ErrorInfo(ErrorCode::OUT_OF_VOCAB_RANGE, "selected output token is outside the logits vocabulary");
                break;
            }
        }
    }
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
    if (stream->calculateSoftmaxProbs() && selected_logprobs_cpu.defined()) {
        update_info.raw_logprobs = selected_logprobs_cpu.narrow(0, batch_idx_out, next_batch_size);
    }
    stream->update(update_info);
}

}  // namespace rtp_llm
