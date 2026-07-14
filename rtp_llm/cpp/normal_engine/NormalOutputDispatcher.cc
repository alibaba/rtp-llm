#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

torch::Tensor NormalOutputDispatcher::calculateSelectedTokenProbs(const torch::Tensor& logits,
                                                                  const torch::Tensor& token_ids,
                                                                  const torch::Tensor& src_batch_indices) const {
    RTP_LLM_CHECK(logits.dim() == 2);
    RTP_LLM_CHECK(token_ids.numel() > 0);

    auto token_ids_cpu = token_ids.to(torch::kCPU, torch::kLong).reshape({-1}).contiguous();
    RTP_LLM_CHECK(token_ids_cpu.min().item<int64_t>() >= 0);
    RTP_LLM_CHECK(token_ids_cpu.max().item<int64_t>() < logits.size(1));

    torch::Tensor src_indices_cpu;
    if (src_batch_indices.defined()) {
        src_indices_cpu = src_batch_indices.to(torch::kCPU, torch::kLong).reshape({-1}).contiguous();
        RTP_LLM_CHECK(src_indices_cpu.numel() == token_ids_cpu.numel());
        RTP_LLM_CHECK(src_indices_cpu.min().item<int64_t>() >= 0);
        RTP_LLM_CHECK(src_indices_cpu.max().item<int64_t>() < logits.size(0));
    } else {
        RTP_LLM_CHECK(token_ids_cpu.numel() == logits.size(0));
        src_indices_cpu = torch::arange(logits.size(0), torch::kLong);
    }

    auto logits_fp32              = logits.to(torch::kFloat32).contiguous();
    auto token_ids_device         = token_ids_cpu.to(logits.device());
    auto src_indices_device       = src_indices_cpu.to(logits.device());
    auto flat_indices             = src_indices_device * logits.size(1) + token_ids_device;
    auto selected_logits          = logits_fp32.reshape({-1}).index_select(0, flat_indices);
    auto log_normalizers          = at::logsumexp(logits_fp32, {1}, false);
    auto selected_log_normalizers = log_normalizers.index_select(0, src_indices_device);
    return torch::exp(selected_logits - selected_log_normalizers).reshape({-1, 1}).cpu();
}

absl::Status NormalOutputDispatcher::dispatch(const StreamGroups& stream_groups,
                                              const MergedOutput& merge_outputs) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto&  sampler_output       = merge_outputs.sampler_output;
    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == (size_t)sampler_output.token_ids.size(0));
    // token_ids and success may be CUDA tensors (Sampler keeps them on GPU to avoid D2H sync during sampling).
    // Move to CPU once here so dispatchSingleStream can use data_ptr safely.
    const torch::Tensor token_ids_cpu =
        sampler_output.token_ids.defined() ? sampler_output.token_ids.cpu() : torch::Tensor();
    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", tensorDebugStringWithData<int32_t>(token_ids_cpu).c_str());
    const torch::Tensor success_cpu = sampler_output.success.defined() ? sampler_output.success.cpu() : torch::Tensor();
    int                 batch_idx_in     = 0;
    int                 batch_idx_out    = 0;
    int                 token_offset     = 0;
    bool                return_all_probs = stream_groups.needReturnAllProbs() != ReturnAllProbsMode::NONE;
    auto                new_tokens_all   = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);

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

    auto batch_new_all_token_ids = new_all_token_ids.narrow(0, batch_idx_out, next_batch_size);

    bool has_beam_search = stream->currentNumBeams() > 1 || stream->nextNumBeams() > 1;
    bool has_var_batch   = stream->currentBatchSize() != stream->nextBatchSize();

    // construct mapping from output batches to input batches
    torch::Tensor src_batch_indices;
    if (has_beam_search) {
        // beam search
        src_batch_indices = sampler_output.beam_index.narrow(0, batch_idx_out, next_batch_size);
    } else if (has_var_batch) {
        // from context stream to decode straem, there might be other cases in future
        src_batch_indices = torch::zeros({(int64_t)next_batch_size}, torch::kInt32);
    }
    // construct update info
    torch::Tensor batch_hidden_states;
    if (stream->generateConfig()->return_hidden_states) {
        batch_hidden_states = model_output.hidden_states.narrow(0, batch_idx_in, cur_batch_size);
    }

    torch::Tensor batch_logits;
    if (stream->returnLogits() || stream->calculateSoftmaxProbs() || has_beam_search) {
        batch_logits = model_output.logits.narrow(0, batch_idx_in, cur_batch_size);
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
            auto log_sum_exp   = sliced_logits.logsumexp(-1, /*keepdim=*/true);
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

    auto new_tokens = new_tokens_all.narrow(0, batch_idx_out, next_batch_size);
    for (size_t i = 0; i < next_batch_size; ++i) {
        new_tokens.data_ptr<int32_t>()[i] =
            new_all_token_ids.data_ptr<int32_t>()[(batch_idx_out + i) * token_stride + token_stride - 1];
    }

    torch::Tensor current_softmax_result;
    if (stream->calculateSoftmaxProbs()) {
        current_softmax_result = calculateSelectedTokenProbs(batch_logits, new_tokens, src_batch_indices);
    }

    for (int i = 0; i < cur_batch_size; ++i) {
        if (success_cpu.defined() && !(success_cpu.data_ptr<bool>()[batch_idx_in + i])) {
            stream->reportError(ErrorCode::UNKNOWN_ERROR, "sampler generate token id failed");
        }
    }

    RTP_LLM_LOG_DEBUG("stream [%ld], new_tokens size = [%ld]", stream->streamId(), new_tokens.numel());

    stream->update({has_beam_search ? batch_new_all_token_ids : new_tokens,
                    1,
                    batch_hidden_states,
                    batch_logits,
                    current_softmax_result,
                    batch_cum_log_probs,
                    all_probs,
                    loss,
                    src_batch_indices,
                    all_hidden_states,
                    true,
                    false,
                    prompt_logits_output});
}

}  // namespace rtp_llm
