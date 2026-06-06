#include "rtp_llm/cpp/normal_engine/NormalOutputDispatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "ATen/cuda/CUDAContext.h"
#endif

namespace rtp_llm {

NormalOutputDispatcher::~NormalOutputDispatcher() {
    close();
}

void NormalOutputDispatcher::close() noexcept {
    if (!triton_bitmask_ops_) {
        return;
    }
    if (Py_IsInitialized()) {
        // py::module_() swap runs Py_DECREF on the old handle; needs the GIL.
        py::gil_scoped_acquire acquire;
        triton_bitmask_ops_ = py::module_();
    } else {
        // Post-finalize: GIL acquire is UB. Deliberately leak.
        (void)triton_bitmask_ops_.release();
    }
}

absl::Status NormalOutputDispatcher::dispatch(const StreamGroups& stream_groups,
                                              const MergedOutput& merge_outputs) const {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    const auto&  sampler_output       = merge_outputs.sampler_output;
    const size_t total_batch_size_out = stream_groups.totalSamplerBatchSizeOut();
    RTP_LLM_CHECK(total_batch_size_out == (size_t)sampler_output.token_ids.size(0));
    // Sampler keeps token_ids/success on GPU to avoid D2H sync mid-sampling;
    // move to CPU once here for the per-stream loop + grammar accept.
    const torch::Tensor token_ids_cpu =
        sampler_output.token_ids.defined() ? sampler_output.token_ids.cpu() : torch::Tensor();
    RTP_LLM_LOG_DEBUG("new_all_token_ids = [%s]", tensorDebugStringWithData<int32_t>(token_ids_cpu).c_str());
    const torch::Tensor success_cpu = sampler_output.success.defined() ? sampler_output.success.cpu() : torch::Tensor();
    int                 batch_idx_in     = 0;
    int                 batch_idx_out    = 0;
    int                 token_offset     = 0;
    bool                return_all_probs = stream_groups.needReturnAllProbs();
    auto                new_tokens_all   = torch::empty({(int64_t)total_batch_size_out, 1}, torch::kInt32);

    bool any_grammar = false;
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

        // Piggyback grammar pre-check: no-grammar path pays only a null check.
        if (!any_grammar && stream->hasGrammarMatcher()) {
            any_grammar = true;
        }

        batch_idx_in += cur_batch_size;
        batch_idx_out += next_batch_size;
        token_offset += token_size;
    }

    if (any_grammar) {
        batchAcceptGrammarTokens(stream_groups, token_ids_cpu);
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
    const auto get_src_idx = [&](int32_t dst_idx) {
        return src_batch_indices.defined() ? src_batch_indices.data_ptr<int32_t>()[dst_idx] : dst_idx;
    };

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
        auto batch_softmax_input = batch_logits.to(torch::kFloat32).contiguous();
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
                    all_hidden_states});
}

void NormalOutputDispatcher::invokeBatchAcceptTokens(
    const StreamGroups&                                                  stream_groups,
    const std::function<std::vector<int32_t>(const GenerateStreamPtr&)>& extract_tokens,
    const char*                                                          log_prefix) const {
    RTP_LLM_PROFILE_SCOPE("grammar.acceptTokens");
    // The closure must run for every stream (even matcher-less ones) to keep
    // per-stream offset counters in lock-step with stream_groups iteration order.
    // Empty vector = skip.
    for (auto& stream : stream_groups.allStreams()) {
        std::vector<int32_t> token_ids = extract_tokens(stream);
        if (token_ids.empty()) {
            continue;
        }
        RtpGrammarMatcher* matcher = stream->tryGetGrammarMatcher();
        if (matcher == nullptr) {
            continue;
        }

        for (int32_t tok : token_ids) {
            if (!matcher->acceptToken(tok)) {
                RTP_LLM_LOG_WARNING(
                    "[%s] stream [%ld] grammar parser rejected token %d", log_prefix, stream->streamId(), tok);
                stream->reportError(ErrorCode::INVALID_PARAMS,
                                    "grammar accept_token error: parser rejected token "
                                        + std::to_string(tok));
                break;
            }
        }
        if (matcher->isTerminated()) {
            matcher->markFinished();
            if (stream->isActive()) {
                stream->reportEvent(StreamEvents::GenerateDone);
            }
        } else if (!stream->isActive()) {
            matcher->markFinished();
        }
    }
}

void NormalOutputDispatcher::batchAcceptGrammarTokens(const StreamGroups&  stream_groups,
                                                      const torch::Tensor& token_ids_cpu) const {
    if (!token_ids_cpu.defined()) {
        return;
    }

    const size_t   token_stride  = token_ids_cpu.size(1);
    const int32_t* data          = token_ids_cpu.data_ptr<int32_t>();
    int            batch_idx_out = 0;

    invokeBatchAcceptTokens(
        stream_groups,
        [&](const GenerateStreamPtr& stream) -> std::vector<int32_t> {
            const int  next_batch_size = stream->nextBatchSize();
            const bool has_beam_search = stream->currentNumBeams() > 1 || stream->nextNumBeams() > 1;
            std::vector<int32_t> out;
            // Beam search + grammar is unsupported → skip via empty vector.
            if (!has_beam_search) {
                out.push_back(data[batch_idx_out * token_stride + token_stride - 1]);
            }
            batch_idx_out += next_batch_size;
            return out;
        },
        "xgrammar batch_accept");
}

}  // namespace rtp_llm
