#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include <chrono>
#include <functional>
#include <mutex>

namespace rtp_llm {

namespace {
// The queue's own waitNotEmpty() slices on autil's DEF_WAIT_TIME (1 s). Wait on our own condition
// variable instead so the slice is short enough to re-check the caller's cancellation predicate
// promptly: a producer's notify keeps an arriving output immediate, while the bounded slice bounds how
// long a stream producing nothing -- queued behind admission, stalled, or a non-streaming PD decode
// that only emits once it finishes -- can park before the predicate is evaluated again. 100 ms keeps
// that far under a second at a handful of wakeups per blocked stream, so it is a timed wait and not a
// busy-poll.
constexpr auto kOutputWaitSlice = std::chrono::milliseconds(100);
}  // namespace

ErrorResult<GenerateOutputs> NormalGenerateStream::nextOutput() {
    return nextOutput(std::function<bool()>());
}

ErrorResult<GenerateOutputs> NormalGenerateStream::nextOutput(const std::function<bool()>& is_cancelled) {
    // TODO(xinfei.sxf) 某些case下会出现1s的等待
    while ((!hasError()) && getStatus() != StreamState::FINISHED && generate_outputs_queue_.isEmpty()) {
        // Without this, a cancellation could only be observed AFTER an output arrived -- which for a
        // non-streaming PD decode never happens until the generation ends, so an abandoned request
        // held its admission slot, KV blocks and per-rank capacity for the whole generation. The RPC
        // layer passes its gRPC context's IsCancelled(); the propagation set up on the prefill's
        // downstream ClientContext is what makes that flag become true.
        if (is_cancelled && is_cancelled()) {
            return ErrorInfo(ErrorCode::CANCELLED, "request cancelled while waiting for an output");
        }
        checkTimeout();
        // Clear the flag before re-testing emptiness so a push landing between the test and the wait
        // still wakes us; a notify missed entirely is bounded by the slice anyway.
        output_wait_->wake.store(false, std::memory_order_release);
        if (generate_outputs_queue_.isEmpty()) {
            std::unique_lock<std::mutex> lock(output_wait_->mu);
            output_wait_->cv.wait_for(lock, kOutputWaitSlice, [this] {
                return output_wait_->wake.load(std::memory_order_acquire) || !generate_outputs_queue_.isEmpty();
            });
        }
    }
    if (hasError()) {
        return statusInfo();
    }
    if (generate_outputs_queue_.isEmpty()) {
        if (isFinished()) {
            return ErrorInfo(ErrorCode::FINISHED, "finished");
        } else {
            return ErrorInfo(ErrorCode::OUTPUT_QUEUE_IS_EMPTY, "output queue is empty");
        }
    }
    return generate_outputs_queue_.getAndPopFront();
}

bool NormalGenerateStream::hasOutput() {
    return !generate_outputs_queue_.isEmpty();
}

GenerateOutputs NormalGenerateStream::prepareGenerateOutput(const StreamUpdateInfo& update_info) {
    size_t          output_len = seqLength() - last_output_pos_;
    GenerateOutputs generate_results;
    generate_results.request_id = request_id_;

    for (int i = 0; i < nextBatchSize(); i++) {
        GenerateOutput generate_output;
        generate_output.aux_info.iter_count = iter_count_;
        generate_output.output_ids          = torch::empty({1, (int64_t)output_len}, torch::kInt32);

        // TODO(xinfei.sxf) optimize this copy : only copy last token
        complete_token_ids_->copyTokensTo(
            i, generate_output.output_ids.data_ptr<int32_t>(), last_output_pos_, output_len);
        if (returnLogits() && update_info.logits.defined()) {
            torch::Tensor logits_result;
            const auto&   select_tokens_id = generate_input_->generate_config->select_tokens_id;
            if (!select_tokens_id.empty()) {
                auto out_of_bound_token_id =
                    std::find_if_not(select_tokens_id.begin(), select_tokens_id.end(), [this](int select_token_id) {
                        return select_token_id >= 0 && select_token_id < vocabSize();
                    });
                if (out_of_bound_token_id == select_tokens_id.end()) {
                    auto select_indices =
                        torch::tensor(std::vector<int64_t>(select_tokens_id.begin(), select_tokens_id.end()),
                                      torch::kLong)
                            .to(update_info.logits.device());
                    logits_result = update_info.logits.index_select(1, select_indices);
                } else {
                    RTP_LLM_LOG_WARNING("select_token_id out of bound, expected >= 0 and < vocab size [%d], found [%d]",
                                        vocabSize(),
                                        *out_of_bound_token_id);
                    logits_result = torch::empty({0, 0}, update_info.logits.options());
                }
            } else {
                logits_result = update_info.logits;
            }
            if (logits_result.size(0) <= 1) {
                generate_output.logits = logits_result.cpu().clone();
            } else {
                generate_output.logits = logits_result.narrow(0, i, 1).cpu().clone();
            }
        }

        if (generate_input_->generate_config->return_hidden_states && update_info.hidden_states.defined()) {
            if (update_info.hidden_states.size(0) == 1) {
                generate_output.hidden_states = update_info.hidden_states.cpu();
            } else {
                generate_output.hidden_states = update_info.hidden_states.narrow(0, i, 1).cpu();
            }
        }
        if (generate_input_->generate_config->return_all_hidden_states && update_info.all_hidden_states.defined()
            && iter_count_ == 1) {
            generate_output.all_hidden_states = update_info.all_hidden_states.cpu();
        }
        if (loss_.defined()) {
            RTP_LLM_CHECK_WITH_INFO(loss_index_ == inputLength() - 1,
                                    "loss index should be input len [%d] - 1 but is [%d]",
                                    inputLength(),
                                    loss_index_);
            if (generate_input_->generate_config->calculate_loss == 1) {
                generate_output.loss = torch::mean(loss_).exp().cpu().unsqueeze(0);
            } else {
                generate_output.loss = loss_;
            }
        }

        generate_output.finished = isSubGenerateDoneWithoutLock(i);
        if (generate_input_->generate_config->aux_info) {
            generate_output.aux_info.iter_count   = iter_count_;
            generate_output.aux_info.cost_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
            generate_output.aux_info.first_token_cost_time_us = complete_token_ids_->firstTokenLatencyUs();
            generate_output.aux_info.wait_time_us             = wait_time_us_;
            generate_output.aux_info.input_len                = generate_input_->promptLength();
            generate_output.aux_info.prefix_len               = generate_input_->prefix_length;
            // TODO(xinfei.sxf) 提前结束的query，output len要设置正确
            generate_output.aux_info.output_len                          = seqLength() - generate_input_->inputLength();
            generate_output.aux_info.step_output_len                     = output_len;
            generate_output.aux_info.reuse_len                           = initial_reuse_length_;
            generate_output.aux_info.pd_sep                              = queryPdSep();
            generate_output.aux_info.local_reuse_len                     = local_reuse_length_;
            generate_output.aux_info.remote_reuse_len                    = remote_reuse_length_;
            generate_output.aux_info.memory_reuse_len                    = memory_reuse_length_;
            generate_output.aux_info.speculative_draft_rounds            = sp_iter_count_;
            generate_output.aux_info.speculative_accepted_tokens_per_pos = speculative_accepted_tokens_per_pos_;
            if (generate_input_->generate_config->return_softmax_probs && softmax_probs_.defined()) {
                generate_output.aux_info.softmax_probs =
                    softmax_probs_[i].narrow(0, last_output_pos_, output_len).clone();
            }
            if (update_info.cum_log_probs.defined()) {
                generate_output.aux_info.cum_log_probs = cum_log_probs_.narrow(0, i, 1).cpu().clone();
            }
            if (generate_input_->generate_config->return_all_probs) {
                if (!update_info.all_probs.defined()) {
                    throw std::runtime_error("all_probs is not while generate_config return_all_probs is true");
                }
                generate_output.aux_info.all_probs = all_probs_.narrow(0, i, 1).clone();
            }
        }
        // hidden_states post process
        if (generate_output.finished && generate_input_->generate_config->return_hidden_states
            && generate_output.hidden_states.has_value()
            && (generate_input_->generate_config->hidden_states_cut_dim > 0
                || generate_input_->generate_config->normalized_hidden_states)) {
            auto hidden_states_tensor = generate_output.hidden_states.value();
            if (generate_input_->generate_config->hidden_states_cut_dim > 0) {
                hidden_states_tensor = hidden_states_tensor.index(
                    {torch::indexing::Slice(),
                     torch::indexing::Slice(0, generate_input_->generate_config->hidden_states_cut_dim)});
            }
            if (generate_input_->generate_config->normalized_hidden_states) {
                hidden_states_tensor = torch::nn::functional::normalize(
                    hidden_states_tensor, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
            }
            generate_output.hidden_states = hidden_states_tensor.cpu().clone();
        }

        generate_results.generate_outputs.emplace_back(std::move(generate_output));
    }
    return generate_results;
}

void NormalGenerateStream::enqueueGenerateOutput(GenerateOutputs&& generate_results) {
    if (generate_outputs_queue_.getSize() >= generate_outputs_queue_.getCapacity()) {
        /* No matter if the queue is full for any reason,
           the stream will be set to stop directly to prevent the push to queue from getting stuck. */
        reportEventWithoutLock(StreamEvents::Error, ErrorCode::OUTPUT_QUEUE_FULL, "output queue is full");
    } else {
        generate_outputs_queue_.push(std::move(generate_results));
    }
    // Wake a nextOutput() waiter on BOTH branches: the error branch is a terminal transition, so a
    // waiter must not sit out the rest of its slice before observing it. An atomic store plus a notify
    // keeps this off the producer's lock path -- this runs while the stream mutex_ is held.
    output_wait_->wake.store(true, std::memory_order_release);
    output_wait_->cv.notify_all();
}

void NormalGenerateStream::updateOutput(const StreamUpdateInfo& update_info) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // TODO(xinfei.sxf) consider the case of pd-sep first token finished.

    if (update_info.loss.defined()) {
        setLoss(update_info.loss);
    }

    // TODO(wangyin.yx): check behaviour of update_info.hidden_states under mtp/eagle model
    if (needReturnHiddenStates() && update_info.all_hidden_states.defined()) {
        last_hidden_states_ = update_info.all_hidden_states;
    }

    if (generate_input_->generate_config->return_softmax_probs && update_info.softmax_probs.defined()) {
        RTP_LLM_CHECK(update_info.softmax_probs.dim() == 2);
        RTP_LLM_CHECK(update_info.softmax_probs.size(1) == update_info.num_new_tokens);
        setSoftmaxProbs(update_info.softmax_probs, seqLength() - update_info.num_new_tokens);
    }

    finished_ = needFinish();
    if (finished_) {
        reportEventWithoutLock(StreamEvents::GenerateDone);
        fillSubGenerateStatus(StreamState::FINISHED);
    }
    if (update_info.cum_log_probs.defined()) {
        cum_log_probs_ = update_info.cum_log_probs.cpu();
    }
    if (update_info.all_probs.defined()) {
        all_probs_ = update_info.all_probs.cpu();
    }

    // TODO: move it to better position
    RTP_LLM_LOG_DEBUG("stream [%s] finished: %d, pd_sep: %d, is_streaming: %d, need_remote_generate: %d",
                      streamLogTag().c_str(),
                      finished_,
                      queryPdSep(),
                      isStreaming(),
                      update_info.update_remote_generate);

    if (queryPdSep() && update_info.update_remote_generate) {
        RTP_LLM_LOG_DEBUG("stream [%s] hold kv cache for pd-sep", streamLogTag().c_str());
        holdKVCacheForPDSep();
        if (!finished_) {
            RTP_LLM_LOG_DEBUG("stream [%s] set need_remote_generate", streamLogTag().c_str());
            reportEventWithoutLock(StreamEvents::NeedRemoteGenerate);
            reportEventWithoutLock(StreamEvents::GenerateDone);
        }
    }

    bool pd_sep_first_token = queryPdSep();
    bool need_update        = pd_sep_first_token || isStreaming() || finished_;
    if (!need_update) {
        return;
    }

    if (seqLength() - last_output_pos_ == 0) {
        return;
    }

    RTP_LLM_LOG_DEBUG("stream [%s] enqueue generate output", streamLogTag().c_str());
    enqueueGenerateOutput(prepareGenerateOutput(update_info));

    if (hasError()) {
        return;
    }

    last_output_pos_ = seqLength();
}
};  // namespace rtp_llm
