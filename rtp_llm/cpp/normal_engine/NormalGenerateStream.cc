#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

#include <algorithm>
#include <chrono>
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

ErrorResult<GenerateOutputs> NormalGenerateStream::nextOutput(int64_t wait_timeout_ms) {
    auto result = nextOutputForRpc(wait_timeout_ms);
    if (result.ok() && result.value().batched_output) {
        // nextOutputForRpc has released mutex_. Materialization only reads the
        // owned snapshot, so it cannot delay scheduling or safe KV reclamation.
        return materializeTerminalOutput(std::move(result.value()));
    }
    return result;
}

ErrorResult<GenerateOutputs> NormalGenerateStream::nextOutputForRpc(int64_t wait_timeout_ms) {
    RTP_LLM_CHECK_WITH_INFO(wait_timeout_ms >= 0, "nextOutput wait_timeout_ms must be non-negative");

    const auto stream_timeout_ms = getTimeoutMs();
    auto       stream_deadline   = std::chrono::steady_clock::time_point::max();

    std::unique_lock<std::mutex> lock(*mutex_);

    if (stream_timeout_ms > 0) {
        const auto elapsed_us   = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
        const auto remaining_us = std::max<int64_t>(stream_timeout_ms * 1000 - elapsed_us, 0);
        stream_deadline         = std::chrono::steady_clock::now() + std::chrono::microseconds(remaining_us);
    }

    if (!consumerReadyWithoutLock()) {
        if (wait_timeout_ms == 0 && stream_timeout_ms <= 0) {
            consumer_cv_->wait(lock, [this] { return consumerReadyWithoutLock(); });
        } else {
            auto wait_deadline = stream_deadline;
            if (wait_timeout_ms > 0) {
                wait_deadline = std::min(stream_deadline,
                                         std::chrono::steady_clock::now() + std::chrono::milliseconds(wait_timeout_ms));
            }

            if (!consumer_cv_->wait_until(lock, wait_deadline, [this] { return consumerReadyWithoutLock(); })) {
                if (stream_timeout_ms > 0 && std::chrono::steady_clock::now() >= stream_deadline) {
                    const auto running_time_ms =
                        (autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_) / 1000;
                    reportTimeoutWithoutLock(running_time_ms, stream_timeout_ms);
                } else {
                    return ErrorInfo(ErrorCode::OUTPUT_QUEUE_NO_UPDATE,
                                     "output queue has no update within " + std::to_string(wait_timeout_ms) + " ms");
                }
            }
        }
    }

    // Preserve existing precedence: terminal errors override queued output.
    if (hasErrorWithoutLock()) {
        return statusInfoWithoutLock();
    }

    // Normal completion is reported only after the final output is drained.
    if (!generate_outputs_.empty()) {
        auto output = std::move(generate_outputs_.front());
        generate_outputs_.pop_front();
        if (output.batched_output) {
            RTP_LLM_PROFILE_SCOPE_DYNAMIC("output.dequeue_terminal(id=%ld)", output.request_id);
        }
        return output;
    }

    if (consumerFinishedWithoutLock()) {
        return ErrorInfo(ErrorCode::FINISHED, "finished");
    }

    RTP_LLM_FAIL("consumer is ready without an error, output, or finished state");
}

bool NormalGenerateStream::hasOutput() {
    std::lock_guard<std::mutex> lock(*mutex_);
    return !generate_outputs_.empty();
}

bool NormalGenerateStream::consumerReadyWithoutLock() const {
    return hasErrorWithoutLock() || !generate_outputs_.empty() || consumerFinishedWithoutLock();
}

bool NormalGenerateStream::canDeferTerminalOutput(const StreamUpdateInfo& update_info) const {
    const auto& config = *generate_input_->generate_config;
    // First implementation: ordinary non-streaming terminal tokens/scores.
    // PD handoff and optional tensor outputs keep their established behavior.
    return finished_ && !config.is_streaming && !queryPdSep() && !config.return_logits && !config.return_prompt_logits
           && !config.return_hidden_states && !config.return_all_hidden_states && !config.return_softmax_probs
           && config.return_all_probs == ReturnAllProbsMode::NONE && config.calculate_loss == 0 && !loss_.defined()
           && !update_info.prompt_logits.has_value()
           && (!update_info.cum_log_probs.defined()
               || (cum_log_probs_.device().is_cpu() && cum_log_probs_.scalar_type() == torch::kFloat32
                   && cum_log_probs_.dim() >= 1));
}

GenerateOutputs NormalGenerateStream::snapshotTerminalOutput(const StreamUpdateInfo& update_info) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("output.snapshot_terminal(id=%ld,beams=%d)", request_id_, currentBatchSize());
    const int64_t batch_size = currentBatchSize();
    const int64_t output_len = seqLength() - last_output_pos_;
    auto          snapshot   = std::make_shared<BatchedGenerateOutput>();
    // Copy only the returned suffix (e.g. 1024 x 3 ints), not the 500-token
    // prompt/history. Independent storage also protects against forced updates
    // and speculative stream copies retaining aliases of CompleteTokenIds.
    snapshot->output_ids = complete_token_ids_->completeTokenIds()
                               .narrow(0, 0, batch_size)
                               .narrow(1, last_output_pos_, output_len)
                               .clone()
                               .unsqueeze(1);
    auto& aux      = snapshot->aux_info;
    aux.iter_count = iter_count_;
    if (generate_input_->generate_config->aux_info) {
        aux.cost_time_us             = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
        aux.first_token_cost_time_us = complete_token_ids_->firstTokenLatencyUs();
        aux.wait_time_us             = wait_time_us_;
        aux.input_len                = generate_input_->promptLength();
        aux.prefix_len               = generate_input_->prefix_length;
        aux.output_len               = seqLength() - generate_input_->inputLength();
        aux.step_output_len          = output_len;
        aux.reuse_len                = initial_reuse_length_;
        aux.pd_sep                   = queryPdSep();
        aux.local_reuse_len          = local_reuse_length_;
        aux.remote_reuse_len         = remote_reuse_length_;
        aux.memory_reuse_len         = memory_reuse_length_;
        aux.multimodal_lengths       = generate_input_->multimodalLengths();
        if (update_info.cum_log_probs.defined()) {
            // Dispatcher has completed D2H before entering update(). Clone only
            // this request's final scores to detach from reusable sampler input.
            snapshot->cum_log_probs = cum_log_probs_.narrow(0, 0, batch_size).clone(at::MemoryFormat::Contiguous);
        }
    }
    GenerateOutputs result;
    result.request_id     = request_id_;
    result.batched_output = std::move(snapshot);
    return result;
}

GenerateOutputs NormalGenerateStream::materializeTerminalOutput(GenerateOutputs output) {
    RTP_LLM_PROFILE_SCOPE_DYNAMIC("output.materialize_terminal(id=%ld)", output.request_id);
    const auto    snapshot   = std::move(output.batched_output);
    const int64_t batch_size = snapshot->output_ids.size(0);
    output.generate_outputs.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
        GenerateOutput row;
        row.finished   = true;
        row.output_ids = snapshot->output_ids.select(0, i);
        row.aux_info   = snapshot->aux_info;
        if (snapshot->cum_log_probs.defined()) {
            row.aux_info.cum_log_probs = snapshot->cum_log_probs.narrow(0, i, 1);
        }
        output.generate_outputs.emplace_back(std::move(row));
    }
    return output;
}

GenerateOutputs NormalGenerateStream::prepareGenerateOutput(const StreamUpdateInfo& update_info) {
    RTP_LLM_PROFILE_SCOPE("output.build_generate_output");
    size_t          output_len = seqLength() - last_output_pos_;
    GenerateOutputs generate_results;
    generate_results.request_id = request_id_;
    torch::Tensor all_hidden_states_cpu;

    // CompleteTokenIds has already applied this step, so currentBatchSize is the output row count.
    for (int i = 0; i < currentBatchSize(); i++) {
        GenerateOutput generate_output;
        generate_output.aux_info.iter_count = iter_count_;
        generate_output.output_ids          = torch::empty({1, (int64_t)output_len}, torch::kInt32);
        generate_output.finished            = isSubGenerateDoneWithoutLock(i);

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
        if (generate_input_->generate_config->return_all_hidden_states) {
            const torch::Tensor* all_hidden_states = nullptr;
            if (update_info.all_hidden_states.defined() && iter_count_ == 1) {
                all_hidden_states = &update_info.all_hidden_states;
            } else if (!isStreaming() && generate_output.finished && all_hidden_states_.defined()) {
                all_hidden_states = &all_hidden_states_;
            }
            if (all_hidden_states != nullptr) {
                if (!all_hidden_states_cpu.defined()) {
                    all_hidden_states_cpu = all_hidden_states->cpu();
                }
                generate_output.all_hidden_states = all_hidden_states_cpu;
            }
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

        if (update_info.prompt_logits.has_value()) {
            generate_output.prompt_logits = update_info.prompt_logits;
        }
        if (generate_input_->generate_config->aux_info) {
            generate_output.aux_info.iter_count   = iter_count_;
            generate_output.aux_info.cost_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
            generate_output.aux_info.first_token_cost_time_us = complete_token_ids_->firstTokenLatencyUs();
            generate_output.aux_info.wait_time_us             = wait_time_us_;
            generate_output.aux_info.input_len                = generate_input_->promptLength();
            generate_output.aux_info.prefix_len               = generate_input_->prefix_length;
            // TODO(xinfei.sxf) 提前结束的query，output len要设置正确
            generate_output.aux_info.output_len       = seqLength() - generate_input_->inputLength();
            generate_output.aux_info.step_output_len  = output_len;
            generate_output.aux_info.reuse_len        = initial_reuse_length_;
            generate_output.aux_info.pd_sep           = queryPdSep();
            generate_output.aux_info.local_reuse_len  = local_reuse_length_;
            generate_output.aux_info.remote_reuse_len = remote_reuse_length_;
            generate_output.aux_info.memory_reuse_len = memory_reuse_length_;

            generate_output.aux_info.multimodal_lengths = generate_input_->multimodalLengths();

            if (calculateSoftmaxProbs() && softmax_probs_.defined()) {
                generate_output.aux_info.softmax_probs =
                    softmax_probs_[i].narrow(0, last_output_pos_, output_len).clone();
            }
            if (update_info.cum_log_probs.defined()) {
                generate_output.aux_info.cum_log_probs = cum_log_probs_.narrow(0, i, 1).cpu().clone();
            }
        }
        // all_probs is returned as an independent tensor in FlattenOutputPB.
        // Disabling auxiliary metadata must not discard this explicit output.
        if (generate_input_->generate_config->return_all_probs != ReturnAllProbsMode::NONE) {
            if (!update_info.all_probs.defined()) {
                throw std::runtime_error("all_probs is not while generate_config return_all_probs is true");
            }
            generate_output.aux_info.all_probs = all_probs_.narrow(0, i, 1).clone();
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
    if (generate_outputs_.size() >= kOutputCapacity) {
        /* No matter if the queue is full for any reason,
           the stream will be set to stop directly to prevent the push to queue from getting stuck. */
        reportEventWithoutLock(StreamEvents::Error, ErrorCode::OUTPUT_QUEUE_FULL, "output queue is full");
    } else {
        generate_outputs_.push_back(std::move(generate_results));
        consumer_cv_->notify_all();
    }
}

void NormalGenerateStream::updateOutput(const StreamUpdateInfo& update_info) {
    RTP_LLM_PROFILE_SCOPE("output.update_result");
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // TODO(xinfei.sxf) consider the case of pd-sep first token finished.

    if (update_info.loss.defined()) {
        setLoss(update_info.loss);
    }

    // TODO(wangyin.yx): check behaviour of update_info.hidden_states under mtp/eagle model
    if (needReturnHiddenStates() && update_info.all_hidden_states.defined()) {
        last_hidden_states_ = update_info.all_hidden_states;
    }
    if (generate_input_->generate_config->return_all_hidden_states && update_info.all_hidden_states.defined()
        && !all_hidden_states_.defined()) {
        all_hidden_states_ = update_info.all_hidden_states;
    }

    if (calculateSoftmaxProbs() && update_info.softmax_probs.defined()) {
        RTP_LLM_CHECK(update_info.softmax_probs.dim() == 2);
        RTP_LLM_CHECK(update_info.softmax_probs.size(1) == update_info.num_new_tokens);
        setSoftmaxProbs(
            update_info.softmax_probs, seqLength() - update_info.num_new_tokens, update_info.src_batch_indices);
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
    RTP_LLM_LOG_DEBUG("stream [%ld] finished: %d, pd_sep: %d, is_streaming: %d, need_remote_generate: %d",
                      streamId(),
                      finished_,
                      queryPdSep(),
                      isStreaming(),
                      update_info.update_remote_generate);

    if (queryPdSep() && update_info.update_remote_generate) {
        // Hold KV cache even when the stream already finished in prefill
        // (e.g. stop words hit): the decode role still issues RemoteLoad for
        // these blocks and would hang if they were freed here.
        holdKVCacheForPDSep();
        if (!finished_) {
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

    RTP_LLM_LOG_DEBUG("stream [%ld] enqueue generate output", streamId());
    if (canDeferTerminalOutput(update_info)) {
        auto result = snapshotTerminalOutput(update_info);
        RTP_LLM_PROFILE_SCOPE_DYNAMIC("output.publish_terminal(id=%ld)", request_id_);
        // GenerateDone and the result become visible together when update()
        // unlocks the stream mutex. Queue draining precedes FINISHED.
        enqueueGenerateOutput(std::move(result));
    } else {
        enqueueGenerateOutput(prepareGenerateOutput(update_info));
    }

    if (hasErrorWithoutLock()) {
        return;
    }

    last_output_pos_ = seqLength();
}
};  // namespace rtp_llm
