#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {

ErrorResult<GenerateOutputs> NormalGenerateStream::nextOutput() {
    // TODO(xinfei.sxf) 某些case下会出现1s的等待
    while ((!hasError()) && getStatus() != StreamState::FINISHED && generate_outputs_queue_.isEmpty()) {
        checkTimeout();
        generate_outputs_queue_.waitNotEmpty();
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
        if (generate_input_->generate_config->return_logprobs) {
            RTP_LLM_CHECK(last_output_pos_ >= (size_t)inputLength());
            const int64_t total_output_len   = seqLength() - inputLength();
            const int64_t chunk_output_start = last_output_pos_ - inputLength();
            RTP_LLM_CHECK(logprobs_history_size_ <= total_output_len);
            const int64_t content_output_start = total_output_len - logprobs_history_size_;
            const int64_t compact_output_start = std::max<int64_t>(chunk_output_start, content_output_start);
            const int64_t logprobs_offset      = compact_output_start - chunk_output_start;
            const int64_t logprobs_count       = total_output_len - compact_output_start;
            RTP_LLM_CHECK(logprobs_offset >= 0 && logprobs_offset <= static_cast<int64_t>(output_len));
            RTP_LLM_CHECK(logprobs_count >= 0 && logprobs_offset + logprobs_count == static_cast<int64_t>(output_len));
            generate_output.logprobs_offset = static_cast<int32_t>(logprobs_offset);
            generate_output.logprobs_count  = static_cast<int32_t>(logprobs_count);

            if (logprobs_count > 0) {
                RTP_LLM_CHECK(token_logprobs_.defined());
                RTP_LLM_CHECK(top_logprob_token_ids_.defined());
                RTP_LLM_CHECK(top_logprobs_.defined());
                const int64_t history_start = compact_output_start - content_output_start;
                RTP_LLM_CHECK(history_start + logprobs_count <= logprobs_history_size_);
                RTP_LLM_CHECK(logprobs_history_size_ <= token_logprobs_.size(1));
                generate_output.token_logprobs = token_logprobs_[i].narrow(0, history_start, logprobs_count).clone();
                generate_output.top_logprob_token_ids =
                    top_logprob_token_ids_[i].narrow(0, history_start, logprobs_count).clone();
                generate_output.top_logprobs = top_logprobs_[i].narrow(0, history_start, logprobs_count).clone();
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
            generate_output.aux_info.output_len       = seqLength() - generate_input_->inputLength();
            generate_output.aux_info.step_output_len  = output_len;
            generate_output.aux_info.reuse_len        = initial_reuse_length_;
            generate_output.aux_info.pd_sep           = queryPdSep();
            generate_output.aux_info.local_reuse_len  = local_reuse_length_;
            generate_output.aux_info.remote_reuse_len = remote_reuse_length_;
            generate_output.aux_info.memory_reuse_len = memory_reuse_length_;
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

    finished_ = needFinish(update_info.num_new_tokens);
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
    if (generate_input_->generate_config->return_logprobs) {
        RTP_LLM_CHECK(update_info.logprobs_offset >= 0);
        RTP_LLM_CHECK(update_info.logprobs_offset <= update_info.num_new_tokens);
        const int start_pos =
            update_info.output_start_pos >= 0 ? update_info.output_start_pos : seqLength() - update_info.num_new_tokens;
        const int committed_num_new_tokens = std::max(0, seqLength() - start_pos);
        RTP_LLM_CHECK(committed_num_new_tokens <= update_info.num_new_tokens);
        const int committed_logprobs_offset = std::min(update_info.logprobs_offset, committed_num_new_tokens);
        const int committed_logprobs_count  = committed_num_new_tokens - committed_logprobs_offset;
        if (update_info.token_logprobs.defined()) {
            RTP_LLM_CHECK(update_info.top_logprob_token_ids.defined());
            RTP_LLM_CHECK(update_info.top_logprobs.defined());
            const int expected_logprobs_count = update_info.num_new_tokens - update_info.logprobs_offset;
            RTP_LLM_CHECK(update_info.token_logprobs.size(1) == expected_logprobs_count);
            RTP_LLM_CHECK(update_info.top_logprob_token_ids.size(1) == expected_logprobs_count);
            RTP_LLM_CHECK(update_info.top_logprobs.size(1) == expected_logprobs_count);
        }
        if (committed_logprobs_count > 0) {
            RTP_LLM_CHECK(update_info.token_logprobs.defined());
            setLogProbs(update_info.token_logprobs.narrow(1, 0, committed_logprobs_count),
                        update_info.top_logprob_token_ids.narrow(1, 0, committed_logprobs_count),
                        update_info.top_logprobs.narrow(1, 0, committed_logprobs_count),
                        update_info.src_batch_indices,
                        start_pos + committed_logprobs_offset);
        }
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
