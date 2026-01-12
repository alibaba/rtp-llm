#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

ErrorResult<GenerateOutputs> NormalGenerateStream::nextOutput() {
    // TODO(xinfei.sxf) 某些case下会出现1s的等待
    while ((!stopped()) && !finished() && generate_outputs_queue_.isEmpty()) {
        checkTimeout();
        generate_outputs_queue_.waitNotEmpty();
    }
    if (stopped()) {
        return statusInfo();
    }
    if (generate_outputs_queue_.isEmpty()) {
        if (finished()) {
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
        generate_output.output_ids          = SAFE_CACHED_HOST_BUF(TYPE_INT32, {1lu, output_len});

        // TODO(xinfei.sxf) optimize this copy : only copy last token
        complete_token_ids_->copyTokensTo(i, generate_output.output_ids->data(), last_output_pos_, output_len);
        if (returnLogits() && update_info.logits) {
            rtp_llm::BufferPtr logits_result;
            const auto&        select_tokens_id = generate_input_->generate_config->select_tokens_id;
            if (!select_tokens_id.empty()) {
                auto out_of_bound_token_id =
                    std::find_if_not(select_tokens_id.begin(), select_tokens_id.end(), [this](int select_token_id) {
                        return select_token_id >= 0 && select_token_id < vocabSize();
                    });
                if (out_of_bound_token_id == select_tokens_id.end()) {
                    auto select_buf = rtp_llm::vector2Buffer(generate_input_->generate_config->select_tokens_id);
                    logits_result   = device_->select({*update_info.logits, *select_buf, 1});
                } else {
                    RTP_LLM_LOG_WARNING("select_token_id out of bound, expected >= 0 and < vocab size [%d], found [%d]",
                                        vocabSize(),
                                        *out_of_bound_token_id);
                    logits_result =
                        device_->allocateBuffer({update_info.logits->type(), {0, 0}, rtp_llm::AllocationType::DEVICE});
                }
            } else {
                logits_result = update_info.logits;
            }
            if (logits_result->shape()[0] <= 1) {
                generate_output.logits = device_->clone({*logits_result, rtp_llm::AllocationType::HOST});
            } else {
                generate_output.logits = device_->clone({logits_result->view(i, 1), rtp_llm::AllocationType::HOST});
            }
        }

        if (generate_input_->generate_config->return_hidden_states && update_info.hidden_states) {
            if (update_info.hidden_states->shape()[0] == 1) {
                generate_output.hidden_states =
                    device_->clone({*update_info.hidden_states, rtp_llm::AllocationType::HOST});
            } else {
                generate_output.hidden_states =
                    device_->clone({update_info.hidden_states->view(i, 1), rtp_llm::AllocationType::HOST});
            }
        }
        if (generate_input_->generate_config->return_all_hidden_states && update_info.all_hidden_states
            && iter_count_ == 1) {
            generate_output.all_hidden_states =
                device_->clone({*update_info.all_hidden_states, rtp_llm::AllocationType::HOST});
        }
        if (loss_) {
            RTP_LLM_CHECK_WITH_INFO(loss_index_ == inputLength() - 1,
                                    "loss index should be input len [%d] - 1 but is [%d]",
                                    inputLength(),
                                    loss_index_);
            auto loss = loss_;
            if (generate_input_->generate_config->calculate_loss == 1) {
                loss = device_->clone(
                    {*rtp_llm::torchTensor2Buffer(torch::mean(rtp_llm::Buffer2torchTensor(*loss_)).exp()),
                     rtp_llm::AllocationType::HOST});
            }
            generate_output.loss = loss;
        }

        generate_output.finished = sub_generate_status_[i].status == StreamState::FINISHED;
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
            generate_output.aux_info.reuse_len        = reuse_info_->reuse_length;
            generate_output.aux_info.pd_sep           = queryPdSep();
            generate_output.aux_info.local_reuse_len  = reuse_info_->local_reuse_length;
            generate_output.aux_info.remote_reuse_len = reuse_info_->remote_reuse_length;

            if (generate_input_->generate_config->return_softmax_probs && softmax_probs_) {
                generate_output.aux_info.softmax_probs = device_->clone(
                    {(*softmax_probs_)[i].view(last_output_pos_, output_len), rtp_llm::AllocationType::HOST});
            }
            if (update_info.cum_log_probs) {
                generate_output.aux_info.cum_log_probs = SAFE_CACHED_HOST_BUF(TYPE_FP32, {1lu});
                memcpy(generate_output.aux_info.cum_log_probs.value()->data(),
                       cum_log_probs_->dataWithOffset<float>(i),
                       sizeof(float));
            }
            if (generate_input_->generate_config->return_all_probs) {
                if (!update_info.all_probs) {
                    throw std::runtime_error("all_probs is not while generate_config return_all_probs is true");
                }
                generate_output.aux_info.all_probs =
                    device_->clone({all_probs_->view(i, 1), rtp_llm::AllocationType::HOST});
            }
        }
        // hidden_states post process
        if (generate_output.finished && generate_input_->generate_config->return_hidden_states
            && generate_output.hidden_states.has_value()
            && (generate_input_->generate_config->hidden_states_cut_dim > 0
                || generate_input_->generate_config->normalized_hidden_states)) {
            auto buffer               = generate_output.hidden_states.value();
            auto hidden_states_tensor = rtp_llm::Buffer2torchTensor(buffer);
            if (generate_input_->generate_config->hidden_states_cut_dim > 0) {
                hidden_states_tensor = hidden_states_tensor.index(
                    {torch::indexing::Slice(),
                     torch::indexing::Slice(0, generate_input_->generate_config->hidden_states_cut_dim)});
            }
            if (generate_input_->generate_config->normalized_hidden_states) {
                hidden_states_tensor = torch::nn::functional::normalize(
                    hidden_states_tensor, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
            }
            generate_output.hidden_states =
                device_->clone({*rtp_llm::torchTensor2Buffer(hidden_states_tensor), rtp_llm::AllocationType::HOST});
        }

        generate_results.generate_outputs.emplace_back(std::move(generate_output));
    }
    return generate_results;
}

void NormalGenerateStream::enqueueGenerateOutput(GenerateOutputs&& generate_results) {
    if (generate_outputs_queue_.getSize() >= generate_outputs_queue_.getCapacity()) {
        /* No matter if the queue is full for any reason,
           the stream will be set to stop directly to prevent the push to queue from getting stuck. */
        setStopWithoutLock(ErrorCode::OUTPUT_QUEUE_FULL, "output queue is full");
    } else {
        generate_outputs_queue_.push(std::move(generate_results));
    }
}

void NormalGenerateStream::updateOutput(const StreamUpdateInfo& update_info) {
    RTP_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // TODO(xinfei.sxf) consider the case of pd-sep first token finished.

    if (update_info.loss) {
        setLoss(*update_info.loss);
    }

    // TODO(wangyin.yx): check behaviour of update_info.hidden_states under mtp/eagle model
    if (needReturnHiddenStates() && update_info.all_hidden_states) {
        RTP_LLM_CHECK(update_info.all_hidden_states != nullptr);
        last_hidden_states_ = update_info.all_hidden_states;
    }

    if (generate_input_->generate_config->return_softmax_probs && update_info.softmax_probs) {
        RTP_LLM_CHECK(update_info.softmax_probs->dim() == 2);
        RTP_LLM_CHECK(update_info.softmax_probs->shape()[1] == update_info.num_new_tokens);
        setSoftmaxProbs(*update_info.softmax_probs, seqLength() - update_info.num_new_tokens);
    }

    finished_ = needFinish();
    if (finished_) {
        setFinishedWithoutLock();
    }
    if (update_info.cum_log_probs) {
        cum_log_probs_ = device_->clone({*update_info.cum_log_probs, rtp_llm::AllocationType::HOST});
    }
    if (update_info.all_probs) {
        all_probs_ = device_->clone({*update_info.all_probs, rtp_llm::AllocationType::HOST});
    }

    // TODO: move it to better position
    RTP_LLM_LOG_DEBUG("stream [%ld] finished: %d, pd_sep: %d, is_streaming: %d, need_remote_generate: %d",
                      streamId(),
                      finished_,
                      queryPdSep(),
                      isStreaming(),
                      update_info.update_remote_generate);
    if (!finished_ && queryPdSep() && update_info.update_remote_generate) {
        RTP_LLM_LOG_DEBUG("stream [%ld] set need_remote_generate", streamId());
        setNeedRemoteGenerateWithoutLock(true);
    }

    bool pd_sep_first_token = queryPdSep();
    bool need_update        = pd_sep_first_token || isStreaming() || finished_;
    if (!need_update) {
        return;
    }

    RTP_LLM_LOG_INFO("seq len is %d last_output_pos %d", seqLength(), last_output_pos_);
    if (seqLength() - last_output_pos_ == 0) {
        return;
    }

    RTP_LLM_LOG_DEBUG("stream [%ld] enqueue generate output", streamId());
    enqueueGenerateOutput(prepareGenerateOutput(update_info));

    if (stoppedWithoutLock()) {
        return;
    }

    last_output_pos_ = seqLength();
}
};  // namespace rtp_llm
