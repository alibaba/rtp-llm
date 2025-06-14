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

    for (int i = 0; i < tileNum(); i++) {
        GenerateOutput generate_output;
        generate_output.aux_info.iter_count      = iter_count_;
        generate_output.aux_info.fallback_tokens = fallback_blocks_ * seqSizePerBlock();
        generate_output.aux_info.fallback_times  = fallback_times_;
        generate_output.output_ids = SAFE_CACHED_HOST_BUF(TYPE_INT32, {1lu, output_len});

        // TODO(xinfei.sxf) optimize this copy : only copy last token
        complete_token_ids_->copyTokensTo(i, generate_output.output_ids->data(), last_output_pos_, output_len);
        if (returnLogits() && update_info.logits) {
            rtp_llm::BufferPtr host_logits;
            if (update_info.logits->shape()[0] == 1) {
                host_logits = device_->clone({*update_info.logits, rtp_llm::AllocationType::HOST});
            } else {
                host_logits = device_->clone({update_info.logits->view(i, 1), rtp_llm::AllocationType::HOST});
            }
            if (!generate_input_->generate_config->select_tokens_id.empty()) {
                auto select_buf        = rtp_llm::vector2Buffer(generate_input_->generate_config->select_tokens_id);
                generate_output.logits = device_->select({*host_logits, *select_buf, 1});
            } else {
                // TODO(xinfei.sxf) not set logits in middle step for streaming
                generate_output.logits = host_logits;
            }
        }

        if (generate_input_->generate_config->return_hidden_states && update_info.hidden_states) {
            if (update_info.hidden_states->shape()[0] == 1) {
                generate_output.hidden_states = device_->clone({*update_info.hidden_states, rtp_llm::AllocationType::HOST});
            } else {
                generate_output.hidden_states = device_->clone({update_info.hidden_states->view(i, 1), rtp_llm::AllocationType::HOST});
            }
        }
        if (loss_) {
            RTP_LLM_CHECK_WITH_INFO(loss_index_ == inputLength() - 1,
                               "loss index should be input len [%d] - 1 but is [%d]",
                               inputLength(),
                               loss_index_);
            auto loss = loss_;
            if (generate_input_->generate_config->calculate_loss == 1) {
                loss = device_->clone({*rtp_llm::torchTensor2Buffer(torch::mean(rtp_llm::Buffer2torchTensor(*loss_)).exp()),
                                       rtp_llm::AllocationType::HOST});
            }
            generate_output.loss = loss;
        }

        if (generate_input_->generate_config->return_softmax_probs && softmax_probs_) {
            generate_output.aux_info.softmax_probs = device_->clone({(*softmax_probs_)[i].view(last_output_pos_, output_len), rtp_llm::AllocationType::HOST});
        }

        generate_output.finished              = sub_generate_status_[i].status == StreamState::FINISHED;
        generate_output.aux_info.cost_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
        generate_output.aux_info.first_token_cost_time_us = complete_token_ids_->firstTokenLatencyUs();
        generate_output.aux_info.wait_time_us = wait_time_us_;
        generate_output.aux_info.input_len    = generate_input_->promptLength();
        generate_output.aux_info.prefix_len   = generate_input_->prefix_length;
        // TODO(xinfei.sxf) 提前结束的query，output len要设置正确
        generate_output.aux_info.output_len         = seqLength() - generate_input_->inputLength();
        generate_output.aux_info.step_output_len    = output_len;
        generate_output.aux_info.reuse_len          = reuse_length_;
        generate_output.aux_info.pd_sep             = queryPdSep();
        generate_output.aux_info.cum_log_probs = SAFE_CACHED_HOST_BUF(TYPE_FP32, {1lu});

        if (update_info.cum_log_probs) {
            memcpy(generate_output.aux_info.cum_log_probs.value()->data(),
                   cum_log_probs_->dataWithOffset<float>(i),
                   sizeof(float));
        }

        if (generate_input_->generate_config->return_all_probs) {
            if (!update_info.all_probs) {
                throw std::runtime_error("all_probs is not while generate_config return_all_probs is true");
            }
            generate_output.aux_info.all_probs = device_->clone(
                {all_probs_->view(i, 1), rtp_llm::AllocationType::HOST});
        }

        generate_results.generate_outputs.emplace_back(std::move(generate_output));
    }
    return generate_results;
}

void NormalGenerateStream::enqueueGenerateOutput(GenerateOutputs &&generate_results) {
    if (generate_outputs_queue_.getSize() >= generate_outputs_queue_.getCapacity()) {
        /* No matter if the queue is full for any reason,
           the stream will be set to stop directly to prevent the push to queue from getting stuck. */
        setStop(ErrorCode::OUTPUT_QUEUE_FULL, "output queue is full");
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

    if (needReturnHiddenStates()) {
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
        device_->copy({*cum_log_probs_, *update_info.cum_log_probs});
    }
    if (update_info.all_probs) {
        all_probs_ = device_->clone({*update_info.all_probs, rtp_llm::AllocationType::HOST});
    }

    //TODO: move it to better position
    if (!finished_ && queryPdSep() && update_info.update_remote_generate) {
        need_remote_generate_ = true;
    }

    bool pd_sep_first_token = queryPdSep();
    bool need_update = pd_sep_first_token || isStreaming() || finished_;
    if (!need_update) {
        return;
    }

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
