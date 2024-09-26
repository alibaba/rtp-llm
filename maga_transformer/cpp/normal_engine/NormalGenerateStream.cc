#include "maga_transformer/cpp/normal_engine/NormalGenerateStream.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

namespace ft = fastertransformer;

namespace rtp_llm {

absl::StatusOr<GenerateOutputs> NormalGenerateStream::nextOutput() {
    // TODO(xinfei.sxf) 某些case下会出现1s的等待
    while ((!stopped()) && !finished() && generate_outputs_queue_.isEmpty()) {
        generate_outputs_queue_.waitNotEmpty();
    }
    if (stopped()) {
        std::lock_guard<std::mutex> lock(*output_mutex_);
        return absl::Status(generate_status_.error_code, generate_status_.error_info);
    }
    if (generate_outputs_queue_.isEmpty()) {
        return absl::InternalError("no output any more");
    }
    return generate_outputs_queue_.getAndPopFront();
}

void NormalGenerateStream::updateOutput(const ft::BufferPtr& new_tokens,
                                        const ft::BufferPtr& hidden_states,
                                        const ft::BufferPtr& logits,
                                        const ft::BufferPtr& cum_log_probs,
                                        const ft::BufferPtr& all_probs) {
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    bool finished = needFinish();
    if (finished) {
        setFinishedWithoutLock();
    }

    if (!isStreaming() && !finished) {
        return;
    }

    if (cum_log_probs) {
        device_->copy({*cum_log_probs_, *cum_log_probs});
    }

    size_t output_len = seq_length_ - last_output_pos_;
    generate_outputs_->generate_outputs.clear();
    for (int i = 0; i < tileNum(); i++) {
        GenerateOutput generate_output;
        generate_output.aux_info.iter_count      = iter_count_;
        generate_output.aux_info.fallback_tokens = fallback_blocks_ * seqSizePerBlock();
        generate_output.aux_info.fallback_times  = fallback_times_;

        generate_output.output_ids =
            device_->allocateBuffer({ft::DataType::TYPE_INT32, {1lu, output_len}, ft::AllocationType::HOST}, {});
        // TODO(xinfei.sxf) optimize this copy : only copy last token
        memcpy(generate_output.output_ids->data(),
               complete_token_ids_->view(i, 1).dataWithOffset<int32_t>(last_output_pos_),
               sizeof(int32_t) * output_len);
        if (generate_input_->generate_config->return_logits && logits) {
            ft::BufferPtr host_logits;
            if (logits->shape()[0] == 1) {
                host_logits = device_->clone({*logits, ft::AllocationType::HOST});
            } else {
                host_logits = device_->clone({logits->view(i, 1), ft::AllocationType::HOST});
            }
            if (!generate_input_->generate_config->select_tokens_id.empty()) {
                auto select_buf        = ft::vector2Buffer(generate_input_->generate_config->select_tokens_id);
                generate_output.logits = device_->select({*host_logits, *select_buf, 1});
            } else {
                // TODO(xinfei.sxf) not set logits in middle step for streaming
                generate_output.logits = host_logits;
            }
        }

        if (generate_input_->generate_config->return_hidden_states && hidden_states) {
            if (hidden_states->shape()[0] == 1) {
                generate_output.hidden_states = device_->clone({*hidden_states, ft::AllocationType::HOST});
            } else {
                generate_output.hidden_states = device_->clone({hidden_states->view(i, 1), ft::AllocationType::HOST});
            }
        }
        if (loss_) {
            FT_CHECK_WITH_INFO(loss_index_ == inputLength() - 1, "loss index should be input len [%d] - 1 but is [%d]", inputLength(), loss_index_);
            auto loss = loss_;
            if (generate_input_->generate_config->calculate_loss == 1) {
                loss = device_->clone({*ft::torchTensor2Buffer(torch::mean(ft::Buffer2torchTensor(*loss_)).exp()), ft::AllocationType::HOST});
            }
            generate_output.loss = loss;
        }

        generate_output.finished              = sub_generate_status_[i].status == GenerateState::FINISHED;
        generate_output.aux_info.cost_time_us = autil::TimeUtility::currentTimeInMicroSeconds() - begin_time_us_;
        generate_output.aux_info.input_len    = generate_input_->promptLength();
        generate_output.aux_info.prefix_len   = generate_input_->prefix_length;
        // TODO(xinfei.sxf) 提前结束的query，output len要设置正确
        generate_output.aux_info.output_len   = seq_length_ - generate_input_->inputLength();
        generate_output.aux_info.step_output_len = output_len;;
        generate_output.aux_info.reuse_len    = reuse_length_;

        generate_output.aux_info.cum_log_probs =
            device_->allocateBuffer({ft::DataType::TYPE_FP32, {1lu}, ft::AllocationType::HOST}, {});

        if (cum_log_probs) {
            memcpy(generate_output.aux_info.cum_log_probs.value()->data(),
                cum_log_probs_->dataWithOffset<float>(i),
                sizeof(float));
        }

        generate_outputs_->generate_outputs.push_back(generate_output);
    }
    if (generate_outputs_queue_.getSize() >= generate_outputs_queue_.getCapacity()) {
        /* No matter if the queue is full for any reason,
           the stream will be set to stop directly to prevent the push to queue from getting stuck. */
        setStop("queue is full");
        return;
    } else {
        generate_outputs_queue_.push(*generate_outputs_);
    }
    last_output_pos_ = seq_length_;
}
};  // namespace rtp_llm