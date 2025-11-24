#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <cstddef>

namespace rtp_llm {

// limits:
// this stream can only be created from norm stream with hidden states.
// we use mtp_index from norm stream to check which token is already pass mtp model with hidden states.
// we need to check hidden states num is equal to token num that split from mtp_index
// we also need to care about prefix_len/reuse_len/
// etc.
class MTPStream: public GenerateStream {
public:
    MTPStream(const GenerateStream& stream, size_t propose_step): GenerateStream(stream) {
        RTP_LLM_CHECK(maxBatchSize() == 1);
        std::shared_ptr<GenerateConfig>& generate_config = generateConfig();
        if (!generate_config->top1()) {
            setReturnAllProbs(true);
        }
        if (generate_config->top_k == 0 && generate_config->top_p > 0.0) {
            generate_config->top_k = 20;
        }
        current_step_                   = 0;
        sp_output_buffer_               = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_output_buffer_->propose_step = propose_step;

        complete_token_ids_ = std::make_shared<CompleteTokenIds>(*stream.getCompleteTokenIds(), true, 1);
        complete_token_ids_->setSeqLength(stream.seqLength() - 1);

        last_hidden_states_ = stream.getLastHiddenStates();
        if (last_hidden_states_ != nullptr) {
            updateStream(stream, propose_step);
        }

        setMetricsReporter(nullptr);
        setGenTimeline(false);
    }

    void updateStream(const GenerateStream& stream, size_t propose_step) {
        if (propose_step > 1) {
            complete_token_ids_ = std::make_shared<CompleteTokenIds>(*stream.getCompleteTokenIds(), true, 1);
        }
        current_step_                   = 0;
        last_hidden_states_             = stream.getLastHiddenStates();
        sp_output_buffer_->propose_step = propose_step;
        if (propose_step > history_max_propose_len_) {
            sp_output_buffer_->tokens = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, propose_step}, rtp_llm::AllocationType::HOST},
                {"mtp propose tokens"});
        }

        RTP_LLM_CHECK(last_hidden_states_ != nullptr);

        // get token ids
        size_t total_token_num = stream.seqLength();
        size_t input_token_num = total_token_num - mtp_token_index_ - 1;
        RTP_LLM_LOG_DEBUG("total_token_num %d, mtp_token_index: %d, input_token_num: %d",
                          total_token_num,
                          mtp_token_index_,
                          input_token_num);
        // check hidden states num is equal to token ids
        RTP_LLM_CHECK_WITH_INFO(
            last_hidden_states_->shape()[0] == input_token_num,
            "hidden states num: %d, total_token_num: %d, execute token num: %d, input_token_num: %d, mtp_token_index_: %d, raw stream msg: %s, stream msg: %s",
            last_hidden_states_->shape()[0],
            total_token_num,
            currentExecuteTokenSize(),
            input_token_num,
            mtp_token_index_,
            stream.debugString().c_str(),
            debugString().c_str());

        if (input_token_num > 1) {
            setReuseLength(mtp_token_index_);
            setIsContextStream(true);
        }

        complete_token_ids_->setSeqLength(stream.seqLength() - 1);
        // after success create mtp stream, we need to change mtp_token_index_
        mtp_token_index_ += input_token_num;
        history_max_propose_len_ = std::max(propose_step, history_max_propose_len_);

        RTP_LLM_CHECK_WITH_INFO(
            last_hidden_states_->shape()[0] == currentExecuteTokenSize(),
            "hidden states num: %d, total_token_num: %d, execute token num: %d, input_token_num: %d, mtp_token_index: %d, raw stream msg: %s, stream msg: %s",
            last_hidden_states_->shape()[0],
            total_token_num,
            currentExecuteTokenSize(),
            input_token_num,
            mtp_token_index_,
            stream.debugString().c_str(),
            debugString().c_str());
    }

    ErrorResult<GenerateOutputs> nextOutput() override {
        RTP_LLM_FAIL("MTPStream::nextOutput should not be called");
        return ErrorInfo::OkStatus();
    };

    void updateOutput(const StreamUpdateInfo& update_info) override {
        size_t propose_step = sp_output_buffer_->propose_step;
        if (update_info.all_probs) {
            // lazy allocate buffer
            if (!sp_output_buffer_->all_probs) {
                size_t vocab_size            = update_info.all_probs->shape()[1];
                sp_output_buffer_->all_probs = device_->allocateBuffer(
                    {rtp_llm::DataType::TYPE_FP32, {propose_step, vocab_size}, rtp_llm::AllocationType::DEVICE},
                    {"mtp_all_probs"});
            }
            device_->copy({sp_output_buffer_->all_probs->view(current_step_, 1), *update_info.all_probs});
        }
        *(sp_output_buffer_->tokens->dataWithOffset<int>(current_step_)) = *((int*)update_info.new_tokens->data());
        current_step_++;

        RTP_LLM_CHECK(update_info.all_hidden_states != nullptr);
        last_hidden_states_ = update_info.all_hidden_states;
    };

    void shiftRightOneToken(const GenerateStream& stream) {
        complete_token_ids_ = std::make_shared<CompleteTokenIds>(*stream.getCompleteTokenIds(), true, 1);
        // get token ids
        size_t last_hidden_size = last_hidden_states_->shape()[0];
        size_t last_mtp_index   = mtp_token_index_ - last_hidden_size;
        size_t total_token_num  = seqLength();
        size_t input_token_num  = total_token_num - last_mtp_index - 1;
        RTP_LLM_LOG_DEBUG("last_mtp_index: %d, input_token_num: %d", last_mtp_index, input_token_num);

        RTP_LLM_CHECK_WITH_INFO(
            last_hidden_states_->shape()[0] == input_token_num,
            "hidden states num: %d, total_token_num: %d, execute token num: %d, input_token_num: %d, mtp_token_index: %d, raw stream msg: %s, stream msg: %s",
            last_hidden_states_->shape()[0],
            total_token_num,
            currentExecuteTokenSize(),
            input_token_num,
            mtp_token_index_,
            stream.debugString().c_str(),
            debugString().c_str());

        complete_token_ids_->setSeqLength(stream.seqLength() - 1);

        if (input_token_num > 1) {
            setReuseLength(last_mtp_index);
            setIsContextStream(true);
        }

        RTP_LLM_CHECK_WITH_INFO(
            last_hidden_states_->shape()[0] == currentExecuteTokenSize(),
            "hidden states num: %d, total_token_num: %d, execute token num: %d, input_token_num: %d, mtp_token_index: %d, raw stream msg: %s, stream msg: %s",
            last_hidden_states_->shape()[0],
            total_token_num,
            currentExecuteTokenSize(),
            input_token_num,
            mtp_token_index_,
            stream.debugString().c_str(),
            debugString().c_str());
    }

protected:
    size_t history_max_propose_len_ = 0;
    size_t current_step_            = 0;
};
}  // namespace rtp_llm
