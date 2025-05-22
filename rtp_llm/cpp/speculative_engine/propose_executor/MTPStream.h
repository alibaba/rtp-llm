#pragma once

#include "rtp_llm/cpp/stream/GenerateStream.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/ProposeOutput.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
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
    MTPStream(const GenerateStream& stream,
              ProposeOutputPtr      propose_output,
              size_t                propose_step):
              GenerateStream(stream),
              propose_output_(propose_output),
              // here MTP only support next next token predict.
              propose_step_(propose_step)
    {
        // we need to check hidden states is not empty,
        // that this stream pass pre-base-model compute.
        RTP_LLM_CHECK(getLastHiddenStates() != nullptr);
        RTP_LLM_CHECK(!isContextStream());

        // we do not support chunk prefill and batch stream now.
        RTP_LLM_CHECK(!isChunkStream());
        RTP_LLM_CHECK(tileNum() == 1);


        // get token ids
        size_t total_token_num = seqLength();
        size_t input_token_num = total_token_num - mtp_token_index_ - 1;
        // check hidden states num is equal to token ids
        RTP_LLM_CHECK_WITH_INFO(getLastHiddenStates()->shape()[0] == input_token_num,
            "hidde_states_num: %d, input_token_num: %d",
            getLastHiddenStates()->shape()[0], input_token_num);
        // change prefix len
        if (input_token_num > 1) {
            setReuseLength(mtp_token_index_);
            setFallbackPrefixLength(0);
            setIsContextStream(true);
        }

        allocateOutputBuffer(propose_step);
        shiftRightOneToken();
        // after success create mtp stream, we need to change mtp_token_index_
        GenerateStream& stream_ = const_cast<GenerateStream&>(stream);
        mtp_token_index_ += input_token_num;
        stream_.setMtpTokenIndex(mtp_token_index_);
        setMetricsReporter(nullptr);
        setNeedReleaseResource(false);
        setGenerateConfig();
        RTP_LLM_LOG_DEBUG("\nsuccess create MTP stream: %s", debugString().c_str());

    };

    ~MTPStream() {
        if (old_top_k_ >= 0) {
            generateConfig()->top_k = old_top_k_;
        }
    };

    ErrorResult<GenerateOutputs> nextOutput() override {
        RTP_LLM_FAIL("MTPStream::nextOutput should not be called");
        return ErrorInfo::OkStatus();
    };

    void updateOutput(const StreamUpdateInfo& update_info) override {
        SpeculativeExecutorStreamOutputPtr output_buffer = propose_output_->outputs[streamId()];
        if (update_info.all_probs) {
            // lazy allocate buffer
            if (!output_buffer->all_probs) {
                size_t vocab_size         = update_info.all_probs->shape()[1];
                output_buffer->all_probs  = device_->allocateBuffer(
                    {rtp_llm::DataType::TYPE_FP32, {propose_step_, vocab_size}, rtp_llm::AllocationType::DEVICE}, {"mtp_all_probs"});
            }
            device_->copy({output_buffer->all_probs->view(current_step_, 1), *update_info.all_probs});
        }
        *((*output_buffer->tokens)[0].dataWithOffset<int>(current_step_)) = ((int*)update_info.new_tokens->data())[0];
        current_step_++;

        RTP_LLM_CHECK(update_info.all_hidden_states != nullptr);
        last_hidden_states_ = device_->clone(
            {*update_info.all_hidden_states, rtp_llm::AllocationType::DEVICE});
    };


    void shiftRightOneToken() {
        size_t total_token_num = seqLength();
        // next next mtp need to delete first token
        auto new_token_ids = getLatestTokens(total_token_num - 1);
        // re init token ids
        std::shared_ptr<GenerateInput> fake_input = std::make_shared<GenerateInput>();
        fake_input->generate_config               = std::make_shared<GenerateConfig>();
        fake_input->input_ids                     = device_->clone({*rtp_llm::vector2Buffer<int32_t>(new_token_ids),
                                                                    rtp_llm::AllocationType::HOST});
        complete_token_ids_                       = std::make_shared<CompleteTokenIds>(device_,
                                                                                       tileNum(),
                                                                                       max_seq_len_,
                                                                                       seqSizePerBlock());
        complete_token_ids_->init(fake_input);
        RTP_LLM_LOG_DEBUG("\nMTP stream fake input is %s", fake_input->debugString().c_str());
    }

    void updatePrefixLen() {
        // get token ids
        size_t last_hidden_size = getLastHiddenStates()->shape()[0];
        size_t last_mtp_index = mtp_token_index_ - last_hidden_size;
        size_t total_token_num = seqLength();
        size_t input_token_num = total_token_num - last_mtp_index - 1;
        RTP_LLM_LOG_DEBUG("updatePrefixLen: last_mtp_index: %d, input_token_num: %d", last_mtp_index, input_token_num);
        if (input_token_num < last_hidden_size) {
            setReuseLength(last_mtp_index);
            setFallbackPrefixLength(0);
            setIsContextStream(true);
        }
    }

    void setGenerateConfig() {
        std::shared_ptr<GenerateConfig>& generate_config = generateConfig();
        old_top_k_= generate_config->top_k;
        if (!generate_config->top1()) {
            setReturnAllProbs(true);
        }
    }

private:
    void allocateOutputBuffer(size_t propose_step) {
        propose_output_->outputs[streamId()]->tokens = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {1, propose_step}, rtp_llm::AllocationType::HOST}, {"mtp propose tokens"});
    }

private:
    int old_top_k_ = -1;
    ProposeOutputPtr propose_output_;
    size_t current_step_ = 0;
    size_t propose_step_ = 0;

};
}  // namespace rtp_llm
