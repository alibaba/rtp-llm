#pragma once
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <cstddef>

namespace rtp_llm {

class ProposeStream: public GenerateStream {
public:
    ProposeStream(const GenerateStream& stream, size_t propose_step): GenerateStream(stream) {
        // WARNING: VanillaStream currently only support batch_size = 1
        RTP_LLM_CHECK(maxBatchSize() == 1);
        std::shared_ptr<GenerateConfig>& generate_config = generateConfig();
        if (!generate_config->top1()) {
            setReturnAllProbs(true);
        }
        if (generate_config->top_k == 0 && generate_config->top_p > 0.0) {
            generate_config->top_k = 20;
        }
        sp_output_buffer_               = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_output_buffer_->propose_step = propose_step;
        CopyOnWrite(stream, false, true);

        updateStream(stream, propose_step);

        setMetricsReporter(nullptr);
        setGenTimeline(false);
    }

    ErrorResult<GenerateOutputs> nextOutput() override {
        RTP_LLM_FAIL("VanillaStream::nextOutput should not be called");
        return ErrorInfo::OkStatus();
    };

    void updateStream(const GenerateStream& stream, size_t propose_step) {
        current_step_                   = 0;
        sp_output_buffer_->propose_step = propose_step;

        if (propose_step > history_max_propose_len_) {
            sp_output_buffer_->tokens = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, propose_step}, rtp_llm::AllocationType::HOST}, {"propose tokens"});
        }

        complete_token_ids_->setSeqLength(stream.seqLength());
        handleBounsToken();
        history_max_propose_len_ = std::max(propose_step, history_max_propose_len_);
    }

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
    }

private:
    void handleBounsToken() {
        if (acceped_bouns_token_) {
            setReuseLength(seqLength() - 2);
            setIsContextStream(true);
        }
    }

protected:
    size_t current_step_            = 0;
    size_t history_max_propose_len_ = 0;
};

}  // namespace rtp_llm
