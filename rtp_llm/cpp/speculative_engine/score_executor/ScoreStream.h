#pragma once
#include "rtp_llm/cpp/stream/GenerateStream.h"
#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include <cstddef>



namespace rtp_llm {

class ScoreStream: public GenerateStream {
public:
    ScoreStream(const GenerateStream&                     stream,
                size_t                                    propose_step,
                rtp_llm::BufferPtr*                            propose_tokens,
                ScoreOutput*                              score_output):
        GenerateStream(stream),
        propose_step_(propose_step),
        score_len_(propose_step == 0 ? 1 : propose_step + 1),
        score_output_(score_output) {
        // WARNING: ScoreStream currently only support batch_size = 1
        RTP_LLM_CHECK(tileNum() == 1);
        CopyOnWrite(stream);
        updateProposeTokens(propose_tokens, propose_step_);
        allocateOutputBuffer();
        setNeedReleaseResource(false);

        if (!stream.generateConfig()->top1()) {
            setReturnAllProbs(true);
        }
    }

    ~ScoreStream() {}

    ErrorResult<GenerateOutputs> nextOutput() override {
        RTP_LLM_FAIL("ScoreStream::nextOutput should not be called");
        return ErrorInfo::OkStatus();
    };

    void updateOutput(const StreamUpdateInfo& update_info) override {
        SpeculativeExecutorStreamOutputPtr output_buffer = score_output_->outputs[streamId()];
        device_->copy({(*output_buffer->tokens)[0], (*update_info.new_tokens)[0]});

        // TODO(xyz): optimize deepclone
        if (update_info.all_probs) {
            output_buffer->all_probs = device_->clone({*update_info.all_probs, rtp_llm::AllocationType::DEVICE, {"score_all_probs"}});
        }

        if (generate_input_->generate_config->return_logits) {
            output_buffer->logits = device_->clone({*update_info.logits, rtp_llm::AllocationType::DEVICE, {"score_logits"}});
        }
        if (generate_input_->generate_config->return_softmax_probs) {
            output_buffer->softmax_probs = device_->clone({*update_info.softmax_probs, rtp_llm::AllocationType::DEVICE, {"softmax_probs"}});
        }

        if (needReturnHiddenStates()) {
            RTP_LLM_CHECK(update_info.all_hidden_states != nullptr);
            output_buffer->hidden_states = device_->clone({*update_info.all_hidden_states, rtp_llm::AllocationType::DEVICE, {"score_hidden_states"}});
        }

        if (update_info.loss) {
            output_buffer->loss = device_->clone({*update_info.loss, rtp_llm::AllocationType::DEVICE, {"score_loss"}});
        }
    }

    size_t scoreLen() const override {
        return score_len_;
    }

private:
    void updateProposeTokens(rtp_llm::BufferPtr* propose_tokens, size_t propose_step) {
        if (!propose_tokens) {
            return;
        }
        complete_token_ids_->appendTokens(0, propose_step, **propose_tokens);
    }

    void allocateOutputBuffer() {
        score_output_->outputs[streamId()]->tokens = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, scoreLen()}, rtp_llm::AllocationType::HOST}, {"score_tokens"});
        if (scoreLen() > 1) {
            setIsContextStream(true);
        }
    }

protected:
    size_t                             propose_step_ = 0;
    size_t                             score_len_    = 0;
    ScoreOutput*                       score_output_;
};

}  // namespace rtp_llm
