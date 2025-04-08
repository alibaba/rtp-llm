#pragma once
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include <cstddef>

namespace ft = fastertransformer;

namespace rtp_llm {

class ScoreStream: public GenerateStream {
public:
    ScoreStream(const GenerateStream&                     stream,
                size_t                                    propose_step,
                ft::BufferPtr*                            propose_tokens,
                ScoreOutput*                              score_output):
        GenerateStream(stream),
        propose_step_(propose_step),
        score_len_(propose_step == 0 ? 1 : propose_step + 1),
        score_output_(score_output) {
        // WARNING: ScoreStream currently only support batch_size = 1
        FT_CHECK(tileNum() == 1);
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
        FT_FAIL("ScoreStream::nextOutput should not be called");
        return ErrorInfo::OkStatus();
    };

    void updateOutput(const StreamUpdateInfo& update_info) override {
        SpeculativeExecutorStreamOutputPtr output_buffer = score_output_->outputs[streamId()];
        device_->copy({(*output_buffer->tokens)[0], (*update_info.new_tokens)[0]});

        // TODO(xyz): optimize deepclone
        if (update_info.all_probs) {
            output_buffer->all_probs = device_->clone({*update_info.all_probs, ft::AllocationType::DEVICE, {"score_all_probs"}});
        }

        if (generate_input_->generate_config->return_logits) {
            output_buffer->logits = device_->clone({*update_info.logits, ft::AllocationType::DEVICE, {"score_logits"}});
        }

        if (generate_input_->generate_config->return_hidden_states) {
            output_buffer->hidden_states = device_->clone({*update_info.hidden_states, ft::AllocationType::DEVICE, {"score_hidden_states"}});
        }

        if (update_info.loss) {
            output_buffer->loss = device_->clone({*update_info.loss, ft::AllocationType::DEVICE, {"score_loss"}});
        }
    }

    size_t scoreLen() const override {
        return score_len_;
    }

private:
    void updateProposeTokens(ft::BufferPtr* propose_tokens, size_t propose_step) {
        if (!propose_tokens) {
            return;
        }
        complete_token_ids_->appendTokens(0, propose_step, **propose_tokens);
    }

    void allocateOutputBuffer() {
        score_output_->outputs[streamId()]->tokens = device_->allocateBuffer(
                {ft::DataType::TYPE_INT32, {1, scoreLen()}, ft::AllocationType::HOST}, {"score_tokens"});
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