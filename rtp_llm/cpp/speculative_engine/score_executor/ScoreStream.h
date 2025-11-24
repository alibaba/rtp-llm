#pragma once
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include <cstddef>

namespace rtp_llm {
class ScoreStream: public GenerateStream {
public:
    ScoreStream(const GenerateStream& stream): GenerateStream(stream) {
        // WARNING: ScoreStream currently only support batch_size = 1
        RTP_LLM_CHECK(maxBatchSize() == 1);
        CopyOnWrite(stream, true, true);
        sp_output_buffer_ = std::make_shared<SpeculativeExecutorStreamOutput>();

        updateStream(stream);
        setMetricsReporter(nullptr);
        setGenTimeline(false);
        if (!stream.generateConfig()->top1()) {
            setReturnAllProbs(true);
        }
    }

    void updateStream(const GenerateStream& stream) {
        size_t             propose_step   = stream.getProposeStep();
        rtp_llm::BufferPtr propose_tokens = stream.getProposeTokens();

        propose_step_                   = propose_step;
        sp_output_buffer_->propose_step = propose_step;
        score_len_                      = propose_step == 0 ? 1 : propose_step + 1;

        complete_token_ids_->setSeqLength(stream.seqLength());

        if (propose_tokens) {
            complete_token_ids_->appendTokens(0, propose_step, *propose_tokens);
        }
        if (score_len_ > history_max_score_len_) {
            sp_output_buffer_->tokens = device_->allocateBuffer(
                {rtp_llm::DataType::TYPE_INT32, {1, score_len_}, rtp_llm::AllocationType::HOST}, {"score_tokens"});
        }
        if (score_len_ > 1 || device_->initParams().sp_config.force_score_context_attention) {
            setIsContextStream(true);
            setReuseLength(stream.reuseLength());
        }
        history_max_score_len_ = std::max(history_max_score_len_, score_len_);
    }

    ErrorResult<GenerateOutputs> nextOutput() override {
        RTP_LLM_FAIL("ScoreStream::nextOutput should not be called");
        return ErrorInfo::OkStatus();
    };

    void updateOutput(const StreamUpdateInfo& update_info) override {
        memcpy(sp_output_buffer_->tokens->data(), update_info.new_tokens->data(), sizeof(int32_t) * score_len_);

        if (update_info.all_probs) {
            sp_output_buffer_->all_probs = update_info.all_probs;
        }

        if (generate_input_->generate_config->return_logits) {
            sp_output_buffer_->logits =
                device_->clone({*update_info.logits, rtp_llm::AllocationType::DEVICE, {"score_logits"}});
        }

        if (generate_input_->generate_config->return_softmax_probs) {
            sp_output_buffer_->softmax_probs =
                device_->clone({*update_info.softmax_probs, rtp_llm::AllocationType::DEVICE, {"softmax_probs"}});
        }

        if (needReturnHiddenStates()) {
            RTP_LLM_CHECK(update_info.all_hidden_states != nullptr);
            sp_output_buffer_->hidden_states = update_info.all_hidden_states;
        }

        if (update_info.loss) {
            sp_output_buffer_->loss =
                device_->clone({*update_info.loss, rtp_llm::AllocationType::DEVICE, {"score_loss"}});
        }
    }

    size_t scoreLen() const override {
        return score_len_;
    }

protected:
    size_t history_max_score_len_ = 0;
};

}  // namespace rtp_llm
