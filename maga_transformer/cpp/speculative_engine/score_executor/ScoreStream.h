#pragma once
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "src/fastertransformer/utils/assert_utils.h"
#include <cstddef>

namespace ft = fastertransformer;

namespace rtp_llm {

class ScoreStream: public GenerateStream {
public:
    ScoreStream(const GenerateStream&                     stream,
                const SpeculativeExecutorStreamOutputPtr& stream_input,
                const SpeculativeExecutorStreamOutputPtr& stream_output,
                size_t                                    propose_step):
        GenerateStream(stream),
        propose_step_(propose_step),
        score_len_(propose_step == 0 ? 1 : propose_step),
        output_buffer_(stream_output) {
        // WARNING: ScoreStream currently only support batch_size = 1
        FT_CHECK(tileNum() == 1);
        CopyOnWrite(stream);
        updateProposeTokens(stream_input);
        allocateOutputBuffer(propose_step);
    }

    ~ScoreStream() {}

    absl::StatusOr<GenerateOutputs> nextOutput() override {
        FT_FAIL("ScoreStream::nextOutput should not be called");
        return absl::OkStatus();
    };

    void updateOutput(const ft::BufferPtr& new_tokens,
                      const ft::BufferPtr& hidden_states,
                      const ft::BufferPtr& logits,
                      const ft::BufferPtr& cum_log_probs) override {
        device_->copy({(*output_buffer_->tokens)[0], (*new_tokens)[0]});
    }

    size_t scoreLen() const override {
        return score_len_;
    }

private:
    void updateProposeTokens(const SpeculativeExecutorStreamOutputPtr& stream_input) {
        auto& propose_tokens = stream_input->tokens;
        device_->copy({((*complete_token_ids_)[0]).view(seq_length_, propose_tokens->size()), *propose_tokens});
        setSeqLength(seqLength() + propose_tokens->size());
    }

    void allocateOutputBuffer(size_t propose_step) {
        if (propose_step > 0) {
            output_buffer_->tokens = device_->allocateBuffer(
                {ft::DataType::TYPE_INT32, {1, propose_step}, ft::AllocationType::HOST}, {"score_tokens"});
            setIsContextStream(true);
        }
    }

protected:
    size_t                             propose_step_ = 0;
    size_t                             score_len_    = 0;
    SpeculativeExecutorStreamOutputPtr output_buffer_;
};

}  // namespace rtp_llm