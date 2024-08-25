#pragma once
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeStreamOutput.h"
#include "src/fastertransformer/utils/assert_utils.h"

namespace ft = fastertransformer;

namespace rtp_llm {

class VanillaStream: public GenerateStream {
public:
    VanillaStream(const GenerateStream&                     stream,
                  const SpeculativeExecutorStreamOutputPtr& stream_output,
                  size_t                                    propose_step):
        GenerateStream(stream), output_buffer_(stream_output) {
        // WARNING: VanillaStream currently only support batch_size = 1
        FT_CHECK(tileNum() == 1);
        CopyOnWrite(stream);
        setMetricsReporter(nullptr);
        allocateOutputBuffer(propose_step);
    }

    ~VanillaStream() {}

    absl::StatusOr<GenerateOutputs> nextOutput() override {
        FT_FAIL("VanillaStream::nextOutput should not be called");
        return absl::OkStatus();
    };

    void updateOutput(const ft::BufferPtr& new_tokens,
                      const ft::BufferPtr& hidden_states,
                      const ft::BufferPtr& logits,
                      const ft::BufferPtr& cum_log_probs) override {
        *((*output_buffer_->tokens)[0].dataWithOffset<int>(current_step_)) = ((int*)new_tokens->data())[0];
        current_step_++;
    }

private:
    void allocateOutputBuffer(size_t propose_step) {
        output_buffer_->tokens = device_->allocateBuffer(
            {ft::DataType::TYPE_INT32, {1, propose_step}, ft::AllocationType::HOST}, {"score_tokens"});
    }

protected:
    size_t                             current_step_ = 0;
    SpeculativeExecutorStreamOutputPtr output_buffer_;
};
}  // namespace rtp_llm
