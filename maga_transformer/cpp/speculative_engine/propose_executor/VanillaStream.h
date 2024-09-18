#pragma once
#include "maga_transformer/cpp/dataclass/GenerateStream.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeStreamOutput.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/utils/assert_utils.h"
#include <cstddef>

namespace ft = fastertransformer;

namespace rtp_llm {

class VanillaStream: public GenerateStream {
public:
    VanillaStream(const GenerateStream&                     stream,
                  const SpeculativeExecutorStreamOutputPtr& stream_output,
                  size_t                                    propose_step):
        GenerateStream(stream), output_buffer_(stream_output), propose_step_(propose_step) {
        // WARNING: VanillaStream currently only support batch_size = 1
        FT_CHECK(tileNum() == 1);
        CopyOnWrite(stream);
        setMetricsReporter(nullptr);
        allocateOutputBuffer(propose_step);
        setNeedReleaseResource(false);
        setGenerateConfig();
    }

    ~VanillaStream() {}

    absl::StatusOr<GenerateOutputs> nextOutput() override {
        FT_FAIL("VanillaStream::nextOutput should not be called");
        return absl::OkStatus();
    };

    void updateOutput(const ft::BufferPtr& new_tokens,
                      const ft::BufferPtr& hidden_states,
                      const ft::BufferPtr& logits,
                      const ft::BufferPtr& cum_log_probs,
                      const ft::BufferPtr& all_probs) override {
        // TODO(xyz): optimize deepclone
        if (all_probs) {
            // lazy allocate buffer
            if (!output_buffer_->all_probs) {
                size_t vocab_size         = all_probs->shape()[1];
                output_buffer_->all_probs = device_->allocateBuffer(
                    {ft::DataType::TYPE_FP32, {propose_step_, vocab_size}, ft::AllocationType::DEVICE}, {"vanilla_all_probs"});
            }
            device_->copy({output_buffer_->all_probs->view(current_step_, 1), *all_probs});
        }
        *((*output_buffer_->tokens)[0].dataWithOffset<int>(current_step_)) = ((int*)new_tokens->data())[0];
        current_step_++;
    }

private:
    void allocateOutputBuffer(size_t propose_step) {
        output_buffer_->tokens = device_->allocateBuffer(
            {ft::DataType::TYPE_INT32, {1, propose_step}, ft::AllocationType::HOST}, {"score_tokens"});
    }

    void setGenerateConfig() {
        std::shared_ptr<GenerateConfig>& generate_config = generateConfig();
        if (!generate_config->top1()) {
            setReturnAllProbs(true);
        } 
        if (generate_config->top_k == 0 && generate_config->top_p > 0.0) {
            generate_config->top_k = 20;
        }
    }

protected:
    SpeculativeExecutorStreamOutputPtr output_buffer_;
    size_t                             current_step_ = 0;
    size_t                             propose_step_ = 0;
};
}  // namespace rtp_llm
