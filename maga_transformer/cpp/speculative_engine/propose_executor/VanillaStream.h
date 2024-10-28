#pragma once
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeStreamOutput.h"
#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
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
        CopyOnWrite(stream, false);
        setMetricsReporter(nullptr);
        allocateOutputBuffer(propose_step);
        setNeedReleaseResource(false);
        setGenerateConfig();
        handleBounsToken();
    }

    ~VanillaStream() {
        if (old_top_k_ >= 0) {
            generateConfig()->top_k = old_top_k_;
        }
    }

    ErrorResult<GenerateOutputs> nextOutput() override {
        FT_FAIL("VanillaStream::nextOutput should not be called");
        return ErrorInfo::OkStatus();
    };

    void updateOutput(const ft::BufferPtr& new_tokens,
                      const ft::BufferPtr& hidden_states,
                      const ft::BufferPtr& logits,
                      const ft::BufferPtr& cum_log_probs,
                      const ft::BufferPtr& all_probs,
                      const ft::BufferPtr& loss,
                      bool                 update_queue = true) override {
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
            {ft::DataType::TYPE_INT32, {1, propose_step}, ft::AllocationType::HOST}, {"vanilla_propose_tokens"});
    }

    void handleBounsToken() {
        if (acceped_bouns_token_) {
            setReuseLength(seqLength() - 2);
            setFallbackPrefixLength(reuseLength());
            setIsContextStream(true);
        }
    }

    void setGenerateConfig() {
        std::shared_ptr<GenerateConfig>& generate_config = generateConfig();
        old_top_k_= generate_config->top_k;
        if (!generate_config->top1()) {
            setReturnAllProbs(true);
        } 
        if (generate_config->top_k == 0 && generate_config->top_p > 0.0) {
            generate_config->top_k = 20;
        }
    }

protected:
    int                                old_top_k_    = -1;              
    SpeculativeExecutorStreamOutputPtr output_buffer_;
    size_t                             current_step_ = 0;
    size_t                             propose_step_ = 0;
};
}  // namespace rtp_llm
