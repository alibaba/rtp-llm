#pragma once
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeOutput.h"
#include "maga_transformer/cpp/core/Buffer.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include <cstddef>



namespace rtp_llm {

class VanillaStream: public GenerateStream {
public:
    VanillaStream(const GenerateStream&                     stream,
                  ProposeOutputPtr                          propose_output,
                  size_t                                    propose_step):
        GenerateStream(stream), propose_output_(propose_output), propose_step_(propose_step) {
        // WARNING: VanillaStream currently only support batch_size = 1
        RTP_LLM_CHECK(tileNum() == 1);
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
        RTP_LLM_FAIL("VanillaStream::nextOutput should not be called");
        return ErrorInfo::OkStatus();
    };

    bool checkFinish() {
        SpeculativeExecutorStreamOutputPtr output_buffer = propose_output_->outputs[streamId()];
        output_buffer->propose_step = current_step_;
        return needFinish() || stoppedWithoutLock();
    };

    void updateOutput(const StreamUpdateInfo& update_info) override {
        SpeculativeExecutorStreamOutputPtr output_buffer = propose_output_->outputs[streamId()];
        // TODO(xyz): optimize deepclone
        if (update_info.all_probs) {
            // lazy allocate buffer
            if (!output_buffer->all_probs) {
                size_t vocab_size         = update_info.all_probs->shape()[1];
                output_buffer->all_probs = device_->allocateBuffer(
                    {rtp_llm::DataType::TYPE_FP32, {propose_step_, vocab_size}, rtp_llm::AllocationType::DEVICE}, {"vanilla_all_probs"});
            }
            device_->copy({output_buffer->all_probs->view(current_step_, 1), *update_info.all_probs});
        }
        *((*output_buffer->tokens)[0].dataWithOffset<int>(current_step_)) = ((int*)update_info.new_tokens->data())[0];
        current_step_++;
    }

private:
    void allocateOutputBuffer(size_t propose_step) {
        propose_output_->outputs[streamId()]->tokens = device_->allocateBuffer(
            {rtp_llm::DataType::TYPE_INT32, {1, propose_step}, rtp_llm::AllocationType::HOST}, {"vanilla_propose_tokens"});
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
    }

protected:
    int                                old_top_k_    = -1;
    ProposeOutputPtr                   propose_output_;
    size_t                             current_step_ = 0;
    size_t                             propose_step_ = 0;
};
}  // namespace rtp_llm
