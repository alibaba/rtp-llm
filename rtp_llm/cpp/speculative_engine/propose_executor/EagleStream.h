#include "rtp_llm/cpp/speculative_engine/propose_executor/MTPStream.h"

namespace rtp_llm {

class EagleStream: public MTPStream {
public:
    EagleStream(const GenerateStream& stream, size_t propose_step): MTPStream(stream, propose_step) {}

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

        // only update last hidden states for eagle method
        RTP_LLM_CHECK(update_info.all_hidden_states != nullptr);
        auto token_num      = update_info.all_hidden_states->shape()[0];
        last_hidden_states_ = update_info.all_hidden_states->slice(token_num - 1, 1, false);
        last_hidden_states_->updateParent(update_info.all_hidden_states);
    };
};
};  // namespace rtp_llm