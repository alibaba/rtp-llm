#include "rtp_llm/cpp/models/logits_processor/BeamDedupLogitsProcessor.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::shared_ptr<BeamDedupLogitsProcessor>
BeamDedupLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input) {
    if (!generate_input->generate_config->beam_dedup_idx.has_value()) {
        return nullptr;
    }

    if (!generate_input->generate_config->hasNumBeams()) {
        RTP_LLM_LOG_WARNING("beam_dedup_idx is set but num_beams <= 1, ignoring beam_dedup_idx");
        return nullptr;
    }

    int beam_dedup_idx = generate_input->generate_config->beam_dedup_idx.value();
    if (beam_dedup_idx < 0) {
        RTP_LLM_LOG_WARNING("beam_dedup_idx must be >= 0, got %d, ignoring", beam_dedup_idx);
        return nullptr;
    }

    int max_new_tokens = generate_input->generate_config->max_new_tokens;
    if (beam_dedup_idx >= max_new_tokens) {
        RTP_LLM_LOG_WARNING("beam_dedup_idx (%d) >= max_new_tokens (%d), processor will never trigger, ignoring",
                           beam_dedup_idx, max_new_tokens);
        return nullptr;
    }

    auto processor_ptr = std::make_shared<BeamDedupLogitsProcessor>();
    processor_ptr->beam_dedup_idx_ = beam_dedup_idx;
    processor_ptr->current_step_ = 0;

    return processor_ptr;
}

void BeamDedupLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (current_step_ != beam_dedup_idx_) {
        return;
    }

    size_t batch_size = inputs.logits.size(0);
    size_t vocab_size = inputs.logits.size(1);

    auto logits = inputs.logits;
    auto max_indices = std::get<1>(logits.max(/*dim=*/1, /*keepdim=*/true));  // [batch_size, 1]
    
    // Create mask with UInt8 type (Byte) as required by maskLogits
    // mask value 1 means to mask (set to -inf), 0 means to keep
    auto logit_mask = torch::ones({(int64_t)batch_size, (int64_t)vocab_size},
                                   torch::TensorOptions().dtype(torch::kUInt8).device(logits.device()));
    logit_mask.scatter_(/*dim=*/1, max_indices, /*value=*/0);

    maskLogits(logits, logit_mask);
}

void BeamDedupLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    // do nothing
}

void BeamDedupLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    current_step_ += num_new_tokens;
}

}  // namespace rtp_llm
