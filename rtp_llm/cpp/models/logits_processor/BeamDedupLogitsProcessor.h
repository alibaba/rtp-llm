#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

class BeamDedupLogitsProcessor : public BaseLogitsProcessor {
public:
    BeamDedupLogitsProcessor() = default;
    virtual ~BeamDedupLogitsProcessor() {}

public:
    static std::shared_ptr<BeamDedupLogitsProcessor> fromGenerateInput(std::shared_ptr<GenerateInput> generate_input);

public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

private:
    int beam_dedup_idx_ = -1;
    int current_step_ = 0;
};

using BeamDedupLogitsProcessorPtr = std::shared_ptr<BeamDedupLogitsProcessor>;

}  // namespace rtp_llm
