#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

class MultiSeqLogitsProcessor: public BaseLogitsProcessor {
public:
    MultiSeqLogitsProcessor() = default;
    virtual ~MultiSeqLogitsProcessor() {};

public:
    static std::shared_ptr<MultiSeqLogitsProcessor> fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                      int64_t                        eos_token_id);

public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;

protected:
    size_t eos_token_id_;
};

using MultiSeqLogitsProcessorPtr = std::shared_ptr<MultiSeqLogitsProcessor>;

}  // namespace rtp_llm