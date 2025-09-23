#pragma once

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

class BeamSearchLogitsProcessor: public BaseLogitsProcessor {
public:
    BeamSearchLogitsProcessor(rtp_llm::DeviceBase* device);
    virtual ~BeamSearchLogitsProcessor() {};

public:
    static std::shared_ptr<BeamSearchLogitsProcessor>
    fromGenerateInput(rtp_llm::DeviceBase* device, std::shared_ptr<GenerateInput> generate_input, int64_t eos_token_id);

public:
    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void beamSearchLogitProcessorUpdate(const std::vector<int>& beam_idx_vec) override;
    void updateLogitProcessorStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) override;

protected:
    size_t eos_token_id_;
};

using BeamSearchLogitsProcessorPtr = std::shared_ptr<BeamSearchLogitsProcessor>;

}  // namespace rtp_llm