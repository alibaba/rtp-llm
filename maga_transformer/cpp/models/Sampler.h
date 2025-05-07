#pragma once

#include "maga_transformer/cpp/core/Buffer.h"
#include "maga_transformer/cpp/utils/DFAUtil.h"
#include "maga_transformer/cpp/models/ThinkModeLogitsProcessor.h"
#include "maga_transformer/cpp/models/SampleInfos.h"
#include "maga_transformer/cpp/core/Types.h"
#include "maga_transformer/cpp/devices/DeviceBase.h"



namespace rtp_llm {
// Sampler would split logits into appropriate groups (mostly, based on beam size)
// and calls device sampling apis (greedy, beam search, etc) for each group
class Sampler {
public:
    Sampler(const SamplerInitParams& params);
    ~Sampler(){};

    SamplerOutput forward(const SamplerInputs& inputs);

private:
    void preprocessLogits(const SamplerInputs& inputs);
    void updateGrammarStatus(const SamplerInputs& inputs);

private:
    rtp_llm::DeviceBase* device_;
    rtp_llm::BufferPtr eos_ids_;
    rtp_llm::BufferPtr eos_ids_host_;
};

}  // namespace rtp_llm
