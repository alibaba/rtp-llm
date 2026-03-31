#pragma once

#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/core/Types.h"

namespace rtp_llm {
// Sampler would split logits into appropriate groups (mostly, based on beam size)
// and calls device sampling apis (greedy, beam search, etc) for each group
class Sampler {
public:
    Sampler(const SamplerInitParams& params);
    ~Sampler() {};

    virtual SamplerOutput forward(const SamplerInputs& inputs);

private:
    void preprocessLogits(const SamplerInputs& inputs);
};

}  // namespace rtp_llm
