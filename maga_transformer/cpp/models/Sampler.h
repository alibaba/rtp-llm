#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "maga_transformer/cpp/utils/DFAUtil.h"
#include "maga_transformer/cpp/models/ThinkModeLogitsProcessor.h"
#include "maga_transformer/cpp/models/SampleInfos.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace ft = fastertransformer;

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
    ft::DeviceBase* device_;
    ft::BufferPtr eos_ids_;
    ft::BufferPtr eos_ids_host_;
};

}  // namespace rtp_llm
