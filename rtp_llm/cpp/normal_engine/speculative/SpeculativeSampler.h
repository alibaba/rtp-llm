#pragma once

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm {

namespace speculative {

struct SpeculativeSamplerOutput {
public:
    std::vector<torch::Tensor> accept_tokens;
    std::vector<int>           accept_len;
};

struct FastTopKSamplerOutput {
    torch::Tensor all_probs;
    torch::Tensor token_ids;
};

struct ValidatedSpeculativeSamplerInputs {
    torch::Tensor draft_token_ids_cpu;
    size_t        token_stride;
    size_t        vocab_size;
};

ValidatedSpeculativeSamplerInputs validateSpeculativeSamplerInputs(size_t               batch_size,
                                                                   size_t               propose_step,
                                                                   const SamplerOutput& draft_sampler_output,
                                                                   const SamplerOutput& target_sampler_output,
                                                                   bool copy_draft_token_ids_to_cpu = false);

int validateSpeculativeEmittedTokenCount(int emitted_token_count, size_t propose_step);

class FastTopKSampler {
public:
    FastTopKSampler() {}
    virtual ~FastTopKSampler() = default;

    virtual FastTopKSamplerOutput forward(const torch::Tensor& logits, int top_k = 1);
};

class SpeculativeSampler {
public:
    SpeculativeSampler(size_t propose_step): propose_step_(propose_step) {}
    virtual ~SpeculativeSampler() = default;

    virtual SpeculativeSamplerOutput forward(const std::list<GenerateStreamPtr>& streams,
                                             SamplerOutput&                      draft_sampler_output,
                                             SamplerOutput&                      target_sampler_output);

private:
    void batchSample(SpeculativeSamplerOutput&           sample_output,
                     const std::list<GenerateStreamPtr>& streams,
                     SamplerOutput&                      draft_sampler_output,
                     SamplerOutput&                      target_sampler_output) const;
    void streamSample(SpeculativeSamplerOutput&           sample_output,
                      const std::list<GenerateStreamPtr>& streams,
                      SamplerOutput&                      draft_sampler_output,
                      SamplerOutput&                      target_sampler_output) const;

protected:
    size_t propose_step_;
};

}  // namespace speculative
}  // namespace rtp_llm
