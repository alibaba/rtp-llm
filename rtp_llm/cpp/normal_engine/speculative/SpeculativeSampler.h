#pragma once

#include "absl/status/statusor.h"
#include "c10/core/Event.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/ModelTypes.h"

namespace rtp_llm {

namespace speculative {

struct SpeculativeSamplerOutput {
public:
    torch::Tensor accept_tokens;
    torch::Tensor accept_len;

    torch::Tensor accept_tokens_cpu;
    torch::Tensor accept_len_cpu;

    std::shared_ptr<torch::Event> transfer_done_event;

    // Per-stream verify errors from SpecLogitsVerifyRunner (main #1006 contract).
    std::vector<std::optional<ErrorInfo>> processor_errors;

    SpeculativeSamplerOutput(): transfer_done_event(std::make_shared<torch::Event>(cuda_graph::makeGraphEvent())) {}
};

struct FastTopKSamplerOutput {
    // Exact proposal distribution in draft-vocabulary space.
    torch::Tensor all_probs;
    torch::Tensor token_ids;
};

class FastTopKSampler {
public:
    // Default: no draft-to-target vocab mapping (execMappingDraft2Target
    // no-ops on an undefined map). Keeps main's ctor contract for tests.
    FastTopKSampler() = default;
    FastTopKSampler(torch::Tensor d2t_map, size_t target_vocab_size);
    virtual ~FastTopKSampler() {}

    virtual FastTopKSamplerOutput forward(const torch::Tensor& logits, int top_k = 1);

private:
    torch::Tensor d2t_map_;
    size_t        target_vocab_size_ = 0;
};

class SpeculativeSampler {
public:
    SpeculativeSampler(torch::Tensor d2t_map, size_t propose_step):
        d2t_map_(std::move(d2t_map)), propose_step_(propose_step) {}

    static torch::Tensor mapDraftProbsToTarget(const torch::Tensor& draft_probs,
                                               const torch::Tensor& d2t_map,
                                               int64_t              target_vocab_size,
                                               torch::Tensor*       target_probs_buffer = nullptr);

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
    torch::Tensor         d2t_map_;
    size_t                propose_step_;
    mutable TensorHolder  buffer_holder_;
    mutable torch::Tensor draft_probs_target_buffer_;
};

}  // namespace speculative
}  // namespace rtp_llm
