#pragma once

#include <array>

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

    SpeculativeSamplerOutput(): transfer_done_event(std::make_shared<torch::Event>(cuda_graph::makeGraphEvent())) {}
};

struct FastTopKSamplerOutput {
    torch::Tensor all_probs;
    torch::Tensor token_ids;
};

class FastTopKSampler {
public:
    FastTopKSampler(torch::Tensor d2t_map): d2t_map_(d2t_map) {}
    virtual ~FastTopKSampler() {}

    virtual FastTopKSamplerOutput forward(const torch::Tensor& logits, int top_k = 1);

private:
    torch::Tensor d2t_map_;
};

class SpeculativeSampler {
public:
    SpeculativeSampler(torch::Tensor d2t_map, size_t propose_step): d2t_map_(d2t_map), propose_step_(propose_step) {}

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
    torch::Tensor        d2t_map_;
    size_t               propose_step_;
    mutable TensorHolder buffer_holder_;

    // Reusable buffer for draft_probs vocab-padding when draft/target vocab sizes differ.
    // Grow-only; reused across batchSample calls to avoid per-forward GPU allocation in hot path.
    mutable torch::Tensor draft_probs_padding_buffer_;

    // Ping-pong pinned host destinations for the per-round accept D2H: a
    // pageable destination makes cudaMemcpyAsync stage synchronously.  Two
    // slots because the async bookkeeping worker may still be reading last
    // round's tensors (it is synced before the next sampler input build, so
    // at most one round is in flight).  Grow-only within each slot.
    mutable std::array<torch::Tensor, 2> accept_tokens_cpu_slots_;
    mutable std::array<torch::Tensor, 2> accept_len_cpu_slots_;
    mutable size_t                       accept_cpu_slot_ = 0;
};

}  // namespace speculative
}  // namespace rtp_llm
