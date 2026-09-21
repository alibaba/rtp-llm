#pragma once

#include "absl/status/statusor.h"
#include "c10/core/Event.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
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
    torch::Tensor all_probs;
    torch::Tensor token_ids;
};

class FastTopKSampler {
public:
    // Default: no draft-to-target vocab mapping (execMappingDraft2Target
    // no-ops on an undefined map). Keeps main's ctor contract for tests.
    FastTopKSampler() = default;
    explicit FastTopKSampler(torch::Tensor d2t_map): d2t_map_(std::move(d2t_map)) {}
    virtual ~FastTopKSampler() {}

    virtual FastTopKSamplerOutput forward(const torch::Tensor& logits, int top_k = 1);

private:
    torch::Tensor d2t_map_;
};

struct SpeculativeSamplingParams {
    torch::Tensor do_sample;  // CPU bool [B], with the same !top1() semantics as streams.
    torch::Tensor force_accept;  // CPU bool [B].
    std::vector<at::Generator> generators;
};

class SpeculativeSampler {
public:
    SpeculativeSampler(torch::Tensor d2t_map, size_t propose_step): d2t_map_(d2t_map), propose_step_(propose_step) {}

    virtual SpeculativeSamplerOutput forward(const SpeculativeSamplingParams& params,
                                             SamplerOutput&                   draft_sampler_output,
                                             SamplerOutput&                   target_sampler_output);

private:
    void batchSample(SpeculativeSamplerOutput&         sample_output,
                     const SpeculativeSamplingParams& params,
                     SamplerOutput&                   draft_sampler_output,
                     SamplerOutput&                   target_sampler_output) const;

protected:
    torch::Tensor        d2t_map_;
    size_t               propose_step_;
    mutable TensorHolder buffer_holder_;

    // Reusable buffer for draft_probs vocab-padding when draft/target vocab sizes differ.
    // Grow-only; reused across batchSample calls to avoid per-forward GPU allocation in hot path.
    mutable torch::Tensor draft_probs_padding_buffer_;
};

}  // namespace speculative
}  // namespace rtp_llm
