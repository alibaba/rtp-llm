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
    // Produced with the final tokens; consumed after the existing ready event.
    torch::Tensor accept_metric_rows;
    torch::Tensor accept_metric_total;

    torch::Tensor accept_tokens_cpu;
    torch::Tensor accept_len_cpu;

    std::shared_ptr<torch::Event> transfer_done_event;

    SpeculativeSamplerOutput(): transfer_done_event(std::make_shared<torch::Event>(cuda_graph::makeGraphEvent())) {}
};

struct MtpDraftSampleContext {
    torch::Tensor sequence_lengths;
    torch::Tensor hidden_to_zero;
    bool          zero_tokens = false;
};

struct FastTopKSamplerOutput {
    torch::Tensor all_probs;
    torch::Tensor token_ids;
    // MTP decode consumes int32 IDs; retain the original int64 API as well.
    torch::Tensor token_ids_int32       = {};
    torch::Tensor next_sequence_lengths = {};
    bool          fake_inputs_prepared  = false;
};

class FastTopKSampler {
public:
    FastTopKSampler(torch::Tensor d2t_map, bool allow_compact_point_mass = false):
        d2t_map_(d2t_map), allow_compact_point_mass_(allow_compact_point_mass) {}
    virtual ~FastTopKSampler() {}

    bool compactPointMassEnabled() const;

    virtual FastTopKSamplerOutput
    forward(const torch::Tensor& logits, int top_k = 1, const MtpDraftSampleContext* context = nullptr);

private:
    torch::Tensor d2t_map_;
    bool          allow_compact_point_mass_ = false;
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
};

}  // namespace speculative
}  // namespace rtp_llm
