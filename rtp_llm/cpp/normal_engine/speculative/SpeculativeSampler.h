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
struct DSparkSamplingGraphs;
struct RejectionSamplingGraphs;

// Shared DSpARK Target/Draft setting. Zero disables fixed-width truncation.
int dsparkForceTopK();

struct SpeculativeSamplerOutput {
public:
    torch::Tensor accept_tokens;
    torch::Tensor accept_len;

    torch::Tensor accept_tokens_cpu;
    torch::Tensor accept_len_cpu;

    std::shared_ptr<torch::Event> transfer_done_event;
    // Retain graph-owned pinned results until asynchronous bookkeeping ends.
    std::shared_ptr<void> graph_buffers;

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

class SpeculativeSampler {
public:
    SpeculativeSampler(torch::Tensor d2t_map, size_t propose_step): d2t_map_(d2t_map), propose_step_(propose_step) {}

    virtual SpeculativeSamplerOutput forward(const std::list<GenerateStreamPtr>& streams,
                                             SamplerOutput&                      draft_sampler_output,
                                             SamplerOutput&                      target_sampler_output);

    // Per-position calibration statistics, FP32 [gamma, 6]: valid attempted
    // count, accepted count, prediction sum, absolute-error sum, squared-error
    // sum, invalid-prediction count. Rejection censors later positions; the
    // bonus/cap boundary is not a draft verification. Optional eligible_rows
    // excludes forced/error rows. This performs no device-to-host transfer.
    static torch::Tensor buildConfidenceStats(const torch::Tensor& confidence_rates,
                                              const torch::Tensor& accept_len,
                                              const torch::Tensor& valid_lengths = torch::Tensor(),
                                              const torch::Tensor& eligible_rows = torch::Tensor());

    // Select exactly total_draft_budget prefix positions from FP32 [B, gamma]
    // rates. Invalid rates become zero; equal scores use stable request identity
    // then position. stable_order_indices is an int64 [B] permutation of original
    // rows in ascending unique identity order, validated by the CPU caller before
    // any proposal draw. Budget and identities must not depend on this round's
    // proposals. Returns contiguous int32 [B] in original row order, on the same
    // CPU/CUDA device; no device values are read back to the host.
    static torch::Tensor buildValidDraftLengths(const torch::Tensor& confidence_rates,
                                                const torch::Tensor& stable_order_indices,
                                                int64_t              total_draft_budget);

    // batch_all_greedy: every stream in the batch is non-stochastic, i.e. the
    // caller pinned all draft temperatures to kMinDraftTemperature and greedy
    // rejection accepts on exact token match only. Return first-argmax tokens
    // as a point mass (token_ids_are_point_mass=true, all_probs undefined).
    // This may change proposals at ties and RNG consumption. Any stochastic
    // row keeps the dense-q sampling path.
    // With copy_token_ids=false, graph output is borrowed until the next sample;
    // the caller must take an owned copy before publishing it to other work.
    // confidence_hidden_logits / confidence_head_w / confidence_temperature are optional.
    // When both tensors are defined the sampler also evaluates the DSpARK
    // confidence head for the Markov conditioning tokens and returns per-position
    // acceptance rates in ``SamplerOutput::confidence_rates``; tokens are
    // unaffected. confidence_hidden_logits is the pre-norm hidden projection
    // h*w_hidden, contiguous CUDA FP32 [B*gamma, 1] in request-major order.
    // confidence_head_w retains [1, hidden+markov_rank]; only its last
    // markov_rank columns are consumed here. Both tensors must be present or
    // both absent. Confidence temperature is independent of sampling temperature.
    SamplerOutput sampleDSparkDraft(const torch::Tensor& base_logits,
                                    const torch::Tensor& anchors,
                                    const torch::Tensor& temperature,
                                    const torch::Tensor& top_k,
                                    const torch::Tensor& top_p,
                                    const torch::Tensor& markov_w1,
                                    const torch::Tensor& markov_w2,
                                    size_t               draft_vocab_size,
                                    bool                 batch_all_greedy,
                                    int                  force_topk               = 0,
                                    int                  joint_max_top_k          = 0,
                                    bool                 copy_token_ids           = true,
                                    const torch::Tensor& confidence_hidden_logits = torch::Tensor(),
                                    const torch::Tensor& confidence_head_w        = torch::Tensor(),
                                    double               confidence_temperature   = 1.0) const;

    void prepareDSparkDraftGraphs(const torch::Tensor&    logits_buffer,
                                  const torch::Tensor&    markov_w1,
                                  const torch::Tensor&    markov_w2,
                                  size_t                  draft_vocab_size,
                                  const std::vector<int>& batch_sizes);
    void prepareRejectionGraphs(const torch::Device& device, int vocab_size, const std::vector<int>& batch_sizes);

private:
    SamplerOutput                            sampleDSparkDraftEager(const torch::Tensor& base_logits,
                                                                    const torch::Tensor& anchors,
                                                                    const torch::Tensor& temperature,
                                                                    const torch::Tensor& top_k,
                                                                    const torch::Tensor& top_p,
                                                                    const torch::Tensor& markov_w1,
                                                                    const torch::Tensor& markov_w2,
                                                                    size_t               draft_vocab_size,
                                                                    bool                 batch_all_greedy,
                                                                    int                  force_topk,
                                                                    int                  joint_max_top_k) const;
    std::shared_ptr<DSparkSamplingGraphs>    dspark_graphs_;
    std::shared_ptr<RejectionSamplingGraphs> rejection_graphs_;
    bool                                     replayRejectionGraph(SpeculativeSamplerOutput& output,
                                                                  const torch::Tensor&      masks,
                                                                  const torch::Tensor&      draft_ids,
                                                                  const torch::Tensor&      target_ids,
                                                                  const torch::Tensor&      uniform,
                                                                  const torch::Tensor&      target_probs,
                                                                  bool                      point_mass,
                                                                  const torch::Tensor&      valid_draft_lengths = torch::Tensor()) const;
    void                                     batchSample(SpeculativeSamplerOutput&           sample_output,
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
