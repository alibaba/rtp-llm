#pragma once

#include <optional>
#include <vector>

#include "rtp_llm/cpp/engine_base/stream/SamplingState.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"

namespace rtp_llm {

// Dispatch-time geometry: live fastgen cursors can be several rounds ahead
// when the first stage consumes a PP result. Never infer finality at result time.
struct PPStreamRoundSnapshot {
    int64_t batch_size         = 0;
    int64_t execute_token_size = 0;
    bool    intermediate_chunk = false;
    // Paired with the scheduler reservation, independent of mutable stream phase.
    bool tracks_chunk_result = false;
};

class PPBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    PPBatchStreamProcessor(const ModelConfig&                 model_config,
                           const PDSepConfig&                 pd_sep_config,
                           const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                           const CacheConfig&                 cache_config,
                           bool                               warm_up,
                           SpeculativeType                    sp_type = SP_TYPE_NONE);

    PPSamplingPlan gatherSamplingPlan(const StreamGroups& stream_groups) const;

    PPOutputConfig gatherOutputConfig(const StreamGroups& stream_groups) const;

    void initSamplingStates(const PPSamplingPlan& sampling_plan,
                            SamplingStates&       sampling_states,
                            PPExecutionResult&    result) const;

    SamplerInputs gatherSamplerInputs(const PPSamplingPlan& sampling_plan,
                                      const PPOutputConfig& output_config,
                                      const torch::Tensor&  logits,
                                      const SamplingStates& sampling_states,
                                      bool                  score_batch   = false,
                                      size_t                propose_step  = 0,
                                      const torch::Tensor&  verify_tokens = {}) const;

    /** Fills sampling outputs while preserving the initialized request fields and earlier errors. */
    void fillExecutionResult(const PPExecutionPlan& plan,
                             const GptModelOutputs& model_output,
                             const SamplerOutput&   sampler_output,
                             PPExecutionResult&     result) const;

    // Synchronous callers only; PPExecutor always passes its captured snapshot.
    absl::Status dispatchExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;
    absl::Status dispatchExecutionResult(const StreamGroups&                       stream_groups,
                                         const PPExecutionResult&                  result,
                                         const std::vector<PPStreamRoundSnapshot>& round_snapshot) const;

    torch::Tensor gatherDraftNextPositionIds(const StreamGroups&   stream_groups,
                                             const GptModelInputs& model_input) const;

    absl::StatusOr<GptModelInputs> gatherTargetVerifyModelInput(const StreamGroups& stream_groups,
                                                                size_t              propose_step,
                                                                TensorHolder&       host_holder) const;

private:
    SamplerInputs allocateSamplerInputs(const PPSamplingPlan& sampling_plan,
                                        const PPOutputConfig& output_config,
                                        size_t                total_batch_size,
                                        size_t                propose_step) const;

    void fillSamplerInputs(SamplerInputs&        sampler_inputs,
                           const PPSamplingPlan& sampling_plan,
                           const SamplingStates& sampling_states,
                           bool                  score_batch,
                           size_t                propose_step,
                           const torch::Tensor&  verify_tokens) const;

    std::optional<ErrorInfo> initLogitsProcessors(std::vector<BaseLogitsProcessorPtr>& processors,
                                                  const PPSamplingPlan&                sampling_plan,
                                                  int64_t                              stream_idx,
                                                  int64_t                              sequence_offset) const;

    void validateExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;

    absl::Status dispatchNormalExecutionResult(const StreamGroups&                       stream_groups,
                                               const PPExecutionResult&                  result,
                                               const std::vector<PPStreamRoundSnapshot>& round_snapshot) const;

    void dispatchNormalSingleStream(const GenerateStreamPtr& stream,
                                    const PPExecutionResult& result,
                                    int64_t                  stream_idx,
                                    int64_t                  batch_idx,
                                    int64_t                  stream_batch_size,
                                    int64_t                  token_offset,
                                    int64_t                  token_size,
                                    int64_t                  loss_offset,
                                    int64_t                  loss_size,
                                    bool                     intermediate_chunk) const;

    absl::Status dispatchSpeculativeExecutionResult(const StreamGroups&      stream_groups,
                                                    const PPExecutionResult& result) const;

private:
    const bool                 sp_enabled_;
    const std::vector<int64_t> output_vocab_ids_;
    const int64_t              processor_eos_token_id_;
};

}  // namespace rtp_llm
