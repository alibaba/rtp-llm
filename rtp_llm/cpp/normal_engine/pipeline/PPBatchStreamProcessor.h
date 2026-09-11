#pragma once

#include <optional>
#include <vector>

#include "rtp_llm/cpp/engine_base/stream/SamplingState.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"

namespace rtp_llm {

// Per-stream geometry of one dispatched PP round, captured at dispatch time.
// The result for round k is consumed up to pp_size rounds later, by which point
// a fastgen stream's chunk cursors have moved on, so anything the result path
// derives from the live stream describes the wrong round.
struct PPStreamRoundSnapshot {
    int64_t batch_size         = 0;
    int64_t execute_token_size = 0;
    // The round carried a non-final fastgen chunk: its sampled token is
    // mid-prompt and must not be appended or flip the stream out of context.
    bool    intermediate_chunk = false;
};

class PPBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    PPBatchStreamProcessor(const ModelConfig&                 model_config,
                           const PDSepConfig&                 pd_sep_config,
                           const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                           const CacheConfig&                 cache_config,
                           bool                               warm_up,
                           bool                               mtp_enabled = false);

    PPSamplingPlan gatherSamplingPlan(const StreamGroups& stream_groups) const;

    PPOutputConfig gatherOutputConfig(const StreamGroups& stream_groups) const;

    absl::StatusOr<SamplerInputs> gatherSamplerInputs(const PPSamplingPlan& sampling_plan,
                                                    const PPOutputConfig& output_config,
                                                    const torch::Tensor&  logits,
                                                    SamplingStates&       sampling_states,
                                                    bool                  score_batch  = false,
                                                    size_t                propose_step = 0) const;

    absl::StatusOr<PPExecutionResult> makeExecutionResult(const PPExecutionPlan& plan,
                                                          const GptModelOutputs& model_output,
                                                          const SamplerOutput&   sampler_output) const;

    // `round_snapshot` carries the per-stream geometry of the round being
    // dispatched, captured by PPExecutor at dispatch time. Under fastgen chunk
    // overlap the result for round k is consumed up to pp_size rounds later, so
    // the live stream cursors no longer describe this round.
    absl::Status dispatchExecutionResult(const StreamGroups&                       stream_groups,
                                         const PPExecutionResult&                  result,
                                         const std::vector<PPStreamRoundSnapshot>& round_snapshot) const;

    torch::Tensor gatherDraftNextPositionIds(const StreamGroups& stream_groups, const GptModelInputs& model_input) const;

    absl::StatusOr<GptModelInputs>
    gatherTargetVerifyModelInput(const StreamGroups& stream_groups, size_t propose_step, TensorHolder& host_holder) const;

private:
    SamplerInputs allocateSamplerInputs(const PPSamplingPlan& sampling_plan,
                                       const PPOutputConfig& output_config,
                                       size_t                total_batch_size,
                                       size_t                propose_step) const;

    absl::Status fillSamplerInputs(SamplerInputs&        sampler_inputs,
                                  const PPSamplingPlan& sampling_plan,
                                  SamplingStates&       sampling_states,
                                  bool                  score_batch,
                                  size_t                propose_step) const;

    absl::StatusOr<SamplingState> createSamplingState(const PPSamplingPlan& sampling_plan,
                                                    int64_t               stream_idx,
                                                    int64_t               sequence_offset) const;

    struct ExecutionResultLayout {
        int64_t batch_size = 0;
        int64_t token_size = 0;
        int64_t loss_size  = 0;
    };

    // Geometry authority for one dispatched round is the snapshot, not the live
    // stream: by result time a fastgen stream's cursors have advanced past it.
    ExecutionResultLayout validateExecutionResult(const std::list<GenerateStreamPtr>&       all_streams,
                                                  const PPExecutionResult&                  result,
                                                  const std::vector<PPStreamRoundSnapshot>& round_snapshot) const;
    void validateRequestedOutputs(const PPOutputConfig&        output_config,
                                  const PPExecutionResult&     result,
                                  const ExecutionResultLayout& layout) const;

    absl::Status dispatchNormalExecutionResult(const StreamGroups&                       stream_groups,
                                               const PPExecutionResult&                  result,
                                               const std::vector<PPStreamRoundSnapshot>& round_snapshot) const;

    void dispatchNormalSingleStream(const GenerateStreamPtr&     stream,
                                    const PPExecutionResult&     result,
                                    const PPStreamRoundSnapshot& round,
                                    int64_t                      stream_idx,
                                    int64_t                      batch_idx,
                                    int64_t                      token_offset,
                                    int64_t                      loss_offset,
                                    std::optional<ErrorInfo>     error_info) const;

    void validateMtpExecutionResult(const std::list<GenerateStreamPtr>& all_streams,
                                    const PPExecutionResult&            result,
                                    const ExecutionResultLayout&        layout) const;

    absl::Status dispatchMtpExecutionResult(const StreamGroups&                       stream_groups,
                                            const PPExecutionResult&                  result,
                                            const std::vector<PPStreamRoundSnapshot>& round_snapshot) const;

private:
    const bool                mtp_enabled_;
    const std::vector<int64_t> output_vocab_ids_;
    const int64_t             processor_eos_token_id_;
};

}  // namespace rtp_llm
