#pragma once

#include <optional>
#include <vector>

#include "rtp_llm/cpp/engine_base/stream/SamplingState.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"

namespace rtp_llm {

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

    absl::Status dispatchExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;

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

    ExecutionResultLayout validateExecutionResult(const std::list<GenerateStreamPtr>& all_streams,
                                                  const PPExecutionResult&            result) const;
    void validateRequestedOutputs(const PPOutputConfig&        output_config,
                                  const PPExecutionResult&     result,
                                  const ExecutionResultLayout& layout) const;

    absl::Status dispatchNormalExecutionResult(const StreamGroups& stream_groups,
                                               const PPExecutionResult& result) const;

    void dispatchNormalSingleStream(const GenerateStreamPtr&  stream,
                                    const PPExecutionResult& result,
                                    int64_t                  stream_idx,
                                    int64_t                  batch_idx,
                                    int64_t                  stream_batch_size,
                                    int64_t                  token_offset,
                                    int64_t                  loss_offset,
                                    std::optional<ErrorInfo> error_info) const;

    void validateMtpExecutionResult(const std::list<GenerateStreamPtr>& all_streams,
                                    const PPExecutionResult&            result,
                                    const ExecutionResultLayout&        layout) const;

    absl::Status dispatchMtpExecutionResult(const StreamGroups& stream_groups,
                                           const PPExecutionResult& result) const;

private:
    const bool                mtp_enabled_;
    const std::vector<int64_t> output_vocab_ids_;
    const int64_t             processor_eos_token_id_;
};

}  // namespace rtp_llm
