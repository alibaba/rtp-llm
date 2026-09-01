#pragma once

#include <optional>
#include <vector>

#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/normal_engine/pipeline/PPTypes.h"

namespace rtp_llm {

class PPBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    PPBatchStreamProcessor(const ModelConfig&                 model_config,
                           const PDSepConfig&                 pd_sep_config,
                           const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                           const CacheConfig&                 cache_config,
                           bool                               warm_up);

    PPSamplingPlan gatherSamplingPlan(const StreamGroups& stream_groups) const;
    PPOutputConfig gatherOutputConfig(const StreamGroups& stream_groups) const;

    absl::StatusOr<PPExecutionResult> makeExecutionResult(const PPExecutionPlan& plan,
                                                          const GptModelOutputs& model_output,
                                                          const SamplerOutput&   sampler_output) const;

    absl::Status dispatchExecutionResult(const StreamGroups& stream_groups, const PPExecutionResult& result) const;

private:
    void dispatchSingleStream(const GenerateStreamPtr& stream,
                              const PPExecutionResult& result,
                              int64_t                  index,
                              int64_t                  token_offset,
                              int64_t                  loss_offset,
                              std::optional<ErrorInfo> error_info) const;

private:
    const std::vector<int64_t> output_vocab_ids_;
};

}  // namespace rtp_llm
