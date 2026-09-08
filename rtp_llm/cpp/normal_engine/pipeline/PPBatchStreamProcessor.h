#pragma once

#include <optional>
#include <vector>

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
                           bool                               warm_up);

    PPSamplingPlan gatherSamplingPlan(const StreamGroups& stream_groups) const;
    PPOutputConfig gatherOutputConfig(const StreamGroups& stream_groups) const;

    absl::StatusOr<PPExecutionResult> makeExecutionResult(const PPExecutionPlan& plan,
                                                          const GptModelOutputs& model_output,
                                                          const SamplerOutput&   sampler_output) const;

    absl::Status dispatchExecutionResult(const StreamGroups&                      stream_groups,
                                         const PPExecutionResult&                 result,
                                         const std::vector<PPStreamRoundSnapshot>& round_snapshot) const;

private:
    void validateExecutionResult(const std::list<GenerateStreamPtr>&       all_streams,
                                 const PPOutputConfig&                     output_config,
                                 const PPExecutionResult&                  result,
                                 const std::vector<PPStreamRoundSnapshot>& round_snapshot) const;

    void dispatchSingleStream(const GenerateStreamPtr&     stream,
                              const PPExecutionResult&     result,
                              const PPStreamRoundSnapshot& round,
                              int64_t                      stream_idx,
                              int64_t                      batch_idx,
                              int64_t                      token_offset,
                              int64_t                      loss_offset,
                              std::optional<ErrorInfo>     error_info) const;

private:
    const std::vector<int64_t> output_vocab_ids_;
};

}  // namespace rtp_llm
