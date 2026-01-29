#pragma once
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
namespace rtp_llm {
class ScoreBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    ScoreBatchStreamProcessor(const ModelConfig&                 model_config,
                              const PDSepConfig&                 pd_sep_config,
                              const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                              const CacheConfig&                 cache_config,
                              bool                               warm_up):
        NormalBatchStreamProcessor(
            nullptr, model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up) {}

    virtual absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const override;
    virtual absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const override;
    virtual absl::StatusOr<SamplerInputs>  gatherSamplerInput(const StreamGroups&    stream_groups,
                                                              const GptModelInputs&  model_inputs,
                                                              const GptModelOutputs& model_output) const override;
};
}  // namespace rtp_llm