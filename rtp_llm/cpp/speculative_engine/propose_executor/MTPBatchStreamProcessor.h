#pragma once
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"

namespace rtp_llm {

class MTPBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    MTPBatchStreamProcessor(const ModelConfig& model_config,
                               const PDSepConfig& pd_sep_config,
                               const ProfilingDebugLoggingConfig& profiling_debug_logging_config,
                               const CacheConfig& cache_config,
                               bool warm_up):
        NormalBatchStreamProcessor(model_config, pd_sep_config, profiling_debug_logging_config, cache_config, warm_up) {};

    absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const override;
};
}  // namespace rtp_llm