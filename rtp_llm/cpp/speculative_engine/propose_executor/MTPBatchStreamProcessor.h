#pragma once
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"

namespace rtp_llm {

class MTPBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    MTPBatchStreamProcessor(const rtp_llm::GptInitParameter& params, const CacheConfig& cache_config, bool warm_up):
        NormalBatchStreamProcessor(params, cache_config, nullptr, warm_up) {};

    absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const override;
};
}  // namespace rtp_llm