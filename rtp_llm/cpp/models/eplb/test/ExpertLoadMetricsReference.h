#pragma once

#include "rtp_llm/cpp/models/eplb/ExpertMetricUtils.h"

namespace rtp_llm {

// CPU oracle for the production GPU reduction; linked only by tests.
ExpertLoadWindowMetrics reduceExpertLoadWindowReference(
    const std::vector<int64_t>& packed, size_t sample_count, size_t layer_count, size_t world_size, size_t ep_size);

}  // namespace rtp_llm
