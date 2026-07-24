#pragma once

#include "rtp_llm/cpp/models/eplb/stats/ExpertStats.h"

namespace rtp_llm {

// Allocates per-layer GPU buffers for routed-expert load tracking.
OverallExpertStats execCreateMoeExpertStates(const ExpertStatsParams& params);

}  // namespace rtp_llm
