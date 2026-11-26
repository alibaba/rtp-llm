#pragma once

#include "aios/navi_sched/common/TPUPrefillPP4Forecast.h"

namespace navi_sched {

// Test-only reference predating Native request-slot accounting.
class TPUPrefillPP4ForecastBaseline {
public:
    static TPUPrefillPP4ForecastOutput predict(
        const TPUPrefillPP4State &state,
        const std::vector<TPUPrefillPP4Request> &requests,
        const TPUPrefillExecutionPolicyConfig &policy,
        const TPUPrefillPP4Timing::Parameters &parameters, size_t maxTransitions,
        bool captureDispatches = false);
};

} // namespace navi_sched
