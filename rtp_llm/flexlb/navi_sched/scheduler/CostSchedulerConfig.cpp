#include "rtp_llm/flexlb/navi_sched/scheduler/CostSchedulerConfig.h"

#include <cmath>

namespace navi_sched {

bool CostSchedulerConfig::valid() const
{
    return std::isfinite(lambda) && lambda >= 0.0 && lambda <= 1.0 &&
           std::isfinite(alpha) && alpha > 0.0 &&
           std::isfinite(alphaDecay) && alphaDecay > 0.0 &&
           alphaDecay <= 1.0 && std::isfinite(minAlpha) &&
           minAlpha >= 0.0 && minAlpha <= alpha &&
           maxLoopCount > 0 && maxLoopCount <= MAX_LOOP_COUNT &&
           minLoopCount > 0 && minLoopCount <= MAX_LOOP_COUNT &&
           optimizationTimeBudgetUs >= -1 &&
           parallelism >= 0 && parallelism <= MAX_PARALLELISM &&
           std::isfinite(latencyWeightConcavity) &&
           (latencyWeightConcavity == 0.0 || latencyWeightConcavity > 1.0) &&
           std::isfinite(latencyTokenNormalizationPower) &&
           latencyTokenNormalizationPower >= 0.0 &&
           latencyTokenNormalizationPower <= 1.0 &&
           discreteCheckInterval > 0 &&
           discreteCheckInterval <= MAX_LOOP_COUNT &&
           std::isfinite(discreteGapThreshold) &&
           discreteGapThreshold >= 0.0 &&
           std::isfinite(discreteRepairAlphaDecay) &&
           discreteRepairAlphaDecay > 0.0 &&
           discreteRepairAlphaDecay < 1.0;
}

} // namespace navi_sched
