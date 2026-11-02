#include "aios/navi_sched/data/AgentLatencyToMasterData.h"

#include <cmath>

namespace navi_sched {

std::string getNaviSchedLatencyFormulaName(int32_t formula)
{
    if (NAVI_SCHED_LATENCY_FORMULA_UNKNOWN == formula) {
        return "unknown";
    }
    if (NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL == formula) {
        return "nonlinear_prefill";
    }
    return "reserved_" + std::to_string(formula);
}

void NaviSchedLatencyOptimizerState::serialize(
    autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(adamMoment1);
    dataBuffer.write(adamMoment2);
    dataBuffer.write(step);
}

void NaviSchedLatencyOptimizerState::deserialize(
    autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(adamMoment1);
    dataBuffer.read(adamMoment2);
    dataBuffer.read(step);
}

bool NaviSchedLatencyOptimizerState::valid(size_t parameterCount) const
{
    if (parameterCount == 0 || adamMoment1.size() != parameterCount ||
        adamMoment2.size() != parameterCount || step <= 0) {
        return false;
    }
    for (size_t index = 0; index < parameterCount; ++index) {
        if (!std::isfinite(adamMoment1[index]) ||
            !std::isfinite(adamMoment2[index]) || adamMoment2[index] < 0.0) {
            return false;
        }
    }
    return true;
}

void AgentLatencyToMasterData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(formula);
    dataBuffer.write(binaryVersion);
    dataBuffer.write(modelRevision);
    dataBuffer.write(parameters);
    dataBuffer.write(optimizerState);
    dataBuffer.write(residualVersion);
    dataBuffer.write(residualParameters);
    dataBuffer.write(residualLastUpdateTimeUs);
    dataBuffer.write(zeroPoint);
    dataBuffer.write(sessionSampleCount);
}

void AgentLatencyToMasterData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(formula);
    dataBuffer.read(binaryVersion);
    dataBuffer.read(modelRevision);
    dataBuffer.read(parameters);
    dataBuffer.read(optimizerState);
    dataBuffer.read(residualVersion);
    dataBuffer.read(residualParameters);
    dataBuffer.read(residualLastUpdateTimeUs);
    dataBuffer.read(zeroPoint);
    dataBuffer.read(sessionSampleCount);
}

}
