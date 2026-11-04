#include "rtp_llm/flexlb/navi_sched/data/LatencyModelGlobalData.h"

#include "rtp_llm/flexlb/navi_sched/data/AgentLatencyToMasterData.h"

#include <cmath>

namespace navi_sched {

bool LatencyModelGlobalData::valid() const
{
    return masterEpoch > 0 && version > 0 &&
           formula == NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL && binaryVersion > 0 &&
           parameterCount > 0 && std::isfinite(globalZeroPoint) && globalZeroPoint >= 0.0;
}

void LatencyModelGlobalData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(masterEpoch);
    dataBuffer.write(version);
    dataBuffer.write(formula);
    dataBuffer.write(binaryVersion);
    dataBuffer.write(parameterCount);
    dataBuffer.write(globalZeroPoint);
}

void LatencyModelGlobalData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(masterEpoch);
    dataBuffer.read(version);
    dataBuffer.read(formula);
    dataBuffer.read(binaryVersion);
    dataBuffer.read(parameterCount);
    dataBuffer.read(globalZeroPoint);
}

} // namespace navi_sched
