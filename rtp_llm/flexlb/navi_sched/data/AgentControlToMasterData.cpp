#include "aios/navi_sched/data/AgentControlToMasterData.h"

namespace navi_sched {

void AgentControlToMasterData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(serverInfo);
    dataBuffer.write(agentStartTimeUs);
    dataBuffer.write(pdSeparated);
    dataBuffer.write(unifiedLatencyPrediction);
    dataBuffer.write(executionGeometryEnabled);
    dataBuffer.write(projectedExecutionGeometryEnabled);
    dataBuffer.write(peerCacheEnabled);
    if (executionGeometryEnabled) {
        dataBuffer.write(prefillExecutionPolicy);
    }
}

void AgentControlToMasterData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(serverInfo);
    dataBuffer.read(agentStartTimeUs);
    dataBuffer.read(pdSeparated);
    dataBuffer.read(unifiedLatencyPrediction);
    dataBuffer.read(executionGeometryEnabled);
    dataBuffer.read(projectedExecutionGeometryEnabled);
    dataBuffer.read(peerCacheEnabled);
    prefillExecutionPolicy = {};
    if (executionGeometryEnabled) {
        dataBuffer.read(prefillExecutionPolicy);
    }
}

}
