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
    dataBuffer.write(tairKvcmConfig);
    dataBuffer.write(nativePrefillPipelineStages);
    if (executionGeometryEnabled || nativePrefillPipelineStages > 0) {
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
    dataBuffer.read(tairKvcmConfig);
    dataBuffer.read(nativePrefillPipelineStages);
    prefillExecutionPolicy = {};
    if (executionGeometryEnabled || nativePrefillPipelineStages > 0) {
        dataBuffer.read(prefillExecutionPolicy);
    }
}

}
