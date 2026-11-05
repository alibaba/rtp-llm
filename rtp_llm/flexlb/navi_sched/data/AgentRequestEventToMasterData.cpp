#include "aios/navi_sched/data/AgentRequestEventToMasterData.h"

namespace navi_sched {

void AgentRequestEventToMasterData::serialize(autil::DataBuffer &dataBuffer) const
{
    dataBuffer.write(requestId);
    dataBuffer.write(cacheHitTokens);
    dataBuffer.write(cacheMissTokens);
    dataBuffer.write(inflightStatsRevision);
    dataBuffer.write(inflightRequests);
    dataBuffer.write(inflightPredictedLatencySumMs);
    dataBuffer.write(batchWaitingRequests);
    dataBuffer.write(queuedBatchRequests);
    dataBuffer.write(runningRequests);
    dataBuffer.write(completedRequestCount);
    dataBuffer.write(batchTimeoutCount);
    dataBuffer.write(abnormalBatchFinishCount);
    dataBuffer.write(abnormalBatchFinished);
    dataBuffer.write(latencyMs);
    dataBuffer.write(queueTotalPredictedLatencyMs);
    dataBuffer.write(queueDecayHeadRemainingMs);
    dataBuffer.write(queueModel);
    dataBuffer.write(latencyValidation);
    dataBuffer.write(localBatch);
    dataBuffer.write(agentPredictedExecutionLatencyMs);
    dataBuffer.write(backendQueueLatencyMs);
    dataBuffer.write(latencyValidationStats);
}

void AgentRequestEventToMasterData::deserialize(autil::DataBuffer &dataBuffer)
{
    dataBuffer.read(requestId);
    dataBuffer.read(cacheHitTokens);
    dataBuffer.read(cacheMissTokens);
    dataBuffer.read(inflightStatsRevision);
    dataBuffer.read(inflightRequests);
    dataBuffer.read(inflightPredictedLatencySumMs);
    dataBuffer.read(batchWaitingRequests);
    dataBuffer.read(queuedBatchRequests);
    dataBuffer.read(runningRequests);
    dataBuffer.read(completedRequestCount);
    dataBuffer.read(batchTimeoutCount);
    dataBuffer.read(abnormalBatchFinishCount);
    dataBuffer.read(abnormalBatchFinished);
    dataBuffer.read(latencyMs);
    dataBuffer.read(queueTotalPredictedLatencyMs);
    dataBuffer.read(queueDecayHeadRemainingMs);
    dataBuffer.read(queueModel);
    dataBuffer.read(latencyValidation);
    dataBuffer.read(localBatch);
    dataBuffer.read(agentPredictedExecutionLatencyMs);
    dataBuffer.read(backendQueueLatencyMs);
    dataBuffer.read(latencyValidationStats);
}

}
