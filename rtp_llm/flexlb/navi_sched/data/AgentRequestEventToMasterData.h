#pragma once

#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include <autil/DataBuffer.h>
#include "navi/common.h"
#include <vector>

namespace navi_sched {

struct AgentRequestEventToMasterData {
public:
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

public:
    int64_t cacheHitTokens = 0;
    int64_t cacheMissTokens = 0;
    uint64_t inflightStatsRevision = 1;
    int32_t inflightRequests = 0;
    double inflightPredictedLatencySumMs = 0.0;
    int32_t batchWaitingRequests = 0;
    int32_t queuedBatchRequests = 0;
    int32_t runningRequests = 0;
    int32_t completedRequestCount = 0;
    uint64_t batchTimeoutCount = 0;
    uint64_t abnormalBatchFinishCount = 0;
    bool abnormalBatchFinished = false;
    double latencyMs = 0.0;
    double queueTotalPredictedLatencyMs = 0.0;
    double queueDecayHeadRemainingMs = 0.0;
    uint64_t requestId = 0;
    bool latencyValidation = false;
    NaviSchedLocalBatchInfo localBatch;
    double agentPredictedExecutionLatencyMs = 0.0;
    double backendQueueLatencyMs = 0.0;
    NaviSchedLatencyValidationStats latencyValidationStats;
};

NAVI_TYPEDEF_PTR(AgentRequestEventToMasterData);

}
