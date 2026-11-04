#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedCommon.h"
#include "rtp_llm/flexlb/navi_sched/service/NaviSchedRoleSummary.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace navi_sched {

struct NaviSchedAgentSummary : public NaviSchedRoleSummary {
public:
    void Jsonize(JsonWrapper &json) override;
    std::string format() const;

public:
    size_t collectorCount = 0;
    std::string serverIp;
    int32_t serverPort = 0;
    bool pdSeparated = true;
    NaviSchedWorkloadSnapshot workload;
    size_t nativeWorkloadBatchCount = 0;
    size_t nativeWorkloadRequestCount = 0;
    size_t nativeUnbatchedRequestCount = 0;
    int64_t rawObserverEventTimeUs = 0;
    size_t pendingBatchCount = 0;
    uint64_t inflightStatsRevision = 0;
    int64_t inflightRequests = 0;
    double inflightPredictedLatencySumMs = 0.0;
    uint64_t inflightInvalidAddCount = 0;
    uint64_t inflightDuplicateAddCount = 0;
    uint64_t inflightMissingRemoveCount = 0;
    uint64_t inflightRecomputeCount = 0;
    uint64_t inflightRecomputedRequestCount = 0;
    uint64_t inflightRecomputeFailureCount = 0;
    uint64_t inflightInvalidStateTransitionCount = 0;
    size_t cachedBlockCount = 0;
    uint64_t cacheRevision = 0;
    uint64_t cacheHashChecksum = 0;
    size_t fullReadyBlockCount = 0;
    uint64_t fullReadyHashChecksum = 0;
    size_t tailReadyBlockCount = 0;
    uint64_t tailReadyHashChecksum = 0;
    int64_t rtpCachedBlockCount = -1;
    uint64_t cacheHitPathRequestCount = 0;
    double cacheHitMasterPredictedMean = 0.0;
    double cacheHitAgentArrivalMean = 0.0;
    double cacheHitAgentPendingAwareMean = 0.0;
    double cacheHitRtpActualMean = 0.0;
    double cacheHitMasterToAgentBiasMean = 0.0;
    double cacheHitAgentToRtpBiasMean = 0.0;
    double cacheHitMasterToAgentPendingAwareBiasMean = 0.0;
    double cacheHitAgentPendingAwareToRtpBiasMean = 0.0;
    double cacheHitMasterToRtpBiasMean = 0.0;
    uint64_t cacheHitAlignedRequestCount = 0;
    double cacheHitAlignedAgentToRtpBiasMean = 0.0;
    double cacheHitAlignedAgentToRtpMae = 0.0;
    uint64_t cacheHitUnalignedRequestCount = 0;
    double cacheHitUnalignedAgentToRtpBiasMean = 0.0;
    double cacheHitUnalignedAgentToRtpMae = 0.0;
    uint64_t hybridReuseDebugRequestCount = 0;
    double hybridReuseAgentPredictedBlocksMean = 0.0;
    double hybridReuseCandidateBlocksMean = 0.0;
    double hybridReuseFlatMatchedBlocksMean = 0.0;
    double hybridReuseMinFullBlocksMean = 0.0;
    double hybridReuseSelectedBlocksMean = 0.0;
    double hybridReuseAgentToFullOverpredictionBlocksMean = 0.0;
    double hybridReuseTailRollbackBlocksMean = 0.0;
    uint64_t hybridReuseInvalidOrderCount = 0;
    uint64_t readyWithoutRunningGapCount = 0;
    double readyWithoutRunningTotalMs = 0.0;
    double readyWithoutRunningMeanMs = 0.0;
    double readyWithoutRunningMaxMs = 0.0;
    double readyWithoutRunningCurrentMs = 0.0;
};

} // namespace navi_sched
