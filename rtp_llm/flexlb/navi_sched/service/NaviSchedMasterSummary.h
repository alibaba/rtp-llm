#pragma once

#include "aios/navi_sched/service/NaviSchedRoleSummary.h"
#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace navi_sched {

struct NaviSchedMasterNodeSummary : public autil::legacy::Jsonizable {
public:
    void Jsonize(JsonWrapper &json) override;
    std::string format() const;

public:
    uint64_t agentIdHash = 0;
    std::string agentId;
    std::string bizName;
    std::string collectorAddress;
    std::string serverIp;
    int32_t serverPort = 0;
    std::string serverAddress;
    bool online = false;
    bool pdSeparated = true;
    bool peerCacheEnabled = false;
    // Live Master diagnostics; intentionally excluded from persisted model/replay wire
    // data.
    bool peerCacheReadAvailable = false;
    uint64_t peerCacheReadRevision = 0;
    int64_t peerCacheReadAgentStartTimeUs = 0;
    int64_t totalBlocks = 0;
    size_t cachedBlockCount = 0;
    uint64_t agentCacheRevision = 0;
    int64_t agentCachedBlockCount = -1;
    uint64_t agentCacheHashChecksum = 0;
    int64_t agentFullReadyBlockCount = -1;
    uint64_t agentFullReadyHashChecksum = 0;
    int64_t agentTailReadyBlockCount = -1;
    uint64_t agentTailReadyHashChecksum = 0;
    uint64_t masterCacheHashChecksum = 0;
    size_t masterFullReadyBlockCount = 0;
    uint64_t masterFullReadyHashChecksum = 0;
    size_t masterTailReadyBlockCount = 0;
    uint64_t masterTailReadyHashChecksum = 0;
    int64_t rtpCachedBlockCount = -1;
    uint64_t cacheRevisionGapCount = 0;
    uint64_t cacheStateMismatchCount = 0;
    uint64_t rtpCacheMirrorSampleCount = 0;
    uint64_t rtpCacheMirrorMismatchCount = 0;
    std::map<std::string, uint64_t> v6dCacheStats;
    size_t pendingCacheBatchCount = 0;
    size_t pendingCacheBlockCount = 0;
    size_t pendingCacheScheduledBatchCount = 0;
    size_t pendingCacheAgentBatchCount = 0;
    uint64_t pendingCacheAgentRevision = 0;
    uint64_t pendingCacheAgentChecksum = 0;
    uint64_t pendingCacheRevisionGapCount = 0;
    uint64_t pendingCacheStateMismatchCount = 0;
    uint64_t pendingCacheResyncTriggerCount = 0;
    bool pendingCacheSnapshotReady = false;
    uint64_t inflightStatsRevision = 0;
    int32_t inflightRequests = 0;
    double inflightPredictedLatencySumMs = 0.0;
    int32_t batchWaitingRequests = 0;
    int32_t queuedBatchRequests = 0;
    int32_t runningRequests = 0;
    double executionUtilization = -1.0;
    double executionBusyUtilization = -1.0;
    uint64_t batchTimeoutCount = 0;
    uint64_t abnormalBatchFinishCount = 0;
    uint64_t masterSentRequestCount = 0;
    uint64_t masterSentBatchCount = 0;
    double masterSentRequestsPerBatchMean = 0.0;
    size_t reservationBatchCount = 0;
    size_t reservationRequestCount = 0;
    size_t reservationWaitingRequestCount = 0;
    size_t reservationDispatchedRequestCount = 0;
    size_t dispatchWaiterCount = 0;
    size_t liveDispatchWaiterCount = 0;
    double lastRequestLatencyMs = 0.0;
    double queueTotalPredictedLatencyMs = 0.0;
    int64_t queueReportReceiveTimeUs = 0;
    NaviSchedWorkloadSnapshot workload;
    double workloadEffectiveRunningElapsedTimeMs = 0.0;
    int64_t workloadReportReceiveTimeUs = 0;
    uint64_t workloadStaleUpdateCount = 0;
    std::string lastEventType;
    int64_t lastEventTimeUs = 0;
    std::string lastRequestEventType;
    int64_t lastRequestEventTimeUs = 0;
    int64_t lastReceiveTimeUs = 0;
    int32_t latencyFormula = 0;
    int64_t latencyBinaryVersion = 0;
    int64_t latencyModelRevision = 0;
    size_t latencyParameterCount = 0;
    std::vector<double> latencyParameters;
    int32_t latencyResidualVersion = 0;
    std::vector<double> latencyResidualParameters;
    int64_t latencyResidualLastUpdateTimeUs = 0;
    std::string latencyModelFingerprint;
    bool latencyModelTrusted = false;
    bool latencyModelWarmStarted = false;
    uint32_t latencyModelConsecutiveRejectedCount = 0;
    std::string latencyModelLastRejectReason;
    uint64_t schedulerConfigGeneration = 0;
    std::string schedulerConfigHash;
    std::string schedulerConfigJson;
    NaviSchedLatencyValidationStats latencyValidation;
};

struct NaviSchedMasterSummary : public NaviSchedRoleSummary {
public:
    void Jsonize(JsonWrapper &json) override;
    std::string format() const;

public:
    std::string masterBizName;
    size_t collectorCount = 0;
    bool scheduleGraphRunning = false;
    std::string scheduleGraphState = "stopped";
    uint64_t scheduleGraphRestartCount = 0;
    uint64_t scheduleRequestPipeGeneration = 0;
    uint64_t scheduleSubmitCount = 0;
    size_t schedulePendingRequestCount = 0;
    size_t scheduleProcessingRequestCount = 0;
    size_t activeRequestCount = 0;
    uint64_t scheduleRoundCount = 0;
    uint64_t scheduleRoundRequestCount = 0;
    double scheduleRoundRequestsMean = 0.0;
    size_t scheduleRoundRequestsMin = 0;
    size_t scheduleRoundRequestsMax = 0;
    std::map<std::string, uint64_t> scheduleRoundRequestCountHistogram;
    size_t maxWaitScheduleCount = 0;
    int64_t maxWaitScheduleTimeMs = 0;
    int64_t queueOverloadStartCycleMultiplier = 0;
    int64_t queueOverloadFullCycleMultiplier = 0;
    uint64_t queueOverloadEvaluatedRoundCount = 0;
    uint64_t queueOverloadEvaluatedRequestCount = 0;
    uint64_t queueOverloadRejectedRequestCount = 0;
    double queueOverloadLatestAveragePredictedLatencyMs = 0.0;
    double queueOverloadLatestDropProbability = 0.0;
    bool executionUtilizationControlEnabled = false;
    double executionUtilizationRawAverage = 0.0;
    uint64_t executionUtilizationRunning = 0;
    uint64_t executionUtilizationWaiting = 0;
    bool executionUtilizationGlobalScaling = false;
    double executionUtilizationAverage = 0.0;
    double executionUtilizationIntegral = 0.0;
    size_t executionUtilizationReportingNodes = 0;
    uint64_t schedulerConfigGeneration = 0;
    std::string schedulerConfigHash;
    std::string schedulerConfigJson;
    uint64_t agentRequestReceivedCount = 0;
    uint64_t agentRequestFinishedCount = 0;
    uint64_t agentWorkloadUpdateCount = 0;
    int64_t agentRequestCacheHitTokens = 0;
    int64_t agentRequestCacheMissTokens = 0;
    int64_t agentRequestTotalLatencyUs = 0;
    std::vector<NaviSchedMasterNodeSummary> nodes;
};

}
