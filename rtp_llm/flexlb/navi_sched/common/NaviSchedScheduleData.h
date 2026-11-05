#pragma once

#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "aios/navi_sched/data/NaviSchedLatencyPrediction.h"
#include "aios/navi_sched/data/NaviSchedLatencySample.h"
#include "aios/navi_sched/data/NaviSchedQueueModel.h"
#include <autil/legacy/json.h>

namespace navi_sched {

struct NaviSchedServerInfo {
    bool valid() const;
    std::string getAddress(const std::string &defaultIp = "") const;
    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);

    std::string ip;
    int32_t port = 0;
};

struct NaviSchedTarget {
    int32_t version = NAVI_SCHED_DATA_VERSION;
    std::string bizName;
    std::string address;
    std::string agentAddress;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

struct NaviSchedDecision {
    int32_t version = NAVI_SCHED_DATA_VERSION;
    uint64_t requestId = 0;
    uint64_t scheduleId = 0;
    double scheduleLoss = 0.0;
    int64_t scheduleLatencyUs = 0;
    int64_t scheduleTimeUs = 0;
    std::string reason;
    int64_t batchId = 0;
    int64_t expectedBatchSize = 0;
    int64_t predictedCacheHitTokenCount = 0;
    double predictedQueueLatencyMs = 0.0;
    double predictedExecutionLatencyMs = 0.0;
    NaviSchedLatencyPredictionConstPtr latencyPrediction;
    NaviSchedLatencyTrainingPhase latencyTrainingPhase =
        NaviSchedLatencyTrainingPhase::NORMAL;
    NaviSchedTarget target;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

struct NaviSchedExecutionLatencyBucketStats {
    uint64_t sampleCount = 0;
    double masterExecutionMaeMs = 0.0;
    double masterExecutionMapePct = 0.0;
    double agentExecutionMaeMs = 0.0;
    double agentExecutionMapePct = 0.0;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

struct NaviSchedLatencyValidationStats {
    uint64_t sampleCount = 0;
    uint64_t unmatchedBatchCount = 0;
    double batchRequestCountMean = 0.0;
    double predictedCacheHitTokensMean = 0.0;
    double actualCacheHitTokensMean = 0.0;
    double cacheHitTokensBiasMean = 0.0;
    double cacheHitTokensMae = 0.0;
    double masterPredictedQueueMeanMs = 0.0;
    double actualQueueMeanMs = 0.0;
    double backendQueueMeanMs = 0.0;
    double queueBiasMeanMs = 0.0;
    double queueMaeMs = 0.0;
    double queueRawPredictedMeanMs = 0.0;
    double queueRawBiasMeanMs = 0.0;
    double queueRawMaeMs = 0.0;
    double agentPredictedQueueMeanMs = 0.0;
    double agentQueueBiasMeanMs = 0.0;
    double agentQueueMaeMs = 0.0;
    double queueFeatureBatchCountMean = 0.0;
    std::vector<double> queueModelWeights;
    uint64_t queueModelFitStep = 0;
    uint64_t zeroPredictedQueueSampleCount = 0;
    double zeroPredictedQueueActualMeanMs = 0.0;
    double zeroPredictedQueueMaeMs = 0.0;
    uint64_t positivePredictedQueueSampleCount = 0;
    double positivePredictedQueueMeanMs = 0.0;
    double positivePredictedQueueActualMeanMs = 0.0;
    double positivePredictedQueueBiasMeanMs = 0.0;
    double positivePredictedQueueMaeMs = 0.0;
    double masterPredictedExecutionMeanMs = 0.0;
    double agentPredictedExecutionMeanMs = 0.0;
    double actualExecutionMeanMs = 0.0;
    double masterExecutionBiasMeanMs = 0.0;
    double masterExecutionMaeMs = 0.0;
    double masterExecutionMapePct = 0.0;
    double agentExecutionBiasMeanMs = 0.0;
    double agentExecutionMaeMs = 0.0;
    double agentExecutionMapePct = 0.0;
    double masterPredictedTotalMeanMs = 0.0;
    double actualTotalMeanMs = 0.0;
    double totalBiasMeanMs = 0.0;
    double totalMaeMs = 0.0;
    double totalMapePct = 0.0;
    std::vector<NaviSchedExecutionLatencyBucketStats> executionLatencyBuckets;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

struct NaviSchedRequestTiming {
    double batchWaitMs = 0.0;
    double queueWaitMs = 0.0;
    double executionLatencyMs = 0.0;
    double clientTotalLatencyMs = 0.0;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

using NaviSchedRequestTimingPtr = std::shared_ptr<NaviSchedRequestTiming>;

struct NaviSchedScheduleReplay;
using NaviSchedScheduleReplayPtr = std::shared_ptr<const NaviSchedScheduleReplay>;

struct NaviSchedScheduleResult {
    int32_t errorCode = 0;
    std::string errorMessage;
    NaviSchedDecision decision;
    NaviSchedScheduleReplayPtr replay;
    NaviSchedRequestTimingPtr requestTiming;
};

struct NaviSchedNodeInfo {
    uint64_t agentIdHash = 0;
    std::string agentId;
    std::string bizName;
    std::string collectorAddress;
    NaviSchedServerInfo serverInfo;
    std::string serverAddress;
    int64_t totalBlocks = 0;
    uint64_t inflightStatsRevision = 0;
    int32_t inflightRequests = 0;
    double inflightPredictedLatencySumMs = 0.0;
    int32_t batchWaitingRequests = 0;
    int32_t queuedBatchRequests = 0;
    int32_t runningRequests = 0;
    bool pdSeparated = true;
    bool unifiedLatencyPrediction = false;
    NaviSchedWorkloadSnapshot workload;
    NaviSchedLatencyFeatures reservedWorkloadFeatures;
    bool workloadFeatureSnapshotValid = false;
    NaviSchedLatencyFeatures workloadFeatures;
    int32_t maxRunningRequestCount = 0;
    int32_t workloadRunningCapacitySlotCount = 0;
    int32_t reservedRunningCapacitySlotCount = 0;
    int64_t workloadReportReceiveTimeUs = 0;
    uint64_t workloadStaleUpdateCount = 0;
    uint64_t batchTimeoutCount = 0;
    uint64_t abnormalBatchFinishCount = 0;
    double lastRequestLatencyMs = 0.0;
    double queueTotalPredictedLatencyMs = 0.0;
    double queueDecayHeadRemainingMs = 0.0;
    NaviSchedQueueModelState queueModel;
    int64_t queueReportReceiveTimeUs = 0;
    bool online = false;
    std::string lastEventType;
    int64_t lastEventTimeUs = 0;
    std::string lastRequestEventType;
    int64_t lastRequestEventTimeUs = 0;
    int64_t lastReceiveTimeUs = 0;
    int32_t latencyFormula = 0;
    int64_t latencyBinaryVersion = 0;
    int64_t latencyModelRevision = 0;
    std::vector<double> latencyParameters;
    int32_t latencyResidualVersion = NAVI_SCHED_LATENCY_RESIDUAL_VERSION;
    std::vector<double> latencyResidualParameters =
        std::vector<double>(NAVI_SCHED_LATENCY_RESIDUAL_PARAMETER_COUNT, 0.0);
    int64_t latencyResidualLastUpdateTimeUs = 0;
    std::string latencyModelFingerprint;
    bool executionGeometryEnabled = false;
    bool projectedExecutionGeometryEnabled = false;
    bool peerCacheEnabled = false;
    // Live Master diagnostics; intentionally excluded from persisted model/replay wire
    // data.
    bool peerCacheReadAvailable = false;
    uint64_t peerCacheReadRevision = 0;
    int64_t peerCacheReadAgentStartTimeUs = 0;
    TPUPrefillExecutionPolicyConfig prefillExecutionPolicy;
    bool latencyModelTrusted = false;
    bool latencyModelWarmStarted = false;
    uint32_t latencyModelConsecutiveRejectedCount = 0;
    std::string latencyModelLastRejectReason;
    uint64_t schedulerConfigGeneration = 0;
    std::string schedulerConfigHash;
    std::string schedulerConfigJson;
    NaviSchedLatencyValidationStats latencyValidation;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

struct NaviSchedScheduleRequest {
    bool valid() const;

    int32_t version = NAVI_SCHED_DATA_VERSION;
    uint64_t requestId = 0;
    std::string bizName;
    int64_t promptTokenCount = 0;
    // Explicit runtime checkpoint, before node-specific block alignment.
    int64_t explicitCacheCheckpointPosition = 0;
    int64_t blockSize = 256;
    NaviSchedBlockHashesConstPtr blockHashes;
    int64_t timeoutMs = 0;
    std::string forwardPayload;
    bool routeOnly = false;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

struct NaviSchedForwardRequest {
    NaviSchedScheduleRequest request;
    NaviSchedDecision decision;
};

struct NaviSchedScheduleItem {
    NaviSchedScheduleRequest request;
    int64_t waitTimeMs = 0;
    int64_t remainingTimeoutMs = 0;
    bool timeoutTriggered = false;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

struct NaviSchedReplayNodeMatch {
    int64_t matchedBlockCount = 0;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

struct NaviSchedScheduleReplay {
    int32_t version = NAVI_SCHED_DATA_VERSION;
    int64_t scheduleTimeUs = 0;
    std::string bizName;
    std::vector<NaviSchedScheduleItem> requests;
    std::vector<NaviSchedNodeInfo> nodes;
    std::vector<std::vector<NaviSchedReplayNodeMatch>> nodeMatches;
    std::vector<NaviSchedDecision> decisions;
    autil::legacy::json::JsonMap schedulerInfo;

    void serialize(autil::DataBuffer &dataBuffer) const;
    void deserialize(autil::DataBuffer &dataBuffer);
};

}
