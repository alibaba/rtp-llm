#pragma once

#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/data/NaviSchedLatencySample.h"
#include "aios/navi_sched/resource/LatencyResidualModel.h"

#include <memory>
#include <string>
#include <vector>

namespace navi_sched {

struct NaviSchedNodeLoad {
    int64_t totalBlocks = 0;
    uint64_t inflightStatsRevision = 0;
    int32_t inflightRequests = 0;
    double inflightPredictedLatencySumMs = 0.0;
    int32_t batchWaitingRequests = 0;
    int32_t queuedBatchRequests = 0;
    int32_t runningRequests = 0;
    uint64_t batchTimeoutCount = 0;
    uint64_t abnormalBatchFinishCount = 0;
    double queueTotalPredictedLatencyMs = 0.0;
    double queueDecayHeadRemainingMs = 0.0;
    int64_t queueReportReceiveTimeUs = 0;
};

struct NaviSchedLatencyModel {
    int32_t formula = 0;
    int64_t binaryVersion = 0;
    int64_t revision = 0;
    std::vector<double> parameters;
    int32_t residualVersion = NAVI_SCHED_LATENCY_RESIDUAL_VERSION;
    std::vector<double> residualParameters =
        std::vector<double>(NAVI_SCHED_LATENCY_RESIDUAL_PARAMETER_COUNT, 0.0);
    int64_t residualLastUpdateTimeUs = 0;
    bool overloadPredictionTrusted = false;
    bool warmStarted = false;
    uint64_t sessionSampleCount = 0;
    uint32_t consecutiveRejectedCount = 0;
    std::string lastRejectReason;
};

struct NaviSchedNodeSnapshot {
    int64_t runtimeLockWaitUs = 0;
    int64_t runtimeLockHoldUs = 0;
    std::shared_ptr<const NaviSchedNodeInfo> info;
    std::shared_ptr<const NaviSchedLatencyModel> latencyModel;
    LatencyResidualModel::RuntimeConfig residualRuntimeConfig;
    NaviSchedNodeLoad load;
    NaviSchedWorkloadSnapshot workload;
    NaviSchedLatencyFeatures workloadFeatures;
    NaviSchedLatencyFeatures reportedWorkloadFeatures;
    NaviSchedLatencyFeatures reservedWorkloadFeatures;
    int32_t maxRunningRequestCount = 0;
    int32_t workloadRunningCapacitySlotCount = 0;
    int32_t reservedRunningCapacitySlotCount = 0;
    bool unifiedLatencyPrediction = false;
    int64_t workloadReportReceiveTimeUs = 0;
    int64_t agentStartTimeUs = 0;
};

class NaviSchedNode {
public:
    virtual ~NaviSchedNode() = default;

public:
    virtual double getExecutionUtilization() const { return -1.0; }
    virtual double getExecutionBusyUtilization() const { return -1.0; }
    virtual NaviSchedNodeInfo getInfo() const = 0;
    virtual bool getScheduleSnapshot(const std::string &bizName,
                                     NaviSchedNodeSnapshot &snapshot) const = 0;
    virtual int64_t getMatchedBlockCount(
        const NaviSchedBlockHashes &blockHashes) const = 0;
    virtual int64_t getMatchedBlockCountForRequest(
        const NaviSchedScheduleRequest &request) const
    {
        static const NaviSchedBlockHashes empty;
        return getMatchedBlockCount(request.blockHashes ? *request.blockHashes : empty);
    }
    virtual void getMatchedBlockCounts(
        const std::vector<NaviSchedScheduleItem> &requests,
        int64_t *matchedBlockCounts) const = 0;
    virtual void getMatchedBlockCounts(
        const std::vector<NaviSchedScheduleItem> &requests,
        int64_t *matchedBlockCounts,
        const std::vector<size_t> &commonPrefixes) const
    {
        getMatchedBlockCounts(requests, matchedBlockCounts);
    }
};

using NaviSchedNodeConstPtr = std::shared_ptr<const NaviSchedNode>;

}
