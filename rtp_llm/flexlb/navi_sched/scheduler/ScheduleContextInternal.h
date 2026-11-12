#pragma once

#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/common/TPUPrefillExecutionPolicy.h"
#include "aios/navi_sched/data/NaviSchedLatencySample.h"
#include "aios/navi_sched/resource/LatencyResidualModel.h"

#include <memory>
#include <optional>
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
    NaviSchedQueueModelState queueModel;
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
    bool peerCacheReadAvailable = false;
    uint64_t peerCacheReadRevision = 0;
    std::optional<TPUPrefillExecutionPolicyConfig> prefillExecutionPolicy;
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

// One immutable READY-only lookup per receiver and schedule round.
// Each row maps a receiver's existing local prefix to the usable total prefix.
struct NaviSchedPeerCacheSuffixLookup {
    std::vector<std::vector<int64_t>> endBlocks;
    int64_t alignmentBlocks = 1;
    int64_t extend(size_t requestIndex, int64_t localBlocks) const;
    static std::shared_ptr<const NaviSchedPeerCacheSuffixLookup> build(
        const std::vector<NaviSchedScheduleItem> &requests,
        const std::vector<std::vector<size_t>> &keyIndexes,
        const std::vector<uint8_t> &readyStates, const NaviSchedCachePolicy &policy);
};

class NaviSchedNode {
public:
    virtual ~NaviSchedNode() = default;
    // Query only this round's requested keys; bits are Full READY=1, Tail READY=2.
    // No pending or planned publication is exported across nodes.
    virtual bool getReadyExternalCacheKeyStates(const NaviSchedBlockHashes &,
                                                std::vector<uint8_t> &,
                                                NaviSchedCachePolicy &) const
    {
        return false;
    }
    virtual bool getReadyExternalCacheCounts(const std::vector<NaviSchedScheduleItem> &,
                                             std::vector<int64_t> &,
                                             NaviSchedCachePolicy &) const
    {
        return false;
    }

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
    // One cache observation supplies both the usable hit and Full-only prefix.
    // False means unavailable, never an observed zero boundary.
    virtual bool getExecutionCacheInputs(const NaviSchedScheduleRequest &, int64_t &,
                                         int64_t &) const
    {
        return false;
    }
    // Optional suffix output shares the batch observation; local = total - suffix.
    // Output arrays have requests.size() entries and must not overlap.
    virtual void getMatchedBlockCounts(
        const std::vector<NaviSchedScheduleItem> &requests, int64_t *matchedBlockCounts,
        int64_t *externalSuffixBlockCounts = nullptr) const = 0;
    virtual bool getExecutionCacheInputs(
        const std::vector<NaviSchedScheduleItem> &requests,
        std::vector<TPUPrefillRequestInput> &inputs,
        int64_t *externalSuffixBlockCounts = nullptr,
        ::navi_sched::TPUPrefillCacheObservationBatch *observation = nullptr) const = 0;
    virtual void getMatchedBlockCounts(const std::vector<NaviSchedScheduleItem> &requests,
                                       int64_t *matchedBlockCounts,
                                       const std::vector<size_t> &commonPrefixes) const
    {
        getMatchedBlockCounts(requests, matchedBlockCounts, nullptr);
    }
};

using NaviSchedNodeConstPtr = std::shared_ptr<const NaviSchedNode>;

}
