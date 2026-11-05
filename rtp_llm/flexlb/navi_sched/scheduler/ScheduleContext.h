#pragma once

#include "aios/navi_sched/common/NaviSchedScheduleData.h"
#include "aios/navi_sched/data/LatencyModelGlobalData.h"
#include "aios/navi_sched/data/MasterDispatchHandle.h"
#include "aios/navi_sched/data/NaviSchedLatencySample.h"
#include "aios/navi_sched/resource/LatencyResidualModel.h"
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace navi_sched {

class MasterResponseToClientData;
class MasterScheduleRequestData;
class MasterScheduleResultKernel;
class MasterStateR;
class MasterLatencyModelR;
class CostSchedulerGraphState;
class CostSchedulerPrepareKernel;
class CostSchedulerKernel;
class CostSchedulerColdStartStrategy;
class NaviSchedNode;
class SchedulerBaseKernel;
struct NaviSchedLatencyModel;

struct NaviSchedColdStartRoutingStats {
    struct TargetRandomDecision {
        size_t nodeIndex = 0;
        double targetInflight = 0.0;
        double admittedTargetInflight = 0.0;
        uint64_t sample = 0;
        double probability = 0.0;
        bool roundedUp = false;
    };
    struct RandomDecision {
        size_t requestIndex = 0;
        uint64_t sample = 0;
        double probability = 0.0;
        bool selected = false;
    };

    double targetInflight = 0.0;
    size_t trustedNodeCount = 0;
    size_t untrustedNodeCount = 0;
    size_t trustedRequestCount = 0;
    size_t untrustedRequestCount = 0;
    size_t rejectedRequestCount = 0;
    size_t untrustedCapacity = 0;
    size_t rampingNodeCount = 0;
    size_t unsafeWaitingMarginalNodeCount = 0;
    double minimumRampProgress = 1.0;
    double minimumRampedTargetInflight = 0.0;
    std::vector<RandomDecision> randomDecisions;
    std::vector<TargetRandomDecision> targetRandomDecisions;
    bool evaluated = false;
};

struct NaviSchedQueueOverloadRandomDecision {
    size_t requestIndex = 0;
    uint64_t sample = 0;
    double probability = 0.0;
    bool accepted = false;
};
struct NaviSchedNodeSnapshot;
using MasterScheduleRequestDataPtr = std::shared_ptr<MasterScheduleRequestData>;
using NaviSchedNodeConstPtr = std::shared_ptr<const NaviSchedNode>;

class NaviSchedScheduleNode {
public:
    double getExecutionUtilization() const;
    double getExecutionBusyUtilization() const;
    int32_t getExecutionRunningRequests() const { return _workload.runningRequestCount; }
    int32_t getExecutionWaitingRequests() const { return _workload.waitingRequestCount; }
    uint64_t getAgentIdHash() const;
    const std::string &getAgentId() const;
    int64_t getTotalBlocks() const;
    int32_t getInflightRequests() const;
    double getInflightPredictedLatencySumMs() const;
    int32_t getBatchWaitingRequests() const;
    int32_t getQueuedBatchRequests() const;
    int32_t getRunningRequests() const;
    uint64_t getBatchTimeoutCount() const;
    uint64_t getAbnormalBatchFinishCount() const;
    double getEstimatedQueueWaitMs() const;
    double getEstimatedQueueWaitMsAt(int64_t predictionTimeUs) const;
    double getEstimatedBacklogLatencyMs() const;
    bool hasLatencyModel() const;
    bool hasValidLatencyModel() const;
    int32_t getLatencyFormula() const;
    int64_t getLatencyBinaryVersion() const;
    int64_t getLatencyModelRevision() const;
    uint64_t getLatencySessionSampleCount() const;
    int32_t getLatencyResidualVersion() const;
    int64_t getLatencyResidualLastUpdateTimeUs() const;
    const std::string &getLatencyModelFingerprint() const;
    bool isLatencyModelTrusted() const;
    bool isBacklogPredictionTrusted() const;
    const std::vector<double> &getLatencyParameters() const;
    double applyLatencyResidual(double basePredictionMs,
                                double *derivativeFactor = nullptr) const;
    double predictLatency(const NaviSchedLatencyFeatures &features) const;
    bool usesUnifiedLatencyPrediction() const;
    const NaviSchedLatencyFeatures &getWorkloadFeatures() const;
    const NaviSchedLatencyFeatures &getReportedWorkloadFeatures() const;
    const NaviSchedLatencyFeatures &getReservedWorkloadFeatures() const;
    bool getUnifiedDecayFloorFeatures(NaviSchedLatencyFeatures &features) const;
    int32_t getMaxRunningRequestCount() const;
    int32_t getWorkloadRunningCapacitySlotCount() const;
    int32_t getReservedRunningCapacitySlotCount() const;
    double getExistingUnifiedLatencyMs() const;
    int64_t getAgentStartTimeUs() const;
    double getReportedRunningElapsedTimeMs() const;
    int32_t getRunningCapacityPrefillRequestCount() const;
    uint64_t getWorkloadRevision() const;
    int64_t getWorkloadEventTimeUs() const;
    int64_t getWorkloadReportReceiveTimeUs() const;
    double getWorkloadReportAgeMs() const;
    bool usesRunningElapsedProgress() const;
    const LatencyResidualModel::RuntimeConfig &getLatencyResidualRuntimeConfig() const;
    double decayUnifiedLatency(double latencyWithReportedWorkloadMs,
                               double latencyWithoutReportedWorkloadMs,
                               bool *usesReportedWorkload = nullptr) const;

private:
    friend class NaviSchedScheduleContext;
    friend class MasterStateR;
    friend class MasterScheduleResultKernel;
    NaviSchedScheduleNode(NaviSchedNodeConstPtr node, NaviSchedNodeSnapshot snapshot,
                          int64_t scheduleTimeUs, double timeCompression = 1.0);
    void refreshExistingUnifiedLatency();

private:
    NaviSchedNodeConstPtr _node;
    std::shared_ptr<const NaviSchedNodeInfo> _info;
    std::shared_ptr<const NaviSchedLatencyModel> _latencyModel;
    LatencyResidualModel::RuntimeConfig _residualRuntimeConfig;
    int64_t _totalBlocks = 0;
    int32_t _inflightRequests = 0;
    double _inflightPredictedLatencySumMs = 0.0;
    int32_t _batchWaitingRequests = 0;
    int32_t _queuedBatchRequests = 0;
    int32_t _runningRequests = 0;
    uint64_t _batchTimeoutCount = 0;
    uint64_t _abnormalBatchFinishCount = 0;
    double _queueTotalPredictedLatencyMs = 0.0;
    double _queueDecayHeadRemainingMs = 0.0;
    int64_t _queueReportReceiveTimeUs = 0;
    int64_t _scheduleTimeUs = 0;
    double _timeCompression = 1.0;
    bool _unifiedLatencyPrediction = false;
    NaviSchedWorkloadSnapshot _workload;
    NaviSchedLatencyFeatures _workloadFeatures;
    NaviSchedLatencyFeatures _reportedWorkloadFeatures;
    NaviSchedLatencyFeatures _reservedWorkloadFeatures;
    int32_t _maxRunningRequestCount = 0;
    int32_t _workloadRunningCapacitySlotCount = 0;
    int32_t _reservedRunningCapacitySlotCount = 0;
    int64_t _workloadReportReceiveTimeUs = 0;
    bool _runningElapsedProgressApplied = false;
    double _existingUnifiedLatencyMs = 0.0;
    int64_t _agentStartTimeUs = 0;
};

class NaviSchedScheduleContext {
private:
    struct DeferredResponse {
        MasterScheduleRequestDataPtr request;
        NaviSchedDecision decision;
    };

public:
    NaviSchedScheduleContext() = default;
    NaviSchedScheduleContext(const NaviSchedScheduleContext &) = delete;
    NaviSchedScheduleContext &operator=(const NaviSchedScheduleContext &) = delete;
    NaviSchedScheduleContext(NaviSchedScheduleContext &&) = default;
    NaviSchedScheduleContext &operator=(NaviSchedScheduleContext &&) = default;

public:
    int64_t getScheduleTimeUs() const;
    int64_t getScheduleBeginTimeUs() const { return _scheduleBeginTimeUs; }
    int64_t getScheduleDeadlineUs() const { return _scheduleDeadlineUs; }
    int64_t getReplayCollectionTimeUs() const { return _replayCollectionTimeUs; }
    void addReplayCollectionTimeUs(int64_t elapsedUs) const
    {
        if (elapsedUs > 0) {
            _replayCollectionTimeUs += elapsedUs;
        }
    }
    size_t requestCount() const;
    size_t nodeCount() const;
    const NaviSchedScheduleItem &getRequest(size_t requestIndex) const;
    const NaviSchedScheduleNode &getNode(size_t nodeIndex) const;
    int64_t getCacheHitBlockCount(size_t requestIndex, size_t nodeIndex) const;
    bool getCacheHitBlockCounts(size_t nodeIndex,
                                std::vector<int64_t> &matchedBlockCounts) const;
    void prepareCacheMatchPrefixes();
    bool isCollectingReplay() const;
    void setSchedulerInfo(autil::legacy::json::JsonMap schedulerInfo);
    bool setScheduleLoss(double loss);
    void recordDiscreteSearchLosses(double initialLoss, double finalLoss) const;
    void recordCostOptimizerProgress(size_t loops, size_t maxLoops,
                                     bool timeBudgetExhausted) const
    {
        _costOptimizerLoopCount = static_cast<int64_t>(loops);
        _costOptimizerMaxLoopCount = static_cast<int64_t>(maxLoops);
        _costOptimizerTimeBudgetExhausted = timeBudgetExhausted;
    }
    std::optional<double> discreteSearchImprovementPct() const;
    void recordCostGradientIteration(int64_t elapsedUs) const
    {
        ++_costGradientIterationCount;
        _costGradientIterationTotalUs += elapsedUs;
        if (elapsedUs > _costGradientIterationMaxUs) {
            _costGradientIterationMaxUs = elapsedUs;
        }
    }
    void recordDiscreteSearch(int64_t elapsedUs, size_t evaluations, size_t moves) const
    {
        _discreteSearchElapsedUs = elapsedUs;
        _discreteSearchEvaluations = evaluations;
        _discreteSearchMoves = moves;
    }
    double estimateBatchExecutionLatencyMs(
        size_t nodeIndex, const std::vector<size_t> &requestIndexes,
        NaviSchedLatencyPrediction *latencyPrediction = nullptr) const;
    bool selectNode(size_t requestIndex, size_t nodeIndex, const std::string &reason,
                    NaviSchedLatencyTrainingPhase latencyTrainingPhase =
                        NaviSchedLatencyTrainingPhase::NORMAL);
    bool reject(size_t requestIndex, const std::string &reason);

private:
    double estimateBatchExecutionLatencyMs(
        size_t nodeIndex, const std::vector<size_t> &requestIndexes,
        NaviSchedLatencyPrediction *latencyPrediction,
        const std::vector<int64_t> *matchedBlockCounts) const;

    friend class MasterScheduleContextKernel;
    friend class MasterScheduleResultKernel;
    friend class MasterScheduleResultPrepareKernel;
    friend class MasterScheduleResultState;
    friend class MasterStateR;
    friend class SchedulerBaseKernel;
    friend class CostSchedulerGraphState;
    friend class CostSchedulerPrepareKernel;
    friend class CostSchedulerKernel;
    friend class CostSchedulerColdStartStrategy;
    static constexpr size_t INVALID_NODE_INDEX = std::numeric_limits<size_t>::max();
    bool valid() const;
    void addRequest(const MasterScheduleRequestDataPtr &request);
    void setNodes(std::vector<NaviSchedNodeConstPtr> nodes,
                  std::vector<NaviSchedNodeSnapshot> nodeSnapshots);
    void setReplay(const std::shared_ptr<NaviSchedScheduleReplay> &replay);
    bool prepare();
    bool activateScheduleView(const std::vector<size_t> &requestIndexes,
                              const std::vector<size_t> &nodeIndexes);
    void clearScheduleView();
    size_t resolveRequestIndex(size_t requestIndex) const;
    size_t resolveNodeIndex(size_t nodeIndex) const;
    bool getReplayMatch(size_t storageRequestIndex, size_t storageNodeIndex,
                        int64_t &matchedBlockCount) const;
    bool prepareResult(int64_t queuePredictionTimeUs);
    bool finish();
    void fail(const std::string &reason);
    void attachLatencyModelGlobals(const MasterLatencyModelR &latencyModels);
    static int64_t makeBatchId(uint64_t hash);
    std::shared_ptr<MasterResponseToClientData> makeResponse(
        const NaviSchedDecision &decision,
        size_t requestIndex = INVALID_NODE_INDEX) const;

private:
    int64_t _scheduleTimeUs = 0;
    int64_t _scheduleBeginTimeUs = 0;
    int64_t _scheduleDeadlineUs = 0;
    int64_t _queuePredictionTimeUs = 0;
    std::shared_ptr<void> _scheduleRoundCompletion;
    std::shared_ptr<void> _scheduleRequestCompletion;
    uint64_t _scheduleId = 0;
    double _timeCompression = 1.0;
    double _scheduleLoss = 0.0;
    int64_t _scheduleLatencyUs = 0;
    int64_t _masterPrepareUs = -1;
    int64_t _snapshotPrepareUs = -1;
    int64_t _snapshotRuntimeLockWaitUs = -1;
    int64_t _snapshotRuntimeLockHoldUs = -1;
    int64_t _replayPrepareUs = -1;
    mutable int64_t _replayCollectionTimeUs = 0;
    int64_t _admissionPrepareUs = -1;
    int64_t _contextPrepareUs = -1;
    int64_t _costPrepareUs = -1;
    int64_t _resultPrepareUs = -1;
    size_t _resultWorkerCount = 1;
    size_t _resultDeliveryWorkerCount = 1;
    int64_t _resultAuditUs = -1;
    int64_t _resultReserveUs = -1;
    int64_t _resultReportReplayUs = -1;
    int64_t _resultDeliveryUs = -1;
    int64_t _resultReplayLogUs = -1;
    int64_t _resultFailureCleanupUs = -1;
    int64_t _resultPartitionUs = -1;
    int64_t _resultFailureCollectUs = -1;
    int64_t _resultMergeUs = -1;
    int64_t _resultPredictionReportUs = -1;
    int64_t _resultReserveStageUs = -1;
    int64_t _resultDeliveryStageUs = -1;
    int64_t _resultCleanupStageUs = -1;
    int64_t _resultReserveDispatchDelayUs = -1;
    int64_t _resultDeliveryDispatchDelayUs = -1;
    int64_t _resultCleanupDispatchDelayUs = -1;
    int64_t _resultFinishUs = -1;
    int64_t _roundTotalUs = -1;
    mutable std::optional<std::pair<double, double>> _discreteSearchLosses;
    mutable int64_t _discreteSearchElapsedUs = -1;
    mutable size_t _discreteSearchEvaluations = 0;
    mutable size_t _discreteSearchMoves = 0;
    mutable int64_t _costOptimizerLoopCount = -1;
    mutable int64_t _costOptimizerMaxLoopCount = 0;
    mutable bool _costOptimizerTimeBudgetExhausted = false;
    mutable size_t _costGradientIterationCount = 0;
    mutable int64_t _costGradientIterationTotalUs = 0;
    mutable int64_t _costGradientIterationMaxUs = 0;
    std::vector<NaviSchedScheduleItem> _requests;
    std::vector<NaviSchedScheduleNode> _scheduleNodes;
    std::vector<MasterScheduleRequestDataPtr> _responseRequests;
    std::shared_ptr<NaviSchedScheduleReplay> _replay;
    std::unordered_map<uint64_t, size_t> _replayMatchRequestIndexes;
    std::vector<NaviSchedDecision> _decisions;
    std::vector<MasterDispatchHandle> _dispatchHandles;
    std::vector<size_t> _selectedNodeIndexes;
    std::vector<size_t> _responseDeliveryFailedIndexes;
    std::vector<NaviSchedQueueOverloadRandomDecision> _queueOverloadRandomDecisions;
    bool _queueOverloadEvaluated = false;
    double _queueOverloadAveragePredictedLatencyMs = 0.0;
    double _queueOverloadDropProbability = 0.0;
    int64_t _queueOverloadScheduleCycleMs = 0;
    int64_t _queueOverloadStartCycleMultiplier = 0;
    int64_t _queueOverloadFullCycleMultiplier = 0;
    size_t _queueOverloadRequestCount = 0;
    size_t _queueOverloadRejectedRequestCount = 0;
    std::vector<DeferredResponse> _deferredResponses;
    std::unordered_map<std::string, LatencyModelGlobalData> _latencyModelGlobals;
    std::vector<size_t> _scheduleViewRequestIndexes;
    std::vector<size_t> _scheduleViewNodeIndexes;
    std::vector<NaviSchedScheduleItem> _scheduleViewRequests;
    std::vector<size_t> _cacheMatchCommonPrefixes;
    bool _scheduleViewUsesAllRequests = false;
    NaviSchedColdStartRoutingStats _coldStartRoutingStats;
    bool _scheduleViewActive = false;
    bool _prepared = false;
    bool _resultPrepared = false;
};

using NaviSchedScheduleContextPtr = std::shared_ptr<NaviSchedScheduleContext>;

}
