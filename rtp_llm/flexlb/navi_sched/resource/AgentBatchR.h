#pragma once

#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "aios/navi_sched/resource/PendingCacheUtil.h"
#include "aios/navi_sched/scheduler/ExecutionUtilizationController.h"
#include "autil/Lock.h"
#include "autil/LockFreeQueue.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include "navi/log/NaviLogger.h"

namespace navi_sched {

class AgentBatchR : public navi::ExternalResource {
public:
    struct CacheHitPathSample {
        uint64_t requestId = 0;
        int64_t masterPredictedTokens = 0;
        int64_t agentPhysicalTokens = 0;
        int64_t agentPendingAwareTokens = 0;
        int64_t rtpActualTokens = 0;
        int64_t masterToAgentPhysicalErrorTokens = 0;
        int64_t agentPhysicalToRtpErrorTokens = 0;
        int64_t masterToAgentPendingAwareErrorTokens = 0;
        int64_t agentPendingAwareToRtpErrorTokens = 0;
        int64_t masterToRtpErrorTokens = 0;
        bool blockAligned = false;
    };

    struct ExecutionState {
        int32_t batchWaitingRequests = 0;
        int32_t queuedBatchRequests = 0;
        int32_t runningRequests = 0;
        int32_t batchRequestCount = 0;
        int64_t batchId = 0;
        int64_t scheduleBatchId = 0;
        std::vector<uint64_t> requestIds;
        NaviSchedLocalBatchInfo localBatch;
        int64_t eventTimeUs = 0;
        double batchLatencyMs = 0.0;
        double backendQueueLatencyMs = 0.0;
        double predictedQueueLatencyMs = 0.0;
        double predictedExecutionLatencyMs = 0.0;
        int64_t predictedCacheHitTokenCount = 0;
        int64_t actualCacheHitTokenCount = 0;
        int32_t cacheHitMismatchRequestCount = 0;
        int64_t cacheHitAbsoluteErrorTokenCount = 0;
        int64_t cacheHitMaxAbsoluteErrorTokenCount = 0;
        std::vector<CacheHitPathSample> cacheHitPathSamples;
        bool latencySampleValid = true;
        bool latencyValidationValid = true;
    };

    struct GraphRecoveryBatch {
        NaviSchedBatchPtr batch;
        int64_t startTimeUs = 0;
        bool running = false;
    };

    struct GraphRecoveryState {
        std::vector<GraphRecoveryBatch> batches;
        int32_t batchWaitingRequests = 0;
        int32_t queuedBatchRequests = 0;
        int32_t runningRequests = 0;
    };

    struct ReadyWithoutRunningStats {
        uint64_t gapCount = 0;
        int64_t totalTimeUs = 0;
        int64_t maxTimeUs = 0;
        int64_t currentTimeUs = 0;
    };

    struct BatchFormationStats {
        uint64_t batchCount = 0;
        int64_t totalTimeUs = 0;
        int64_t maxTimeUs = 0;
    };

    struct BackendBatchExecutionSample {
        int64_t wallTimeUs = 0;
        int64_t executeModelTimeUs = 0;
        int64_t executionStepCount = 0;
        int64_t prefillTokenCount = 0;
        int64_t decodeRequestStepCount = 0;
        int64_t decodeTokenCount = 0;
        int64_t admissionToFirstExecutionTimeUs = -1;
    };

    struct BackendBatchExecutionStats {
        uint64_t batchCount = 0;
        int64_t wallTimeUs = 0;
        int64_t maxWallTimeUs = 0;
        int64_t executeModelTimeUs = 0;
        int64_t interStepGapTimeUs = 0;
        int64_t maxInterStepGapTimeUs = 0;
        int64_t executionStepCount = 0;
        int64_t prefillTokenCount = 0;
        int64_t decodeRequestStepCount = 0;
        int64_t decodeTokenCount = 0;
        uint64_t admissionToFirstExecutionSampleCount = 0;
        int64_t admissionToFirstExecutionTimeUs = 0;
        int64_t maxAdmissionToFirstExecutionTimeUs = 0;
    };

    struct BackendExecutionStepSample {
        int64_t elapsedTimeUs = 0;
        bool pureDecode = false;
    };

    struct BackendExecutionStepStats {
        uint64_t stepCount = 0;
        int64_t totalTimeUs = 0;
        int64_t maxTimeUs = 0;
        uint64_t pureDecodeStepCount = 0;
        int64_t pureDecodeTotalTimeUs = 0;
        int64_t pureDecodeMaxTimeUs = 0;
    };

    struct CoreStepSample {
        int64_t elapsedTimeUs = 0;
        bool futureWait = false;
    };

    struct CoreStepStats {
        uint64_t stepCount = 0;
        int64_t totalTimeUs = 0;
        int64_t maxTimeUs = 0;
        uint64_t futureWaitCount = 0;
        int64_t futureWaitTotalTimeUs = 0;
        int64_t futureWaitMaxTimeUs = 0;
    };

    struct CacheHitPathStats {
        uint64_t batchCount = 0;
        uint64_t requestCount = 0;
        int64_t masterPredictedTokens = 0;
        int64_t agentArrivalTokens = 0;
        int64_t agentPendingAwareTokens = 0;
        int64_t rtpActualTokens = 0;
        uint64_t masterToAgentAbsoluteErrorTokens = 0;
        uint64_t agentToRtpAbsoluteErrorTokens = 0;
        uint64_t masterToAgentPendingAwareAbsoluteErrorTokens = 0;
        uint64_t agentPendingAwareToRtpAbsoluteErrorTokens = 0;
        uint64_t masterToRtpAbsoluteErrorTokens = 0;
        uint64_t alignedRequestCount = 0;
        int64_t alignedAgentToRtpErrorTokens = 0;
        uint64_t alignedAgentToRtpAbsoluteErrorTokens = 0;
        uint64_t unalignedRequestCount = 0;
        int64_t unalignedAgentToRtpErrorTokens = 0;
        uint64_t unalignedAgentToRtpAbsoluteErrorTokens = 0;
        uint64_t hybridDebugRequestCount = 0;
        int64_t hybridAgentPredictedBlocks = 0;
        int64_t hybridCandidateBlocks = 0;
        int64_t hybridFlatMatchedBlocks = 0;
        int64_t hybridMinFullReuseBlocks = 0;
        int64_t hybridSelectedReuseBlocks = 0;
        int64_t hybridAgentToFullOverpredictionBlocks = 0;
        int64_t hybridTailRollbackBlocks = 0;
        uint64_t hybridInvalidOrderCount = 0;
    };

    struct CacheKeyAudit {
        NaviSchedBlockHashesConstPtr blockHashes;
        int64_t physicalCacheHitTokenCount = -1;
        int64_t arrivalTimeUs = 0;
        uint64_t cacheRevision = 0;
        size_t cachedBlockCount = 0;
        uint64_t cacheHashChecksum = 0;
        size_t fullReadyBlockCount = 0;
        uint64_t fullReadyHashChecksum = 0;
        size_t tailReadyBlockCount = 0;
        uint64_t tailReadyHashChecksum = 0;
        bool hybridTailRequired = false;
        uint64_t startCacheRevision = 0;
        size_t startCachedBlockCount = 0;
        uint64_t startCacheHashChecksum = 0;
        size_t startFullReadyBlockCount = 0;
        uint64_t startFullReadyHashChecksum = 0;
        size_t startTailReadyBlockCount = 0;
        uint64_t startTailReadyHashChecksum = 0;
        int64_t startRtpCachedBlockCount = -1;
    };

    AgentBatchR();
    AgentBatchR(int64_t batchTimeoutMs, int64_t closedBatchKeepMs);

public:
    bool beginLifecycleCallback();
    void endLifecycleCallback();
    bool addRequest(const NaviSchedRequestStats &request, int64_t batchId,
                    int64_t expectedBatchSize,
                    const NaviSchedReadyBatchCallback &callback,
                    const std::function<void()> &acceptedCallback = {});
    bool addRequest(
        const NaviSchedRequestStats &request, int64_t batchId, int64_t expectedBatchSize,
        const NaviSchedReadyBatchCallback &callback,
        const std::function<void()> &acceptedCallback, const CacheKeyAudit &cacheKeyAudit,
        const std::function<void(const ExecutionState &)> &acceptedFailureCallback = {},
        int64_t batchTimeoutMs = 0);
    bool startBatch(int64_t batchId, const std::vector<uint64_t> &requestIds,
                    ExecutionState &state);
    bool startBatch(int64_t batchId, const std::vector<uint64_t> &requestIds,
                    int64_t executionStartTimeUs, ExecutionState &state);
    bool recordCacheStateAtStart(int64_t batchId, uint64_t cacheRevision,
                                 size_t cachedBlockCount, uint64_t cacheHashChecksum,
                                 size_t fullReadyBlockCount,
                                 uint64_t fullReadyHashChecksum,
                                 size_t tailReadyBlockCount,
                                 uint64_t tailReadyHashChecksum,
                                 int64_t rtpCachedBlockCount);
    bool makePendingCacheReservation(int64_t batchId, const NaviSchedCachePolicy &policy,
                                     NaviSchedPendingCacheReservation &reservation) const;
    bool markPendingCacheTracked(int64_t batchId);
    bool finishBatch(int64_t batchId, const std::vector<NaviSchedRequestStats> &requests,
                     ExecutionState &state,
                     std::vector<NaviSchedRequestStats> *completedRequests = nullptr,
                     double nativePreparationMs = 0.0);
    bool abortBatchRequests(int64_t batchId, const std::vector<uint64_t> &requestIds,
                            ExecutionState &state,
                            std::vector<NaviSchedRequestStats> &abortedRequests,
                            NaviSchedBatchPtr &remainingBatch,
                            NaviSchedPendingCacheReservation &remainingPendingCache);
    bool abortPendingBatch(int64_t scheduleBatchId, ExecutionState &state);
    bool abortBatch(int64_t batchId, ExecutionState &state);
    bool getExecutionState(ExecutionState &state) const;
    bool beginBatchExecution(int64_t batchId);
    struct ExecutionUtilizationSnapshot {
        double busyEma = 0.0;
        double busyWindow = 0.0;
        int64_t windowBusyUs = 0;
        int64_t windowElapsedUs = 0;
        int64_t windowSampleMonoUs = 0;
        size_t activeCount = 0;
        int64_t sampleBusyUs = 0;
        int64_t sampleElapsedUs = 0;
        int32_t runningRequests = 0;
        int32_t queuedRequests = 0;
        double utilization = 0.0;
        double executionCapacity = 0.0;
        double executionCapacityDenominator = 0.0;
        int32_t gpuReadyQueuedRequests = 0;
    };
    ExecutionUtilizationSnapshot getExecutionUtilizationSnapshot() const;
    const std::shared_ptr<std::atomic<double>> &getExecutionUtilizationSource() const
    {
        return _executionUtilization;
    }
    const std::shared_ptr<std::atomic<double>> &getExecutionBusyUtilizationSource() const
    {
        return _backendExecutionUtilization;
    }
    GraphRecoveryState getGraphRecoveryState() const;
    bool setBatchTimeoutMs(int64_t batchTimeoutMs);
    void setRuntimeConfig(int64_t batchTimeoutMs, int64_t closedBatchKeepMs);
    void expireBatches();
    std::vector<ExecutionState> stop();
    bool prepareGraphStart();
    uint64_t beginGraphPipeGeneration();
    bool setGraphPipe(uint64_t generation, const navi::AsyncPipePtr &graphPipe);
    bool waitGraphPipe(uint64_t generation, int64_t timeoutMs) const;
    void clearGraphPipe(uint64_t generation);
    bool finishGraph();
    uint64_t getGraphPipeGeneration() const;
    int64_t getNextBatchTimeoutMs() const;
    int64_t getLatestLocalBatchId() const;
    size_t getPendingBatchCount() const;
    uint64_t getTimeoutBatchCount() const;
    uint64_t getTimeoutMissingRequestCount() const;
    uint64_t getLateRequestCount() const;
    uint64_t getAbnormalBatchFinishCount() const;
    ReadyWithoutRunningStats getReadyWithoutRunningStats() const;
    BatchFormationStats takeBatchFormationWindowStats() const;
    void recordBackendBatchExecution(const BackendBatchExecutionSample &sample) const;
    BackendBatchExecutionStats takeBackendBatchExecutionWindowStats() const;
    void recordBackendExecutionStep(const BackendExecutionStepSample &sample) const;
    BackendExecutionStepStats takeBackendExecutionStepWindowStats() const;
    void recordCoreStep(const CoreStepSample &sample) const;
    CoreStepStats takeCoreStepWindowStats() const;
    CacheHitPathStats getCacheHitPathStats() const;
    int64_t getExecutionBusyTimeUs(int64_t currentTimeUs) const;
    bool beginBackendExecution(int64_t eventTimeUs = 0, int32_t requestCount = 0);
    void updateBackendWaitingRequests(int32_t requestCount, int32_t runningRequests = 0);
    bool endBackendExecution(int64_t eventTimeUs = 0);
    bool hasBackendExecutionObservation() const;
    int64_t getBackendExecutionBusyTimeUs(int64_t currentTimeUs) const;
    bool sampleBackendExecutionUtilization(int64_t nowMonoUs = 0);
    static const std::vector<NaviSchedRequestStats> &getRequestStats(
        const NaviSchedBatch &batch);
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    enum class AppendResult {
        ACCEPTED,
        INVALID_REQUEST,
        PREDICTION_MISMATCH,
        DUPLICATE_REQUEST,
    };

    struct CallbackBatch {
        struct CacheHitPathTokens {
            int64_t masterTokens = 0;
            int64_t agentPhysicalTokens = 0;
            int64_t agentPendingAwareTokens = 0;
            int64_t totalTokens = 0;
            int64_t blockSize = 0;
            CacheKeyAudit audit;
        };

        int64_t batchId = 0;
        int64_t scheduleBatchId = 0;
        int64_t expectedBatchSize = 0;
        int64_t deadlineTimeUs = 0;
        int64_t firstRequestTimeUs = 0;
        int64_t lastRequestTimeUs = 0;
        int64_t timeoutDetectedTimeUs = 0;
        size_t timeoutReceivedRequestCount = 0;
        int64_t startTimeUs = 0;
        int64_t readyTimeUs = 0;
        int64_t readyAcceptedCallbackTimeUs = 0;
        double backendQueueLatencyMs = 0.0;
        double predictedQueueLatencyMs = 0.0;
        double predictedExecutionLatencyMs = 0.0;
        int64_t predictedCacheHitTokenCount = 0;
        size_t pendingAcceptedRequestCount = 0;
        bool pendingCacheTracked = false;
        NaviSchedCachePolicy pendingCachePolicy;
        bool abortRequested = false;
        bool callbackDispatched = false;
        bool executionBusy = false;
        bool latencySampleValid = true;
        bool latencyValidationValid = true;
        std::unordered_set<uint64_t> requestIdSet;
        std::vector<NaviSchedRequestStats> requestStats;
        std::unordered_map<uint64_t, CacheHitPathTokens> cacheHitPathTokens;
        NaviSchedBatchPtr batch;
        NaviSchedReadyBatchCallback callback;
        std::function<void(const ExecutionState &)> acceptedFailureCallback;
    };
    NAVI_TYPEDEF_PTR(CallbackBatch);

    struct ClosedBatch {
        int64_t expireTimeUs = 0;
        bool aborted = false;
    };

private:
    static CallbackBatchPtr createCallbackBatch(
        int64_t expectedBatchSize, const NaviSchedReadyBatchCallback &callback,
        const std::function<void(const ExecutionState &)> &acceptedFailureCallback);
    static AppendResult appendRequest(const CallbackBatchPtr &batch,
                                      const NaviSchedRequestStats &request,
                                      const CacheKeyAudit &cacheKeyAudit);
    CallbackBatchPtr markRequestAccepted(const CallbackBatchPtr &batch);
    static CallbackBatchPtr takeReadyCallback(const CallbackBatchPtr &batch);
    static NaviSchedLocalBatchInfo makeLocalBatchInfo(const CallbackBatch &batch);
    bool abortPendingBatch(int64_t scheduleBatchId, ExecutionState &state,
                           const CallbackBatchPtr &expectedBatch);
    void finishCallbackWorkLocked();
    void finishCallbackWork();
    void publishReadyBatch(const CallbackBatchPtr &batch);
    void flushExpiredCallbackBatches(int64_t nowTimeUs,
                                     std::vector<CallbackBatchPtr> &readyBatches,
                                     std::vector<CallbackBatchPtr> &timedOutBatches);
    void logBatchTimeout(const CallbackBatchPtr &batch);
    void recycleClosedBatches(int64_t nowTimeUs);
    void makeCallbackBatchReady(int64_t groupBatchId, CallbackBatchPtr batch,
                                bool timedOut, int64_t nowTimeUs);
    int64_t makeBatchId();
    bool runCallbackBatch(const CallbackBatchPtr &batch);
    void makeExecutionState(ExecutionState &state) const;
    void updateReadyWithoutRunningLocked(int64_t nowTimeUs);
    void finishExecutionBusyTimeLocked(int64_t nowTimeUs);
    void beginUtilizationExecutionLocked(CallbackBatch &batch);
    void finishUtilizationExecutionLocked(CallbackBatch &batch);
    void refreshExecutionUtilizationLocked() const;

private:
    DECLARE_LOGGER();
    int64_t _batchTimeoutMs;
    int64_t _closedBatchKeepMs;
    mutable autil::ThreadCond _lock;
    std::unordered_map<int64_t, CallbackBatchPtr> _callbackBatches;
    std::unordered_map<int64_t, ClosedBatch> _closedBatches;
    std::unordered_map<int64_t, CallbackBatchPtr> _readyBatches;
    std::unordered_map<int64_t, CallbackBatchPtr> _runningBatches;
    int32_t _batchWaitingRequestCount = 0;
    int32_t _queuedBatchRequestCount = 0;
    int32_t _runningRequestCount = 0;
    uint64_t _readyWithoutRunningGapCount = 0;
    int64_t _readyWithoutRunningTotalTimeUs = 0;
    int64_t _readyWithoutRunningMaxTimeUs = 0;
    int64_t _readyWithoutRunningStartTimeUs = 0;
    int64_t _executionBusyTimeUs = 0;
    int64_t _executionBusyStartTimeUs = 0;
    ExecutionUtilizationEstimator _executionUtilizationEstimator;
    ExecutionUtilizationWindow _executionUtilizationWindow;
    std::shared_ptr<std::atomic<double>> _backendExecutionUtilization =
        std::make_shared<std::atomic<double>>(0.0);
    std::atomic<double> _backendExecutionCapacity{0.0};
    std::atomic<int32_t> _backendWaitingRequests{0};
    int32_t _utilizationRunningRequests = 0;
    std::shared_ptr<std::atomic<double>> _executionUtilization =
        std::make_shared<std::atomic<double>>(0.0);
    mutable autil::ThreadMutex _backendExecutionLock;
    int64_t _backendExecutionBusyTimeUs = 0;
    int64_t _backendExecutionBusyStartTimeUs = 0;
    size_t _activeBackendExecutionCount = 0;
    bool _backendExecutionObserved = false;
    size_t _activeCallbackCount = 0;
    size_t _activeLifecycleCallbackCount = 0;
    std::atomic<int64_t> _nextBatchId{0};
    bool _stopped = false;
    bool _graphFinishRequested = false;
    navi::AsyncPipePtr _graphPipe;
    uint64_t _graphPipeOwnerGeneration = 0;
    std::atomic<uint64_t> _graphPipeGeneration{0};
    mutable std::atomic<size_t> _callbackFailureCount{0};
    mutable std::atomic<uint64_t> _timeoutBatchCount{0};
    mutable std::atomic<uint64_t> _timeoutMissingRequestCount{0};
    mutable std::atomic<uint64_t> _lateRequestCount{0};
    mutable std::atomic<uint64_t> _expectedSizeMismatchCount{0};
    mutable std::atomic<uint64_t> _predictionMismatchCount{0};
    mutable std::atomic<uint64_t> _duplicateRequestCount{0};
    mutable std::atomic<uint64_t> _invalidRequestCount{0};
    mutable std::atomic<uint64_t> _abnormalBatchFinishCount{0};
    mutable autil::LockFreeQueue<int64_t> _batchFormationTimeQueue;
    mutable autil::LockFreeQueue<BackendBatchExecutionSample> _backendBatchExecutionQueue;
    mutable autil::LockFreeQueue<BackendExecutionStepSample> _backendExecutionStepQueue;
    mutable autil::LockFreeQueue<CoreStepSample> _coreStepQueue;
    CacheHitPathStats _cacheHitPathStats;
};

NAVI_TYPEDEF_PTR(AgentBatchR);

}
