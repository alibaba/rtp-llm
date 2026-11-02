#pragma once

#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "aios/navi_sched/data/NaviSchedLatencySample.h"
#include "autil/Lock.h"
#include "autil/LockFreeQueue.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include <atomic>
#include <deque>
#include <vector>

namespace navi_sched {

enum class AgentLatencyInputType {
    BATCH_READY,
    BATCH_STARTED,
    BATCH_FINISHED,
    BATCH_ABORTED,
    BATCH_REQUESTS_ABORTED,
    OBSERVED_BATCH_FINISHED,
    EXECUTION_LOAD,
};

struct AgentLatencyInput {
    static void appendFeatures(NaviSchedLatencyFeatures &features,
                               const NaviSchedRequestStats &query);
    NaviSchedLatencySample makeLatencySample() const;
    NaviSchedLatencyTrainingPhase getLatencyTrainingPhase() const;

    AgentLatencyInputType type = AgentLatencyInputType::BATCH_READY;
    std::vector<NaviSchedRequestStats> queries;
    NaviSchedBatchPtr batch;
    NaviSchedLocalBatchInfo localBatch;
    int64_t batchId = 0;
    double latencyMs = 0.0;
    double modelLatencyMs = 0.0;
    bool unifiedLatencySample = false;
    bool latencySampleValid = true;
    bool latencyValidationValid = true;
    NaviSchedLatencyFeatures precedingFeatures;
    double backendQueueLatencyMs = 0.0;
    double predictedQueueLatencyMs = 0.0;
    double predictedExecutionLatencyMs = 0.0;
    int64_t predictedCacheHitTokenCount = 0;
    int64_t actualCacheHitTokenCount = 0;
    int32_t cacheHitMismatchRequestCount = 0;
    int64_t cacheHitAbsoluteErrorTokenCount = 0;
    int64_t cacheHitMaxAbsoluteErrorTokenCount = 0;
    int32_t inflightRequests = 0;
    int32_t batchWaitingRequests = 0;
    int32_t queuedBatchRequests = 0;
    int32_t runningRequests = 0;
    int32_t completedRequestCount = 0;
    int64_t eventTimeUs = 0;
    bool graphRecovery = false;
};

class AgentLatencyInputR : public navi::ExternalResource {
public:
    struct Stats {
        uint64_t submittedInputCount = 0;
        uint64_t rejectedInputCount = 0;
        uint64_t processedInputCount = 0;
        uint64_t submittedFinishedRequestCount = 0;
        uint64_t processedFinishedRequestCount = 0;
        uint64_t submittedAbortedRequestCount = 0;
        uint64_t processedAbortedRequestCount = 0;
    };

    struct ProcessingLagStats {
        uint64_t sampleCount = 0;
        int64_t totalTimeUs = 0;
        int64_t maxTimeUs = 0;
    };

    AgentLatencyInputR();

public:
    bool submitBatchReady(const NaviSchedBatchPtr &batch, int32_t batchWaitingRequests,
                          int32_t queuedBatchRequests, int32_t runningRequests,
                          int64_t eventTimeUs);
    bool submitBatchStarted(int64_t batchId, int32_t batchWaitingRequests,
                            int32_t queuedBatchRequests, int32_t runningRequests,
                            int64_t eventTimeUs);
    bool submitBatchFinished(const std::vector<NaviSchedRequestStats> &requests,
                             double latencyMs, double backendQueueLatencyMs,
                             double predictedQueueLatencyMs,
                             double predictedExecutionLatencyMs,
                             int64_t predictedCacheHitTokenCount,
                             int64_t actualCacheHitTokenCount,
                             int32_t cacheHitMismatchRequestCount,
                             int64_t cacheHitAbsoluteErrorTokenCount,
                             int64_t cacheHitMaxAbsoluteErrorTokenCount,
                             int64_t batchId, int32_t inflightRequests,
                             int32_t batchWaitingRequests, int32_t queuedBatchRequests,
                             int32_t runningRequests, int64_t eventTimeUs,
                             const NaviSchedLatencyFeatures &precedingFeatures = {},
                             double unifiedLatencyMs = -1.0,
                             bool latencySampleValid = true,
                             bool latencyValidationValid = true,
                             const NaviSchedLocalBatchInfo &localBatch = {});
    bool submitBatchAborted(int32_t completedRequestCount, int64_t batchId,
                            int32_t inflightRequests, int32_t batchWaitingRequests,
                            int32_t queuedBatchRequests, int32_t runningRequests,
                            int64_t eventTimeUs,
                            const NaviSchedLocalBatchInfo &localBatch);
    bool submitBatchRequestsAborted(
        const std::vector<NaviSchedRequestStats> &requests,
        const NaviSchedBatchPtr &remainingBatch, int64_t batchId,
        int32_t inflightRequests, int32_t batchWaitingRequests,
        int32_t queuedBatchRequests, int32_t runningRequests,
        int64_t eventTimeUs, const NaviSchedLocalBatchInfo &localBatch);
    bool submitObservedBatchFinished(const std::vector<NaviSchedRequestStats> &requests,
                                     double latencyMs, int32_t inflightRequests,
                                     int64_t eventTimeUs);
    bool submitExecutionLoad(int32_t batchWaitingRequests, int32_t queuedBatchRequests,
                             int32_t runningRequests, int64_t eventTimeUs);
    size_t getPendingInputCount() const;
    Stats getStats() const;
    ProcessingLagStats takeBatchStartedProcessingLagWindowStats() const;
    void take(std::vector<AgentLatencyInput> &inputs);
    void replaceGraphRecoveryInputs(std::vector<AgentLatencyInput> inputs);
    void prepareGraphStart();
    uint64_t beginInputPipeGeneration();
    bool setInputPipe(uint64_t generation,
                      const navi::AsyncPipePtr &inputPipe);
    bool waitInputPipe(uint64_t generation, int64_t timeoutMs) const;
    void clearInputPipe(uint64_t generation);
    bool finish();
    uint64_t getInputPipeGeneration() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    bool submit(AgentLatencyInput &&input);

private:
    mutable autil::ThreadMutex _lock;
    std::deque<AgentLatencyInput> _inputs;
    navi::AsyncPipePtr _inputPipe;
    uint64_t _inputPipeOwnerGeneration = 0;
    bool _finishRequested = false;
    std::atomic<uint64_t> _inputPipeGeneration{0};
    std::atomic<uint64_t> _submittedInputCount{0};
    std::atomic<uint64_t> _rejectedInputCount{0};
    std::atomic<uint64_t> _processedInputCount{0};
    std::atomic<uint64_t> _submittedFinishedRequestCount{0};
    std::atomic<uint64_t> _processedFinishedRequestCount{0};
    std::atomic<uint64_t> _submittedAbortedRequestCount{0};
    std::atomic<uint64_t> _processedAbortedRequestCount{0};
    mutable autil::LockFreeQueue<int64_t> _batchStartedProcessingLagQueue;
};

NAVI_TYPEDEF_PTR(AgentLatencyInputR);

}
