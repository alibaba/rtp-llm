#pragma once

#include "navi_sched/NaviSchedCommon.h"
#include <memory>
#include <string>
#include <vector>

namespace navi_sched {

class NaviSchedAgentImpl;
class NaviSchedLatencyFeatures;

class NAVI_SCHED_API NaviSchedBatchFinishReporter {
public:
    ~NaviSchedBatchFinishReporter();

public:
    bool beginExecution();
    bool startBatch(const std::vector<uint64_t> &requestIds);
    bool startBatch(const std::vector<uint64_t> &requestIds,
                    int64_t executionStartTimeUs);
    bool finishBatch(const std::vector<NaviSchedRequestStats> &requests);
    bool finishBatch(const std::vector<NaviSchedRequestStats> &requests,
                     double wallTimeMs, double executeModelTimeMs,
                     int64_t executionStepCount, int64_t prefillTokenCount,
                     int64_t decodeRequestStepCount, int64_t decodeTokenCount,
                     double admissionToFirstExecutionMs = -1.0);
    bool finishBatch(const std::vector<NaviSchedRequestStats> &requests,
                     double wallTimeMs, double executeModelTimeMs,
                     int64_t executionStepCount, int64_t prefillTokenCount,
                     int64_t decodeRequestStepCount, int64_t decodeTokenCount,
                     double admissionToFirstExecutionMs, double nativePreparationMs);
    bool finishBatchWithUnifiedLatency(
        const std::vector<NaviSchedRequestStats> &requests, double wallTimeMs,
        double executeModelTimeMs, int64_t executionStepCount,
        int64_t prefillTokenCount, int64_t decodeRequestStepCount,
        int64_t decodeTokenCount, const NaviSchedLatencyFeatures &precedingFeatures,
        double unifiedLatencyMs, double admissionToFirstExecutionMs = -1.0);
    bool finishBatchWithUnifiedLatency(
        const std::vector<NaviSchedRequestStats> &requests, double wallTimeMs,
        double executeModelTimeMs, int64_t executionStepCount,
        int64_t prefillTokenCount, int64_t decodeRequestStepCount,
        int64_t decodeTokenCount, const NaviSchedLatencyFeatures &precedingFeatures,
        double unifiedLatencyMs, double admissionToFirstExecutionMs,
        double nativePreparationMs);
    bool abortRequests(const std::vector<uint64_t> &requestIds);

private:
    friend class NaviSchedAgentImpl;
    struct State;
    explicit NaviSchedBatchFinishReporter(const std::shared_ptr<State> &state);

private:
    std::shared_ptr<State> _state;
};

class NAVI_SCHED_API NaviSchedAgent final {
public:
    NaviSchedAgent(const NaviSchedAgentConfig &agentConfig, int32_t serverPort = 0,
                   const std::string &serverIp = "", int64_t totalBlocks = 0);
    ~NaviSchedAgent();

public:
    bool init();
    bool addBatchRequest(const NaviSchedBatchRequest &request, int64_t batchId,
                         int64_t expectedBatchSize,
                         const NaviSchedReadyBatchCallback &callback);
    bool abortPendingBatch(int64_t scheduleBatchId);
    bool setForwardBatchHandler(const NaviSchedForwardBatchHandler &handler,
                                bool streamResponse = true,
                                const NaviSchedForwardRequestAcceptedHandler
                                    &requestAcceptedHandler = {});
    bool publishForwardResponse(uint64_t requestId, std::vector<std::string> frames,
                                bool terminal);
    bool failForwardResponse(uint64_t requestId, const std::string &errorMessage);
    void setForwardCancelHandler(const std::function<void(uint64_t)> &handler);
    bool recordObservedBatch(const std::vector<NaviSchedRequestStats> &requests,
                             double latencyMs, int64_t eventTimeUs,
                             const std::vector<NaviSchedBatchFinishReporterPtr>
                                 &finishReporters = {}) const;
    bool reportObservedExecutionLoad(int32_t queuedBatchRequests,
                                     int32_t runningRequests,
                                     int64_t eventTimeUs) const;
    bool reportObservedWorkload(const NaviSchedWorkloadSnapshot &snapshot) const;
    bool reportObservedWorkload(const NaviSchedWorkloadSnapshot &snapshot,
                               const std::vector<int64_t> &coveredNativeBatchIds) const;
    bool beginObservedBackendExecution(int32_t requestCount = 0) const;
    bool endObservedBackendExecution() const;
    bool reportObservedExecutionStep(double elapsedMs, int64_t prefillTokens,
                                     int64_t decodeTokens) const;
    bool reportObservedCoreStep(double elapsedMs, bool futureWait) const;
    void reportCacheInserted(const NaviSchedBlockHashesConstPtr &blockHashes) const;
    void reportCacheRemoved(const NaviSchedBlockHashesConstPtr &blockHashes) const;
    void reportCacheUpdated(int64_t totalBlocks,
                            int64_t blockSize,
                            int64_t linearStep,
                            const NaviSchedBlockHashesConstPtr &addedBlockHashes,
                            const NaviSchedBlockHashesConstPtr &removedBlockHashes,
                            const NaviSchedBlockHashesConstPtr &addedFullReadyBlockHashes,
                            const NaviSchedBlockHashesConstPtr &removedFullReadyBlockHashes,
                            const NaviSchedBlockHashesConstPtr &addedTailReadyBlockHashes,
                            const NaviSchedBlockHashesConstPtr &removedTailReadyBlockHashes,
                            int8_t hybridTailRequired,
                            int64_t eventTimeUs,
                            int64_t rtpCachedBlockCount = -1) const;
    void stop();
    int32_t getGrpcPort() const;

private:
    friend class NaviSchedAgentImpl;

private:
    std::unique_ptr<NaviSchedAgentImpl> _impl;
};

}
