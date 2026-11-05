#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "navi_sched/NaviSchedCommon.h"

namespace navi_sched {

class NaviSchedAgent;

enum class RtpLLMAgentInitStatus {
    OK,
    INVALID_CONFIG,
    INIT_FAILED,
};

enum class RtpLLMReadyBatchRetainStatus {
    OK,
    INVALID_BATCH,
    ZERO_REQUEST_ID,
    DUPLICATE_REQUEST_ID,
};

struct RtpLLMReadyBatchRetainResult {
    RtpLLMReadyBatchRetainStatus status = RtpLLMReadyBatchRetainStatus::OK;
    uint64_t requestId = 0;
    size_t retainedRequestCount = 0;

    explicit operator bool() const { return status == RtpLLMReadyBatchRetainStatus::OK; }
};

class NAVI_SCHED_API RtpLLMRetainedBatchLease {
public:
    ~RtpLLMRetainedBatchLease();

public:
    void abort();
    void dismiss();

private:
    friend std::shared_ptr<RtpLLMRetainedBatchLease> acquireRtpLLMRetainedBatchLease(
        uint64_t requestId);
    explicit RtpLLMRetainedBatchLease(uint64_t requestId);

private:
    uint64_t _requestId = 0;
    std::atomic<bool> _active{true};
};

using RtpLLMRetainedBatchLeasePtr = std::shared_ptr<RtpLLMRetainedBatchLease>;

NAVI_SCHED_API RtpLLMAgentInitStatus createRtpLLMAgentFromEnvironment(
    bool enableForRole, const std::string &serverIp, int32_t serverPort,
    std::shared_ptr<NaviSchedAgent> &agent, std::string &errorMessage);
NAVI_SCHED_API int32_t
getRtpLLMAgentGrpcPort(const std::shared_ptr<NaviSchedAgent> &agent);
NAVI_SCHED_API void stopRtpLLMAgent(const std::shared_ptr<NaviSchedAgent> &agent);
NAVI_SCHED_API bool setRtpLLMForwardBatchHandler(
    const std::shared_ptr<NaviSchedAgent> &agent,
    const NaviSchedForwardBatchHandler &handler);
NAVI_SCHED_API bool addRtpLLMBatchRequest(const std::shared_ptr<NaviSchedAgent> &agent,
                                          const NaviSchedBatchRequest &request,
                                          int64_t batchId, int64_t expectedBatchSize,
                                          const NaviSchedReadyBatchCallback &callback);
NAVI_SCHED_API bool startRtpLLMBatch(
    const NaviSchedBatchFinishReporterPtr &finishReporter,
    const std::vector<uint64_t> &requestIds);
NAVI_SCHED_API bool finishRtpLLMBatch(
    const NaviSchedBatchFinishReporterPtr &finishReporter,
    const std::vector<NaviSchedRequestStats> &requests);
NAVI_SCHED_API bool retainRtpLLMReadyBatch(const NaviSchedBatchPtr &batch);
NAVI_SCHED_API RtpLLMReadyBatchRetainResult
retainRtpLLMReadyBatchWithResult(const NaviSchedBatchPtr &batch);
NAVI_SCHED_API RtpLLMReadyBatchRetainResult retainRtpLLMForwardBatchWithResult(
    const NaviSchedBatchPtr &batch, const std::vector<uint64_t> &engineRequestIds);
NAVI_SCHED_API RtpLLMRetainedBatchLeasePtr
acquireRtpLLMRetainedBatchLease(uint64_t requestId);
NAVI_SCHED_API bool abortRetainedRtpLLMBatch(uint64_t requestId);
NAVI_SCHED_API bool startRetainedRtpLLMBatches(
    const std::vector<NaviSchedLatencyQuery> &queries);
NAVI_SCHED_API bool startRetainedRtpLLMBatches(
    const std::vector<NaviSchedLatencyQuery> &queries, int64_t executionStartTimeUs);
NAVI_SCHED_API void reportRtpLLMCacheInserted(
    const std::shared_ptr<NaviSchedAgent> &agent,
    const NaviSchedBlockHashesConstPtr &addedBlockHashes);
NAVI_SCHED_API void reportRtpLLMCacheRemoved(
    const std::shared_ptr<NaviSchedAgent> &agent,
    const NaviSchedBlockHashesConstPtr &removedBlockHashes);
NAVI_SCHED_API void recordRtpLLMPrefillBatch(
    const std::shared_ptr<NaviSchedAgent> &agent,
    const std::vector<NaviSchedLatencyQuery> &queries, int64_t latencyUs,
    int64_t eventTimeUs,
    const std::vector<NaviSchedBatchFinishReporterPtr> &finishReporters = {});
NAVI_SCHED_API void recordRtpLLMObservedPrefillBatch(
    const std::shared_ptr<NaviSchedAgent> &agent,
    const std::vector<NaviSchedLatencyQuery> &queries, int64_t latencyUs,
    int64_t eventTimeUs);
NAVI_SCHED_API void reportRtpLLMExecutionLoad(
    const std::shared_ptr<NaviSchedAgent> &agent,
    const std::vector<NaviSchedLatencyQuery> &queuedBatchRequests,
    int32_t runningRequests, int64_t eventTimeUs);
NAVI_SCHED_API void reportRtpLLMCacheStatus(const std::shared_ptr<NaviSchedAgent> &agent,
                                            int64_t totalBlocks,
                                            int64_t blockSize,
                                            int64_t linearStep,
                                            bool hybridTailRequired,
                                            int64_t eventTimeUs);
NAVI_SCHED_API void reportRtpLLMCacheStateChanged(
    const std::shared_ptr<NaviSchedAgent> &agent,
    const NaviSchedBlockHashesConstPtr &changedBlockHashes,
    const NaviSchedBlockHashesConstPtr &flatPresentBlockHashes,
    const NaviSchedBlockHashesConstPtr &fullReadyBlockHashes,
    const NaviSchedBlockHashesConstPtr &tailReadyBlockHashes,
    bool hybridTailRequired, int64_t eventTimeUs,
    int64_t rtpCachedBlockCount = -1);
NAVI_SCHED_API void resetRtpLLMCacheReportLatency();
NAVI_SCHED_API int64_t takeRtpLLMCacheReportLatencyUs();

}
