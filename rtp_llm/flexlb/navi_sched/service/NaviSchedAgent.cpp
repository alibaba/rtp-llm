#include "aios/navi_sched/service/NaviSchedAgent.h"
#include "aios/navi_sched/data/NaviSchedRequestTrace.h"
#include "aios/navi_sched/navi_sched_git_version.h"
#include "aios/navi_sched/service/NaviSchedAgentImpl.h"
#include "aios/navi_sched/graph/AgentGraph.h"
#include "aios/navi_sched/resource/AgentBatchR.h"
#include "aios/navi_sched/resource/AgentCacheR.h"
#include "aios/navi_sched/resource/AgentHistoryR.h"
#include "aios/navi_sched/resource/AgentInstanceR.h"
#include "aios/navi_sched/resource/AgentLatencyInputR.h"
#include "aios/navi_sched/resource/AgentReportR.h"
#include "aios/navi_sched/resource/AgentRequestR.h"
#include "aios/navi_sched/resource/AgentResponseR.h"
#include "aios/navi_sched/resource/LatencyModelR.h"
#include "aios/navi_sched/resource/RequestLatencyTracker.h"
#include "aios/navi_sched/config/NaviSchedConfigBuilder.h"
#include "aios/navi_sched/resource/AgentSchedulerConfigR.h"
#include "aios/navi_sched/service/NaviSchedAgentSummary.h"
#include "aios/navi_sched/data/AgentResponseToClientData.h"
#include "aios/navi_sched/scheduler/ExecutionUtilizationController.h"
#include "autil/TimeUtility.h"
#include "autil/Log.h"
#include "autil/Scope.h"
#include "autil/legacy/jsonizable.h"
#include "kmonitor/client/MetricsReporter.h"
#include "kmonitor/client/core/MetricsTags.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace navi_sched {
AUTIL_DECLARE_AND_SETUP_LOGGER(navi_sched, NaviSchedAgentImplRuntime);
namespace {

class BatchLifecycleGuard {
public:
    explicit BatchLifecycleGuard(const AgentBatchRPtr &batchR)
        : _batchR(batchR && batchR->beginLifecycleCallback() ? batchR : nullptr)
    {
    }
    ~BatchLifecycleGuard()
    {
        if (_batchR) _batchR->endLifecycleCallback();
    }
    BatchLifecycleGuard(const BatchLifecycleGuard &) = delete;
    BatchLifecycleGuard &operator=(const BatchLifecycleGuard &) = delete;
    explicit operator bool() const { return static_cast<bool>(_batchR); }

private:
    AgentBatchRPtr _batchR;
};

RequestLatencyTracker::Snapshot removeTrackedRequests(
    const std::shared_ptr<ConcurrentRequestLatencyTracker> &tracker,
    const std::vector<uint64_t> &requestIds, const std::shared_ptr<AgentReportR> &reportR,
    const char *phase, bool diagnoseMissing = true)
{
    RequestLatencyTracker::Snapshot snapshot;
    if (!tracker) {
        return snapshot;
    }
    std::vector<uint64_t> missing;
    const auto removed = diagnoseMissing ? tracker->remove(requestIds, snapshot, &missing)
                                         : tracker->removePresent(requestIds, snapshot);
    if (diagnoseMissing && removed != requestIds.size()) {
        AUTIL_LOG(ERROR,
                  "remove inflight requests incomplete, phase [%s], requested [%lu], "
                  "removed [%lu], missing [%lu], revision [%lu], remaining [%d]",
                  phase ? phase : "unknown", requestIds.size(), removed, missing.size(),
                  snapshot.revision, snapshot.requestCount);
    }
    if (reportR && !reportR->setInflightStats(snapshot)) {
        AUTIL_LOG(ERROR,
                  "publish invalid inflight snapshot failed, phase [%s], "
                  "revision [%lu], count [%d], latencySumMs [%.6f]",
                  phase ? phase : "unknown", snapshot.revision, snapshot.requestCount,
                  snapshot.totalPredictedLatencyMs);
    }
    return snapshot;
}

std::string makeHistoryBinaryFingerprint(const std::string &frameworkFingerprint)
{
    autil::legacy::json::JsonMap values;
    values["navi_sched_commit"] = std::string(navi_sched_build::GIT_COMMIT_SHA);
    values["framework_binary"] = frameworkFingerprint;
    return autil::legacy::ToJsonString(values, true);
}

const int64_t AGENT_STOP_ACK_TIMEOUT_US = 20 * 1000 * 1000;

double validationWindowMean(double currentMean, uint64_t currentCount,
                            double previousMean, uint64_t previousCount)
{
    const auto windowCount = currentCount - previousCount;
    return windowCount > 0
               ? (currentMean * currentCount - previousMean * previousCount) / windowCount
               : 0.0;
}

bool reportPendingTerminal(const std::shared_ptr<AgentReportR> &report,
                           const AgentBatchR::ExecutionState &state, bool completed)
{
    if (!report) {
        return !state.localBatch.pendingCacheTracked;
    }
    bool success = true;
    for (const auto type : state.pendingCacheTypes) {
        const bool reported =
            completed
                ? report->reportPendingCacheCompleted(state.localBatch.localBatchId, type)
                : report->reportPendingCacheAborted(state.localBatch.localBatchId, type);
        success = reported && success;
    }
    return success;
}

} // namespace

void NaviSchedAgentImpl::reportCacheHitPathSamples(
    const std::shared_ptr<kmonitor::MetricsReporter> &reporter,
    const std::vector<AgentBatchR::CacheHitPathSample> &samples)
{
    if (!reporter) {
        return;
    }
    const auto report = [&reporter](const std::string &name, double value) {
        reporter->report(value, name, kmonitor::MetricType::GAUGE, nullptr,
                         kmonitor::MetricLevel::FATAL);
    };
    for (const auto &sample : samples) {
        report("agent.cache_hit_path_master_predicted_tokens",
               sample.masterPredictedTokens);
        report("agent.cache_hit_path_agent_arrival_tokens", sample.agentPhysicalTokens);
        report("agent.cache_hit_path_agent_pending_aware_tokens",
               sample.agentPendingAwareTokens);
        report("agent.cache_hit_path_rtp_actual_tokens", sample.rtpActualTokens);

        report("agent.cache_hit_path_master_to_agent_error_tokens",
               sample.masterToAgentPhysicalErrorTokens);
        report("agent.cache_hit_path_master_to_agent_absolute_error_tokens",
               std::abs(sample.masterToAgentPhysicalErrorTokens));
        report("agent.cache_hit_path_agent_to_rtp_error_tokens",
               sample.agentPhysicalToRtpErrorTokens);
        report("agent.cache_hit_path_agent_to_rtp_absolute_error_tokens",
               std::abs(sample.agentPhysicalToRtpErrorTokens));
        report("agent.cache_hit_path_master_to_agent_pending_aware_error_tokens",
               sample.masterToAgentPendingAwareErrorTokens);
        report("agent.cache_hit_path_master_to_agent_pending_aware_absolute_error_tokens",
               std::abs(sample.masterToAgentPendingAwareErrorTokens));
        report("agent.cache_hit_path_agent_pending_aware_to_rtp_error_tokens",
               sample.agentPendingAwareToRtpErrorTokens);
        report("agent.cache_hit_path_agent_pending_aware_to_rtp_absolute_error_tokens",
               std::abs(sample.agentPendingAwareToRtpErrorTokens));
        report("agent.cache_hit_path_master_to_rtp_error_tokens",
               sample.masterToRtpErrorTokens);
        report("agent.cache_hit_path_master_to_rtp_absolute_error_tokens",
               std::abs(sample.masterToRtpErrorTokens));

        const auto alignment = sample.blockAligned ? "aligned" : "unaligned";
        report(std::string("agent.cache_hit_path_") + alignment +
                   "_agent_to_rtp_error_tokens",
               sample.agentPhysicalToRtpErrorTokens);
        report(std::string("agent.cache_hit_path_") + alignment +
                   "_agent_to_rtp_absolute_error_tokens",
               std::abs(sample.agentPhysicalToRtpErrorTokens));
    }
}

class AgentForwardState {
public:
    struct PendingRequest {
        NaviSchedForwardRequest request;
        navi::AsyncPipePtr completionPipe;
        int64_t arrivalTimeUs = 0;
    };

    std::mutex mutex;
    std::condition_variable condition;
    std::unordered_map<uint64_t, std::shared_ptr<const PendingRequest>> requests;
    NaviSchedForwardBatchHandler handler;
    NaviSchedForwardRequestAcceptedHandler requestAcceptedHandler;
    bool streamResponse = true;
    size_t activeHandlerCount = 0;
};

struct NaviSchedBatchFinishReporter::State {
    State(int64_t batchId, std::vector<uint64_t> requestIds,
          const AgentBatchRPtr &agentBatchR,
          const AgentLatencyInputRPtr &agentLatencyInputR,
          const AgentReportRPtr &agentReportR,
          const std::shared_ptr<ConcurrentRequestLatencyTracker> &inflightState,
          const std::shared_ptr<kmonitor::MetricsReporter> &metricsReporter)
        : batchId(batchId)
        , ownedRequestIds(std::move(requestIds))
        , agentBatchR(agentBatchR)
        , agentLatencyInputR(agentLatencyInputR)
        , agentReportR(agentReportR)
        , inflightState(inflightState)
        , metricsReporter(metricsReporter)
    {
    }

    bool beginExecution()
    {
        auto batchR = agentBatchR.lock();
        BatchLifecycleGuard lifecycle(batchR);
        if (!lifecycle) return false;
        std::lock_guard<std::mutex> lock(mutex);
        return !finished && batchR->beginBatchExecution(batchId);
    }

    bool start(const std::vector<uint64_t> &requestIds, int64_t executionStartTimeUs)
    {
        auto batchR = agentBatchR.lock();
        BatchLifecycleGuard lifecycle(batchR);
        if (!lifecycle) return false;
        auto latencyInputR = agentLatencyInputR.lock();
        auto reportR = agentReportR.lock();
        AgentBatchR::ExecutionState executionState;
        std::vector<uint64_t> cleanupRequestIds;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!batchR || !latencyInputR || !reportR || !inflightState) {
                return false;
            }
            if (finished || started || requestIds.empty()) {
                return false;
            }
            for (auto requestId : requestIds) {
                if (0 == requestId) {
                    return false;
                }
            }
            if (batchId <= 0 ||
                !batchR->startBatch(batchId, requestIds, executionStartTimeUs,
                                    executionState)) {
                return false;
            }
            const auto cacheState = reportR->getCacheState();
            batchR->recordCacheStateAtStart(
                batchId, cacheState.revision, cacheState.cachedBlockCount,
                cacheState.hashChecksum, cacheState.fullReadyBlockCount,
                cacheState.fullReadyHashChecksum, cacheState.tailReadyBlockCount,
                cacheState.tailReadyHashChecksum, cacheState.rtpCachedBlockCount);
            RequestLatencyTracker::Snapshot inflight;
            if (!inflightState->markRunning(requestIds, inflight)) {
                AUTIL_LOG(ERROR,
                          "mark batch inflight requests running failed, "
                          "batchId [%ld], requestCount [%lu], inflightRevision [%lu]",
                          batchId, requestIds.size(), inflight.revision);
                finished = true;
                cleanupRequestIds = ownedRequestIds;
                ownedRequestIds.clear();
            } else {
                started = true;
            }
            if (started &&
                latencyInputR->submitBatchStarted(
                    executionState.batchId, executionState.batchWaitingRequests,
                    executionState.queuedBatchRequests, executionState.runningRequests,
                    executionState.eventTimeUs)) {
                return true;
            }
            finished = true;
            cleanupRequestIds = ownedRequestIds;
            ownedRequestIds.clear();
        }
        if (!reportR->reportNativeBatchRemoved(batchId)) {
            AUTIL_LOG(ERROR,
                      "clear native workload after start failure failed, batchId [%ld]",
                      batchId);
        }
        AgentBatchR::ExecutionState abortedState;
        if (batchR->abortBatch(batchId, abortedState)) {
            if (abortedState.localBatch.pendingCacheTracked) {
                reportPendingTerminal(reportR, abortedState, false);
            }
            const auto inflight =
                removeTrackedRequests(inflightState, abortedState.requestIds, reportR,
                                      "batch_start_report_failed");
            latencyInputR->submitBatchAborted(
                abortedState.batchRequestCount, abortedState.batchId,
                inflight.requestCount, abortedState.batchWaitingRequests,
                abortedState.queuedBatchRequests, abortedState.runningRequests, 0,
                abortedState.localBatch);
        } else {
            AUTIL_LOG(ERROR,
                      "abort batch after start report failure failed, batchId [%ld], "
                      "fallbackRequestCount [%lu]",
                      batchId, cleanupRequestIds.size());
            removeTrackedRequests(inflightState, cleanupRequestIds, reportR,
                                  "batch_start_report_failed_after_external_abort",
                                  false);
        }
        return false;
    }

    void filterFailedPublications(NaviSchedPendingCacheReservation &reservation,
                                  NaviSchedCacheKeyType type) const
    {
        const auto it = failedPublications.find(type);
        if (it == failedPublications.end()) {
            return;
        }
        const auto &full = it->second.first;
        const auto &tail = it->second.second;
        auto &requiredFull = reservation.fullReadyBlockHashes;
        auto &requiredTail = reservation.tailReadyBlockHashes;
        requiredFull.erase(std::remove_if(requiredFull.begin(), requiredFull.end(),
                                          [&](uint64_t h) { return full.count(h); }),
                           requiredFull.end());
        requiredTail.erase(
            std::remove_if(requiredTail.begin(), requiredTail.end(),
                           [&](uint64_t h) { return full.count(h) || tail.count(h); }),
            requiredTail.end());
    }

    bool resolveExecutionCache(const std::vector<NaviSchedRequestStats> &requests)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto batchR = agentBatchR.lock();
        auto inputR = agentLatencyInputR.lock();
        if (!started || finished || !batchR || !inputR) {
            return false;
        }
        AgentBatchR::ExecutionState state;
        NaviSchedBatchPtr resolved;
        if (!batchR->resolveExecutionCache(batchId, requests, state, resolved)) {
            return false;
        }
        AgentLatencyInput input;
        input.batchWaitingRequests = state.batchWaitingRequests;
        input.queuedBatchRequests = state.queuedBatchRequests;
        input.runningRequests = state.runningRequests;
        input.eventTimeUs = state.eventTimeUs;
        return inputR->submitBatchCacheResolved(resolved, input);
    }

    bool publicationFailed(NaviSchedCacheKeyType type, const NaviSchedBlockHashes &full,
                           const NaviSchedBlockHashes &tail)
    {
        if (!validCacheKeyType(type)) {
            return false;
        }
        std::lock_guard<std::mutex> publicationLock(publicationMutex);
        auto report = agentReportR.lock();
        auto batch = agentBatchR.lock();
        if (!report || !batch) {
            return false;
        }
        auto &failed = failedPublications[type];
        failed.first.insert(full.begin(), full.end());
        failed.second.insert(tail.begin(), tail.end());
        bool retired = false;
        if (!report->removePendingCachePublications(batchId, full, tail, type, retired)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex);
        return !retired || finished || batch->unmarkPendingCacheTracked(batchId, type);
    }

    bool finish(const std::vector<NaviSchedRequestStats> &requests,
                const AgentBatchR::BackendBatchExecutionSample *backendExecutionSample,
                const NaviSchedLatencyFeatures *precedingFeatures = nullptr,
                double unifiedLatencyMs = -1.0, double nativePreparationMs = 0.0)
    {
        std::lock_guard<std::mutex> publicationLock(publicationMutex);
        auto batchR = agentBatchR.lock();
        BatchLifecycleGuard lifecycle(batchR);
        if (!lifecycle) return false;
        auto latencyInputR = agentLatencyInputR.lock();
        auto reportR = agentReportR.lock();
        AgentBatchR::ExecutionState executionState;
        std::vector<NaviSchedRequestStats> completedRequests;
        NaviSchedPendingCacheReservation publicationReservation;
        bool publicationRangeKnown = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!batchR || !latencyInputR || !reportR || !inflightState) {
                return false;
            }
            if (finished || !started || requests.empty() ||
                (backendExecutionSample &&
                 (backendExecutionSample->globalPaddedTokens > 0 ||
                  backendExecutionSample->projectedExecutionStepCount > 0) &&
                 !executionGeometryEligible)) {
                return false;
            }
            for (const auto &request : requests) {
                if (!request.valid()) {
                    return false;
                }
            }
            if (reportR->hasCacheKeyType(NaviSchedCacheKeyType::V6D)) {
                const auto cache = reportR->getCacheState(NaviSchedCacheKeyType::V6D);
                if (cache.policy.valid(NaviSchedCacheKeyType::V6D)) {
                    std::map<uint64_t, int64_t> publicationStarts;
                    for (const auto &request : requests) {
                        publicationStarts.emplace(request.requestId,
                                                  request.cacheHitTokenCount);
                    }
                    if (!batchR->makePendingCacheReservation(
                            batchId, cache.policy, publicationReservation,
                            NaviSchedCacheKeyType::V6D, publicationStarts)) {
                        return false;
                    }
                    publicationRangeKnown = true;
                    filterFailedPublications(publicationReservation,
                                             NaviSchedCacheKeyType::V6D);
                }
            }
            if (!batchR->finishBatch(
                    batchId, requests, executionState, &completedRequests,
                    nativePreparationMs,
                    backendExecutionSample ? backendExecutionSample->executionFinishTimeUs
                                           : 0)) {
                return false;
            }
            finished = true;
        }
        if (backendExecutionSample) {
            batchR->recordBackendBatchExecution(*backendExecutionSample);
        }
        NaviSchedAgentImpl::reportCacheHitPathSamples(metricsReporter,
                                                      executionState.cacheHitPathSamples);
        bool pendingRefined = true;
        const auto externalPending =
            std::find(executionState.pendingCacheTypes.begin(),
                      executionState.pendingCacheTypes.end(), NaviSchedCacheKeyType::V6D);
        if (publicationRangeKnown &&
            externalPending != executionState.pendingCacheTypes.end()) {
            // V6D advances its store cursor past reused/loaded tokens. Those
            // tokens may remain in HBM after their V6D objects were evicted,
            // and this execution will never publish them again.
            const bool empty = publicationReservation.fullReadyBlockHashes.empty();
            pendingRefined = reportR->replacePendingCache(
                executionState.localBatch.localBatchId,
                empty ? nullptr : &publicationReservation, NaviSchedCacheKeyType::V6D);
            if (pendingRefined && empty) {
                executionState.pendingCacheTypes.erase(externalPending);
            }
        }
        const bool pendingCompleted = reportR->reportNativeBatchCompleted(
            batchId,
            executionState.localBatch.pendingCacheTracked
                ? executionState.localBatch.localBatchId : 0,
            executionState.pendingCacheTypes);
        const auto inflight = removeTrackedRequests(
            inflightState, executionState.requestIds, reportR, "batch_finished");
        const bool hasGeometry =
            backendExecutionSample && backendExecutionSample->globalPaddedTokens > 0;
        const bool hasProjectedGeometry =
            backendExecutionSample &&
            backendExecutionSample->projectedExecutionStepCount > 0;
        const bool latencySubmitted =
            hasProjectedGeometry
                ? latencyInputR->submitBatchFinishedWithProjectedExecutionGeometry(
                      completedRequests, executionState.batchLatencyMs,
                      executionState.backendQueueLatencyMs,
                      executionState.predictedQueueLatencyMs,
                      executionState.predictedExecutionLatencyMs,
                      executionState.predictedCacheHitTokenCount,
                      executionState.actualCacheHitTokenCount,
                      executionState.cacheHitMismatchRequestCount,
                      executionState.cacheHitAbsoluteErrorTokenCount,
                      executionState.cacheHitMaxAbsoluteErrorTokenCount,
                      executionState.batchId, inflight.requestCount,
                      executionState.batchWaitingRequests,
                      executionState.queuedBatchRequests, executionState.runningRequests,
                      0,
                      precedingFeatures ? *precedingFeatures : NaviSchedLatencyFeatures(),
                      unifiedLatencyMs, executionState.latencySampleValid,
                      executionState.latencyValidationValid, executionState.localBatch,
                      backendExecutionSample->executionStepCount,
                      backendExecutionSample->globalPaddedTokens,
                      backendExecutionSample->projectedExecutionStepCount,
                      backendExecutionSample->projectedGlobalPaddedTokens)

            : hasGeometry
                ? latencyInputR->submitBatchFinishedWithExecutionGeometry(
                      completedRequests, executionState.batchLatencyMs,
                      executionState.backendQueueLatencyMs,
                      executionState.predictedQueueLatencyMs,
                      executionState.predictedExecutionLatencyMs,
                      executionState.predictedCacheHitTokenCount,
                      executionState.actualCacheHitTokenCount,
                      executionState.cacheHitMismatchRequestCount,
                      executionState.cacheHitAbsoluteErrorTokenCount,
                      executionState.cacheHitMaxAbsoluteErrorTokenCount,
                      executionState.batchId, inflight.requestCount,
                      executionState.batchWaitingRequests,
                      executionState.queuedBatchRequests, executionState.runningRequests,
                      0,
                      precedingFeatures ? *precedingFeatures : NaviSchedLatencyFeatures(),
                      unifiedLatencyMs, executionState.latencySampleValid,
                      executionState.latencyValidationValid, executionState.localBatch,
                      backendExecutionSample->executionStepCount,
                      backendExecutionSample->globalPaddedTokens)
                : latencyInputR->submitBatchFinished(
                      completedRequests, executionState.batchLatencyMs,
                      executionState.backendQueueLatencyMs,
                      executionState.predictedQueueLatencyMs,
                      executionState.predictedExecutionLatencyMs,
                      executionState.predictedCacheHitTokenCount,
                      executionState.actualCacheHitTokenCount,
                      executionState.cacheHitMismatchRequestCount,
                      executionState.cacheHitAbsoluteErrorTokenCount,
                      executionState.cacheHitMaxAbsoluteErrorTokenCount,
                      executionState.batchId, inflight.requestCount,
                      executionState.batchWaitingRequests,
                      executionState.queuedBatchRequests, executionState.runningRequests,
                      0,
                      precedingFeatures ? *precedingFeatures : NaviSchedLatencyFeatures(),
                      unifiedLatencyMs, executionState.latencySampleValid,
                      executionState.latencyValidationValid, executionState.localBatch);

        return pendingRefined && pendingCompleted && latencySubmitted;
    }

    bool abortRequests(const std::vector<uint64_t> &requestIds)
    {
        std::lock_guard<std::mutex> publicationLock(publicationMutex);
        auto batchR = agentBatchR.lock();
        BatchLifecycleGuard lifecycle(batchR);
        if (!lifecycle) return false;
        auto latencyInputR = agentLatencyInputR.lock();
        auto reportR = agentReportR.lock();
        AgentBatchR::ExecutionState executionState;
        std::vector<NaviSchedRequestStats> abortedRequests;
        NaviSchedBatchPtr remainingBatch;
        NaviSchedPendingCacheReservation remainingPendingCache;
        std::map<NaviSchedCacheKeyType, NaviSchedPendingCacheReservation> typedRemaining;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!batchR || !latencyInputR || !reportR || !inflightState || finished ||
                requestIds.empty() ||
                !batchR->abortBatchRequests(batchId, requestIds, executionState,
                                            abortedRequests, remainingBatch,
                                            remainingPendingCache, &typedRemaining)) {
                return false;
            }
            executionGeometryEligible = false;
            for (auto requestId : executionState.requestIds) {
                ownedRequestIds.erase(std::remove(ownedRequestIds.begin(),
                                                  ownedRequestIds.end(), requestId),
                                      ownedRequestIds.end());
            }
        }
        bool pendingUpdated = true;
        reportR->reportNativeRequestsRemoved(batchId, executionState.requestIds);
        if (executionState.localBatch.pendingCacheTracked) {
            for (const auto type : executionState.pendingCacheTypes) {
                const auto it = typedRemaining.find(type);
                if (it != typedRemaining.end()) {
                    filterFailedPublications(it->second, type);
                }
                const auto *remaining =
                    it != typedRemaining.end() &&
                            PendingCacheUtil::validReservation(it->second)
                        ? &it->second
                        : nullptr;
                if (!remaining && remainingBatch && failedPublications.count(type)) {
                    pendingUpdated = batchR->unmarkPendingCacheTracked(batchId, type) &&
                                     pendingUpdated;
                }
                const bool updated = reportR->replacePendingCache(
                    executionState.localBatch.localBatchId, remaining, type);
                pendingUpdated = updated && pendingUpdated;
            }
        }
        const auto inflight = removeTrackedRequests(
            inflightState, executionState.requestIds, reportR, "batch_requests_aborted");
        const auto reported = latencyInputR->submitBatchRequestsAborted(
            abortedRequests, remainingBatch, executionState.batchId,
            inflight.requestCount, executionState.batchWaitingRequests,
            executionState.queuedBatchRequests, executionState.runningRequests,
            executionState.eventTimeUs, executionState.localBatch);
        return pendingUpdated && reported;
    }

    bool finishObserved()
    {
        std::lock_guard<std::mutex> publicationLock(publicationMutex);
        auto batchR = agentBatchR.lock();
        BatchLifecycleGuard lifecycle(batchR);
        if (!lifecycle) return false;
        auto reportR = agentReportR.lock();
        AgentBatchR::ExecutionState executionState;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!batchR || !reportR || !inflightState || finished || started ||
                !batchR->abortBatch(batchId, executionState)) {
                return false;
            }
            finished = true;
        }
        reportR->reportNativeBatchRemoved(batchId);
        if (executionState.localBatch.pendingCacheTracked) {
            reportPendingTerminal(reportR, executionState, false);
        }
        removeTrackedRequests(inflightState, executionState.requestIds, reportR,
                              "observed_batch_finished");
        return true;
    }

    void finishAbnormally()
    {
        std::lock_guard<std::mutex> publicationLock(publicationMutex);
        auto batchR = agentBatchR.lock();
        BatchLifecycleGuard lifecycle(batchR);
        if (batchR && !lifecycle) return;
        auto latencyInputR = agentLatencyInputR.lock();
        auto reportR = agentReportR.lock();
        AgentBatchR::ExecutionState executionState;
        std::vector<uint64_t> cleanupRequestIds;
        bool batchAborted = false;
        bool missingInflightState = false;
        bool batchAbortFailed = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (finished) {
                return;
            }
            finished = true;
            cleanupRequestIds = ownedRequestIds;
            ownedRequestIds.clear();
            if (!inflightState) {
                missingInflightState = true;
            } else if (batchR && batchR->abortBatch(batchId, executionState)) {
                batchAborted = true;
                cleanupRequestIds = executionState.requestIds;
            } else {
                batchAbortFailed = true;
            }
        }
        if (missingInflightState) {
            AUTIL_LOG(ERROR,
                      "cannot clean abnormal batch without lifecycle state, "
                      "batchId [%ld], hasBatchResource [%d], hasInflightState [%d]",
                      batchId, static_cast<bool>(batchR),
                      static_cast<bool>(inflightState));
            return;
        }
        if (reportR) {
            reportR->reportNativeBatchRemoved(batchId);
        }
        if (batchAbortFailed) {
            AUTIL_LOG(ERROR,
                      "abort abnormal batch failed, batchId [%ld], "
                      "hasBatchResource [%d], fallbackRequestCount [%lu]",
                      batchId, static_cast<bool>(batchR), cleanupRequestIds.size());
        }
        if (batchAborted && executionState.localBatch.pendingCacheTracked &&
            (!reportR || !reportPendingTerminal(reportR, executionState, false))) {
            AUTIL_LOG(ERROR,
                      "report abnormal pending cache abort failed, batchId [%ld], "
                      "localBatchId [%ld], hasReportResource [%d]",
                      batchId, executionState.localBatch.localBatchId,
                      static_cast<bool>(reportR));
        }
        const auto inflight =
            removeTrackedRequests(inflightState, cleanupRequestIds, reportR,
                                  "batch_finished_abnormally", batchAborted);
        if (!batchAborted) {
            return;
        }
        if (!latencyInputR ||
            !latencyInputR->submitBatchAborted(
                executionState.batchRequestCount, executionState.batchId,
                inflight.requestCount, executionState.batchWaitingRequests,
                executionState.queuedBatchRequests, executionState.runningRequests, 0,
                executionState.localBatch)) {
            AUTIL_LOG(ERROR,
                      "report abnormal batch cleanup failed, batchId [%ld], "
                      "requestCount [%d], remainingInflight [%d], "
                      "hasLatencyInputResource [%d]",
                      batchId, executionState.batchRequestCount, inflight.requestCount,
                      static_cast<bool>(latencyInputR));
        }
    }

    int64_t batchId = 0;
    std::vector<uint64_t> ownedRequestIds;
    std::weak_ptr<AgentBatchR> agentBatchR;
    std::weak_ptr<AgentLatencyInputR> agentLatencyInputR;
    std::weak_ptr<AgentReportR> agentReportR;
    std::shared_ptr<ConcurrentRequestLatencyTracker> inflightState;
    std::shared_ptr<kmonitor::MetricsReporter> metricsReporter;
    std::mutex mutex;
    std::mutex publicationMutex;
    std::map<NaviSchedCacheKeyType,
             std::pair<std::unordered_set<uint64_t>, std::unordered_set<uint64_t>>>
        failedPublications;
    bool started = false;
    bool finished = false;
    bool executionGeometryEligible = true;
};

NaviSchedBatchFinishReporter::NaviSchedBatchFinishReporter(
    const std::shared_ptr<State> &state)
    : _state(state)
{
}

NaviSchedBatchFinishReporter::~NaviSchedBatchFinishReporter()
{
    if (_state) {
        _state->finishAbnormally();
    }
}

bool NaviSchedBatchFinishReporter::beginExecution()
{
    return _state && _state->beginExecution();
}

bool NaviSchedBatchFinishReporter::startBatch(const std::vector<uint64_t> &requestIds)
{
    return startBatch(requestIds, 0);
}

bool NaviSchedBatchFinishReporter::startBatch(const std::vector<uint64_t> &requestIds,
                                              int64_t executionStartTimeUs)
{
    return _state && _state->start(requestIds, executionStartTimeUs);
}

bool NaviSchedBatchFinishReporter::resolveExecutionCache(
    const std::vector<NaviSchedRequestStats> &requests)
{
    return _state && _state->resolveExecutionCache(requests);
}

bool NaviSchedBatchFinishReporter::finishBatch(
    const std::vector<NaviSchedRequestStats> &requests)
{
    return _state && _state->finish(requests, nullptr);
}

bool NaviSchedBatchFinishReporter::reportCachePublicationFailed(
    NaviSchedCacheKeyType type, const std::vector<uint64_t> &full,
    const std::vector<uint64_t> &tail)
{
    return _state && _state->publicationFailed(type, full, tail);
}

std::function<bool(NaviSchedCacheKeyType, const std::vector<uint64_t> &,
                   const std::vector<uint64_t> &)>
NaviSchedBatchFinishReporter::makeCachePublicationFailureCallback() const
{
    return [state = _state](NaviSchedCacheKeyType type, const std::vector<uint64_t> &full,
                            const std::vector<uint64_t> &tail) {
        return state && state->publicationFailed(type, full, tail);
    };
}

bool NaviSchedBatchFinishReporter::finishBatch(
    const std::vector<NaviSchedRequestStats> &requests, double wallTimeMs,
    double executeModelTimeMs, int64_t executionStepCount, int64_t prefillTokenCount,
    int64_t decodeRequestStepCount, int64_t decodeTokenCount,
    double admissionToFirstExecutionMs, int64_t executionFinishTimeUs)
{
    return finishBatch(requests, wallTimeMs, executeModelTimeMs, executionStepCount,
                       prefillTokenCount, decodeRequestStepCount, decodeTokenCount,
                       admissionToFirstExecutionMs, 0.0, executionFinishTimeUs);
}

bool NaviSchedBatchFinishReporter::finishBatch(
    const std::vector<NaviSchedRequestStats> &requests, double wallTimeMs,
    double executeModelTimeMs, int64_t executionStepCount, int64_t prefillTokenCount,
    int64_t decodeRequestStepCount, int64_t decodeTokenCount,
    double admissionToFirstExecutionMs, double nativePreparationMs,
    int64_t executionFinishTimeUs)
{
    if (!std::isfinite(admissionToFirstExecutionMs) ||
        admissionToFirstExecutionMs < -1.0) {
        return false;
    }
    AgentBatchR::BackendBatchExecutionSample sample;
    sample.executionFinishTimeUs = executionFinishTimeUs;
    sample.wallTimeUs = static_cast<int64_t>(std::max(0.0, wallTimeMs) * 1000.0);
    sample.executeModelTimeUs =
        static_cast<int64_t>(std::max(0.0, executeModelTimeMs) * 1000.0);
    sample.executionStepCount = std::max<int64_t>(0, executionStepCount);
    sample.prefillTokenCount = std::max<int64_t>(0, prefillTokenCount);
    sample.decodeRequestStepCount = std::max<int64_t>(0, decodeRequestStepCount);
    sample.decodeTokenCount = std::max<int64_t>(0, decodeTokenCount);
    if (std::isfinite(admissionToFirstExecutionMs) &&
        admissionToFirstExecutionMs >= 0.0) {
        sample.admissionToFirstExecutionTimeUs =
            static_cast<int64_t>(admissionToFirstExecutionMs * 1000.0);
    }
    return _state &&
           _state->finish(requests, &sample, nullptr, -1.0, nativePreparationMs);
}

bool NaviSchedBatchFinishReporter::finishBatch(
    const std::vector<NaviSchedRequestStats> &requests, double wallTimeMs,
    double executeModelTimeMs, int64_t executionStepCount, int64_t prefillTokenCount,
    int64_t decodeRequestStepCount, int64_t decodeTokenCount)
{
    return finishBatch(requests, wallTimeMs, executeModelTimeMs, executionStepCount,
                       prefillTokenCount, decodeRequestStepCount, decodeTokenCount, -1.0);
}

bool NaviSchedBatchFinishReporter::finishBatchWithExecutionGeometry(
    const std::vector<NaviSchedRequestStats> &requests, double wallTimeMs,
    double executeModelTimeMs, int64_t executionStepCount, int64_t prefillTokenCount,
    int64_t decodeRequestStepCount, int64_t decodeTokenCount,
    double admissionToFirstExecutionMs, int64_t globalPaddedTokens,
    int64_t executionFinishTimeUs, double nativePreparationMs)
{
    const double maxTimeMs =
        static_cast<double>(std::numeric_limits<int64_t>::max()) / 1000.0;
    if (!std::isfinite(wallTimeMs) || wallTimeMs < 0.0 || wallTimeMs >= maxTimeMs ||
        !std::isfinite(executeModelTimeMs) || executeModelTimeMs < 0.0 ||
        executeModelTimeMs >= maxTimeMs || !std::isfinite(admissionToFirstExecutionMs) ||
        admissionToFirstExecutionMs < -1.0 || admissionToFirstExecutionMs >= maxTimeMs ||
        executionStepCount <= 0 || globalPaddedTokens <= 0 ||
        globalPaddedTokens < executionStepCount || prefillTokenCount < 0 ||
        globalPaddedTokens < prefillTokenCount || decodeRequestStepCount != 0 ||
        decodeTokenCount != 0) {
        return false;
    }
    AgentBatchR::BackendBatchExecutionSample sample;
    sample.executionFinishTimeUs = executionFinishTimeUs;
    sample.wallTimeUs = static_cast<int64_t>(wallTimeMs * 1000.0);
    sample.executeModelTimeUs = static_cast<int64_t>(executeModelTimeMs * 1000.0);
    sample.executionStepCount = executionStepCount;
    sample.prefillTokenCount = prefillTokenCount;
    sample.globalPaddedTokens = globalPaddedTokens;
    if (admissionToFirstExecutionMs >= 0.0) {
        sample.admissionToFirstExecutionTimeUs =
            static_cast<int64_t>(admissionToFirstExecutionMs * 1000.0);
    }
    return _state && _state->finish(requests, &sample, nullptr, -1.0, nativePreparationMs);
}

bool NaviSchedBatchFinishReporter::finishBatchWithProjectedExecutionGeometry(
    const std::vector<NaviSchedRequestStats> &requests, double wallTimeMs,
    double executeModelTimeMs, int64_t executionStepCount, int64_t prefillTokenCount,
    int64_t decodeRequestStepCount, int64_t decodeTokenCount,
    double admissionToFirstExecutionMs, int64_t globalPaddedTokens,
    int64_t projectedExecutionStepCount, int64_t projectedGlobalPaddedTokens,
    int64_t executionFinishTimeUs, double nativePreparationMs)
{
    const double maxTimeMs =
        static_cast<double>(std::numeric_limits<int64_t>::max()) / 1000.0;
    if (!std::isfinite(wallTimeMs) || wallTimeMs < 0.0 || wallTimeMs >= maxTimeMs ||
        !std::isfinite(executeModelTimeMs) || executeModelTimeMs < 0.0 ||
        executeModelTimeMs >= maxTimeMs || !std::isfinite(admissionToFirstExecutionMs) ||
        admissionToFirstExecutionMs < -1.0 || admissionToFirstExecutionMs >= maxTimeMs ||
        executionStepCount <= 0 || globalPaddedTokens < 0 ||
        projectedExecutionStepCount <= 0 ||
        projectedGlobalPaddedTokens < projectedExecutionStepCount ||
        (globalPaddedTokens > 0 && (globalPaddedTokens < executionStepCount ||
                                    globalPaddedTokens < prefillTokenCount)) ||
        prefillTokenCount < 0 || decodeRequestStepCount != 0 || decodeTokenCount != 0) {
        return false;
    }
    AgentBatchR::BackendBatchExecutionSample sample;
    sample.executionFinishTimeUs = executionFinishTimeUs;
    sample.wallTimeUs = static_cast<int64_t>(wallTimeMs * 1000.0);
    sample.executeModelTimeUs = static_cast<int64_t>(executeModelTimeMs * 1000.0);
    sample.executionStepCount = executionStepCount;
    sample.prefillTokenCount = prefillTokenCount;
    sample.globalPaddedTokens = globalPaddedTokens;
    sample.projectedExecutionStepCount = projectedExecutionStepCount;
    sample.projectedGlobalPaddedTokens = projectedGlobalPaddedTokens;
    if (admissionToFirstExecutionMs >= 0.0) {
        sample.admissionToFirstExecutionTimeUs =
            static_cast<int64_t>(admissionToFirstExecutionMs * 1000.0);
    }
    return _state && _state->finish(requests, &sample, nullptr, -1.0, nativePreparationMs);
}

bool NaviSchedBatchFinishReporter::finishBatchWithUnifiedLatency(
    const std::vector<NaviSchedRequestStats> &requests, double wallTimeMs,
    double executeModelTimeMs, int64_t executionStepCount, int64_t prefillTokenCount,
    int64_t decodeRequestStepCount, int64_t decodeTokenCount,
    const NaviSchedLatencyFeatures &precedingFeatures, double unifiedLatencyMs,
    double admissionToFirstExecutionMs)
{
    return finishBatchWithUnifiedLatency(
        requests, wallTimeMs, executeModelTimeMs, executionStepCount, prefillTokenCount,
        decodeRequestStepCount, decodeTokenCount, precedingFeatures, unifiedLatencyMs,
        admissionToFirstExecutionMs, 0.0);
}

bool NaviSchedBatchFinishReporter::finishBatchWithUnifiedLatency(
    const std::vector<NaviSchedRequestStats> &requests, double wallTimeMs,
    double executeModelTimeMs, int64_t executionStepCount, int64_t prefillTokenCount,
    int64_t decodeRequestStepCount, int64_t decodeTokenCount,
    const NaviSchedLatencyFeatures &precedingFeatures, double unifiedLatencyMs,
    double admissionToFirstExecutionMs, double nativePreparationMs)
{
    if (!std::isfinite(unifiedLatencyMs) || unifiedLatencyMs < 0.0 ||
        !std::isfinite(admissionToFirstExecutionMs) ||
        admissionToFirstExecutionMs < -1.0) {
        return false;
    }
    AgentBatchR::BackendBatchExecutionSample sample;
    sample.wallTimeUs = static_cast<int64_t>(std::max(0.0, wallTimeMs) * 1000.0);
    sample.executeModelTimeUs =
        static_cast<int64_t>(std::max(0.0, executeModelTimeMs) * 1000.0);
    sample.executionStepCount = std::max<int64_t>(0, executionStepCount);
    sample.prefillTokenCount = std::max<int64_t>(0, prefillTokenCount);
    sample.decodeRequestStepCount = std::max<int64_t>(0, decodeRequestStepCount);
    sample.decodeTokenCount = std::max<int64_t>(0, decodeTokenCount);
    if (std::isfinite(admissionToFirstExecutionMs) &&
        admissionToFirstExecutionMs >= 0.0) {
        sample.admissionToFirstExecutionTimeUs =
            static_cast<int64_t>(admissionToFirstExecutionMs * 1000.0);
    }
    return _state && _state->finish(requests, &sample, &precedingFeatures,
                                    unifiedLatencyMs, nativePreparationMs);
}

bool NaviSchedBatchFinishReporter::abortRequests(const std::vector<uint64_t> &requestIds)
{
    return _state && _state->abortRequests(requestIds);
}

std::string NaviSchedAgentImpl::getLatencyModelStateFile(const NaviSchedConfig &config)
{
    if (!config.latencyModelStateFile.empty()) {
        return config.latencyModelStateFile;
    }
    if (config.enableSnapshot && !config.name.empty()) {
        return config.snapshotDir + "/" + config.name + "/latency_model.state";
    }
    return "";
}

NaviSchedAgentImpl *NaviSchedAgentImpl::getImpl(NaviSchedAgent &agent)
{
    return agent._impl.get();
}

NaviSchedAgentImpl::NaviSchedAgentImpl(const NaviSchedAgentConfig &agentConfig,
                                       int32_t serverPort, const std::string &serverIp,
                                       int64_t totalBlocks)
    : NaviSchedRole(agentConfig.service)
    , _agentBatchR(std::make_shared<AgentBatchR>(agentConfig.service.batchTimeoutMs,
                                                 agentConfig.service.closedBatchKeepMs,
                                                 agentConfig.pdSeparated &&
                                                     !agentConfig.unifiedLatencyPrediction))
    , _agentLatencyInputR(std::make_shared<AgentLatencyInputR>())
    , _agentReportBridgeR(std::make_shared<AgentReportBridgeR>())
    , _agentRequestR(std::make_shared<AgentRequestR>())
    , _agentResponseR(std::make_shared<AgentResponseR>())
    , _agentConfig(agentConfig.runtime)
    , _latencyModelFingerprint(agentConfig.latencyModelFingerprint)
    , _latencyHistoryFingerprint(agentConfig.latencyHistoryFingerprint.empty()
                                     ? agentConfig.latencyModelFingerprint
                                     : agentConfig.latencyHistoryFingerprint)
    , _latencyHistoryBinaryFingerprint(agentConfig.latencyHistoryBinaryFingerprint)
    , _serverIp(serverIp)
    , _serverPort(serverPort)
    , _cacheTotalBlocks(totalBlocks)
    , _pdSeparated(agentConfig.pdSeparated)
    , _unifiedLatencyPrediction(agentConfig.unifiedLatencyPrediction)
    , _executionGeometryEnabled(agentConfig.executionGeometryEnabled)
    , _executionCacheSourceEnabled(agentConfig.executionCacheSourceEnabled)
    , _projectedExecutionGeometryEnabled(agentConfig.projectedExecutionGeometryEnabled)
    , _fullAttentionQueryDropTokens(agentConfig.fullAttentionQueryDropTokens)
    , _mambaReplayStopShift(agentConfig.mambaReplayStopShift)
    , _mambaRetentionInterval(agentConfig.mambaRetentionInterval)
    , _mambaAlign(agentConfig.mambaAlign)
    , _timeCompression(agentConfig.timeCompression)
    , _inflightState(std::make_shared<ConcurrentRequestLatencyTracker>())
    , _forwardState(std::make_shared<AgentForwardState>())
{
    AgentControlToMasterData control;
    control.executionGeometryEnabled = agentConfig.executionGeometryEnabled;
    control.projectedExecutionGeometryEnabled =
        agentConfig.projectedExecutionGeometryEnabled;
    control.prefillExecutionPolicy = agentConfig.prefillExecutionPolicy;
    control.peerCacheEnabled = agentConfig.peerCacheEnabled;
    _agentReportBridgeR->setControlConfig(control);
}

NaviSchedAgentImpl::~NaviSchedAgentImpl() { stop(); }

bool NaviSchedAgentImpl::applyConfig(const std::string &configStr)
{
    auto resourceMap = std::make_shared<navi::ResourceMap>();
    if (!resourceMap->addResource(_agentBatchR) ||
        !resourceMap->addResource(_agentLatencyInputR) ||
        !resourceMap->addResource(_agentReportBridgeR) ||
        !resourceMap->addResource(_agentRequestR) ||
        !resourceMap->addResource(_agentResponseR)) {
        return false;
    }
    return update(configStr, resourceMap);
}

bool NaviSchedAgentImpl::init()
{
    if ((_executionGeometryEnabled &&
         (!_pdSeparated || _unifiedLatencyPrediction ||
          !TPUPrefillExecutionPolicy::validate(
              _agentReportBridgeR->getControlConfig().prefillExecutionPolicy))) ||
        (_projectedExecutionGeometryEnabled && !_executionGeometryEnabled) ||
        (_executionCacheSourceEnabled && !_executionGeometryEnabled) ||
        _cacheTotalBlocks < 0 || (!_pdSeparated && !_unifiedLatencyPrediction) ||
        !std::isfinite(_timeCompression) || _timeCompression <= 0.0 ||
        getConfig().latencyPersistIntervalMs <= 0) {
        return false;
    }
    if (!NaviSchedRole::init()) {
        return false;
    }
    _agentBizName = _agentConfig.bizName;
    if (!_agentBatchR->setBatchTimeoutMs(getConfig().batchTimeoutMs)) {
        return false;
    }
    NaviSchedConfigBuilder::NamedConfigs kernelConfigs;
    autil::legacy::json::JsonMap latencyKernelConfig;
    latencyKernelConfig["queue_learning_rate"] = getConfig().queueLearningRate;
    latencyKernelConfig["time_compression"] = _timeCompression;
    kernelConfigs["navi_sched.agent_latency_k"] =
        autil::legacy::ToJsonString(latencyKernelConfig, true);
    autil::legacy::json::JsonMap historySaveKernelConfig;
    historySaveKernelConfig["persist_interval_ms"] = getConfig().latencyPersistIntervalMs;
    kernelConfigs["navi_sched.agent_history_save_k"] =
        autil::legacy::ToJsonString(historySaveKernelConfig, true);
    std::vector<std::string> kernels = {
        "^navi_sched\\.agent_.*_k$",
    };
    autil::legacy::json::JsonMap historyConfig;
    historyConfig["state_file"] = getLatencyModelStateFile(getConfig());
    historyConfig["formula_fingerprint"] = _latencyModelFingerprint;
    historyConfig["history_fingerprint"] = _latencyHistoryFingerprint;
    historyConfig["binary_fingerprint"] =
        makeHistoryBinaryFingerprint(_latencyHistoryBinaryFingerprint);
    historyConfig["history_limit"] = getConfig().latencyHistoryLimit;
    historyConfig["history_bucket_capacity"] = getConfig().latencyHistoryBucketCapacity;
    historyConfig["fit_batch_size"] = getConfig().latencyFitBatchSize;
    historyConfig["fit_interval"] = getConfig().latencyFitInterval;
    historyConfig["persist_interval"] = getConfig().latencyPersistInterval;
    historyConfig["warmup_sample_count"] = getConfig().latencyWarmupSampleCount;
    NaviSchedConfigBuilder::NamedConfigs resourceConfigs;
    const auto runtimeSchedulerConfig =
        AgentSchedulerConfig::fromServiceConfig(getConfig(), _timeCompression);
    autil::legacy::json::JsonMap runtimeSchedulerResourceConfig;
    runtimeSchedulerResourceConfig["config_json"] =
        AgentSchedulerConfigR::toJson(runtimeSchedulerConfig);
    resourceConfigs[AgentSchedulerConfigR::RESOURCE_ID] =
        autil::legacy::ToJsonString(runtimeSchedulerResourceConfig, true);
    autil::legacy::json::JsonMap cacheConfig;
    cacheConfig["total_blocks"] = _cacheTotalBlocks;
    resourceConfigs[AgentCacheR::RESOURCE_ID] =
        autil::legacy::ToJsonString(cacheConfig, true);
    autil::legacy::json::JsonMap reportConfig;
    reportConfig["server_ip"] = _serverIp;
    reportConfig["server_port"] = _serverPort;
    reportConfig["latency_validation_enabled"] = getConfig().latencyValidationEnabled;
    reportConfig["pd_separated"] = _pdSeparated;
    reportConfig["unified_latency_prediction"] = _unifiedLatencyPrediction;
    reportConfig["full_attention_query_drop_tokens"] = _fullAttentionQueryDropTokens;
    reportConfig["mamba_replay_stop_shift"] = _mambaReplayStopShift;
    reportConfig["mamba_retention_interval"] = _mambaRetentionInterval;
    reportConfig["mamba_align"] = _mambaAlign;
    reportConfig["workload_publish_timing_enabled"] = getConfig().enableKmon;
    reportConfig["latency_model_fingerprint"] = _latencyModelFingerprint;
    resourceConfigs[AgentReportR::RESOURCE_ID] =
        autil::legacy::ToJsonString(reportConfig, true);
    autil::legacy::json::JsonMap nonLinearPrefillModelConfig;
    nonLinearPrefillModelConfig["learning_rate"] = getConfig().latencyLearningRate;
    nonLinearPrefillModelConfig["check_gradient"] = getConfig().latencyCheckGradient;
    nonLinearPrefillModelConfig["zero_anchor_max_ms"] =
        getConfig().latencyZeroAnchorMaxMs;
    nonLinearPrefillModelConfig["pd_separated"] = _pdSeparated;
    nonLinearPrefillModelConfig["execution_geometry_enabled"] = _executionGeometryEnabled;
    nonLinearPrefillModelConfig["projected_execution_geometry_enabled"] =
        _projectedExecutionGeometryEnabled;
    nonLinearPrefillModelConfig["execution_cache_source_enabled"] =
        _executionCacheSourceEnabled;
    nonLinearPrefillModelConfig["use_batch_count_feature"] = _pdSeparated;
    nonLinearPrefillModelConfig["use_capacity_split_feature"] = !_pdSeparated;
    resourceConfigs[NonLinearPrefillModelR::RESOURCE_ID] =
        autil::legacy::ToJsonString(nonLinearPrefillModelConfig, true);
    resourceConfigs[AgentHistoryR::RESOURCE_ID] =
        autil::legacy::ToJsonString(historyConfig, true);
    autil::legacy::json::JsonMap latencyModelConfig;
    latencyModelConfig["latency_model_fingerprint"] = _latencyModelFingerprint;
    latencyModelConfig["cold_start_fit_epochs"] = getConfig().latencyColdStartFitEpochs;
    latencyModelConfig["trust_min_revision"] = getConfig().latencyModelTrustMinRevision;
    latencyModelConfig["zero_anchor_min_samples"] =
        getConfig().latencyZeroAnchorMinSamples;
    resourceConfigs[LatencyModelR::RESOURCE_ID] =
        autil::legacy::ToJsonString(latencyModelConfig, true);
    auto configStr = NaviSchedConfigBuilder::build(
        _agentConfig, kernels, false, kernelConfigs, resourceConfigs,
        {AgentInstanceR::RESOURCE_ID, AgentSchedulerConfigR::RESOURCE_ID,
         AgentReportR::RESOURCE_ID});
    if (!configStr.empty() && applyConfig(configStr)) {
        return true;
    }
    stop();
    return false;
}

bool NaviSchedAgentImpl::addBatchRequest(const NaviSchedBatchRequest &request,
                                         int64_t batchId, int64_t expectedBatchSize,
                                         const NaviSchedReadyBatchCallback &callback,
                                         const std::function<void()> &acceptedCallback)
{
    if (!_agentReportR || !_agentBatchR || !_agentLatencyInputR || !_inflightState) {
        return false;
    }
    if (!request.valid() || !callback || batchId < 0 || expectedBatchSize < 0 ||
        ((0 == batchId) != (0 == expectedBatchSize))) {
        return false;
    }
    const bool tracing = traceRequestBatch(batchId);
    const auto auditBeginTimeUs = tracing ? autil::TimeUtility::currentTime() : 0;
    auto requestAudit = _agentReportR->makeRequestAudit(request);
    const auto auditDoneTimeUs = tracing ? autil::TimeUtility::currentTime() : 0;
    if (!requestAudit.valid) {
        return false;
    }
    const auto &requestStats = requestAudit.stats;
    AgentBatchR::CacheKeyAudit cacheKeyAudit;
    cacheKeyAudit.blockHashes = request.blockHashes;
    cacheKeyAudit.cachePublicationTokenCount = request.cachePublicationTokenCount;
    cacheKeyAudit.physicalCacheHitTokenCount = requestAudit.physicalCacheHitTokenCount;
    cacheKeyAudit.arrivalTimeUs = requestAudit.eventTimeUs;
    cacheKeyAudit.cacheRevision = requestAudit.cacheState.revision;
    cacheKeyAudit.cachedBlockCount = requestAudit.cacheState.cachedBlockCount;
    cacheKeyAudit.cacheHashChecksum = requestAudit.cacheState.hashChecksum;
    cacheKeyAudit.fullReadyBlockCount = requestAudit.cacheState.fullReadyBlockCount;
    cacheKeyAudit.fullReadyHashChecksum = requestAudit.cacheState.fullReadyHashChecksum;
    cacheKeyAudit.tailReadyBlockCount = requestAudit.cacheState.tailReadyBlockCount;
    cacheKeyAudit.tailReadyHashChecksum = requestAudit.cacheState.tailReadyHashChecksum;
    cacheKeyAudit.hybridTailRequired = requestAudit.hybridTailRequired;
    RequestLatencyTracker::Snapshot inflight;
    const auto predictedLatencyMs =
        request.predictedQueueLatencyMs + request.predictedExecutionLatencyMs;
    if (!addInflightRequest(request.requestId, predictedLatencyMs,
                            request.latencyPrediction, inflight)) {
        AUTIL_LOG(ERROR,
                  "reject Agent request after inflight ownership failure, "
                  "requestId [%lu], scheduleBatchId [%ld]",
                  request.requestId, batchId);
        return false;
    }
    bool acceptedCallbackStarted = false;
    bool accepted = false;
    if (tracing) {
        const auto nowTimeUs = autil::TimeUtility::currentTime();
        AUTIL_LOG(INFO,
                  "[navi-request-stage] stage=agent_pre_batch request_id=%lu "
                  "schedule_batch_id=%ld event_time_us=%ld cache_audit_us=%ld "
                  "inflight_prepare_us=%ld",
                  request.requestId, batchId, nowTimeUs,
                  auditDoneTimeUs - auditBeginTimeUs, nowTimeUs - auditDoneTimeUs);
    }
    try {
        accepted = _agentBatchR->addRequest(
            requestStats, batchId, expectedBatchSize,
            [this, callback](const NaviSchedBatchPtr &batch) {
                for (const auto type :
                     {NaviSchedCacheKeyType::BLOCK_POOL, NaviSchedCacheKeyType::V6D}) {
                    const auto cacheState = _agentReportR->getCacheState(type);
                    if (!cacheState.policy.valid(type)) {
                        continue;
                    }
                    NaviSchedPendingCacheReservation pendingReservation;
                    if (_agentBatchR->makePendingCacheReservation(
                            batch->batchId, cacheState.policy, pendingReservation,
                            type) &&
                        _agentReportR->reportPendingCacheReady(pendingReservation,
                                                               type)) {
                        if (!_agentBatchR->markPendingCacheTracked(batch->batchId,
                                                                   type)) {
                            _agentReportR->reportPendingCacheAborted(batch->batchId,
                                                                     type);
                            throw std::runtime_error(
                                "mark ready batch pending cache tracked failed");
                        }
                    }
                }
                AgentBatchR::ExecutionState state;
                _agentBatchR->getExecutionState(state);
                if (!_agentReportR->reportNativeBatchReady(batch)) {
                    throw std::runtime_error("publish native batch workload failed");
                }
                if (!_agentLatencyInputR->submitBatchReady(
                        batch, state.batchWaitingRequests, state.queuedBatchRequests,
                        state.runningRequests, state.eventTimeUs)) {
                    AUTIL_LOG(ERROR,
                              "submit ready batch latency input failed, "
                              "scheduleBatchId [%ld], localBatchId [%ld], "
                              "requestCount [%lu]",
                              batch->scheduleBatchId, batch->batchId,
                              batch->requestIds.size());
                    throw std::runtime_error("submit ready batch latency input failed");
                }
                auto reporter =
                    makeBatchFinishReporter(batch->batchId, batch->requestIds);
                batch->finishReporter = reporter;
                try {
                    callback(batch);
                } catch (...) {
                    reporter->_state->finishAbnormally();
                    batch->finishReporter.reset();
                    throw;
                }
            },
            [this, request, &requestStats, acceptedCallback, &acceptedCallbackStarted]() {
                acceptedCallbackStarted = true;
                AgentBatchR::ExecutionState state;
                _agentBatchR->getExecutionState(state);
                const bool nativeWorkload = !_agentReportR->pdSeparated() &&
                                            _agentReportR->usesUnifiedLatencyPrediction();
                if (nativeWorkload && !_agentReportR->reportNativeRequestReceived(
                                          requestStats, state.batchWaitingRequests,
                                          state.queuedBatchRequests,
                                          state.runningRequests, state.eventTimeUs)) {
                    throw std::runtime_error(
                        "publish native accepted request workload failed");
                }
                if (acceptedCallback) {
                    acceptedCallback();
                }
                if (!nativeWorkload) {
                    _agentBatchR->getExecutionState(state);
                    const auto currentInflight = getInflightStats();
                    _agentReportR->reportRequestReceived(
                        request.requestId, request.totalTokenCount,
                        currentInflight.requestCount, state.batchWaitingRequests,
                        state.queuedBatchRequests, state.runningRequests,
                        state.eventTimeUs);
                }
            },
            cacheKeyAudit,
            [this](const AgentBatchR::ExecutionState &state) {
                handleAcceptedBatchFailure(state);
            },
            request.timeoutMs);
    } catch (...) {
        if (!acceptedCallbackStarted) {
            removeInflightRequests({request.requestId}, "batch_admission_exception");
        }
        throw;
    }
    if (!accepted && !acceptedCallbackStarted) {
        removeInflightRequests({request.requestId}, "batch_admission_rejected");
    }
    return accepted;
}

bool NaviSchedAgentImpl::abortPendingBatch(int64_t scheduleBatchId)
{
    if (!_agentBatchR || !_agentLatencyInputR || !_inflightState ||
        scheduleBatchId <= 0) {
        return false;
    }
    AgentBatchR::ExecutionState state;
    if (!_agentBatchR->abortPendingBatch(scheduleBatchId, state)) {
        return false;
    }
    if (_agentReportR)
        _agentReportR->reportNativeUnbatchedRequestsRemoved(state.requestIds);
    const auto inflight =
        removeInflightRequests(state.requestIds, "pending_batch_aborted");
    return _agentLatencyInputR->submitBatchAborted(
        state.batchRequestCount, state.batchId, inflight.requestCount,
        state.batchWaitingRequests, state.queuedBatchRequests, state.runningRequests,
        state.eventTimeUs, state.localBatch);
}

bool NaviSchedAgentImpl::setForwardBatchHandler(
    const NaviSchedForwardBatchHandler &handler, bool streamResponse,
    const NaviSchedForwardRequestAcceptedHandler &requestAcceptedHandler)
{
    if (!_agentRequestR || !_forwardState) {
        return false;
    }
    if (handler) {
        std::lock_guard<std::mutex> lock(_forwardState->mutex);
        _forwardState->handler = handler;
        _forwardState->requestAcceptedHandler = requestAcceptedHandler;
        _forwardState->streamResponse = streamResponse;
        _agentRequestR->setCompletionHandler(
            [this](const NaviSchedForwardRequest &request,
                   const navi::AsyncPipePtr &completionPipe) {
                return submitForwardRequest(request, completionPipe);
            });
        return true;
    }

    _agentRequestR->setCompletionHandler({});
    std::vector<std::pair<uint64_t, navi::AsyncPipePtr>> completionPipes;
    decltype(_forwardState->requests) releasedRequests;
    std::unordered_set<int64_t> pendingScheduleBatchIds;
    {
        std::unique_lock<std::mutex> lock(_forwardState->mutex);
        _forwardState->handler = {};
        _forwardState->requestAcceptedHandler = {};
        completionPipes.reserve(_forwardState->requests.size());
        for (auto &pair : _forwardState->requests) {
            pendingScheduleBatchIds.emplace(pair.second->request.decision.batchId);
            completionPipes.emplace_back(pair.first, pair.second->completionPipe);
        }
        releasedRequests.swap(_forwardState->requests);
        _forwardState->condition.wait(
            lock, [this] { return _forwardState->activeHandlerCount == 0; });
    }
    for (auto scheduleBatchId : pendingScheduleBatchIds) {
        abortPendingBatch(scheduleBatchId);
    }
    for (const auto &[requestId, pipe] : completionPipes) {
        if (!_agentResponseR->fail(requestId, "agent forwarding disabled") && pipe) {
            pipe->setEof();
        }
    }
    return true;
}

bool NaviSchedAgentImpl::submitForwardRequest(const NaviSchedForwardRequest &request,
                                              const navi::AsyncPipePtr &completionPipe)
{
    const auto arrivalTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
    const bool tracing = traceRequestBatch(request.decision.batchId);
    if (!completionPipe || request.request.forwardPayload.empty() ||
        request.request.requestId == 0 || request.decision.batchId <= 0 ||
        request.decision.expectedBatchSize <= 0) {
        abortForwardBatch(request.decision.batchId, completionPipe);
        return false;
    }
    if (!_agentResponseR->registerRequest(request.request.requestId, completionPipe)) {
        abortForwardBatch(request.decision.batchId, completionPipe);
        return false;
    }
    const auto responseRegisteredTimeUs = tracing ? autil::TimeUtility::currentTime() : 0;
    bool registered = false;
    const auto pendingRequest = std::make_shared<const AgentForwardState::PendingRequest>(
        AgentForwardState::PendingRequest{request, completionPipe, arrivalTimeUs});
    const auto lockBeginTimeUs = tracing ? autil::TimeUtility::currentTime() : 0;
    int64_t lockAcquiredTimeUs = 0;
    {
        std::lock_guard<std::mutex> lock(_forwardState->mutex);
        if (tracing) {
            lockAcquiredTimeUs = autil::TimeUtility::currentTime();
        }
        if (_forwardState->handler) {
            registered =
                _forwardState->requests.emplace(request.request.requestId, pendingRequest)
                    .second;
        }
    }
    if (tracing) {
        const auto nowTimeUs = autil::TimeUtility::currentTime();
        AUTIL_LOG(INFO,
                  "[navi-request-stage] stage=agent_forward_registered request_id=%lu "
                  "schedule_batch_id=%ld event_time_us=%ld submit_begin_us=%ld "
                  "response_register_us=%ld request_clone_us=%ld state_lock_wait_us=%ld "
                  "state_copy_us=%ld "
                  "registered=%d",
                  request.request.requestId, request.decision.batchId, nowTimeUs,
                  arrivalTimeUs, responseRegisteredTimeUs - arrivalTimeUs,
                  lockBeginTimeUs - responseRegisteredTimeUs,
                  lockAcquiredTimeUs - lockBeginTimeUs, nowTimeUs - lockAcquiredTimeUs,
                  registered);
    }
    if (!registered) {
        _agentResponseR->fail(request.request.requestId,
                              "failed to register forwarded request");
        abortForwardBatch(request.decision.batchId, completionPipe);
        return false;
    }
    NaviSchedBatchRequest batchRequest;
    batchRequest.requestId = request.request.requestId;
    batchRequest.timeoutMs = request.request.timeoutMs;
    batchRequest.totalTokenCount = request.request.promptTokenCount;
    batchRequest.blockSize = request.request.blockSize;
    batchRequest.blockHashes = request.request.blockHashes;
    batchRequest.predictedCacheHitTokenCount =
        request.decision.predictedCacheHitTokenCount;
    batchRequest.predictedQueueLatencyMs = request.decision.predictedQueueLatencyMs;
    batchRequest.predictedExecutionLatencyMs =
        request.decision.predictedExecutionLatencyMs;
    batchRequest.latencyPrediction = request.decision.latencyPrediction;
    batchRequest.latencyTrainingPhase = request.decision.latencyTrainingPhase;
    const auto requestAccepted = [this, &request]() {
        NaviSchedForwardRequestAcceptedHandler handler;
        {
            std::lock_guard<std::mutex> lock(_forwardState->mutex);
            handler = _forwardState->requestAcceptedHandler;
        }
        if (!handler) {
            return;
        }
        try {
            handler(request.request.requestId, request.request.forwardPayload);
        } catch (...) {
            AUTIL_LOG(ERROR,
                      "forward request accepted handler failed, requestId [%lu], "
                      "scheduleBatchId [%ld]",
                      request.request.requestId, request.decision.batchId);
        }
    };
    if (addBatchRequest(
            batchRequest, request.decision.batchId, request.decision.expectedBatchSize,
            [this](const NaviSchedBatchPtr &batch) { handleForwardBatch(batch); },
            requestAccepted)) {
        return true;
    }
    abortForwardBatch(request.decision.batchId, completionPipe);
    return false;
}

void NaviSchedAgentImpl::abortForwardBatch(int64_t scheduleBatchId,
                                           const navi::AsyncPipePtr &completionPipe)
{
    if (scheduleBatchId > 0) {
        abortPendingBatch(scheduleBatchId);
    }
    std::vector<std::pair<uint64_t, navi::AsyncPipePtr>> completionPipes;
    std::vector<std::shared_ptr<const AgentForwardState::PendingRequest>>
        releasedRequests;
    {
        std::lock_guard<std::mutex> lock(_forwardState->mutex);
        for (auto it = _forwardState->requests.begin();
             _forwardState->requests.end() != it;) {
            if (it->second->request.decision.batchId != scheduleBatchId) {
                ++it;
                continue;
            }
            completionPipes.emplace_back(it->first, it->second->completionPipe);
            releasedRequests.emplace_back(std::move(it->second));
            it = _forwardState->requests.erase(it);
        }
    }
    for (const auto &[requestId, pipe] : completionPipes) {
        if (!_agentResponseR->fail(requestId, "forward batch aborted") && pipe) {
            pipe->setEof();
        }
    }
}

void NaviSchedAgentImpl::handleForwardBatch(const NaviSchedBatchPtr &batch)
{
    if (!batch) {
        return;
    }
    const bool tracing = traceRequestBatch(batch->scheduleBatchId);
    const auto forwardHandlerStartTimeUs =
        autil::TimeUtility::currentTimeInMicroSeconds();
    NaviSchedForwardBatchHandler handler;
    bool streamResponse = true;
    std::vector<NaviSchedForwardedRequest> forwarded;
    std::vector<std::shared_ptr<const AgentForwardState::PendingRequest>> pendingRequests;
    bool handlerActive = false;
    const auto finishHandler = [this, &handlerActive]() {
        if (!handlerActive) {
            return;
        }
        std::lock_guard<std::mutex> lock(_forwardState->mutex);
        --_forwardState->activeHandlerCount;
        handlerActive = false;
        _forwardState->condition.notify_all();
    };
    autil::ScopeGuard handlerGuard(finishHandler);
    int64_t scheduleBatchId = 0;
    const auto forwardStateLockBeginTimeUs =
        autil::TimeUtility::currentTimeInMicroSeconds();
    int64_t forwardStateLockAcquiredTimeUs = 0;
    {
        std::lock_guard<std::mutex> lock(_forwardState->mutex);
        forwardStateLockAcquiredTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
        handler = _forwardState->handler;
        streamResponse = _forwardState->streamResponse;
        pendingRequests.reserve(batch->requestIds.size());
        for (auto requestId : batch->requestIds) {
            auto it = _forwardState->requests.find(requestId);
            if (it == _forwardState->requests.end()) {
                handler = {};
                break;
            }
            const auto &request = it->second->request;
            if (scheduleBatchId == 0) {
                scheduleBatchId = request.decision.batchId;
            } else if (scheduleBatchId != request.decision.batchId) {
                handler = {};
                break;
            }
            pendingRequests.emplace_back(it->second);
        }
        if (handler) {
            ++_forwardState->activeHandlerCount;
            handlerActive = true;
        }
    }
    const auto payloadBuildBeginTimeUs = tracing ? autil::TimeUtility::currentTime() : 0;
    if (handler) {
        forwarded.reserve(pendingRequests.size());
        for (const auto &pending : pendingRequests) {
            const auto &request = pending->request;
            NaviSchedForwardedRequest value;
            value.requestId = request.request.requestId;
            value.localBatchId = batch->batchId;
            value.scheduleBatchSize = batch->expectedBatchSize;
            value.localBatchSize = static_cast<int64_t>(batch->requestIds.size());
            value.pendingCacheTracked = batch->pendingCacheTracked;
            value.payload = request.request.forwardPayload;
            value.blockHashes = request.request.blockHashes;
            value.expectedBatchSize = batch->expectedBatchSize;
            value.timedOut = batch->timedOut;
            value.scheduleTimeUs = request.decision.scheduleTimeUs;
            value.scheduleLatencyUs = request.decision.scheduleLatencyUs;
            value.agentArrivalTimeUs = pending->arrivalTimeUs;
            value.batchReadyTimeUs = batch->readyTimeUs;
            value.callbackDispatchTimeUs = batch->callbackDispatchTimeUs;
            value.readyAcceptedCallbackTimeUs = batch->readyAcceptedCallbackTimeUs;
            value.forwardHandlerStartTimeUs = forwardHandlerStartTimeUs;
            value.predictedCacheHitTokenCount =
                request.decision.predictedCacheHitTokenCount;
            value.predictedQueueLatencyMs = request.decision.predictedQueueLatencyMs;
            value.predictedExecutionLatencyMs =
                request.decision.predictedExecutionLatencyMs;
            forwarded.emplace_back(std::move(value));
        }
    }
    // RPC arrivals do not preserve the Master batch's iteration order.
    // Match the canonical order used by exact cost geometry and reservations;
    // accepted callbacks and their arrival timestamps remain untouched.
    std::sort(forwarded.begin(), forwarded.end(),
              [](const NaviSchedForwardedRequest &left,
                 const NaviSchedForwardedRequest &right) {
                  return left.requestId < right.requestId;
              });
    const auto forwardBatchBuildDoneTimeUs =
        autil::TimeUtility::currentTimeInMicroSeconds();
    if (tracing) {
        AUTIL_LOG(
            INFO,
            "[navi-request-stage] stage=agent_forward_batch_build "
            "schedule_batch_id=%ld local_batch_id=%ld request_count=%zu "
            "event_time_us=%ld state_lock_wait_us=%ld metadata_lock_and_unlock_us=%ld "
            "payload_build_us=%ld handler_available=%d",
            batch->scheduleBatchId, batch->batchId, forwarded.size(),
            forwardBatchBuildDoneTimeUs,
            forwardStateLockAcquiredTimeUs - forwardStateLockBeginTimeUs,
            payloadBuildBeginTimeUs - forwardStateLockAcquiredTimeUs,
            forwardBatchBuildDoneTimeUs - payloadBuildBeginTimeUs, bool(handler));
    }
    for (auto &request : forwarded) {
        request.forwardStateLockWaitUs =
            forwardStateLockAcquiredTimeUs - forwardStateLockBeginTimeUs;
        request.forwardBatchBuildTimeUs =
            forwardBatchBuildDoneTimeUs - forwardStateLockAcquiredTimeUs;
    }
    bool accepted = false;
    if (handler) {
        try {
            accepted = handler(forwarded, scheduleBatchId, batch->finishReporter);
        } catch (...) {
            accepted = false;
        }
    }
    finishHandler();
    if (!accepted && batch->finishReporter && batch->finishReporter->_state) {
        batch->finishReporter->_state->finishAbnormally();
        batch->finishReporter.reset();
    }
    std::vector<std::pair<uint64_t, navi::AsyncPipePtr>> completionPipes;
    {
        std::lock_guard<std::mutex> lock(_forwardState->mutex);
        for (auto requestId : batch->requestIds) {
            auto it = _forwardState->requests.find(requestId);
            if (it == _forwardState->requests.end()) {
                continue;
            }
            completionPipes.emplace_back(it->first, it->second->completionPipe);
            _forwardState->requests.erase(it);
        }
    }
    for (const auto &[requestId, pipe] : completionPipes) {
        if (!pipe) {
            continue;
        }
        if (accepted) {
            if (!_agentResponseR->acknowledge(requestId, NaviSchedRequestTiming(),
                                              !streamResponse)) {
                pipe->setEof();
            }
        } else if (!_agentResponseR->fail(requestId, "forward handler rejected batch")) {
            pipe->setEof();
        }
    }
}

bool NaviSchedAgentImpl::publishForwardResponse(uint64_t requestId,
                                                std::vector<std::string> frames,
                                                bool terminal) const
{
    return _agentResponseR &&
           _agentResponseR->publish(requestId, std::move(frames), terminal);
}

bool NaviSchedAgentImpl::failForwardResponse(uint64_t requestId,
                                             const std::string &errorMessage) const
{
    return _agentResponseR && _agentResponseR->fail(requestId, errorMessage);
}

void NaviSchedAgentImpl::setForwardCancelHandler(
    const std::function<void(uint64_t)> &handler) const
{
    if (_agentResponseR) {
        _agentResponseR->setCancelHandler(handler);
    }
}

bool NaviSchedAgentImpl::recordObservedBatch(
    const std::vector<NaviSchedRequestStats> &requests, double latencyMs,
    int64_t eventTimeUs,
    const std::vector<NaviSchedBatchFinishReporterPtr> &finishReporters) const
{
    std::unordered_set<const NaviSchedBatchFinishReporter::State *> states;
    bool allFinished = true;
    for (const auto &reporter : finishReporters) {
        if (!reporter || !reporter->_state) {
            allFinished = false;
            continue;
        }
        states.emplace(reporter->_state.get());
    }
    for (const auto &reporter : finishReporters) {
        if (reporter && reporter->_state && states.erase(reporter->_state.get()) > 0 &&
            !reporter->_state->finishObserved()) {
            allFinished = false;
        }
    }
    if (!allFinished) {
        return false;
    }
    return _agentLatencyInputR &&
           _agentLatencyInputR->submitObservedBatchFinished(
               requests, latencyMs, getInflightRequestCount(), eventTimeUs);
}

bool NaviSchedAgentImpl::reportObservedExecutionLoad(int32_t queuedBatchRequests,
                                                     int32_t runningRequests,
                                                     int64_t eventTimeUs) const
{
    return _agentLatencyInputR &&
           _agentLatencyInputR->submitExecutionLoad(0, queuedBatchRequests,
                                                    runningRequests, eventTimeUs);
}

bool NaviSchedAgentImpl::reportPeerCacheReadAvailable(bool available) const
{
    return _agentReportR && _agentReportR->reportPeerCacheReadAvailable(available);
}

bool NaviSchedAgentImpl::reportObservedWorkload(
    const NaviSchedWorkloadSnapshot &snapshot,
    const std::vector<int64_t> &coveredNativeBatchIds,
    const std::vector<uint64_t> *waitingRequestIds) const
{
    if (waitingRequestIds) {
        if (snapshot.waitingRequestCount < 0 ||
            waitingRequestIds->size() > static_cast<size_t>(snapshot.waitingRequestCount))
            return false;
        std::unordered_set<uint64_t> unique;
        for (auto requestId : *waitingRequestIds) {
            if (requestId == 0 || !unique.insert(requestId).second) return false;
        }
    }
    const auto revision = _agentReportR ? _agentReportR->reportWorkloadUpdated(
                                              snapshot, coveredNativeBatchIds)
                                        : 0;
    if (revision > 0 && _agentBatchR) {
        _agentBatchR->updateBackendWaitingRequests(snapshot.waitingRequestCount,
                                                   snapshot.runningRequestCount,
                                                   waitingRequestIds);
    }
    return revision > 0;
}

bool NaviSchedAgentImpl::beginObservedBackendExecution(int64_t eventTimeUs, int32_t requestCount) const
{
    return _agentBatchR && _agentBatchR->beginBackendExecution(eventTimeUs, requestCount);
}

bool NaviSchedAgentImpl::endObservedBackendExecution(int64_t eventTimeUs) const
{
    return _agentBatchR && _agentBatchR->endBackendExecution(eventTimeUs);
}

bool NaviSchedAgentImpl::reportObservedExecutionStep(double elapsedMs,
                                                     int64_t prefillTokens,
                                                     int64_t decodeTokens) const
{
    if (!_agentBatchR || !std::isfinite(elapsedMs) || elapsedMs < 0.0 ||
        elapsedMs > std::numeric_limits<int64_t>::max() / 1000.0 || prefillTokens < 0 ||
        decodeTokens < 0) {
        return false;
    }
    _agentBatchR->recordBackendExecutionStep(
        {static_cast<int64_t>(std::llround(elapsedMs * 1000.0)),
         prefillTokens == 0 && decodeTokens > 0});
    return true;
}

bool NaviSchedAgentImpl::reportObservedCoreStep(double elapsedMs, bool futureWait) const
{
    if (!_agentBatchR || !std::isfinite(elapsedMs) || elapsedMs < 0.0 ||
        elapsedMs > std::numeric_limits<int64_t>::max() / 1000.0) {
        return false;
    }
    _agentBatchR->recordCoreStep(
        {static_cast<int64_t>(std::llround(elapsedMs * 1000.0)), futureWait});
    return true;
}

void NaviSchedAgentImpl::reportCacheInserted(
    const NaviSchedBlockHashesConstPtr &blockHashes,
    NaviSchedCacheKeyType cacheKeyType) const
{
    if (!_agentReportR) {
        return;
    }
    NaviSchedCacheUpdate update;
    update.cacheKeyType = cacheKeyType;
    update.addedBlockHashes = blockHashes;
    _agentReportR->reportCacheUpdated(update);
}

void NaviSchedAgentImpl::reportCacheRemoved(
    const NaviSchedBlockHashesConstPtr &blockHashes,
    NaviSchedCacheKeyType cacheKeyType) const
{
    if (!_agentReportR) {
        return;
    }
    NaviSchedCacheUpdate update;
    update.cacheKeyType = cacheKeyType;
    update.removedBlockHashes = blockHashes;
    _agentReportR->reportCacheUpdated(update);
}

void NaviSchedAgentImpl::reportCacheUpdated(
    int64_t totalBlocks, int64_t blockSize, int64_t linearStep,
    const NaviSchedBlockHashesConstPtr &addedBlockHashes,
    const NaviSchedBlockHashesConstPtr &removedBlockHashes,
    const NaviSchedBlockHashesConstPtr &addedFullReadyBlockHashes,
    const NaviSchedBlockHashesConstPtr &removedFullReadyBlockHashes,
    const NaviSchedBlockHashesConstPtr &addedTailReadyBlockHashes,
    const NaviSchedBlockHashesConstPtr &removedTailReadyBlockHashes,
    int8_t hybridTailRequired, int64_t eventTimeUs, int64_t rtpCachedBlockCount,
    NaviSchedCacheKeyType cacheKeyType, const NaviSchedCachePolicy *typedPolicy) const
{
    const bool hasStatus = blockSize > 0 && linearStep > 0 && hybridTailRequired >= 0;
    if (typedPolicy &&
        (!typedPolicy->valid(cacheKeyType) || typedPolicy->blockSize != blockSize ||
         typedPolicy->linearStep != linearStep ||
         typedPolicy->hybridTailRequired != (hybridTailRequired != 0))) {
        throw std::invalid_argument("invalid cache policy");
    }
    if (hasStatus && cacheKeyType == NaviSchedCacheKeyType::V6D && !typedPolicy) {
        throw std::invalid_argument("V6D cache status requires an independent policy");
    }
    if (!_agentReportR) {
        return;
    }
    NaviSchedCacheUpdate update;
    update.cacheKeyType = cacheKeyType;
    update.totalBlocks = totalBlocks;
    update.blockSize = blockSize;
    update.linearStep = linearStep;
    update.addedBlockHashes = addedBlockHashes;
    update.removedBlockHashes = removedBlockHashes;
    update.addedFullReadyBlockHashes = addedFullReadyBlockHashes;
    update.removedFullReadyBlockHashes = removedFullReadyBlockHashes;
    update.addedTailReadyBlockHashes = addedTailReadyBlockHashes;
    update.removedTailReadyBlockHashes = removedTailReadyBlockHashes;
    update.hybridTailRequired = hybridTailRequired;
    if (typedPolicy) {
        update.fullAttentionQueryDropTokens = typedPolicy->fullAttentionQueryDropTokens;
        update.mambaReplayStopShift = typedPolicy->mambaReplayStopShift ? 1 : 0;
        update.mambaRetentionInterval = typedPolicy->mambaRetentionInterval;
        update.mambaAlign = typedPolicy->mambaAlign;
    } else if (hasStatus && cacheKeyType == NaviSchedCacheKeyType::BLOCK_POOL) {
        update.fullAttentionQueryDropTokens = _fullAttentionQueryDropTokens;
        update.mambaReplayStopShift = _mambaReplayStopShift ? 1 : 0;
        update.mambaRetentionInterval = _mambaRetentionInterval;
        update.mambaAlign = _mambaAlign;
    }
    update.eventTimeUs = eventTimeUs;
    update.rtpCachedBlockCount = rtpCachedBlockCount;
    _agentReportR->reportCacheUpdated(update);
}

int32_t NaviSchedAgentImpl::getInflightRequestCount() const
{
    return getInflightStats().requestCount;
}

size_t NaviSchedAgentImpl::getCachedBlockCount() const
{
    return _agentReportR ? _agentReportR->getCachedBlockCount() : 0;
}

RequestLatencyTracker::Snapshot NaviSchedAgentImpl::getInflightStats() const
{
    return _inflightState ? _inflightState->snapshot()
                          : RequestLatencyTracker::Snapshot();
}

bool NaviSchedAgentImpl::addInflightRequest(
    uint64_t requestId, double predictedLatencyMs,
    const NaviSchedLatencyPredictionConstPtr &latencyPrediction,
    RequestLatencyTracker::Snapshot &snapshot) const
{
    if (!_inflightState || !_inflightState->add(requestId, predictedLatencyMs,
                                                latencyPrediction, snapshot)) {
        return false;
    }
    if (_agentReportR && !_agentReportR->setInflightStats(snapshot)) {
        AUTIL_LOG(ERROR,
                  "update Agent inflight snapshot after add failed, requestId [%lu], "
                  "revision [%lu], count [%d], latencySumMs [%.6f]",
                  requestId, snapshot.revision, snapshot.requestCount,
                  snapshot.totalPredictedLatencyMs);
        RequestLatencyTracker::Snapshot rollbackSnapshot;
        if (!_inflightState->remove(requestId, rollbackSnapshot)) {
            AUTIL_LOG(ERROR,
                      "rollback Agent inflight add failed, requestId [%lu], "
                      "revision [%lu], count [%d], latencySumMs [%.6f]",
                      requestId, rollbackSnapshot.revision, rollbackSnapshot.requestCount,
                      rollbackSnapshot.totalPredictedLatencyMs);
            snapshot = rollbackSnapshot;
            return false;
        }
        snapshot = rollbackSnapshot;
        if (!_agentReportR->setInflightStats(snapshot)) {
            AUTIL_LOG(ERROR,
                      "publish Agent inflight rollback failed, requestId [%lu], "
                      "revision [%lu], count [%d], latencySumMs [%.6f]",
                      requestId, snapshot.revision, snapshot.requestCount,
                      snapshot.totalPredictedLatencyMs);
        }
        return false;
    }
    return true;
}

RequestLatencyTracker::Snapshot NaviSchedAgentImpl::removeInflightRequests(
    const std::vector<uint64_t> &requestIds, const char *phase) const
{
    return removeTrackedRequests(_inflightState, requestIds, _agentReportR, phase);
}

void NaviSchedAgentImpl::handleAcceptedBatchFailure(
    const AgentBatchR::ExecutionState &state) const
{
    if (!_inflightState || state.batchRequestCount <= 0) {
        return;
    }
    if (_agentReportR) {
        _agentReportR->reportNativeUnbatchedRequestsRemoved(state.requestIds);
        _agentReportR->reportNativeBatchRemoved(state.batchId);
    }
    if (state.requestIds.size() != static_cast<size_t>(state.batchRequestCount)) {
        AUTIL_LOG(ERROR,
                  "accepted batch failure has inconsistent request ownership, "
                  "scheduleBatchId [%ld], localBatchId [%ld], requestCount [%d], "
                  "requestIdCount [%lu]",
                  state.scheduleBatchId, state.batchId, state.batchRequestCount,
                  state.requestIds.size());
    }
    if (state.localBatch.pendingCacheTracked &&
        (!_agentReportR || !reportPendingTerminal(_agentReportR, state, false))) {
        AUTIL_LOG(ERROR,
                  "retire accepted batch pending cache after failure failed, "
                  "scheduleBatchId [%ld], localBatchId [%ld], hasReportResource [%d]",
                  state.scheduleBatchId, state.batchId, static_cast<bool>(_agentReportR));
    }
    const auto inflight =
        removeInflightRequests(state.requestIds, "accepted_batch_failed");
    if (!_agentLatencyInputR) {
        AUTIL_LOG(ERROR,
                  "skip accepted batch failure report without latency input, "
                  "scheduleBatchId [%ld], localBatchId [%ld], removed [%lu]",
                  state.scheduleBatchId, state.batchId, state.requestIds.size());
        return;
    }
    if (!_agentLatencyInputR->submitBatchAborted(
            state.batchRequestCount, state.batchId, inflight.requestCount,
            state.batchWaitingRequests, state.queuedBatchRequests, state.runningRequests,
            state.eventTimeUs, state.localBatch)) {
        AUTIL_LOG(ERROR,
                  "report accepted batch failure failed, scheduleBatchId [%ld], "
                  "localBatchId [%ld], requestCount [%d]",
                  state.scheduleBatchId, state.batchId, state.batchRequestCount);
    }
}

NaviSchedBatchFinishReporterPtr NaviSchedAgentImpl::makeBatchFinishReporter(
    int64_t batchId, const std::vector<uint64_t> &requestIds) const
{
    auto state = std::make_shared<NaviSchedBatchFinishReporter::State>(
        batchId, requestIds, _agentBatchR, _agentLatencyInputR, _agentReportR,
        _inflightState, _metricsReporter);
    return NaviSchedBatchFinishReporterPtr(new NaviSchedBatchFinishReporter(state));
}

void NaviSchedAgentImpl::stop()
{
    stopSnapshot();
    setForwardBatchHandler({});
    _agentResponseR->finishAll("agent stopped");
    _agentRequestR->finish();
    const auto abortedStates = _agentBatchR->stop();
    for (const auto &state : abortedStates) {
        handleAcceptedBatchFailure(state);
    }
    auto inflight = getInflightStats();
    if (inflight.requestCount != 0 || inflight.totalPredictedLatencyMs != 0.0) {
        AUTIL_LOG(ERROR,
                  "force clear leaked Agent inflight requests during stop, "
                  "revision [%lu], count [%d], latencySumMs [%.6f]",
                  inflight.revision, inflight.requestCount,
                  inflight.totalPredictedLatencyMs);
        _inflightState->clear(inflight);
        if (_agentReportR && !_agentReportR->setInflightStats(inflight)) {
            AUTIL_LOG(ERROR,
                      "publish cleared Agent inflight snapshot during stop failed, "
                      "revision [%lu]",
                      inflight.revision);
        }
    }
    if (_agentGraph) {
        _agentGraph->stop();
        _agentGraph.reset();
    }
    if (!_stoppingReported && _agentReportR) {
        _agentReportR->beginStop(0);
        _agentReportR->waitForStopAcks(AGENT_STOP_ACK_TIMEOUT_US);
        _agentReportR->finishAll();
        _stoppingReported = true;
    }
    NaviSchedRole::stop();
}

void NaviSchedAgentImpl::onUpdating()
{
    if (_agentGraph) {
        _agentGraph->stop();
        _agentGraph.reset();
    }
    _agentReportR.reset();
    _agentReportBridgeR->setAgentReportR(nullptr);
}

bool NaviSchedAgentImpl::onUpdated()
{
    const auto &navi = getNavi();
    _agentReportR = _agentReportBridgeR->getAgentReportR();
    if (!_agentReportR) {
        return false;
    }
    if (!_agentReportR->setExecutionUtilizationSource(
            _agentBatchR->getExecutionUtilizationSource(),
            _agentBatchR->getExecutionBusyUtilizationSource())) {
        return false;
    }
    if (!_agentReportR->setInflightTracker(_inflightState)) {
        return false;
    }
    _agentGraph = std::make_shared<AgentGraph>(navi, _agentBatchR, _agentLatencyInputR,
                                               _agentBizName, getConfig().enableVis);
    if (!_agentGraph->start()) {
        _agentGraph.reset();
        return false;
    }
    return true;
}

std::string NaviSchedAgentImpl::makeSnapshot() const
{
    NaviSchedAgentSummary summary;
    static_cast<NaviSchedRoleSummary &>(summary) = *makeSnapshotSummary();
    summary.role = "agent";
    summary.collectorCount = _agentReportR ? _agentReportR->getSubscriberCount() : 0;
    auto serverInfo =
        _agentReportR ? _agentReportR->getServerInfo() : NaviSchedServerInfo();
    summary.serverIp = serverInfo.ip;
    summary.serverPort = serverInfo.port;
    summary.pdSeparated = _pdSeparated;
    const auto execution = _agentBatchR->getExecutionUtilizationSnapshot();
    summary.executionCapacity = execution.executionCapacity;
    summary.executionCapacityDenominator = execution.executionCapacityDenominator;
    summary.gpuReadyQueuedRequests = execution.gpuReadyQueuedRequests;
    summary.backendWaitingRequests = execution.backendWaitingRequests;
    summary.readyBatchQueuedRequests = execution.readyBatchQueuedRequests;
    summary.executionBusyWindow = execution.busyWindow;
    summary.executionWindowBusyUs = execution.windowBusyUs;
    summary.executionWindowElapsedUs = execution.windowElapsedUs;
    summary.executionWindowSampleMonoUs = execution.windowSampleMonoUs;
    summary.executionActiveCount = execution.activeCount;
    summary.executionRunningRequests = execution.runningRequests;
    summary.executionQueuedRequests = execution.queuedRequests;
    summary.executionUtilization = execution.utilization;
    const auto workload = _agentReportR->getWorkloadSnapshot();
    if (workload) {
        summary.workload = *workload;
    }
    const auto nativeWorkload = _agentReportR->getWorkloadOwnershipStats();
    summary.nativeWorkloadBatchCount = nativeWorkload.batchCount;
    summary.nativeWorkloadRequestCount = nativeWorkload.requestCount;
    summary.nativeUnbatchedRequestCount = nativeWorkload.unbatchedRequestCount;
    summary.rawObserverEventTimeUs = nativeWorkload.rawObserverEventTimeUs;
    summary.pendingBatchCount = _agentBatchR->getPendingBatchCount();
    const auto inflight = getInflightStats();
    summary.inflightStatsRevision = inflight.revision;
    summary.inflightRequests = inflight.requestCount;
    summary.inflightPredictedLatencySumMs = inflight.totalPredictedLatencyMs;
    summary.inflightInvalidAddCount = inflight.invalidAddCount;
    summary.inflightDuplicateAddCount = inflight.duplicateAddCount;
    summary.inflightMissingRemoveCount = inflight.missingRemoveCount;
    summary.inflightRecomputeCount = inflight.recomputeCount;
    summary.inflightRecomputedRequestCount = inflight.recomputedRequestCount;
    summary.inflightRecomputeFailureCount = inflight.recomputeFailureCount;
    summary.inflightInvalidStateTransitionCount = inflight.invalidStateTransitionCount;
    const auto cacheState = _agentReportR->getCacheState();
    summary.cachedBlockCount = cacheState.cachedBlockCount;
    summary.cacheRevision = cacheState.revision;
    summary.cacheHashChecksum = cacheState.hashChecksum;
    summary.fullReadyBlockCount = cacheState.fullReadyBlockCount;
    summary.fullReadyHashChecksum = cacheState.fullReadyHashChecksum;
    summary.tailReadyBlockCount = cacheState.tailReadyBlockCount;
    summary.tailReadyHashChecksum = cacheState.tailReadyHashChecksum;
    summary.rtpCachedBlockCount = cacheState.rtpCachedBlockCount;
    const auto cacheHitPathStats = _agentBatchR->getCacheHitPathStats();
    summary.cacheHitPathRequestCount = cacheHitPathStats.requestCount;
    if (cacheHitPathStats.requestCount > 0) {
        const auto divisor = static_cast<double>(cacheHitPathStats.requestCount);
        summary.cacheHitMasterPredictedMean =
            cacheHitPathStats.masterPredictedTokens / divisor;
        summary.cacheHitAgentArrivalMean = cacheHitPathStats.agentArrivalTokens / divisor;
        summary.cacheHitAgentPendingAwareMean =
            cacheHitPathStats.agentPendingAwareTokens / divisor;
        summary.cacheHitRtpActualMean = cacheHitPathStats.rtpActualTokens / divisor;
        summary.cacheHitMasterToAgentBiasMean =
            (cacheHitPathStats.agentArrivalTokens -
             cacheHitPathStats.masterPredictedTokens) /
            divisor;
        summary.cacheHitAgentToRtpBiasMean =
            (cacheHitPathStats.rtpActualTokens - cacheHitPathStats.agentArrivalTokens) /
            divisor;
        summary.cacheHitMasterToAgentPendingAwareBiasMean =
            (cacheHitPathStats.agentPendingAwareTokens -
             cacheHitPathStats.masterPredictedTokens) /
            divisor;
        summary.cacheHitAgentPendingAwareToRtpBiasMean =
            (cacheHitPathStats.rtpActualTokens -
             cacheHitPathStats.agentPendingAwareTokens) /
            divisor;
        summary.cacheHitMasterToRtpBiasMean = (cacheHitPathStats.masterPredictedTokens -
                                               cacheHitPathStats.rtpActualTokens) /
                                              divisor;
    }
    summary.cacheHitAlignedRequestCount = cacheHitPathStats.alignedRequestCount;
    if (cacheHitPathStats.alignedRequestCount > 0) {
        const auto divisor = static_cast<double>(cacheHitPathStats.alignedRequestCount);
        summary.cacheHitAlignedAgentToRtpBiasMean =
            cacheHitPathStats.alignedAgentToRtpErrorTokens / divisor;
        summary.cacheHitAlignedAgentToRtpMae =
            cacheHitPathStats.alignedAgentToRtpAbsoluteErrorTokens / divisor;
    }
    summary.cacheHitUnalignedRequestCount = cacheHitPathStats.unalignedRequestCount;
    if (cacheHitPathStats.unalignedRequestCount > 0) {
        const auto divisor = static_cast<double>(cacheHitPathStats.unalignedRequestCount);
        summary.cacheHitUnalignedAgentToRtpBiasMean =
            cacheHitPathStats.unalignedAgentToRtpErrorTokens / divisor;
        summary.cacheHitUnalignedAgentToRtpMae =
            cacheHitPathStats.unalignedAgentToRtpAbsoluteErrorTokens / divisor;
    }
    summary.hybridReuseDebugRequestCount = cacheHitPathStats.hybridDebugRequestCount;
    if (cacheHitPathStats.hybridDebugRequestCount > 0) {
        const auto divisor =
            static_cast<double>(cacheHitPathStats.hybridDebugRequestCount);
        summary.hybridReuseAgentPredictedBlocksMean =
            cacheHitPathStats.hybridAgentPredictedBlocks / divisor;
        summary.hybridReuseCandidateBlocksMean =
            cacheHitPathStats.hybridCandidateBlocks / divisor;
        summary.hybridReuseFlatMatchedBlocksMean =
            cacheHitPathStats.hybridFlatMatchedBlocks / divisor;
        summary.hybridReuseMinFullBlocksMean =
            cacheHitPathStats.hybridMinFullReuseBlocks / divisor;
        summary.hybridReuseSelectedBlocksMean =
            cacheHitPathStats.hybridSelectedReuseBlocks / divisor;
        summary.hybridReuseAgentToFullOverpredictionBlocksMean =
            cacheHitPathStats.hybridAgentToFullOverpredictionBlocks / divisor;
        summary.hybridReuseTailRollbackBlocksMean =
            cacheHitPathStats.hybridTailRollbackBlocks / divisor;
    }
    summary.hybridReuseInvalidOrderCount = cacheHitPathStats.hybridInvalidOrderCount;
    const auto idleStats = _agentBatchR->getReadyWithoutRunningStats();
    summary.readyWithoutRunningGapCount = idleStats.gapCount;
    summary.readyWithoutRunningTotalMs = idleStats.totalTimeUs / 1000.0;
    summary.readyWithoutRunningMeanMs =
        idleStats.gapCount > 0 ? summary.readyWithoutRunningTotalMs / idleStats.gapCount
                               : 0.0;
    summary.readyWithoutRunningMaxMs = idleStats.maxTimeUs / 1000.0;
    summary.readyWithoutRunningCurrentMs = idleStats.currentTimeUs / 1000.0;
    return summary.format();
}

void NaviSchedAgentImpl::reportMetrics() const
{
    NaviSchedRole::reportMetrics();
    reportMetric("agent.pd_separated", _pdSeparated ? 1 : 0);
    reportMetric("agent.collector_count",
                 _agentReportR ? _agentReportR->getSubscriberCount() : 0);
    reportMetric("agent.pending_batches", _agentBatchR->getPendingBatchCount());
    const auto inflight = getInflightStats();
    reportMetric("agent.inflight_requests", inflight.requestCount);
    reportMetric("agent.inflight_predicted_latency_sum_ms",
                 inflight.totalPredictedLatencyMs);
    reportMetric("agent.inflight_predicted_latency_average_ms",
                 inflight.requestCount > 0
                     ? inflight.totalPredictedLatencyMs / inflight.requestCount
                     : 0.0);
    reportCounterQpsMetric("agent.inflight_invalid_add_qps", inflight.invalidAddCount);
    reportCounterQpsMetric("agent.inflight_duplicate_add_qps",
                           inflight.duplicateAddCount);
    reportCounterQpsMetric("agent.inflight_missing_remove_qps",
                           inflight.missingRemoveCount);
    reportCounterQpsMetric("agent.inflight_recompute_qps", inflight.recomputeCount);
    reportCounterQpsMetric("agent.inflight_recomputed_request_qps",
                           inflight.recomputedRequestCount);
    reportCounterQpsMetric("agent.inflight_recompute_failure_qps",
                           inflight.recomputeFailureCount);
    reportCounterQpsMetric("agent.inflight_invalid_state_transition_qps",
                           inflight.invalidStateTransitionCount);
    reportCounterQpsMetric("agent.batch_timeout_qps",
                           _agentBatchR->getTimeoutBatchCount());
    reportCounterQpsMetric("agent.batch_timeout_missing_request_qps",
                           _agentBatchR->getTimeoutMissingRequestCount());
    reportCounterQpsMetric("agent.batch_late_request_qps",
                           _agentBatchR->getLateRequestCount());
    AgentBatchR::ExecutionState executionState;
    _agentBatchR->getExecutionState(executionState);
    reportMetric("agent.requests_batch_waiting", executionState.batchWaitingRequests);
    reportMetric("agent.requests_queued_for_execution",
                 executionState.queuedBatchRequests);
    reportMetric("agent.requests_running", executionState.runningRequests);
    reportMetric("agent.running_batch_request_count", executionState.batchRequestCount);
    const auto executionUtilization = _agentBatchR->getExecutionUtilizationSnapshot();
    if (executionUtilization.utilization >= 0.0) {
        reportMetric("agent.prefill_effective_execution_utilization_pct",
                     executionUtilization.utilization * 100.0);
        reportMetric("agent.execution_busy_window_pct",
                     executionUtilization.busyWindow * 100.0);
        reportMetric("agent.execution_capacity_ema",
                     executionUtilization.executionCapacity);
        reportMetric("agent.gpu_ready_queued_requests",
                     executionUtilization.gpuReadyQueuedRequests);
    }
    const auto cacheState = _agentReportR->getCacheState();
    const auto pendingCacheState = _agentReportR->getPendingCacheState();
    reportMetric("agent.cache_total_blocks", _cacheTotalBlocks);
    reportMetric("agent.cached_block_count", cacheState.cachedBlockCount);
    reportMetric("agent.cache_occupancy_pct",
                 _cacheTotalBlocks > 0
                     ? 100.0 * cacheState.cachedBlockCount / _cacheTotalBlocks
                     : 0.0);
    reportMetric("agent.pending_cache_batch_count", pendingCacheState.batchCount);
    reportMetric("agent.pending_cache_completed_batch_count",
                 pendingCacheState.completedBatchCount);
    reportMetric("agent.pending_cache_oldest_completed_age_ms",
                 pendingCacheState.oldestCompletedAgeMs);
    reportMetric("agent.pending_cache_unique_block_count",
                 pendingCacheState.uniqueBlockCount);
    reportMetric("agent.pending_cache_unique_block_pct",
                 _cacheTotalBlocks > 0
                     ? 100.0 * pendingCacheState.uniqueBlockCount / _cacheTotalBlocks
                     : 0.0);
    reportMetric("agent.full_ready_block_count", cacheState.fullReadyBlockCount);
    reportMetric("agent.tail_ready_block_count", cacheState.tailReadyBlockCount);
    reportMetric("agent.cache_revision", cacheState.revision);
    if (_agentReportR->hasCacheKeyType(NaviSchedCacheKeyType::V6D)) {
        const auto external = _agentReportR->getCacheState(NaviSchedCacheKeyType::V6D);
        const auto pending =
            _agentReportR->getPendingCacheState(NaviSchedCacheKeyType::V6D);
        reportMetric("agent.v6d_cached_block_count", external.cachedBlockCount);
        reportMetric("agent.v6d_full_ready_block_count", external.fullReadyBlockCount);
        reportMetric("agent.v6d_tail_ready_block_count", external.tailReadyBlockCount);
        reportMetric("agent.v6d_cache_revision", external.revision);
        reportMetric("agent.v6d_pending_cache_batch_count", pending.batchCount);
        reportMetric("agent.v6d_pending_cache_completed_batch_count",
                     pending.completedBatchCount);
        reportMetric("agent.v6d_pending_cache_oldest_completed_age_ms",
                     pending.oldestCompletedAgeMs);
        reportMetric("agent.v6d_pending_cache_unique_block_count",
                     pending.uniqueBlockCount);
        reportMetric("agent.v6d_pending_cache_unique_tail_ready_block_count",
                     pending.uniqueTailReadyBlockCount);
    }
    const bool rtpCacheMirrorAvailable = cacheState.rtpCachedBlockCount >= 0;
    reportMetric("agent.rtp_cache_mirror_available", rtpCacheMirrorAvailable);
    if (rtpCacheMirrorAvailable) {
        reportMetric("agent.rtp_cached_block_count", cacheState.rtpCachedBlockCount);
        reportMetric("agent.rtp_cache_mirror_block_count_delta",
                     static_cast<int64_t>(cacheState.cachedBlockCount) -
                         cacheState.rtpCachedBlockCount);
    }
    const auto latencySnapshot = _agentReportR->getLatencyModelSnapshot();
    if (latencySnapshot && latencySnapshot->valid()) {
        reportMetric("agent.latency_model_formula", latencySnapshot->formula);
        reportMetric("agent.latency_model_binary_version",
                     latencySnapshot->binaryVersion);
        reportMetric("agent.latency_model_revision", latencySnapshot->revision);
        reportMetric("agent.latency_model_parameter_count",
                     latencySnapshot->parameters.size());
        reportMetric("agent.latency_model_trusted",
                     latencySnapshot->overloadPredictionTrusted ? 1 : 0);
        reportMetric("agent.latency_model_warm_started",
                     latencySnapshot->warmStarted ? 1 : 0);
        reportMetric("agent.latency_residual_version", latencySnapshot->residualVersion);
        reportMetric("agent.latency_residual_parameter_count",
                     latencySnapshot->residualParameters.size());
        const auto currentTimeUs = autil::TimeUtility::currentTime();
        const auto residualAgeMs =
            latencySnapshot->residualLastUpdateTimeUs > 0
                ? std::max<int64_t>(
                      0, currentTimeUs - latencySnapshot->residualLastUpdateTimeUs) /
                      1000.0
                : 0.0;
        const auto residualDecay = LatencyResidualModel::decayFactor(
            latencySnapshot->residualLastUpdateTimeUs, currentTimeUs,
            _agentReportR->getSchedulerConfigSnapshot().config.residualRuntimeConfig());
        reportMetric("agent.latency_residual_age_ms", residualAgeMs);
        reportMetric("agent.latency_residual_decay_factor", residualDecay);
        reportMetric(
            "agent.latency_model_zero_point",
            NonLinearPrefillModelR::calculateZeroPoint(
                latencySnapshot->parameters, nullptr, latencySnapshot->binaryVersion));
        for (size_t index = 0; index < latencySnapshot->parameters.size(); ++index) {
            reportMetric("agent.latency_model_parameter_" + std::to_string(index),
                         latencySnapshot->parameters[index]);
        }
        for (size_t index = 0; index < latencySnapshot->residualParameters.size();
             ++index) {
            reportMetric("agent.latency_residual_parameter_" + std::to_string(index),
                         latencySnapshot->residualParameters[index]);
            reportMetric(
                "agent.latency_residual_effective_parameter_" + std::to_string(index),
                latencySnapshot->residualParameters[index] * residualDecay);
        }
    }
    const auto cacheHitPathStats = _agentBatchR->getCacheHitPathStats();
    reportMetric("agent.kv_cache_hit_tokens_total", cacheHitPathStats.rtpActualTokens);
    reportMetric("agent.kv_cache_prompt_tokens_total", cacheHitPathStats.promptTokens);
    const auto cacheTokenReportTimeUs = autil::TimeUtility::monotonicTimeUs();
    reportCounterQpsMetricAt("agent.kv_cache_hit_tokens_qps",
                             cacheHitPathStats.rtpActualTokens, cacheTokenReportTimeUs);
    reportCounterQpsMetricAt("agent.kv_cache_prompt_tokens_qps",
                             cacheHitPathStats.promptTokens, cacheTokenReportTimeUs);
    if (cacheHitPathStats.requestCount < _lastCacheHitPathStats.requestCount) {
        _lastCacheHitPathStats = AgentBatchR::CacheHitPathStats();
    }
    const auto cacheHitRequestDelta =
        cacheHitPathStats.requestCount - _lastCacheHitPathStats.requestCount;
    if (cacheHitRequestDelta > 0) {
        const auto promptTokens =
            cacheHitPathStats.promptTokens - _lastCacheHitPathStats.promptTokens;
        const auto hitTokens =
            cacheHitPathStats.rtpActualTokens - _lastCacheHitPathStats.rtpActualTokens;
        if (promptTokens > 0 && hitTokens >= 0 && hitTokens <= promptTokens) {
            reportSampleMetric("agent.kv_cache_hit_rate_pct",
                               100.0 * hitTokens / promptTokens);
        }
        const auto delta = [](auto current, auto previous) { return current - previous; };
        const auto hybridDebugCount =
            delta(cacheHitPathStats.hybridDebugRequestCount,
                  _lastCacheHitPathStats.hybridDebugRequestCount);
        if (hybridDebugCount > 0) {
            const auto hybridDivisor = static_cast<double>(hybridDebugCount);
            reportSampleMetric("agent.hybrid_reuse_window_agent_predicted_blocks_mean",
                               delta(cacheHitPathStats.hybridAgentPredictedBlocks,
                                     _lastCacheHitPathStats.hybridAgentPredictedBlocks) /
                                   hybridDivisor);
            reportSampleMetric("agent.hybrid_reuse_window_candidate_blocks_mean",
                               delta(cacheHitPathStats.hybridCandidateBlocks,
                                     _lastCacheHitPathStats.hybridCandidateBlocks) /
                                   hybridDivisor);
            reportSampleMetric("agent.hybrid_reuse_window_flat_matched_blocks_mean",
                               delta(cacheHitPathStats.hybridFlatMatchedBlocks,
                                     _lastCacheHitPathStats.hybridFlatMatchedBlocks) /
                                   hybridDivisor);
            reportSampleMetric("agent.hybrid_reuse_window_min_full_blocks_mean",
                               delta(cacheHitPathStats.hybridMinFullReuseBlocks,
                                     _lastCacheHitPathStats.hybridMinFullReuseBlocks) /
                                   hybridDivisor);
            reportSampleMetric("agent.hybrid_reuse_window_selected_blocks_mean",
                               delta(cacheHitPathStats.hybridSelectedReuseBlocks,
                                     _lastCacheHitPathStats.hybridSelectedReuseBlocks) /
                                   hybridDivisor);
            reportSampleMetric(
                "agent.hybrid_reuse_window_agent_to_full_overprediction_blocks_mean",
                delta(cacheHitPathStats.hybridAgentToFullOverpredictionBlocks,
                      _lastCacheHitPathStats.hybridAgentToFullOverpredictionBlocks) /
                    hybridDivisor);
            reportSampleMetric("agent.hybrid_reuse_window_tail_rollback_blocks_mean",
                               delta(cacheHitPathStats.hybridTailRollbackBlocks,
                                     _lastCacheHitPathStats.hybridTailRollbackBlocks) /
                                   hybridDivisor);
        }
        _lastCacheHitPathStats = cacheHitPathStats;
    }
    const auto formationStats = _agentBatchR->takeBatchFormationWindowStats();
    if (formationStats.batchCount > 0) {
        reportSampleMetric("agent.batch_formation_window_mean_ms",
                           formationStats.totalTimeUs / 1000.0 /
                               static_cast<double>(formationStats.batchCount));
        reportSampleMetric("agent.batch_formation_window_max_ms",
                           formationStats.maxTimeUs / 1000.0);
        reportSampleMetric("agent.batch_formation_window_batch_count",
                           formationStats.batchCount);
    }
    const auto backendBatchStats = _agentBatchR->takeBackendBatchExecutionWindowStats();
    const auto stepStats = _agentBatchR->takeBackendExecutionStepWindowStats();
    reportSampleMetric("agent.vllm_execution_step_window_count", stepStats.stepCount);
    if (stepStats.stepCount > 0) {
        reportSampleMetric("agent.vllm_execution_step_window_mean_ms",
                           stepStats.totalTimeUs / 1000.0 / stepStats.stepCount);
        reportSampleMetric("agent.vllm_execution_step_window_max_ms",
                           stepStats.maxTimeUs / 1000.0);
    }
    reportSampleMetric("agent.vllm_pure_decode_step_window_count",
                       stepStats.pureDecodeStepCount);
    if (stepStats.pureDecodeStepCount > 0) {
        reportSampleMetric(
            "agent.vllm_pure_decode_step_window_mean_ms",
            stepStats.pureDecodeTotalTimeUs / 1000.0 / stepStats.pureDecodeStepCount);
        reportSampleMetric("agent.vllm_pure_decode_step_window_max_ms",
                           stepStats.pureDecodeMaxTimeUs / 1000.0);
    }
    const auto coreStepStats = _agentBatchR->takeCoreStepWindowStats();
    reportSampleMetric("agent.vllm_core_step_window_count", coreStepStats.stepCount);
    if (coreStepStats.stepCount > 0) {
        reportSampleMetric("agent.vllm_core_step_window_mean_ms",
                           coreStepStats.totalTimeUs / 1000.0 / coreStepStats.stepCount);
        reportSampleMetric("agent.vllm_core_step_window_max_ms",
                           coreStepStats.maxTimeUs / 1000.0);
    }
    reportSampleMetric("agent.vllm_future_wait_window_count",
                       coreStepStats.futureWaitCount);
    if (coreStepStats.futureWaitCount > 0) {
        reportSampleMetric(
            "agent.vllm_future_wait_window_mean_ms",
            coreStepStats.futureWaitTotalTimeUs / 1000.0 / coreStepStats.futureWaitCount);
        reportSampleMetric("agent.vllm_future_wait_window_max_ms",
                           coreStepStats.futureWaitMaxTimeUs / 1000.0);
    }
    if (backendBatchStats.batchCount > 0) {
        const auto batchCount = static_cast<double>(backendBatchStats.batchCount);
        const auto wallTimeUs = static_cast<double>(backendBatchStats.wallTimeUs);
        reportSampleMetric("agent.vllm_batch_wall_window_mean_ms",
                           wallTimeUs / 1000.0 / batchCount);
        reportSampleMetric("agent.vllm_batch_wall_window_max_ms",
                           backendBatchStats.maxWallTimeUs / 1000.0);
        reportSampleMetric("agent.vllm_batch_execute_model_sum_window_mean_ms",
                           backendBatchStats.executeModelTimeUs / 1000.0 / batchCount);
        reportSampleMetric("agent.vllm_batch_inter_step_gap_window_mean_ms",
                           backendBatchStats.interStepGapTimeUs / 1000.0 / batchCount);
        reportSampleMetric("agent.vllm_batch_inter_step_gap_window_max_ms",
                           backendBatchStats.maxInterStepGapTimeUs / 1000.0);
        reportSampleMetric(
            "agent.vllm_batch_inter_step_gap_ratio_pct",
            wallTimeUs > 0 ? static_cast<double>(backendBatchStats.interStepGapTimeUs) /
                                 wallTimeUs * 100.0
                           : 0.0);
        reportSampleMetric("agent.vllm_batch_execution_step_count_window_mean",
                           backendBatchStats.executionStepCount / batchCount);
        reportSampleMetric("agent.vllm_batch_prefill_tokens_window_mean",
                           backendBatchStats.prefillTokenCount / batchCount);
        reportSampleMetric("agent.vllm_batch_decode_request_steps_window_mean",
                           backendBatchStats.decodeRequestStepCount / batchCount);
        reportSampleMetric("agent.vllm_batch_decode_tokens_window_mean",
                           backendBatchStats.decodeTokenCount / batchCount);
        reportSampleMetric("agent.vllm_batch_observation_window_batch_count",
                           backendBatchStats.batchCount);
        if (backendBatchStats.admissionToFirstExecutionSampleCount > 0) {
            const auto waitSampleCount = static_cast<double>(
                backendBatchStats.admissionToFirstExecutionSampleCount);
            reportSampleMetric("agent.vllm_admission_to_first_execution_window_mean_ms",
                               backendBatchStats.admissionToFirstExecutionTimeUs /
                                   1000.0 / waitSampleCount);
            reportSampleMetric(
                "agent.vllm_admission_to_first_execution_window_max_ms",
                backendBatchStats.maxAdmissionToFirstExecutionTimeUs / 1000.0);
            reportSampleMetric(
                "agent.vllm_admission_to_first_execution_window_sample_count",
                backendBatchStats.admissionToFirstExecutionSampleCount);
        }
    }
    reportMetric("agent.latency_input_pending",
                 _agentLatencyInputR->getPendingInputCount());
    const auto inputStats = _agentLatencyInputR->getStats();
    const auto batchStartedLagStats =
        _agentLatencyInputR->takeBatchStartedProcessingLagWindowStats();
    if (batchStartedLagStats.sampleCount > 0) {
        reportSampleMetric("agent.batch_started_processing_lag_window_mean_ms",
                           batchStartedLagStats.totalTimeUs / 1000.0 /
                               static_cast<double>(batchStartedLagStats.sampleCount));
        reportSampleMetric("agent.batch_started_processing_lag_window_max_ms",
                           batchStartedLagStats.maxTimeUs / 1000.0);
    }
    reportMetric("agent.latency_input_submitted_total", inputStats.submittedInputCount);
    reportMetric("agent.latency_input_rejected_total", inputStats.rejectedInputCount);
    reportMetric("agent.latency_input_processed_total", inputStats.processedInputCount);
    reportMetric("agent.latency_finished_requests_submitted_total",
                 inputStats.submittedFinishedRequestCount);
    reportMetric("agent.latency_finished_requests_processed_total",
                 inputStats.processedFinishedRequestCount);
    reportMetric("agent.latency_aborted_requests_submitted_total",
                 inputStats.submittedAbortedRequestCount);
    reportMetric("agent.latency_aborted_requests_processed_total",
                 inputStats.processedAbortedRequestCount);
    if (_agentReportR) {
        const auto reportStats = _agentReportR->getStats();
        reportMetric("agent.report_events_published_total",
                     reportStats.publishedEventCount);
        reportMetric("agent.report_requests_received_total",
                     reportStats.publishedRequestReceivedCount);
        reportMetric("agent.report_requests_finished_normal_total",
                     reportStats.publishedNormalFinishedRequestCount);
        reportMetric("agent.report_requests_finished_abnormal_total",
                     reportStats.publishedAbnormalFinishedRequestCount);
        const auto finishedBatchDelta = reportCounterQpsMetric(
            "agent.executed_batch_qps", reportStats.publishedFinishedBatchCount);
        reportCounterQpsMetric("agent.abnormal_finished_batch_qps",
                               reportStats.publishedAbnormalFinishedBatchCount);
        const auto receivedRequestDelta = reportCounterQpsMetric(
            "agent.request_received_qps", reportStats.publishedRequestReceivedCount);
        const auto promptTokenDelta = reportCounterQpsMetric(
            "agent.prompt_tokens_qps", reportStats.publishedPromptTokenCount);
        if (receivedRequestDelta > 0) {
            reportSampleMetric(
                "agent.request_prompt_tokens_mean",
                static_cast<double>(promptTokenDelta) / receivedRequestDelta);
        }
        const auto normalRequestDelta =
            reportCounterQpsMetric("agent.request_finished_qps",
                                   reportStats.publishedNormalFinishedRequestCount);
        const auto abnormalRequestDelta =
            reportCounterQpsMetric("agent.request_abnormal_finished_qps",
                                   reportStats.publishedAbnormalFinishedRequestCount);
        if (finishedBatchDelta > 0) {
            reportSampleMetric(
                "agent.executed_batch_request_count_mean",
                static_cast<double>(normalRequestDelta + abnormalRequestDelta) /
                    finishedBatchDelta);
        }
        reportMetric("agent.report_messages_total", reportStats.deliveredMessageCount);
        reportMetric("agent.report_combined_message_savings_total",
                     reportStats.combinedMessageSavings);
        reportCounterQpsMetric("agent.report_message_qps",
                               reportStats.deliveredMessageCount);
        reportMetric("agent.report_delivery_success_total",
                     reportStats.deliverySuccessCount);
        reportMetric("agent.report_delivery_failure_total",
                     reportStats.deliveryFailureCount);
        reportMetric("agent.report_no_subscriber_events_total",
                     reportStats.noSubscriberEventCount);
        reportMetric("agent.workload_update_published_total",
                     reportStats.publishedWorkloadUpdateCount);
        reportMetric("agent.latency_validation_suppressed_batch_total",
                     reportStats.suppressedLatencyValidationBatchCount);
        reportCounterQpsMetric("agent.workload_update_qps",
                               reportStats.publishedWorkloadUpdateCount);
        const auto workload = _agentReportR->getWorkloadSnapshot();
        const auto workloadPublishStats = _agentReportR->takeWorkloadPublishWindowStats();
        if (workloadPublishStats.sampleCount > 0) {
            reportSampleMetric("agent.workload_publish_window_count",
                               workloadPublishStats.sampleCount);
            reportSampleMetric("agent.workload_publish_window_mean_ms",
                               workloadPublishStats.totalTimeUs / 1000.0 /
                                   workloadPublishStats.sampleCount);
            reportSampleMetric("agent.workload_publish_window_max_ms",
                               workloadPublishStats.maxTimeUs / 1000.0);
        }
        if (workload) {
            const auto nowTimeUs = autil::TimeUtility::currentTime();
            reportMetric("agent.workload_revision", workload->revision);
            reportMetric("agent.workload_batch_count", workload->batchCount);
            reportMetric("agent.workload_age_ms",
                         (nowTimeUs - workload->eventTimeUs) / 1000.0);
            reportMetric("agent.workload_request_count", workload->requestCount);
            reportMetric("agent.workload_remaining_prefill_request_count",
                         workload->remainingPrefillRequestCount);
            reportMetric("agent.workload_running_request_count",
                         workload->runningRequestCount);
            reportMetric("agent.workload_running_elapsed_time_ms",
                         workload->runningElapsedTimeMs);
            reportMetric("agent.workload_running_decode_request_count",
                         workload->runningDecodeRequestCount);
            reportMetric("agent.workload_running_decode_generated_tokens",
                         workload->runningDecodeGeneratedTokenCount);
            reportMetric("agent.workload_running_decode_remaining_token_budget",
                         workload->runningDecodeRemainingTokenBudget);
            reportMetric("agent.workload_waiting_request_count",
                         workload->waitingRequestCount);
            reportMetric("agent.workload_remaining_prefill_context_tokens",
                         workload->remainingPrefillContextTokenCount);
            reportMetric("agent.workload_remaining_prefill_tokens",
                         workload->remainingPrefillTokenCount);
            reportMetric("agent.workload_remaining_prefill_token_square_sum",
                         workload->remainingPrefillTokenSquareSum);
            reportMetric("agent.workload_context_remaining_prefill_product_sum",
                         workload->contextRemainingPrefillTokenProductSum);
            reportMetric("agent.workload_prefill_capacity_split",
                         workload->prefillCapacitySplit ? 1 : 0);
            reportMetric("agent.workload_max_running_request_count",
                         workload->maxRunningRequestCount);
            reportMetric("agent.workload_running_capacity_slot_count",
                         workload->runningCapacitySlotCount);
            reportMetric("agent.workload_running_capacity_prefill_request_count",
                         workload->runningCapacityPrefillRequestCount);
            reportMetric("agent.workload_running_capacity_prefill_context_tokens",
                         workload->runningCapacityPrefillContextTokenCount);
            reportMetric("agent.workload_running_capacity_prefill_tokens",
                         workload->runningCapacityPrefillTokenCount);
            reportMetric("agent.workload_running_capacity_prefill_token_square_sum",
                         workload->runningCapacityPrefillTokenSquareSum);
            reportMetric("agent.workload_running_capacity_prefill_product_sum",
                         workload->runningCapacityContextRemainingPrefillTokenProductSum);
            reportMetric("agent.workload_waiting_capacity_prefill_request_count",
                         workload->waitingCapacityPrefillRequestCount);
            reportMetric("agent.workload_waiting_capacity_prefill_context_tokens",
                         workload->waitingCapacityPrefillContextTokenCount);
            reportMetric("agent.workload_waiting_capacity_prefill_tokens",
                         workload->waitingCapacityPrefillTokenCount);
            reportMetric("agent.workload_waiting_capacity_prefill_token_square_sum",
                         workload->waitingCapacityPrefillTokenSquareSum);
            reportMetric("agent.workload_waiting_capacity_prefill_product_sum",
                         workload->waitingCapacityContextRemainingPrefillTokenProductSum);
            reportMetric("agent.workload_remaining_total_context_tokens",
                         workload->remainingTotalContextTokenCount);
            reportMetric("agent.workload_remaining_total_tokens",
                         workload->remainingTotalTokenCount);
            reportMetric("agent.workload_remaining_total_token_square_sum",
                         workload->remainingTotalTokenSquareSum);
            reportMetric("agent.workload_context_remaining_total_product_sum",
                         workload->contextRemainingTotalTokenProductSum);
            reportMetric("agent.workload_total_prompt_tokens",
                         workload->totalPromptTokenCount);
            reportMetric("agent.workload_inflight_scheduled_tokens",
                         workload->inFlightScheduledTokenCount);
        }
        {
            const auto nowTimeUs = autil::TimeUtility::currentTime();
            const auto executionBusyTimeUs =
                _agentBatchR->getExecutionBusyTimeUs(nowTimeUs);
            if (_lastExecutionBusyReportTimeUs > 0 &&
                nowTimeUs > _lastExecutionBusyReportTimeUs) {
                const auto busyDeltaUs =
                    executionBusyTimeUs >= _lastExecutionBusyTimeUs
                        ? executionBusyTimeUs - _lastExecutionBusyTimeUs
                        : executionBusyTimeUs;
                const auto elapsedUs = nowTimeUs - _lastExecutionBusyReportTimeUs;
                const auto busyFraction =
                    std::min(1.0, static_cast<double>(busyDeltaUs) /
                                      static_cast<double>(elapsedUs));
                reportMetric("agent.prefill_card_time_utilization_pct",
                             busyFraction * 100.0);
            }
            _lastExecutionBusyTimeUs = executionBusyTimeUs;
            _lastExecutionBusyReportTimeUs = nowTimeUs;
        }
        {
            // Include idle nodes in both execution utilization metrics.
            const auto backendNowTimeUs = autil::TimeUtility::monotonicTimeUs();
            const auto backendExecutionBusyTimeUs =
                _agentBatchR->getBackendExecutionBusyTimeUs(backendNowTimeUs);
            if (_lastBackendExecutionBusyReportTimeUs > 0 &&
                backendNowTimeUs > _lastBackendExecutionBusyReportTimeUs) {
                const auto busyDeltaUs =
                    backendExecutionBusyTimeUs >= _lastBackendExecutionBusyTimeUs
                        ? backendExecutionBusyTimeUs - _lastBackendExecutionBusyTimeUs
                        : backendExecutionBusyTimeUs;
                const auto elapsedUs =
                    backendNowTimeUs - _lastBackendExecutionBusyReportTimeUs;
                reportMetric("agent.vllm_execution_time_utilization_pct",
                             std::min(100.0, static_cast<double>(busyDeltaUs) /
                                                 static_cast<double>(elapsedUs) * 100.0));
            }
            _lastBackendExecutionBusyTimeUs = backendExecutionBusyTimeUs;
            _lastBackendExecutionBusyReportTimeUs = backendNowTimeUs;
        }
        const auto validationDiagnostics =
            _agentReportR->getLatencyValidationDiagnostics();
        const auto &validation = validationDiagnostics.stats;
        const auto validationEnabled = _agentReportR->latencyValidationEnabled();
        const auto validationNowTimeUs = autil::TimeUtility::currentTime();
        const auto updateAgeMs = [&](int64_t updateTimeUs) {
            return updateTimeUs > 0
                       ? std::max<int64_t>(0, validationNowTimeUs - updateTimeUs) / 1000.0
                       : -1.0;
        };
        if (validation.sampleCount < _lastLatencyValidationStats.sampleCount) {
            _lastLatencyValidationStats = NaviSchedLatencyValidationStats();
        }
        const auto previous = _lastLatencyValidationStats;
        const auto validationWindowSampleCount =
            validation.sampleCount >= previous.sampleCount
                ? validation.sampleCount - previous.sampleCount
                : validation.sampleCount;
        reportMetric("agent.latency_validation_enabled", validationEnabled ? 1 : 0);
        if (validationWindowSampleCount > 0) {
            reportSampleMetric("agent.validation_window_new_sample_count",
                               validationWindowSampleCount);
        }
        reportMetric("agent.validation_last_update_time_s",
                     validationDiagnostics.lastUpdateTimeUs / 1000000.0);
        reportMetric("agent.validation_last_update_age_ms",
                     updateAgeMs(validationDiagnostics.lastUpdateTimeUs));
        if (validation.sampleCount > previous.sampleCount) {
            const auto reportWindowMetric = [this](const std::string &name,
                                                   double value) {
                reportSampleMetric(name, value);
            };
            const auto windowMean = [&](double currentMean, double previousMean) {
                return validationWindowMean(currentMean, validation.sampleCount,
                                            previousMean, previous.sampleCount);
            };
            const auto nonNegativeWindowMean = [&](double currentMean,
                                                   double previousMean) {
                return std::max(0.0, windowMean(currentMean, previousMean));
            };
            reportWindowMetric("agent.validation_window_batch_request_count_mean",
                               nonNegativeWindowMean(validation.batchRequestCountMean,
                                                     previous.batchRequestCountMean));
            reportWindowMetric(
                "agent.validation_window_cache_hit_tokens_predicted_mean",
                nonNegativeWindowMean(validation.predictedCacheHitTokensMean,
                                      previous.predictedCacheHitTokensMean));
            reportWindowMetric("agent.validation_window_cache_hit_tokens_actual_mean",
                               nonNegativeWindowMean(validation.actualCacheHitTokensMean,
                                                     previous.actualCacheHitTokensMean));
            reportWindowMetric(
                "agent.validation_window_queue_predicted_mean_ms",
                nonNegativeWindowMean(validation.masterPredictedQueueMeanMs,
                                      previous.masterPredictedQueueMeanMs));
            reportWindowMetric("agent.validation_window_queue_actual_mean_ms",
                               nonNegativeWindowMean(validation.actualQueueMeanMs,
                                                     previous.actualQueueMeanMs));
            reportWindowMetric("agent.validation_window_queue_backend_mean_ms",
                               nonNegativeWindowMean(validation.backendQueueMeanMs,
                                                     previous.backendQueueMeanMs));
            reportWindowMetric("agent.queue_model_window_raw_prediction_mean_ms",
                               nonNegativeWindowMean(validation.queueRawPredictedMeanMs,
                                                     previous.queueRawPredictedMeanMs));
            reportWindowMetric("agent.queue_model_window_prediction_mean_ms",
                               nonNegativeWindowMean(validation.agentPredictedQueueMeanMs,
                                                     previous.agentPredictedQueueMeanMs));
            reportWindowMetric("agent.queue_model_window_actual_mean_ms",
                               nonNegativeWindowMean(validation.actualQueueMeanMs,
                                                     previous.actualQueueMeanMs));
            reportWindowMetric(
                "agent.queue_model_window_preceding_batch_count_mean",
                nonNegativeWindowMean(validation.queueFeatureBatchCountMean,
                                      previous.queueFeatureBatchCountMean));
            reportWindowMetric(
                "agent.validation_window_execution_master_predicted_mean_ms",
                nonNegativeWindowMean(validation.masterPredictedExecutionMeanMs,
                                      previous.masterPredictedExecutionMeanMs));
            reportWindowMetric(
                "agent.validation_window_execution_agent_predicted_mean_ms",
                nonNegativeWindowMean(validation.agentPredictedExecutionMeanMs,
                                      previous.agentPredictedExecutionMeanMs));
            reportWindowMetric("agent.validation_window_execution_actual_mean_ms",
                               nonNegativeWindowMean(validation.actualExecutionMeanMs,
                                                     previous.actualExecutionMeanMs));
            reportWindowMetric(
                "agent.validation_window_total_predicted_mean_ms",
                nonNegativeWindowMean(validation.masterPredictedTotalMeanMs,
                                      previous.masterPredictedTotalMeanMs));
            reportWindowMetric("agent.validation_window_total_actual_mean_ms",
                               nonNegativeWindowMean(validation.actualTotalMeanMs,
                                                     previous.actualTotalMeanMs));
        }
        for (size_t bucketId = 0; bucketId < validation.executionLatencyBuckets.size();
             ++bucketId) {
            const auto currentCount =
                validation.executionLatencyBuckets[bucketId].sampleCount;
            const auto previousCount =
                bucketId < previous.executionLatencyBuckets.size()
                    ? previous.executionLatencyBuckets[bucketId].sampleCount
                    : 0;
            const auto newSampleCount = currentCount >= previousCount
                                            ? currentCount - previousCount
                                            : currentCount;
            const auto lastUpdateTimeUs =
                bucketId < validationDiagnostics.bucketLastUpdateTimeUs.size()
                    ? validationDiagnostics.bucketLastUpdateTimeUs[bucketId]
                    : 0;
            kmonitor::MetricsTags tags("bucket_id", std::to_string(bucketId));
            reportMetric("agent.validation_execution_bucket_enabled",
                         validationEnabled ? 1 : 0, tags);
            if (newSampleCount > 0) {
                reportSampleMetric("agent.validation_execution_bucket_new_sample_count",
                                   newSampleCount, tags);
            }
            reportMetric("agent.validation_execution_bucket_last_update_time_s",
                         lastUpdateTimeUs / 1000000.0, tags);
            reportMetric("agent.validation_execution_bucket_last_update_age_ms",
                         updateAgeMs(lastUpdateTimeUs), tags);
        }
        _lastLatencyValidationStats = validation;
        reportMetric("agent.queue_model_fit_step", validation.queueModelFitStep);
        for (size_t index = 0; index < validation.queueModelWeights.size(); ++index) {
            reportMetric("agent.queue_model_weight_" + std::to_string(index),
                         validation.queueModelWeights[index]);
        }
    }
    const auto idleStats = _agentBatchR->getReadyWithoutRunningStats();
    reportMetric("agent.ready_without_running_gap_count", idleStats.gapCount);
    reportMetric("agent.ready_without_running_total_ms", idleStats.totalTimeUs / 1000.0);
    reportMetric("agent.ready_without_running_mean_ms",
                 idleStats.gapCount > 0
                     ? idleStats.totalTimeUs / 1000.0 / idleStats.gapCount
                     : 0.0);
    reportMetric("agent.ready_without_running_max_ms", idleStats.maxTimeUs / 1000.0);
    reportMetric("agent.ready_without_running_current_ms",
                 idleStats.currentTimeUs / 1000.0);
}

const std::shared_ptr<navi::Navi> &NaviSchedAgentImpl::getRuntimeNavi() const
{
    return getNavi();
}

const std::string &NaviSchedAgentImpl::getRuntimeBizName() const { return _agentBizName; }

int32_t NaviSchedAgentImpl::getGrpcPort() const { return NaviSchedRole::getGrpcPort(); }

NaviSchedAgent::NaviSchedAgent(const NaviSchedAgentConfig &agentConfig,
                               int32_t serverPort, const std::string &serverIp,
                               int64_t totalBlocks)
    : _impl(std::make_unique<NaviSchedAgentImpl>(agentConfig, serverPort, serverIp,
                                                 totalBlocks))
{
}

NaviSchedAgent::~NaviSchedAgent() = default;

bool NaviSchedAgent::init() { return _impl && _impl->init(); }

bool NaviSchedAgent::addBatchRequest(const NaviSchedBatchRequest &request,
                                     int64_t batchId, int64_t expectedBatchSize,
                                     const NaviSchedReadyBatchCallback &callback)
{
    return _impl && _impl->addBatchRequest(request, batchId, expectedBatchSize, callback);
}

bool NaviSchedAgent::abortPendingBatch(int64_t scheduleBatchId)
{
    return _impl && _impl->abortPendingBatch(scheduleBatchId);
}

bool NaviSchedAgent::setForwardBatchHandler(
    const NaviSchedForwardBatchHandler &handler, bool streamResponse,
    const NaviSchedForwardRequestAcceptedHandler &requestAcceptedHandler)
{
    return _impl &&
           _impl->setForwardBatchHandler(handler, streamResponse, requestAcceptedHandler);
}

bool NaviSchedAgent::publishForwardResponse(uint64_t requestId,
                                            std::vector<std::string> frames,
                                            bool terminal)
{
    return _impl && _impl->publishForwardResponse(requestId, std::move(frames), terminal);
}

bool NaviSchedAgent::failForwardResponse(uint64_t requestId,
                                         const std::string &errorMessage)
{
    return _impl && _impl->failForwardResponse(requestId, errorMessage);
}

void NaviSchedAgent::setForwardCancelHandler(const std::function<void(uint64_t)> &handler)
{
    if (_impl) {
        _impl->setForwardCancelHandler(handler);
    }
}

bool NaviSchedAgent::recordObservedBatch(
    const std::vector<NaviSchedRequestStats> &requests, double latencyMs,
    int64_t eventTimeUs,
    const std::vector<NaviSchedBatchFinishReporterPtr> &finishReporters) const
{
    return _impl &&
           _impl->recordObservedBatch(requests, latencyMs, eventTimeUs, finishReporters);
}

bool NaviSchedAgent::reportObservedExecutionLoad(int32_t queuedBatchRequests,
                                                 int32_t runningRequests,
                                                 int64_t eventTimeUs) const
{
    return _impl && _impl->reportObservedExecutionLoad(queuedBatchRequests,
                                                       runningRequests, eventTimeUs);
}

bool NaviSchedAgent::reportPeerCacheReadAvailable(bool available) const
{
    return _impl && _impl->reportPeerCacheReadAvailable(available);
}

bool NaviSchedAgent::reportObservedWorkload(
    const NaviSchedWorkloadSnapshot &snapshot) const
{
    return _impl && _impl->reportObservedWorkload(snapshot);
}

bool NaviSchedAgent::reportObservedWorkload(
    const NaviSchedWorkloadSnapshot &snapshot,
    const std::vector<int64_t> &coveredNativeBatchIds) const
{
    return _impl && _impl->reportObservedWorkload(snapshot, coveredNativeBatchIds);
}

bool NaviSchedAgent::reportObservedWorkload(
    const NaviSchedWorkloadSnapshot &snapshot,
    const std::vector<int64_t> &coveredNativeBatchIds,
    const std::vector<uint64_t> &waitingRequestIds) const
{
    return _impl && _impl->reportObservedWorkload(snapshot, coveredNativeBatchIds,
                                                  &waitingRequestIds);
}

bool NaviSchedAgent::beginObservedBackendExecution(int64_t eventTimeUs, int32_t requestCount) const
{
    return _impl && _impl->beginObservedBackendExecution(eventTimeUs, requestCount);
}

bool NaviSchedAgent::endObservedBackendExecution(int64_t eventTimeUs) const
{
    return _impl && _impl->endObservedBackendExecution(eventTimeUs);
}

bool NaviSchedAgent::reportObservedExecutionStep(double elapsedMs, int64_t prefillTokens,
                                                 int64_t decodeTokens) const
{
    return _impl &&
           _impl->reportObservedExecutionStep(elapsedMs, prefillTokens, decodeTokens);
}

bool NaviSchedAgent::reportObservedCoreStep(double elapsedMs, bool futureWait) const
{
    return _impl && _impl->reportObservedCoreStep(elapsedMs, futureWait);
}

void NaviSchedAgent::reportCacheInserted(const NaviSchedBlockHashesConstPtr &blockHashes,
                                         NaviSchedCacheKeyType cacheKeyType) const
{
    if (_impl) {
        _impl->reportCacheInserted(blockHashes, cacheKeyType);
    }
}

void NaviSchedAgent::reportCacheRemoved(const NaviSchedBlockHashesConstPtr &blockHashes,
                                        NaviSchedCacheKeyType cacheKeyType) const
{
    if (_impl) {
        _impl->reportCacheRemoved(blockHashes, cacheKeyType);
    }
}

void NaviSchedAgent::reportCacheUpdated(
    int64_t totalBlocks, int64_t blockSize, int64_t linearStep,
    const NaviSchedBlockHashesConstPtr &addedBlockHashes,
    const NaviSchedBlockHashesConstPtr &removedBlockHashes,
    const NaviSchedBlockHashesConstPtr &addedFullReadyBlockHashes,
    const NaviSchedBlockHashesConstPtr &removedFullReadyBlockHashes,
    const NaviSchedBlockHashesConstPtr &addedTailReadyBlockHashes,
    const NaviSchedBlockHashesConstPtr &removedTailReadyBlockHashes,
    int8_t hybridTailRequired, int64_t eventTimeUs, int64_t rtpCachedBlockCount,
    NaviSchedCacheKeyType cacheKeyType, const NaviSchedCachePolicy *typedPolicy) const
{
    if (_impl) {
        _impl->reportCacheUpdated(
            totalBlocks, blockSize, linearStep, addedBlockHashes, removedBlockHashes,
            addedFullReadyBlockHashes, removedFullReadyBlockHashes,
            addedTailReadyBlockHashes, removedTailReadyBlockHashes, hybridTailRequired,
            eventTimeUs, rtpCachedBlockCount, cacheKeyType, typedPolicy);
    }
}

void NaviSchedAgent::stop()
{
    if (_impl) {
        _impl->stop();
    }
}

int32_t NaviSchedAgent::getGrpcPort() const { return _impl ? _impl->getGrpcPort() : -1; }

}
