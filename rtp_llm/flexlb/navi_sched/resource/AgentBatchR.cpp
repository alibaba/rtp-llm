#include "aios/navi_sched/resource/AgentBatchR.h"
#include "aios/navi_sched/data/NaviSchedRequestTrace.h"
#include "aios/navi_sched/data/SummaryUtil.h"
#include "autil/TimeUtility.h"
#include "navi/log/NaviLogger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sstream>
#include <thread>

namespace navi_sched {
namespace {

struct CacheKeyAuditLogRecord {
    uint64_t requestId = 0;
    int64_t scheduleBatchId = 0;
    int64_t localBatchId = 0;
    int64_t expectedBatchSize = 0;
    int64_t arrivalTimeUs = 0;
    int64_t readyTimeUs = 0;
    int64_t startTimeUs = 0;
    int64_t finishTimeUs = 0;
    AgentBatchR::CacheKeyAudit audit;
    int64_t totalTokens = 0;
    int64_t blockSize = 0;
    int64_t masterHitTokens = 0;
    int64_t agentPhysicalHitTokens = 0;
    int64_t agentPendingAwareHitTokens = 0;
    NaviSchedRequestStats actual;
    const char *outcome = "unknown";
};

std::string formatBlockHashes(const NaviSchedBlockHashesConstPtr &blockHashes)
{
    if (!blockHashes) {
        return "";
    }
    std::ostringstream stream;
    for (size_t index = 0; index < blockHashes->size(); ++index) {
        if (index > 0) {
            stream << ',';
        }
        stream << (*blockHashes)[index];
    }
    return stream.str();
}

void logCacheKeyAudit(const navi::NaviObjectLogger &logger,
                      const CacheKeyAuditLogRecord &record)
{
    const auto hashes = formatBlockHashes(record.audit.blockHashes);
    NAVI_LOG_INTERNAL(
        logger, INFO,
        "[navi-cache-key-audit] request_id=%lu schedule_batch_id=%ld "
        "local_batch_id=%ld expected_batch_size=%ld arrival_time_us=%ld "
        "ready_time_us=%ld start_time_us=%ld finish_time_us=%ld "
        "cache_revision=%lu cached_blocks=%zu cache_checksum=%lu "
        "full_ready_blocks=%zu full_ready_checksum=%lu tail_ready_blocks=%zu "
        "tail_ready_checksum=%lu hybrid_tail_required=%d total_tokens=%ld "
        "start_cache_revision=%lu start_cached_blocks=%zu start_cache_checksum=%lu "
        "start_full_ready_blocks=%zu start_full_ready_checksum=%lu "
        "start_tail_ready_blocks=%zu start_tail_ready_checksum=%lu "
        "start_rtp_cached_blocks=%ld "
        "block_size=%ld master_hit_tokens=%ld agent_hit_tokens=%ld "
        "agent_pending_aware_hit_tokens=%ld "
        "rtp_hit_tokens=%ld hybrid_candidate_blocks=%ld "
        "hybrid_flat_matched_blocks=%ld hybrid_min_full_reuse_blocks=%ld "
        "hybrid_selected_reuse_blocks=%ld outcome=%s block_hashes=[%s]",
        record.requestId, record.scheduleBatchId, record.localBatchId,
        record.expectedBatchSize, record.arrivalTimeUs, record.readyTimeUs,
        record.startTimeUs, record.finishTimeUs, record.audit.cacheRevision,
        record.audit.cachedBlockCount, record.audit.cacheHashChecksum,
        record.audit.fullReadyBlockCount, record.audit.fullReadyHashChecksum,
        record.audit.tailReadyBlockCount, record.audit.tailReadyHashChecksum,
        record.audit.hybridTailRequired ? 1 : 0, record.totalTokens,
        record.audit.startCacheRevision, record.audit.startCachedBlockCount,
        record.audit.startCacheHashChecksum, record.audit.startFullReadyBlockCount,
        record.audit.startFullReadyHashChecksum, record.audit.startTailReadyBlockCount,
        record.audit.startTailReadyHashChecksum, record.audit.startRtpCachedBlockCount,
        record.blockSize, record.masterHitTokens, record.agentPhysicalHitTokens,
        record.agentPendingAwareHitTokens, record.actual.cacheHitTokenCount,
        record.actual.hybridCandidateBlockCount,
        record.actual.hybridFlatMatchedBlockCount,
        record.actual.hybridMinFullReuseBlockCount,
        record.actual.hybridSelectedReuseBlockCount, record.outcome, hashes.c_str());
}

class CacheKeyAuditLogRecords : public std::vector<CacheKeyAuditLogRecord> {
public:
    explicit CacheKeyAuditLogRecords(const navi::NaviObjectLogger &logger)
        : _logger(logger)
    {
    }

    ~CacheKeyAuditLogRecords()
    {
        for (const auto &record : *this) {
            logCacheKeyAudit(_logger, record);
        }
    }

private:
    const navi::NaviObjectLogger &_logger;
};

}

const int64_t DEFAULT_BATCH_TIMEOUT_MS = 10;
const int64_t DEFAULT_CLOSED_BATCH_KEEP_MS = 60 * 1000;

const std::string AgentBatchR::RESOURCE_ID = "navi_sched.agent_batch_r";

AgentBatchR::AgentBatchR()
    : AgentBatchR(DEFAULT_BATCH_TIMEOUT_MS, DEFAULT_CLOSED_BATCH_KEEP_MS)
{
}

AgentBatchR::AgentBatchR(int64_t batchTimeoutMs, int64_t closedBatchKeepMs)
    : navi::ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL)
    , _batchTimeoutMs(std::max<int64_t>(1, batchTimeoutMs))
    , _closedBatchKeepMs(std::max<int64_t>(_batchTimeoutMs, closedBatchKeepMs))
    , _executionUtilizationEstimator(autil::TimeUtility::monotonicTimeUs())
{
}

bool AgentBatchR::addRequest(const NaviSchedRequestStats &request, int64_t batchId,
                             int64_t expectedBatchSize,
                             const NaviSchedReadyBatchCallback &callback,
                             const std::function<void()> &acceptedCallback)
{
    return addRequest(request, batchId, expectedBatchSize, callback, acceptedCallback,
                      CacheKeyAudit());
}

bool AgentBatchR::addRequest(
    const NaviSchedRequestStats &request, int64_t batchId, int64_t expectedBatchSize,
    const NaviSchedReadyBatchCallback &callback,
    const std::function<void()> &acceptedCallback, const CacheKeyAudit &cacheKeyAudit,
    const std::function<void(const ExecutionState &)> &acceptedFailureCallback,
    int64_t batchTimeoutMs)
{
    if (!callback || batchId < 0 || expectedBatchSize < 0 ||
        ((0 == batchId) != (0 == expectedBatchSize))) {
        _invalidRequestCount.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    auto nowTimeUs = autil::TimeUtility::currentTime();
    const auto addBeginTimeUs = nowTimeUs;
    CallbackBatchPtr callbackBatch;
    CallbackBatchPtr readyBatch;
    const auto recordAppendFailure = [this](AppendResult result) {
        if (AppendResult::PREDICTION_MISMATCH == result) {
            _predictionMismatchCount.fetch_add(1, std::memory_order_relaxed);
        } else if (AppendResult::DUPLICATE_REQUEST == result) {
            _duplicateRequestCount.fetch_add(1, std::memory_order_relaxed);
        } else {
            _invalidRequestCount.fetch_add(1, std::memory_order_relaxed);
        }
    };
    while (true) {
        CallbackBatchPtr expiredReadyBatch;
        CallbackBatchPtr timedOutBatch;
        {
            autil::ScopedLock lock(_lock);
            if (_stopped || _graphFinishRequested) {
                _invalidRequestCount.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            if (_graphPipeGeneration.load(std::memory_order_relaxed) > 0 && !_graphPipe) {
                _invalidRequestCount.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            if (_graphPipe &&
                navi::EC_NONE != _graphPipe->setDataOverride(nullptr, true, true)) {
                _graphPipe.reset();
                _invalidRequestCount.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            nowTimeUs = autil::TimeUtility::currentTime();
            recycleClosedBatches(nowTimeUs);
            const auto existing = _callbackBatches.find(batchId);
            if (existing != _callbackBatches.end() &&
                existing->second->deadlineTimeUs <= nowTimeUs) {
                auto expiredBatch = existing->second;
                makeCallbackBatchReady(batchId, expiredBatch, true, nowTimeUs);
                timedOutBatch = expiredBatch;
                expiredReadyBatch = takeReadyCallback(expiredBatch);
                if (expiredReadyBatch) {
                    ++_activeCallbackCount;
                }
            } else {
                auto immediate = 0 == batchId || expectedBatchSize <= 1;
                const auto closedIt = _closedBatches.find(batchId);
                if (!immediate && _closedBatches.end() != closedIt &&
                    closedIt->second.aborted) {
                    return false;
                }
                auto late = !immediate && _closedBatches.end() != closedIt;
                if (immediate || late) {
                    callbackBatch =
                        createCallbackBatch(immediate ? 1 : expectedBatchSize, callback,
                                            acceptedFailureCallback);
                    callbackBatch->scheduleBatchId = batchId;
                    callbackBatch->firstRequestTimeUs = nowTimeUs;
                    const auto appendResult =
                        appendRequest(callbackBatch, request, cacheKeyAudit);
                    if (AppendResult::ACCEPTED != appendResult) {
                        recordAppendFailure(appendResult);
                        return false;
                    }
                    callbackBatch->lastRequestTimeUs = nowTimeUs;
                    callbackBatch->batch->timedOut = late;
                    if (late) {
                        _lateRequestCount.fetch_add(1, std::memory_order_relaxed);
                    }
                    publishReadyBatch(callbackBatch);
                } else {
                    auto &batch = _callbackBatches[batchId];
                    const bool created = !batch;
                    if (!batch) {
                        batch = createCallbackBatch(expectedBatchSize, callback,
                                                    acceptedFailureCallback);
                        batch->scheduleBatchId = batchId;
                        batch->firstRequestTimeUs = nowTimeUs;
                        const auto waitMs =
                            batchTimeoutMs > 0 ? batchTimeoutMs : _batchTimeoutMs;
                        batch->deadlineTimeUs = nowTimeUs + waitMs * 1000;
                    } else if (batch->expectedBatchSize != expectedBatchSize) {
                        _expectedSizeMismatchCount.fetch_add(1,
                                                             std::memory_order_relaxed);
                        return false;
                    }
                    auto appendResult = appendRequest(batch, request, cacheKeyAudit);
                    if (AppendResult::ACCEPTED != appendResult) {
                        if (created) {
                            _callbackBatches.erase(batchId);
                        }
                        recordAppendFailure(appendResult);
                        return false;
                    }
                    batch->lastRequestTimeUs = nowTimeUs;
                    callbackBatch = batch;
                    ++_batchWaitingRequestCount;
                    if (batch->batch->requestIds.size() >=
                        static_cast<size_t>(batch->expectedBatchSize)) {
                        makeCallbackBatchReady(batchId, batch, false, nowTimeUs);
                    }
                }
                ++_activeCallbackCount;
                break;
            }
        }
        if (timedOutBatch) {
            logBatchTimeout(timedOutBatch);
        }
        if (expiredReadyBatch) {
            runCallbackBatch(expiredReadyBatch);
            finishCallbackWork();
        }
    }
    std::exception_ptr acceptedException;
    const auto acceptedCallbackBeginTimeUs = autil::TimeUtility::currentTime();
    if (traceRequestBatch(batchId)) {
        NAVI_LOG(INFO,
                 "[navi-request-stage] stage=agent_batch_appended request_id=%lu "
                 "schedule_batch_id=%ld event_time_us=%ld add_lock_and_expire_us=%ld",
                 request.requestId, batchId, acceptedCallbackBeginTimeUs,
                 acceptedCallbackBeginTimeUs - addBeginTimeUs);
    }
    if (acceptedCallback) {
        try {
            acceptedCallback();
        } catch (...) {
            acceptedException = std::current_exception();
        }
    }
    const auto acceptedCallbackTimeUs =
        autil::TimeUtility::currentTime() - acceptedCallbackBeginTimeUs;
    if (traceRequestBatch(batchId)) {
        NAVI_LOG(INFO,
                 "[navi-request-stage] stage=agent_request_accepted request_id=%lu "
                 "schedule_batch_id=%ld event_time_us=%ld accepted_callback_us=%ld",
                 request.requestId, batchId,
                 acceptedCallbackBeginTimeUs + acceptedCallbackTimeUs,
                 acceptedCallbackTimeUs);
    }
    bool finishAcceptedAbort = false;
    int64_t acceptedAbortBatchId = 0;
    bool finishAfterAcceptedFailure = false;
    {
        autil::ScopedLock lock(_lock);
        if (acceptedException) {
            callbackBatch->abortRequested = true;
            if (callbackBatch->batchId > 0) {
                acceptedAbortBatchId = callbackBatch->batchId;
            } else {
                const auto it = _callbackBatches.find(callbackBatch->scheduleBatchId);
                if (_callbackBatches.end() != it && it->second == callbackBatch) {
                    const auto closeExpireTimeUs =
                        autil::TimeUtility::currentTime() + _closedBatchKeepMs * 1000;
                    _closedBatches[callbackBatch->scheduleBatchId] =
                        ClosedBatch{closeExpireTimeUs, true};
                    finishAcceptedAbort = true;
                }
            }
        }
        if (callbackBatch->readyTimeUs > 0) {
            callbackBatch->readyAcceptedCallbackTimeUs = acceptedCallbackTimeUs;
            callbackBatch->batch->readyAcceptedCallbackTimeUs = acceptedCallbackTimeUs;
        }
        readyBatch = markRequestAccepted(callbackBatch);
        if (!readyBatch) {
            if (acceptedException) {
                finishAfterAcceptedFailure = true;
            } else {
                finishCallbackWorkLocked();
            }
        }
    }
    if (acceptedException) {
        ExecutionState state;
        bool aborted = false;
        if (acceptedAbortBatchId > 0) {
            aborted = abortBatch(acceptedAbortBatchId, state);
        } else if (finishAcceptedAbort) {
            aborted =
                abortPendingBatch(callbackBatch->scheduleBatchId, state, callbackBatch);
        }
        if (aborted && acceptedFailureCallback) {
            try {
                acceptedFailureCallback(state);
            } catch (...) {
                NAVI_LOG(ERROR,
                         "accepted failure callback threw, batchId [%ld], "
                         "scheduleBatchId [%ld], requestCount [%d]",
                         state.batchId, state.scheduleBatchId, state.batchRequestCount);
            }
        }
        if (finishAfterAcceptedFailure) {
            finishCallbackWork();
        }
        std::rethrow_exception(acceptedException);
    }
    bool result = true;
    if (readyBatch) {
        result = runCallbackBatch(readyBatch);
        finishCallbackWork();
    }
    return result;
}

bool AgentBatchR::setBatchTimeoutMs(int64_t batchTimeoutMs)
{
    if (batchTimeoutMs <= 0) {
        return false;
    }
    autil::ScopedLock lock(_lock);
    if (_batchTimeoutMs == batchTimeoutMs) {
        return true;
    }
    if (!_callbackBatches.empty()) {
        return false;
    }
    _batchTimeoutMs = batchTimeoutMs;
    _closedBatchKeepMs = std::max(_closedBatchKeepMs, _batchTimeoutMs);
    return true;
}

void AgentBatchR::setRuntimeConfig(int64_t batchTimeoutMs, int64_t closedBatchKeepMs)
{
    autil::ScopedLock lock(_lock);
    _batchTimeoutMs = std::max<int64_t>(1, batchTimeoutMs);
    _closedBatchKeepMs = std::max(_batchTimeoutMs, closedBatchKeepMs);
}

void AgentBatchR::expireBatches()
{
    std::vector<CallbackBatchPtr> readyBatches;
    std::vector<CallbackBatchPtr> timedOutBatches;
    {
        autil::ScopedLock lock(_lock);
        if (_stopped || _graphFinishRequested) {
            return;
        }
        auto nowTimeUs = autil::TimeUtility::currentTime();
        flushExpiredCallbackBatches(nowTimeUs, readyBatches, timedOutBatches);
        recycleClosedBatches(nowTimeUs);
        _activeCallbackCount += readyBatches.size();
    }
    for (const auto &batch : timedOutBatches) {
        logBatchTimeout(batch);
    }
    for (const auto &readyBatch : readyBatches) {
        runCallbackBatch(readyBatch);
        finishCallbackWork();
    }
}

bool AgentBatchR::beginLifecycleCallback()
{
    autil::ScopedLock lock(_lock);
    if (_stopped) {
        return false;
    }
    ++_activeLifecycleCallbackCount;
    return true;
}

void AgentBatchR::endLifecycleCallback()
{
    autil::ScopedLock lock(_lock);
    if (--_activeLifecycleCallbackCount == 0) {
        _lock.broadcast();
    }
}

std::vector<AgentBatchR::ExecutionState> AgentBatchR::stop()
{
    std::vector<ExecutionState> abortedStates;
    std::unordered_map<int64_t, CallbackBatchPtr> callbackBatches;
    std::unordered_map<int64_t, CallbackBatchPtr> readyBatches;
    std::unordered_map<int64_t, CallbackBatchPtr> runningBatches;
    {
        autil::ScopedLock lock(_lock);
        _stopped = true;
        while (_activeCallbackCount > 0 || _activeLifecycleCallbackCount > 0) {
            _lock.wait();
        }
        const auto nowTimeUs = autil::TimeUtility::currentTime();
        finishExecutionBusyTimeLocked(nowTimeUs);
        callbackBatches.swap(_callbackBatches);
        readyBatches.swap(_readyBatches);
        runningBatches.swap(_runningBatches);
        for (const auto &entry : readyBatches)
            finishUtilizationExecutionLocked(*entry.second);
        for (const auto &entry : runningBatches)
            finishUtilizationExecutionLocked(*entry.second);
        _batchWaitingRequestCount = 0;
        _queuedBatchRequestCount = 0;
        _runningRequestCount = 0;
        refreshExecutionUtilizationLocked();
        updateReadyWithoutRunningLocked(nowTimeUs);

        const auto collectAborted = [&](const auto &batches) {
            for (const auto &entry : batches) {
                const auto &batch = entry.second;
                if (!batch || batch->requestIdSet.empty()) {
                    continue;
                }
                ExecutionState state;
                state.batchId = batch->batchId > 0 ? batch->batchId : makeBatchId();
                state.scheduleBatchId = batch->scheduleBatchId;
                state.batchRequestCount =
                    static_cast<int32_t>(batch->requestIdSet.size());
                state.requestIds.assign(batch->requestIdSet.begin(),
                                        batch->requestIdSet.end());
                std::sort(state.requestIds.begin(), state.requestIds.end());
                if (batch->scheduleBatchId > 0) {
                    if (batch->batchId <= 0) {
                        batch->batchId = state.batchId;
                    }
                    state.localBatch = makeLocalBatchInfo(*batch);
                }
                state.eventTimeUs = nowTimeUs;
                abortedStates.emplace_back(state);
                _abnormalBatchFinishCount.fetch_add(1, std::memory_order_relaxed);
            }
        };
        collectAborted(callbackBatches);
        collectAborted(readyBatches);
        collectAborted(runningBatches);
    }
    return abortedStates;
}

bool AgentBatchR::startBatch(int64_t batchId, const std::vector<uint64_t> &requestIds,
                             ExecutionState &state)
{
    return startBatch(batchId, requestIds, 0, state);
}

bool AgentBatchR::startBatch(int64_t batchId, const std::vector<uint64_t> &requestIds,
                             int64_t executionStartTimeUs, ExecutionState &state)
{
    autil::ScopedLock lock(_lock);
    auto it = _readyBatches.find(batchId);
    if (_readyBatches.end() == it ||
        it->second->requestIdSet.size() != requestIds.size()) {
        return false;
    }
    std::unordered_set<uint64_t> requestIdSet;
    for (auto requestId : requestIds) {
        if (!requestIdSet.emplace(requestId).second) {
            return false;
        }
        if (it->second->requestIdSet.end() == it->second->requestIdSet.find(requestId)) {
            return false;
        }
    }
    auto batch = it->second;
    const auto nowTimeUs = autil::TimeUtility::currentTime();
    batch->startTimeUs =
        executionStartTimeUs > 0
            ? std::max(batch->readyTimeUs, std::min(executionStartTimeUs, nowTimeUs))
            : nowTimeUs;
    batch->backendQueueLatencyMs =
        std::max<int64_t>(0, batch->startTimeUs - batch->readyTimeUs) / 1000.0;
    if (_runningBatches.empty()) {
        _executionBusyStartTimeUs = batch->startTimeUs;
    }
    _runningBatches.emplace(batchId, batch);
    beginUtilizationExecutionLocked(*batch);
    _readyBatches.erase(it);
    _queuedBatchRequestCount -= batch->requestIdSet.size();
    _runningRequestCount += static_cast<int32_t>(batch->requestIdSet.size());
    updateReadyWithoutRunningLocked(batch->startTimeUs);
    makeExecutionState(state);
    state.batchId = batchId;
    state.scheduleBatchId = batch->scheduleBatchId;
    state.requestIds = requestIds;
    state.localBatch = makeLocalBatchInfo(*batch);
    state.eventTimeUs = batch->startTimeUs;
    return true;
}

bool AgentBatchR::makePendingCacheReservation(
    int64_t batchId, const NaviSchedCachePolicy &policy,
    NaviSchedPendingCacheReservation &reservation) const
{
    autil::ScopedLock lock(_lock);
    const auto readyIt = _readyBatches.find(batchId);
    const auto runningIt = _runningBatches.find(batchId);
    const auto batch = readyIt != _readyBatches.end()
                           ? readyIt->second
                           : (runningIt != _runningBatches.end() ? runningIt->second
                                                                 : CallbackBatchPtr());
    if (!batch || batch->scheduleBatchId <= 0) {
        return false;
    }
    std::vector<PendingCacheReferencedRequest> requests;
    requests.reserve(batch->cacheHitPathTokens.size());
    for (const auto &entry : batch->cacheHitPathTokens) {
        requests.emplace_back(PendingCacheReferencedRequest{
            entry.second.audit.blockHashes.get(), entry.second.totalTokens,
            entry.second.blockSize});
    }
    if (!PendingCacheUtil::makeReservation(batch->batchId, requests, policy,
                                           reservation)) {
        return false;
    }
    batch->pendingCachePolicy = policy;
    reservation.scheduleBatchId = batch->scheduleBatchId;
    return true;
}

bool AgentBatchR::markPendingCacheTracked(int64_t batchId)
{
    autil::ScopedLock lock(_lock);
    const auto readyIt = _readyBatches.find(batchId);
    const auto runningIt = _runningBatches.find(batchId);
    const auto batch = readyIt != _readyBatches.end()
                           ? readyIt->second
                           : (runningIt != _runningBatches.end() ? runningIt->second
                                                                 : CallbackBatchPtr());
    if (!batch) {
        return false;
    }
    batch->pendingCacheTracked = true;
    if (batch->batch) {
        batch->batch->pendingCacheTracked = true;
    }
    return true;
}

bool AgentBatchR::recordCacheStateAtStart(
    int64_t batchId, uint64_t cacheRevision, size_t cachedBlockCount,
    uint64_t cacheHashChecksum, size_t fullReadyBlockCount,
    uint64_t fullReadyHashChecksum, size_t tailReadyBlockCount,
    uint64_t tailReadyHashChecksum, int64_t rtpCachedBlockCount)
{
    autil::ScopedLock lock(_lock);
    const auto it = _runningBatches.find(batchId);
    if (_runningBatches.end() == it) {
        return false;
    }
    for (auto &entry : it->second->cacheHitPathTokens) {
        auto &audit = entry.second.audit;
        audit.startCacheRevision = cacheRevision;
        audit.startCachedBlockCount = cachedBlockCount;
        audit.startCacheHashChecksum = cacheHashChecksum;
        audit.startFullReadyBlockCount = fullReadyBlockCount;
        audit.startFullReadyHashChecksum = fullReadyHashChecksum;
        audit.startTailReadyBlockCount = tailReadyBlockCount;
        audit.startTailReadyHashChecksum = tailReadyHashChecksum;
        audit.startRtpCachedBlockCount = rtpCachedBlockCount;
    }
    return true;
}

bool AgentBatchR::finishBatch(int64_t batchId,
                              const std::vector<NaviSchedRequestStats> &requests,
                              ExecutionState &state,
                              std::vector<NaviSchedRequestStats> *completedRequests,
                              double nativePreparationMs)
{
    if (!std::isfinite(nativePreparationMs) || nativePreparationMs < 0.0) {
        return false;
    }
    int64_t actualCacheHitTokenCount = 0;
    int32_t cacheHitMismatchRequestCount = 0;
    int64_t cacheHitAbsoluteErrorTokenCount = 0;
    int64_t cacheHitMaxAbsoluteErrorTokenCount = 0;
    std::unordered_map<uint64_t, const NaviSchedRequestStats *> completedRequestMap;
    CacheKeyAuditLogRecords auditRecords(_logger);
    completedRequestMap.reserve(requests.size());
    for (const auto &request : requests) {
        actualCacheHitTokenCount += request.cacheHitTokenCount;
        completedRequestMap.emplace(request.requestId, &request);
    }
    autil::ScopedLock lock(_lock);
    auto it = _runningBatches.find(batchId);
    if (_runningBatches.end() == it ||
        it->second->requestIdSet.size() != requests.size()) {
        return false;
    }
    std::unordered_set<uint64_t> requestIdSet;
    for (const auto &request : requests) {
        if (!requestIdSet.emplace(request.requestId).second) {
            return false;
        }
        if (it->second->requestIdSet.end() ==
            it->second->requestIdSet.find(request.requestId)) {
            return false;
        }
    }
    auto batch = it->second;
    std::vector<CacheHitPathSample> cacheHitPathSamples;
    cacheHitPathSamples.reserve(batch->cacheHitPathTokens.size());
    if (completedRequests) {
        *completedRequests = requests;
    }
    for (const auto &scheduledRequest : batch->requestStats) {
        const auto completedIt = completedRequestMap.find(scheduledRequest.requestId);
        if (completedRequestMap.end() == completedIt) {
            return false;
        }
        if (completedRequests) {
            const auto requestIndex = completedIt->second - requests.data();
            (*completedRequests)[requestIndex].latencyTrainingPhase =
                scheduledRequest.latencyTrainingPhase;
        }
    }
    const auto nowTimeUs = autil::TimeUtility::currentTime();
    auditRecords.reserve(batch->cacheHitPathTokens.size());
    ++_cacheHitPathStats.batchCount;
    for (const auto &entry : batch->cacheHitPathTokens) {
        const auto actualIt = completedRequestMap.find(entry.first);
        if (completedRequestMap.end() == actualIt) {
            continue;
        }
        const auto &request = *actualIt->second;
        const auto masterTokens = entry.second.masterTokens;
        const auto agentPhysicalTokens = entry.second.agentPhysicalTokens;
        const auto agentPendingAwareTokens = entry.second.agentPendingAwareTokens;
        const auto rtpTokens = request.cacheHitTokenCount;
        const bool blockAligned = entry.second.blockSize > 0 &&
                                  entry.second.totalTokens % entry.second.blockSize == 0;
        cacheHitPathSamples.emplace_back(CacheHitPathSample{
            entry.first,
            masterTokens,
            agentPhysicalTokens,
            agentPendingAwareTokens,
            rtpTokens,
            agentPhysicalTokens - masterTokens,
            rtpTokens - agentPhysicalTokens,
            agentPendingAwareTokens - masterTokens,
            rtpTokens - agentPendingAwareTokens,
            rtpTokens - masterTokens,
            blockAligned,
        });
        if (entry.second.audit.arrivalTimeUs > 0) {
            auditRecords.emplace_back(CacheKeyAuditLogRecord{
                entry.first,
                batch->scheduleBatchId,
                batch->batchId,
                batch->expectedBatchSize,
                entry.second.audit.arrivalTimeUs,
                batch->readyTimeUs,
                batch->startTimeUs,
                nowTimeUs,
                entry.second.audit,
                entry.second.totalTokens,
                entry.second.blockSize,
                entry.second.masterTokens,
                entry.second.agentPhysicalTokens,
                entry.second.agentPendingAwareTokens,
                request,
                "finished",
            });
        }
        const auto masterToRtpAbsoluteError = std::abs(rtpTokens - masterTokens);
        if (masterToRtpAbsoluteError > 0) {
            ++cacheHitMismatchRequestCount;
        }
        cacheHitAbsoluteErrorTokenCount += masterToRtpAbsoluteError;
        cacheHitMaxAbsoluteErrorTokenCount =
            std::max(cacheHitMaxAbsoluteErrorTokenCount, masterToRtpAbsoluteError);
        ++_cacheHitPathStats.requestCount;
        _cacheHitPathStats.masterPredictedTokens += masterTokens;
        _cacheHitPathStats.agentArrivalTokens += agentPhysicalTokens;
        _cacheHitPathStats.agentPendingAwareTokens += agentPendingAwareTokens;
        _cacheHitPathStats.rtpActualTokens += rtpTokens;
        _cacheHitPathStats.masterToAgentAbsoluteErrorTokens +=
            static_cast<uint64_t>(std::abs(agentPhysicalTokens - masterTokens));
        _cacheHitPathStats.agentToRtpAbsoluteErrorTokens +=
            static_cast<uint64_t>(std::abs(rtpTokens - agentPhysicalTokens));
        _cacheHitPathStats.masterToAgentPendingAwareAbsoluteErrorTokens +=
            static_cast<uint64_t>(std::abs(agentPendingAwareTokens - masterTokens));
        _cacheHitPathStats.agentPendingAwareToRtpAbsoluteErrorTokens +=
            static_cast<uint64_t>(std::abs(rtpTokens - agentPendingAwareTokens));
        _cacheHitPathStats.masterToRtpAbsoluteErrorTokens +=
            static_cast<uint64_t>(std::abs(rtpTokens - masterTokens));
        const auto agentToRtpError = rtpTokens - agentPhysicalTokens;
        if (blockAligned) {
            ++_cacheHitPathStats.alignedRequestCount;
            _cacheHitPathStats.alignedAgentToRtpErrorTokens += agentToRtpError;
            _cacheHitPathStats.alignedAgentToRtpAbsoluteErrorTokens +=
                static_cast<uint64_t>(std::abs(agentToRtpError));
        } else {
            ++_cacheHitPathStats.unalignedRequestCount;
            _cacheHitPathStats.unalignedAgentToRtpErrorTokens += agentToRtpError;
            _cacheHitPathStats.unalignedAgentToRtpAbsoluteErrorTokens +=
                static_cast<uint64_t>(std::abs(agentToRtpError));
        }
        if (request.blockSize > 0 && request.hybridCandidateBlockCount >= 0 &&
            request.hybridFlatMatchedBlockCount >= 0 &&
            request.hybridMinFullReuseBlockCount >= 0 &&
            request.hybridSelectedReuseBlockCount >= 0) {
            ++_cacheHitPathStats.hybridDebugRequestCount;
            const auto agentPredictedBlocks =
                entry.second.agentPhysicalTokens / request.blockSize;
            _cacheHitPathStats.hybridAgentPredictedBlocks += agentPredictedBlocks;
            _cacheHitPathStats.hybridCandidateBlocks += request.hybridCandidateBlockCount;
            _cacheHitPathStats.hybridFlatMatchedBlocks +=
                request.hybridFlatMatchedBlockCount;
            _cacheHitPathStats.hybridMinFullReuseBlocks +=
                request.hybridMinFullReuseBlockCount;
            _cacheHitPathStats.hybridSelectedReuseBlocks +=
                request.hybridSelectedReuseBlockCount;
            _cacheHitPathStats.hybridAgentToFullOverpredictionBlocks += std::max<int64_t>(
                0, agentPredictedBlocks - request.hybridMinFullReuseBlockCount);
            _cacheHitPathStats.hybridTailRollbackBlocks +=
                std::max<int64_t>(0, request.hybridMinFullReuseBlockCount -
                                         request.hybridSelectedReuseBlockCount);
            if (request.hybridCandidateBlockCount < request.hybridFlatMatchedBlockCount ||
                request.hybridFlatMatchedBlockCount <
                    request.hybridMinFullReuseBlockCount ||
                request.hybridMinFullReuseBlockCount <
                    request.hybridSelectedReuseBlockCount) {
                ++_cacheHitPathStats.hybridInvalidOrderCount;
            }
        }
    }
    auto batchLatencyMs = std::max<int64_t>(0, nowTimeUs - batch->startTimeUs) / 1000.0;
    finishUtilizationExecutionLocked(*batch);
    _runningBatches.erase(it);
    _runningRequestCount = std::max<int32_t>(
        0, _runningRequestCount - static_cast<int32_t>(batch->requestIdSet.size()));
    if (_runningBatches.empty()) {
        finishExecutionBusyTimeLocked(nowTimeUs);
    }
    updateReadyWithoutRunningLocked(nowTimeUs);
    makeExecutionState(state);
    state.batchId = batchId;
    state.scheduleBatchId = batch->scheduleBatchId;
    state.requestIds.reserve(requests.size());
    for (const auto &request : requests) {
        state.requestIds.emplace_back(request.requestId);
    }
    state.localBatch = makeLocalBatchInfo(*batch);
    const auto preparationMs =
        std::min(nativePreparationMs, batch->backendQueueLatencyMs);
    state.batchLatencyMs = batchLatencyMs + preparationMs;
    state.backendQueueLatencyMs = batch->backendQueueLatencyMs - preparationMs;
    state.predictedQueueLatencyMs = batch->predictedQueueLatencyMs;
    state.predictedExecutionLatencyMs = batch->predictedExecutionLatencyMs;
    state.predictedCacheHitTokenCount = batch->predictedCacheHitTokenCount;
    state.actualCacheHitTokenCount = actualCacheHitTokenCount;
    state.cacheHitMismatchRequestCount = cacheHitMismatchRequestCount;
    state.cacheHitAbsoluteErrorTokenCount = cacheHitAbsoluteErrorTokenCount;
    state.cacheHitMaxAbsoluteErrorTokenCount = cacheHitMaxAbsoluteErrorTokenCount;
    state.cacheHitPathSamples = std::move(cacheHitPathSamples);
    state.batchRequestCount = static_cast<int32_t>(requests.size());
    state.latencySampleValid = batch->latencySampleValid;
    state.latencyValidationValid = batch->latencyValidationValid;
    return true;
}

bool AgentBatchR::abortBatchRequests(
    int64_t batchId, const std::vector<uint64_t> &requestIds, ExecutionState &state,
    std::vector<NaviSchedRequestStats> &abortedRequests,
    NaviSchedBatchPtr &remainingBatch,
    NaviSchedPendingCacheReservation &remainingPendingCache)
{
    state = ExecutionState();
    abortedRequests.clear();
    remainingBatch.reset();
    remainingPendingCache = NaviSchedPendingCacheReservation();
    if (batchId <= 0 || requestIds.empty()) {
        return false;
    }

    CacheKeyAuditLogRecords auditRecords(_logger);
    autil::ScopedLock lock(_lock);
    auto readyIt = _readyBatches.find(batchId);
    auto runningIt = _runningBatches.find(batchId);
    auto batch = readyIt != _readyBatches.end()
                     ? readyIt->second
                     : (runningIt != _runningBatches.end() ? runningIt->second
                                                           : CallbackBatchPtr());
    if (!batch || requestIds.size() >= batch->requestIdSet.size()) {
        return false;
    }

    std::unordered_set<uint64_t> abortedIds;
    abortedIds.reserve(requestIds.size());
    for (auto requestId : requestIds) {
        if (requestId == 0 || !abortedIds.emplace(requestId).second ||
            batch->requestIdSet.count(requestId) == 0) {
            return false;
        }
    }

    std::vector<NaviSchedRequestStats> remainingRequests;
    remainingRequests.reserve(batch->requestIdSet.size() - abortedIds.size());
    abortedRequests.reserve(abortedIds.size());
    for (const auto &request : batch->requestStats) {
        if (abortedIds.count(request.requestId) != 0) {
            abortedRequests.emplace_back(request);
        } else if (batch->requestIdSet.count(request.requestId) != 0) {
            remainingRequests.emplace_back(request);
        }
    }
    if (abortedRequests.size() != abortedIds.size() ||
        remainingRequests.size() + abortedRequests.size() != batch->requestIdSet.size()) {
        abortedRequests.clear();
        return false;
    }

    for (auto requestId : abortedIds) {
        if (batch->cacheHitPathTokens.count(requestId) == 0) {
            abortedRequests.clear();
            return false;
        }
    }

    remainingBatch = std::make_shared<NaviSchedBatch>();
    remainingBatch->batchId = batchId;
    remainingBatch->scheduleBatchId = batch->scheduleBatchId;
    remainingBatch->expectedBatchSize = batch->expectedBatchSize;
    remainingBatch->timedOut = batch->timeoutDetectedTimeUs > 0;
    remainingBatch->readyTimeUs = batch->readyTimeUs;
    remainingBatch->readyAcceptedCallbackTimeUs = batch->readyAcceptedCallbackTimeUs;
    remainingBatch->pendingCacheTracked = batch->pendingCacheTracked;
    for (const auto &request : remainingRequests) {
        remainingBatch->requestIds.emplace_back(request.requestId);
    }
    remainingBatch->_requestStats = remainingRequests;

    if (batch->pendingCacheTracked) {
        std::vector<PendingCacheReferencedRequest> referencedRequests;
        size_t referencedBlockCount = 0;
        referencedRequests.reserve(batch->cacheHitPathTokens.size() - abortedIds.size());
        for (const auto &entry : batch->cacheHitPathTokens) {
            if (abortedIds.count(entry.first) != 0) {
                continue;
            }
            if (!entry.second.audit.blockHashes) {
                abortedRequests.clear();
                remainingBatch.reset();
                return false;
            }
            referencedBlockCount += entry.second.audit.blockHashes->size();
            referencedRequests.emplace_back(PendingCacheReferencedRequest{
                entry.second.audit.blockHashes.get(), entry.second.totalTokens,
                entry.second.blockSize});
        }
        if (referencedBlockCount > 0) {
            if (!PendingCacheUtil::makeReservation(batchId, referencedRequests,
                                                   batch->pendingCachePolicy,
                                                   remainingPendingCache)) {
                abortedRequests.clear();
                remainingBatch.reset();
                return false;
            }
            remainingPendingCache.scheduleBatchId = batch->scheduleBatchId;
        }
    }

    const auto nowTimeUs = autil::TimeUtility::currentTime();
    batch->latencyValidationValid = false;
    if (runningIt != _runningBatches.end()) {
        batch->latencySampleValid = false;
    }
    auditRecords.reserve(abortedIds.size());
    for (auto requestId : abortedIds) {
        auto cacheIt = batch->cacheHitPathTokens.find(requestId);
        const auto &cache = cacheIt->second;
        if (cache.audit.arrivalTimeUs > 0) {
            CacheKeyAuditLogRecord record;
            record.requestId = requestId;
            record.scheduleBatchId = batch->scheduleBatchId;
            record.localBatchId = batchId;
            record.expectedBatchSize = batch->expectedBatchSize;
            record.arrivalTimeUs = cache.audit.arrivalTimeUs;
            record.readyTimeUs = batch->readyTimeUs;
            record.startTimeUs = batch->startTimeUs;
            record.finishTimeUs = nowTimeUs;
            record.audit = cache.audit;
            record.totalTokens = cache.totalTokens;
            record.blockSize = cache.blockSize;
            record.masterHitTokens = cache.masterTokens;
            record.agentPhysicalHitTokens = cache.agentPhysicalTokens;
            record.agentPendingAwareHitTokens = cache.agentPendingAwareTokens;
            record.actual.cacheHitTokenCount = -1;
            record.outcome = "aborted_request";
            auditRecords.emplace_back(std::move(record));
        }
        batch->predictedCacheHitTokenCount =
            std::max<int64_t>(0, batch->predictedCacheHitTokenCount - cache.masterTokens);
        batch->cacheHitPathTokens.erase(cacheIt);
        batch->requestIdSet.erase(requestId);
    }
    batch->requestStats = std::move(remainingRequests);

    const auto abortedRequestCount = static_cast<int32_t>(abortedIds.size());
    if (batch->executionBusy) {
        _utilizationRunningRequests -= abortedRequestCount;
    }
    if (readyIt != _readyBatches.end()) {
        _queuedBatchRequestCount =
            std::max<int32_t>(0, _queuedBatchRequestCount - abortedRequestCount);
    } else {
        _runningRequestCount =
            std::max<int32_t>(0, _runningRequestCount - abortedRequestCount);
    }
    updateReadyWithoutRunningLocked(nowTimeUs);
    makeExecutionState(state);
    state.batchId = batchId;
    state.scheduleBatchId = batch->scheduleBatchId;
    state.batchRequestCount = abortedRequestCount;
    state.requestIds.assign(abortedIds.begin(), abortedIds.end());
    std::sort(state.requestIds.begin(), state.requestIds.end());
    state.eventTimeUs = nowTimeUs;
    state.localBatch.scheduleBatchId = batch->scheduleBatchId;
    state.localBatch.localBatchId = batchId;
    state.localBatch.requestCount = state.batchRequestCount;
    state.localBatch.pendingCacheTracked = batch->pendingCacheTracked;
    state.localBatch.requestIds.assign(abortedIds.begin(), abortedIds.end());
    std::sort(state.localBatch.requestIds.begin(), state.localBatch.requestIds.end());
    state.latencySampleValid = batch->latencySampleValid;
    state.latencyValidationValid = batch->latencyValidationValid;
    return true;
}

bool AgentBatchR::abortBatch(int64_t batchId, ExecutionState &state)
{
    CacheKeyAuditLogRecords auditRecords(_logger);
    autil::ScopedLock lock(_lock);
    int32_t batchRequestCount = 0;
    CallbackBatchPtr batch;
    auto readyIt = _readyBatches.find(batchId);
    if (_readyBatches.end() != readyIt) {
        batch = readyIt->second;
        batchRequestCount = static_cast<int32_t>(readyIt->second->requestIdSet.size());
        _readyBatches.erase(readyIt);
        _queuedBatchRequestCount -= batchRequestCount;
    } else {
        auto runningIt = _runningBatches.find(batchId);
        if (_runningBatches.end() == runningIt) {
            return false;
        }
        batch = runningIt->second;
        batchRequestCount = static_cast<int32_t>(runningIt->second->requestIdSet.size());
        _runningBatches.erase(runningIt);
        _runningRequestCount = std::max(0, _runningRequestCount - batchRequestCount);
        if (_runningBatches.empty()) {
            finishExecutionBusyTimeLocked(autil::TimeUtility::currentTime());
        }
    }
    const auto nowTimeUs = autil::TimeUtility::currentTime();
    if (batch) finishUtilizationExecutionLocked(*batch);
    updateReadyWithoutRunningLocked(nowTimeUs);
    _abnormalBatchFinishCount.fetch_add(1, std::memory_order_relaxed);
    makeExecutionState(state);
    state.batchId = batchId;
    state.scheduleBatchId = batch ? batch->scheduleBatchId : 0;
    if (batch) {
        state.localBatch = makeLocalBatchInfo(*batch);
        state.requestIds.assign(batch->requestIdSet.begin(), batch->requestIdSet.end());
        std::sort(state.requestIds.begin(), state.requestIds.end());
    }
    state.batchRequestCount = batchRequestCount;
    if (batch) {
        auditRecords.reserve(batch->cacheHitPathTokens.size());
        for (const auto &entry : batch->cacheHitPathTokens) {
            if (entry.second.audit.arrivalTimeUs <= 0) {
                continue;
            }
            CacheKeyAuditLogRecord record;
            record.requestId = entry.first;
            record.scheduleBatchId = batch->scheduleBatchId;
            record.localBatchId = batch->batchId;
            record.expectedBatchSize = batch->expectedBatchSize;
            record.arrivalTimeUs = entry.second.audit.arrivalTimeUs;
            record.readyTimeUs = batch->readyTimeUs;
            record.startTimeUs = batch->startTimeUs;
            record.finishTimeUs = nowTimeUs;
            record.audit = entry.second.audit;
            record.totalTokens = entry.second.totalTokens;
            record.blockSize = entry.second.blockSize;
            record.masterHitTokens = entry.second.masterTokens;
            record.agentPhysicalHitTokens = entry.second.agentPhysicalTokens;
            record.agentPendingAwareHitTokens = entry.second.agentPendingAwareTokens;
            record.actual.cacheHitTokenCount = -1;
            record.outcome = "aborted";
            auditRecords.emplace_back(std::move(record));
        }
    }
    return true;
}

bool AgentBatchR::abortPendingBatch(int64_t scheduleBatchId, ExecutionState &state)
{
    return abortPendingBatch(scheduleBatchId, state, CallbackBatchPtr());
}

bool AgentBatchR::abortPendingBatch(int64_t scheduleBatchId, ExecutionState &state,
                                    const CallbackBatchPtr &expectedBatch)
{
    CacheKeyAuditLogRecords auditRecords(_logger);
    autil::ScopedLock lock(_lock);
    auto it = _callbackBatches.find(scheduleBatchId);
    if (_callbackBatches.end() == it || (expectedBatch && it->second != expectedBatch)) {
        return false;
    }
    auto batch = it->second;
    if (expectedBatch) {
        if (!batch->abortRequested) {
            return false;
        }
    } else {
        if (batch->abortRequested) {
            return false;
        }
        batch->abortRequested = true;
        const auto closeExpireTimeUs =
            autil::TimeUtility::currentTime() + _closedBatchKeepMs * 1000;
        _closedBatches[scheduleBatchId] = ClosedBatch{closeExpireTimeUs, true};
    }
    while (batch->pendingAcceptedRequestCount > 0) {
        _lock.wait();
    }
    it = _callbackBatches.find(scheduleBatchId);
    if (_callbackBatches.end() == it || it->second != batch) {
        return false;
    }
    _callbackBatches.erase(it);
    const auto nowTimeUs = autil::TimeUtility::currentTime();
    const auto batchRequestCount = static_cast<int32_t>(batch->requestIdSet.size());
    _batchWaitingRequestCount =
        std::max<int32_t>(0, _batchWaitingRequestCount - batchRequestCount);
    _abnormalBatchFinishCount.fetch_add(1, std::memory_order_relaxed);
    makeExecutionState(state);
    state.batchId = makeBatchId();
    state.scheduleBatchId = batch->scheduleBatchId;
    state.batchRequestCount = batchRequestCount;
    state.eventTimeUs = nowTimeUs;
    batch->batchId = state.batchId;
    state.requestIds.assign(batch->requestIdSet.begin(), batch->requestIdSet.end());
    std::sort(state.requestIds.begin(), state.requestIds.end());
    state.localBatch = makeLocalBatchInfo(*batch);
    auditRecords.reserve(batch->cacheHitPathTokens.size());
    for (const auto &entry : batch->cacheHitPathTokens) {
        if (entry.second.audit.arrivalTimeUs <= 0) {
            continue;
        }
        CacheKeyAuditLogRecord record;
        record.requestId = entry.first;
        record.scheduleBatchId = batch->scheduleBatchId;
        record.localBatchId = state.batchId;
        record.expectedBatchSize = batch->expectedBatchSize;
        record.arrivalTimeUs = entry.second.audit.arrivalTimeUs;
        record.readyTimeUs = batch->readyTimeUs;
        record.startTimeUs = batch->startTimeUs;
        record.finishTimeUs = nowTimeUs;
        record.audit = entry.second.audit;
        record.totalTokens = entry.second.totalTokens;
        record.blockSize = entry.second.blockSize;
        record.masterHitTokens = entry.second.masterTokens;
        record.agentPhysicalHitTokens = entry.second.agentPhysicalTokens;
        record.agentPendingAwareHitTokens = entry.second.agentPendingAwareTokens;
        record.actual.cacheHitTokenCount = -1;
        record.outcome = "aborted_pending";
        auditRecords.emplace_back(std::move(record));
    }
    return true;
}

bool AgentBatchR::getExecutionState(ExecutionState &state) const
{
    autil::ScopedLock lock(_lock);
    makeExecutionState(state);
    return true;
}

AgentBatchR::GraphRecoveryState AgentBatchR::getGraphRecoveryState() const
{
    autil::ScopedLock lock(_lock);
    GraphRecoveryState state;
    state.batchWaitingRequests = _batchWaitingRequestCount;
    state.queuedBatchRequests = _queuedBatchRequestCount;
    state.runningRequests = _runningRequestCount;
    const auto append = [&state](const CallbackBatchPtr &source, bool running) {
        if (!source || source->requestIdSet.empty()) {
            return;
        }
        auto batch = std::make_shared<NaviSchedBatch>();
        batch->batchId = source->batchId;
        batch->scheduleBatchId = source->scheduleBatchId;
        batch->requestIds.assign(source->requestIdSet.begin(),
                                 source->requestIdSet.end());
        std::sort(batch->requestIds.begin(), batch->requestIds.end());
        batch->expectedBatchSize = source->expectedBatchSize;
        batch->timedOut = source->batch ? source->batch->timedOut : false;
        batch->readyTimeUs = source->readyTimeUs;
        batch->callbackDispatchTimeUs =
            source->batch ? source->batch->callbackDispatchTimeUs : 0;
        batch->readyAcceptedCallbackTimeUs = source->readyAcceptedCallbackTimeUs;
        batch->pendingCacheTracked = source->pendingCacheTracked;
        batch->_requestStats = source->requestStats;
        state.batches.emplace_back(
            GraphRecoveryBatch{std::move(batch), source->startTimeUs, running});
    };
    for (const auto &entry : _readyBatches) {
        append(entry.second, false);
    }
    for (const auto &entry : _runningBatches) {
        append(entry.second, true);
    }
    std::sort(state.batches.begin(), state.batches.end(),
              [](const GraphRecoveryBatch &lhs, const GraphRecoveryBatch &rhs) {
                  if (lhs.batch->readyTimeUs != rhs.batch->readyTimeUs) {
                      return lhs.batch->readyTimeUs < rhs.batch->readyTimeUs;
                  }
                  return lhs.batch->batchId < rhs.batch->batchId;
              });
    return state;
}

bool AgentBatchR::prepareGraphStart()
{
    autil::ScopedLock lock(_lock);
    if (_stopped) {
        return false;
    }
    _graphFinishRequested = false;
    _graphPipe.reset();
    _graphPipeOwnerGeneration = 0;
    return true;
}

uint64_t AgentBatchR::beginGraphPipeGeneration()
{
    autil::ScopedLock lock(_lock);
    if (_graphFinishRequested || _stopped) {
        return 0;
    }
    const auto generation =
        _graphPipeGeneration.fetch_add(1, std::memory_order_relaxed) + 1;
    _graphPipeOwnerGeneration = generation;
    _graphPipe.reset();
    return generation;
}

bool AgentBatchR::setGraphPipe(uint64_t generation, const navi::AsyncPipePtr &graphPipe)
{
    initLogger(_logger, "AgentBatchR");
    if (0 == generation || !graphPipe) {
        NAVI_LOG(ERROR,
                 "reject invalid Agent batch graph pipe, generation [%lu], pipe [%p]",
                 generation, graphPipe.get());
        return false;
    }
    bool finishRequested = false;
    {
        autil::ScopedLock lock(_lock);
        if (_stopped || _graphPipeOwnerGeneration != generation ||
            (_graphPipe && _graphPipe != graphPipe)) {
            NAVI_LOG(ERROR,
                     "reject stale Agent batch graph pipe, generation [%lu], "
                     "ownerGeneration [%lu], stopped [%d], currentPipe [%p], "
                     "incomingPipe [%p]",
                     generation, _graphPipeOwnerGeneration, static_cast<int>(_stopped),
                     _graphPipe.get(), graphPipe.get());
            return false;
        }
        _graphPipe = graphPipe;
        finishRequested = _graphFinishRequested;
    }
    if (!finishRequested) {
        return true;
    }
    return navi::EC_NONE == graphPipe->setDataOverride(nullptr, true) &&
           navi::EC_NONE == graphPipe->setEof();
}

bool AgentBatchR::waitGraphPipe(uint64_t generation, int64_t timeoutMs) const
{
    if (0 == generation || timeoutMs < 0) {
        return false;
    }
    constexpr int64_t READY_POLL_INTERVAL_US = 10 * 1000;
    const auto deadlineUs = autil::TimeUtility::monotonicTimeUs() + timeoutMs * 1000;
    while (true) {
        {
            autil::ScopedLock lock(_lock);
            if (_stopped || _graphFinishRequested ||
                _graphPipeOwnerGeneration != generation) {
                return false;
            }
            if (_graphPipe) {
                return true;
            }
        }
        const auto remainingUs = deadlineUs - autil::TimeUtility::monotonicTimeUs();
        if (remainingUs <= 0) {
            return false;
        }
        std::this_thread::sleep_for(
            std::chrono::microseconds(std::min(remainingUs, READY_POLL_INTERVAL_US)));
    }
}

void AgentBatchR::clearGraphPipe(uint64_t generation)
{
    autil::ScopedLock lock(_lock);
    if (_graphPipeOwnerGeneration != generation) {
        NAVI_LOG(INFO,
                 "ignore stale Agent batch graph pipe cleanup, generation [%lu], "
                 "ownerGeneration [%lu]",
                 generation, _graphPipeOwnerGeneration);
        return;
    }
    _graphPipe.reset();
    _graphPipeOwnerGeneration = 0;
}

bool AgentBatchR::finishGraph()
{
    navi::AsyncPipePtr graphPipe;
    {
        autil::ScopedLock lock(_lock);
        _graphFinishRequested = true;
        graphPipe = _graphPipe;
    }
    const bool pipeFinished =
        !graphPipe || (navi::EC_NONE == graphPipe->setDataOverride(nullptr, true) &&
                       navi::EC_NONE == graphPipe->setEof());
    {
        autil::ScopedLock lock(_lock);
        while (_activeCallbackCount > 0) {
            _lock.wait();
        }
    }
    return pipeFinished;
}

uint64_t AgentBatchR::getGraphPipeGeneration() const
{
    return _graphPipeGeneration.load(std::memory_order_relaxed);
}

int64_t AgentBatchR::getNextBatchTimeoutMs() const
{
    autil::ScopedLock lock(_lock);
    if (_callbackBatches.empty()) {
        return -1;
    }
    auto deadlineTimeUs = std::numeric_limits<int64_t>::max();
    for (const auto &pair : _callbackBatches) {
        deadlineTimeUs = std::min(deadlineTimeUs, pair.second->deadlineTimeUs);
    }
    auto remainingUs = deadlineTimeUs - autil::TimeUtility::currentTime();
    return std::max<int64_t>(1, (remainingUs + 999) / 1000);
}

int64_t AgentBatchR::getLatestLocalBatchId() const
{
    return _nextBatchId.load(std::memory_order_acquire);
}

size_t AgentBatchR::getPendingBatchCount() const
{
    autil::ScopedLock lock(_lock);
    return _callbackBatches.size();
}

uint64_t AgentBatchR::getTimeoutBatchCount() const
{
    return _timeoutBatchCount.load(std::memory_order_relaxed);
}

uint64_t AgentBatchR::getTimeoutMissingRequestCount() const
{
    return _timeoutMissingRequestCount.load(std::memory_order_relaxed);
}

uint64_t AgentBatchR::getLateRequestCount() const
{
    return _lateRequestCount.load(std::memory_order_relaxed);
}

uint64_t AgentBatchR::getAbnormalBatchFinishCount() const
{
    return _abnormalBatchFinishCount.load(std::memory_order_relaxed);
}

AgentBatchR::ReadyWithoutRunningStats AgentBatchR::getReadyWithoutRunningStats() const
{
    autil::ScopedLock lock(_lock);
    ReadyWithoutRunningStats stats;
    stats.gapCount = _readyWithoutRunningGapCount;
    stats.totalTimeUs = _readyWithoutRunningTotalTimeUs;
    stats.maxTimeUs = _readyWithoutRunningMaxTimeUs;
    if (_readyWithoutRunningStartTimeUs > 0) {
        stats.currentTimeUs = std::max<int64_t>(
            0, autil::TimeUtility::currentTime() - _readyWithoutRunningStartTimeUs);
        stats.totalTimeUs += stats.currentTimeUs;
        stats.maxTimeUs = std::max(stats.maxTimeUs, stats.currentTimeUs);
    }
    return stats;
}

AgentBatchR::BatchFormationStats AgentBatchR::takeBatchFormationWindowStats() const
{
    BatchFormationStats stats;
    int64_t formationTimeUs = 0;
    while (_batchFormationTimeQueue.Pop(&formationTimeUs)) {
        ++stats.batchCount;
        stats.totalTimeUs += formationTimeUs;
        stats.maxTimeUs = std::max(stats.maxTimeUs, formationTimeUs);
    }
    return stats;
}

void AgentBatchR::recordBackendBatchExecution(
    const BackendBatchExecutionSample &sample) const
{
    _backendBatchExecutionQueue.Push(sample);
}

AgentBatchR::BackendBatchExecutionStats
AgentBatchR::takeBackendBatchExecutionWindowStats() const
{
    BackendBatchExecutionStats stats;
    BackendBatchExecutionSample sample;
    while (_backendBatchExecutionQueue.Pop(&sample)) {
        ++stats.batchCount;
        stats.wallTimeUs += sample.wallTimeUs;
        stats.maxWallTimeUs = std::max(stats.maxWallTimeUs, sample.wallTimeUs);
        stats.executeModelTimeUs += sample.executeModelTimeUs;
        const auto interStepGapTimeUs =
            std::max<int64_t>(0, sample.wallTimeUs - sample.executeModelTimeUs);
        stats.interStepGapTimeUs += interStepGapTimeUs;
        stats.maxInterStepGapTimeUs =
            std::max(stats.maxInterStepGapTimeUs, interStepGapTimeUs);
        stats.executionStepCount += sample.executionStepCount;
        stats.prefillTokenCount += sample.prefillTokenCount;
        stats.decodeRequestStepCount += sample.decodeRequestStepCount;
        stats.decodeTokenCount += sample.decodeTokenCount;
        if (sample.admissionToFirstExecutionTimeUs >= 0) {
            ++stats.admissionToFirstExecutionSampleCount;
            stats.admissionToFirstExecutionTimeUs +=
                sample.admissionToFirstExecutionTimeUs;
            stats.maxAdmissionToFirstExecutionTimeUs =
                std::max(stats.maxAdmissionToFirstExecutionTimeUs,
                         sample.admissionToFirstExecutionTimeUs);
        }
    }
    return stats;
}

void AgentBatchR::recordBackendExecutionStep(
    const BackendExecutionStepSample &sample) const
{
    _backendExecutionStepQueue.Push(sample);
}

AgentBatchR::BackendExecutionStepStats AgentBatchR::takeBackendExecutionStepWindowStats()
    const
{
    BackendExecutionStepStats stats;
    BackendExecutionStepSample sample;
    while (_backendExecutionStepQueue.Pop(&sample)) {
        ++stats.stepCount;
        stats.totalTimeUs += sample.elapsedTimeUs;
        stats.maxTimeUs = std::max(stats.maxTimeUs, sample.elapsedTimeUs);
        if (sample.pureDecode) {
            ++stats.pureDecodeStepCount;
            stats.pureDecodeTotalTimeUs += sample.elapsedTimeUs;
            stats.pureDecodeMaxTimeUs =
                std::max(stats.pureDecodeMaxTimeUs, sample.elapsedTimeUs);
        }
    }
    return stats;
}

void AgentBatchR::recordCoreStep(const CoreStepSample &sample) const
{
    _coreStepQueue.Push(sample);
}

AgentBatchR::CoreStepStats AgentBatchR::takeCoreStepWindowStats() const
{
    CoreStepStats stats;
    CoreStepSample sample;
    while (_coreStepQueue.Pop(&sample)) {
        if (sample.futureWait) {
            ++stats.futureWaitCount;
            stats.futureWaitTotalTimeUs += sample.elapsedTimeUs;
            stats.futureWaitMaxTimeUs =
                std::max(stats.futureWaitMaxTimeUs, sample.elapsedTimeUs);
        } else {
            ++stats.stepCount;
            stats.totalTimeUs += sample.elapsedTimeUs;
            stats.maxTimeUs = std::max(stats.maxTimeUs, sample.elapsedTimeUs);
        }
    }
    return stats;
}

AgentBatchR::CacheHitPathStats AgentBatchR::getCacheHitPathStats() const
{
    autil::ScopedLock lock(_lock);
    return _cacheHitPathStats;
}

int64_t AgentBatchR::getExecutionBusyTimeUs(int64_t currentTimeUs) const
{
    autil::ScopedLock lock(_lock);
    auto busyTimeUs = _executionBusyTimeUs;
    if (_executionBusyStartTimeUs > 0) {
        busyTimeUs += std::max<int64_t>(0, currentTimeUs - _executionBusyStartTimeUs);
    }
    return busyTimeUs;
}

bool AgentBatchR::beginBackendExecution(int64_t eventTimeUs, int32_t requestCount)
{
    if (eventTimeUs <= 0) {
        eventTimeUs = autil::TimeUtility::monotonicTimeUs();
    }
    autil::ScopedLock lock(_backendExecutionLock);
    if (requestCount > 0) {
        const auto previous = _backendExecutionCapacity.load(std::memory_order_relaxed);
        _backendExecutionCapacity.store(
            previous > 0 ? previous + 0.1 * (requestCount - previous) : requestCount,
            std::memory_order_relaxed);
    }
    _backendExecutionObserved = true;
    _executionUtilizationEstimator.begin(eventTimeUs);
    NAVI_LOG(INFO, "[navi-execution-utilization] phase=step_begin mono_us=%ld active=%lu",
             eventTimeUs, _executionUtilizationEstimator.active());
    if (0 == _activeBackendExecutionCount++) {
        _backendExecutionBusyStartTimeUs = eventTimeUs;
    }
    return true;
}

bool AgentBatchR::endBackendExecution(int64_t eventTimeUs)
{
    if (eventTimeUs <= 0) {
        eventTimeUs = autil::TimeUtility::monotonicTimeUs();
    }
    autil::ScopedLock lock(_backendExecutionLock);
    if (0 == _activeBackendExecutionCount) {
        return false;
    }
    _executionUtilizationEstimator.end(eventTimeUs);
    NAVI_LOG(INFO,
             "[navi-execution-utilization] phase=step_end mono_us=%ld "
             "sample_busy_us=%ld sample_elapsed_us=%ld active=%lu busy_ema=%.9f "
             "capacity_ema=%.6f gpu_ready_queued=%d effective=%.9f",
             eventTimeUs, _executionUtilizationEstimator.sampleBusyUs(),
             _executionUtilizationEstimator.sampleElapsedUs(),
             _executionUtilizationEstimator.active(),
             _executionUtilizationEstimator.value(),
             _backendExecutionCapacity.load(std::memory_order_relaxed),
             _backendWaitingRequests.load(std::memory_order_relaxed),
             _executionUtilization->load(std::memory_order_relaxed));
    if (0 == --_activeBackendExecutionCount) {
        _backendExecutionBusyTimeUs +=
            std::max<int64_t>(0, eventTimeUs - _backendExecutionBusyStartTimeUs);
        _backendExecutionBusyStartTimeUs = 0;
    }
    return true;
}

bool AgentBatchR::sampleBackendExecutionUtilization(int64_t nowMonoUs)
{
    autil::ScopedLock lock(_backendExecutionLock);
    if (nowMonoUs <= 0) nowMonoUs = autil::TimeUtility::monotonicTimeUs();
    auto busyUs = _backendExecutionBusyTimeUs;
    if (_activeBackendExecutionCount > 0) {
        busyUs += std::max<int64_t>(0, nowMonoUs - _backendExecutionBusyStartTimeUs);
    }
    if (!_executionUtilizationWindow.sample(nowMonoUs, busyUs)) return false;
    _backendExecutionUtilization->store(_executionUtilizationWindow.value(),
                                        std::memory_order_relaxed);
    refreshExecutionUtilizationLocked();
    NAVI_LOG(INFO,
             "[navi-execution-utilization] phase=periodic_sample mono_us=%ld "
             "sample_busy_us=%ld sample_elapsed_us=%ld active=%lu busy_window=%.9f "
             "effective=%.9f",
             nowMonoUs, _executionUtilizationWindow.busyUs(),
             _executionUtilizationWindow.elapsedUs(), _activeBackendExecutionCount,
             _executionUtilizationWindow.value(),
             _executionUtilization->load(std::memory_order_relaxed));
    return true;
}

bool AgentBatchR::hasBackendExecutionObservation() const
{
    autil::ScopedLock lock(_backendExecutionLock);
    return _backendExecutionObserved;
}

int64_t AgentBatchR::getBackendExecutionBusyTimeUs(int64_t currentTimeUs) const
{
    autil::ScopedLock lock(_backendExecutionLock);
    auto busyTimeUs = _backendExecutionBusyTimeUs;
    if (_backendExecutionBusyStartTimeUs > 0) {
        busyTimeUs +=
            std::max<int64_t>(0, currentTimeUs - _backendExecutionBusyStartTimeUs);
    }
    return busyTimeUs;
}

void AgentBatchR::finishExecutionBusyTimeLocked(int64_t nowTimeUs)
{
    if (_executionBusyStartTimeUs > 0) {
        _executionBusyTimeUs +=
            std::max<int64_t>(0, nowTimeUs - _executionBusyStartTimeUs);
        _executionBusyStartTimeUs = 0;
    }
}

const std::vector<NaviSchedRequestStats> &AgentBatchR::getRequestStats(
    const NaviSchedBatch &batch)
{
    return batch._requestStats;
}

navi::SummaryErrorCode AgentBatchR::getSummary(const std::string &query,
                                               navi::SummaryBuilder &builder) const
{
    autil::ScopedLock lock(_lock);
    auto overview = builder.newTable("NaviSchedAgentBatchOverview");
    {
        autil::ScopedLock backendLock(_backendExecutionLock);
        TABLE_ADD_COLUMN_VALUE(overview, "execution_busy_ema",
                               _executionUtilizationEstimator.value());
        TABLE_ADD_COLUMN_VALUE(overview, "execution_busy_window",
                               _executionUtilizationWindow.value());
        TABLE_ADD_COLUMN_VALUE(overview, "execution_window_busy_us",
                               _executionUtilizationWindow.busyUs());
        TABLE_ADD_COLUMN_VALUE(overview, "execution_window_elapsed_us",
                               _executionUtilizationWindow.elapsedUs());
        TABLE_ADD_COLUMN_VALUE(overview, "execution_window_sample_mono_us",
                               _executionUtilizationWindow.sampleMonoUs());
        TABLE_ADD_COLUMN_VALUE(overview, "execution_active_count",
                               _executionUtilizationEstimator.active());
        TABLE_ADD_COLUMN_VALUE(overview, "execution_sample_busy_us",
                               _executionUtilizationEstimator.sampleBusyUs());
        TABLE_ADD_COLUMN_VALUE(overview, "execution_sample_elapsed_us",
                               _executionUtilizationEstimator.sampleElapsedUs());
    }
    TABLE_ADD_COLUMN_VALUE(overview, "execution_capacity_ema",
                           _backendExecutionCapacity.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "gpu_ready_queued_requests",
                           _backendWaitingRequests.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "execution_running_requests",
                           _utilizationRunningRequests);
    TABLE_ADD_COLUMN_VALUE(overview, "execution_utilization",
                           _executionUtilization->load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "batch_timeout_ms", _batchTimeoutMs);
    TABLE_ADD_COLUMN_VALUE(overview, "pending_batch_count", _callbackBatches.size());
    TABLE_ADD_COLUMN_VALUE(overview, "callback_failure_count",
                           _callbackFailureCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "timeout_batch_count",
                           _timeoutBatchCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "timeout_missing_request_count",
                           _timeoutMissingRequestCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "late_request_count",
                           _lateRequestCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "expected_size_mismatch_count",
                           _expectedSizeMismatchCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "prediction_mismatch_count",
                           _predictionMismatchCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "duplicate_request_count",
                           _duplicateRequestCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "invalid_request_count",
                           _invalidRequestCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(overview, "cache_hit_path_batch_count",
                           _cacheHitPathStats.batchCount);
    TABLE_ADD_COLUMN_VALUE(overview, "cache_hit_path_request_count",
                           _cacheHitPathStats.requestCount);
    TABLE_ADD_COLUMN_VALUE(overview, "cache_hit_path_master_predicted_tokens",
                           _cacheHitPathStats.masterPredictedTokens);
    TABLE_ADD_COLUMN_VALUE(overview, "cache_hit_path_agent_arrival_tokens",
                           _cacheHitPathStats.agentArrivalTokens);
    TABLE_ADD_COLUMN_VALUE(overview, "cache_hit_path_agent_pending_aware_tokens",
                           _cacheHitPathStats.agentPendingAwareTokens);
    TABLE_ADD_COLUMN_VALUE(overview, "cache_hit_path_rtp_actual_tokens",
                           _cacheHitPathStats.rtpActualTokens);
    TABLE_ADD_COLUMN_VALUE(overview, "cache_hit_path_aligned_request_count",
                           _cacheHitPathStats.alignedRequestCount);
    TABLE_ADD_COLUMN_VALUE(overview, "cache_hit_path_aligned_agent_to_rtp_error_tokens",
                           _cacheHitPathStats.alignedAgentToRtpErrorTokens);
    TABLE_ADD_COLUMN_VALUE(overview, "cache_hit_path_unaligned_request_count",
                           _cacheHitPathStats.unalignedRequestCount);
    TABLE_ADD_COLUMN_VALUE(overview, "cache_hit_path_unaligned_agent_to_rtp_error_tokens",
                           _cacheHitPathStats.unalignedAgentToRtpErrorTokens);
    TABLE_ADD_COLUMN_VALUE(overview, "hybrid_reuse_debug_request_count",
                           _cacheHitPathStats.hybridDebugRequestCount);
    TABLE_ADD_COLUMN_VALUE(overview, "hybrid_reuse_agent_predicted_blocks",
                           _cacheHitPathStats.hybridAgentPredictedBlocks);
    TABLE_ADD_COLUMN_VALUE(overview, "hybrid_reuse_candidate_blocks",
                           _cacheHitPathStats.hybridCandidateBlocks);
    TABLE_ADD_COLUMN_VALUE(overview, "hybrid_reuse_flat_matched_blocks",
                           _cacheHitPathStats.hybridFlatMatchedBlocks);
    TABLE_ADD_COLUMN_VALUE(overview, "hybrid_reuse_min_full_blocks",
                           _cacheHitPathStats.hybridMinFullReuseBlocks);
    TABLE_ADD_COLUMN_VALUE(overview, "hybrid_reuse_selected_blocks",
                           _cacheHitPathStats.hybridSelectedReuseBlocks);
    TABLE_ADD_COLUMN_VALUE(overview, "hybrid_reuse_agent_to_full_overprediction_blocks",
                           _cacheHitPathStats.hybridAgentToFullOverpredictionBlocks);
    TABLE_ADD_COLUMN_VALUE(overview, "hybrid_reuse_tail_rollback_blocks",
                           _cacheHitPathStats.hybridTailRollbackBlocks);
    TABLE_ADD_COLUMN_VALUE(overview, "hybrid_reuse_invalid_order_count",
                           _cacheHitPathStats.hybridInvalidOrderCount);
    if (SummaryUtil::getSummaryLevel(query) < 1) {
        return navi::SEC_OK;
    }
    auto pendingTable = builder.newTable("NaviSchedAgentPendingBatches");
    for (const auto &pair : _callbackBatches) {
        TABLE_ADD_COLUMN_VALUE(pendingTable, "batch_id", pair.first);
        TABLE_ADD_COLUMN_VALUE(pendingTable, "expected_batch_size",
                               pair.second->expectedBatchSize);
        TABLE_ADD_COLUMN_VALUE(pendingTable, "received_request_count",
                               pair.second->requestIdSet.size());
    }
    return navi::SEC_OK;
}

AgentBatchR::CallbackBatchPtr AgentBatchR::createCallbackBatch(
    int64_t expectedBatchSize, const NaviSchedReadyBatchCallback &callback,
    const std::function<void(const ExecutionState &)> &acceptedFailureCallback)
{
    auto batch = std::make_shared<CallbackBatch>();
    batch->batch = std::make_shared<NaviSchedBatch>();
    batch->expectedBatchSize = expectedBatchSize;
    batch->batch->expectedBatchSize = expectedBatchSize;
    batch->callback = callback;
    batch->acceptedFailureCallback = acceptedFailureCallback;
    return batch;
}

AgentBatchR::AppendResult AgentBatchR::appendRequest(const CallbackBatchPtr &batch,
                                                     const NaviSchedRequestStats &request,
                                                     const CacheKeyAudit &cacheKeyAudit)
{
    if (!batch || !request.valid()) {
        return AppendResult::INVALID_REQUEST;
    }
    const auto firstRequest = batch->requestIdSet.empty();
    if (!firstRequest &&
        (batch->predictedQueueLatencyMs != request.predictedQueueLatencyMs ||
         batch->predictedExecutionLatencyMs != request.predictedExecutionLatencyMs ||
         batch->requestStats.front().latencyTrainingPhase !=
             request.latencyTrainingPhase)) {
        return AppendResult::PREDICTION_MISMATCH;
    }
    if (!batch->requestIdSet.emplace(request.requestId).second) {
        return AppendResult::DUPLICATE_REQUEST;
    }
    if (firstRequest) {
        batch->predictedQueueLatencyMs = request.predictedQueueLatencyMs;
        batch->predictedExecutionLatencyMs = request.predictedExecutionLatencyMs;
    }
    batch->batch->requestIds.emplace_back(request.requestId);
    batch->batch->_requestStats.emplace_back(request);
    batch->requestStats.emplace_back(request);
    batch->cacheHitPathTokens.emplace(
        request.requestId, CallbackBatch::CacheHitPathTokens{
                               request.predictedCacheHitTokenCount,
                               cacheKeyAudit.physicalCacheHitTokenCount >= 0
                                   ? cacheKeyAudit.physicalCacheHitTokenCount
                                   : request.cacheHitTokenCount,
                               request.cacheHitTokenCount, request.totalTokenCount,
                               request.blockSize, cacheKeyAudit});
    batch->predictedCacheHitTokenCount += request.predictedCacheHitTokenCount;
    ++batch->pendingAcceptedRequestCount;
    return AppendResult::ACCEPTED;
}

AgentBatchR::CallbackBatchPtr AgentBatchR::markRequestAccepted(
    const CallbackBatchPtr &batch)
{
    if (!batch || 0 == batch->pendingAcceptedRequestCount) {
        return nullptr;
    }
    --batch->pendingAcceptedRequestCount;
    if (batch->abortRequested) {
        if (0 == batch->pendingAcceptedRequestCount) {
            _lock.broadcast();
        }
        return nullptr;
    }
    if (_stopped) {
        return nullptr;
    }
    return takeReadyCallback(batch);
}

AgentBatchR::CallbackBatchPtr AgentBatchR::takeReadyCallback(
    const CallbackBatchPtr &batch)
{
    if (!batch || batch->batchId <= 0 || batch->pendingAcceptedRequestCount > 0 ||
        batch->callbackDispatched) {
        return nullptr;
    }
    batch->callbackDispatched = true;
    return batch;
}

NaviSchedLocalBatchInfo AgentBatchR::makeLocalBatchInfo(const CallbackBatch &batch)
{
    NaviSchedLocalBatchInfo info;
    if (batch.scheduleBatchId <= 0 || batch.batchId <= 0 || batch.requestIdSet.empty() ||
        batch.requestIdSet.size() >
            static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        return info;
    }
    info.scheduleBatchId = batch.scheduleBatchId;
    info.localBatchId = batch.batchId;
    info.requestCount = static_cast<int32_t>(batch.requestIdSet.size());
    info.pendingCacheTracked = batch.pendingCacheTracked;
    if (static_cast<int64_t>(batch.requestIdSet.size()) != batch.expectedBatchSize) {
        info.requestIds.assign(batch.requestIdSet.begin(), batch.requestIdSet.end());
        std::sort(info.requestIds.begin(), info.requestIds.end());
    }
    return info;
}

void AgentBatchR::finishCallbackWorkLocked()
{
    --_activeCallbackCount;
    if (0 == _activeCallbackCount) {
        _lock.broadcast();
    }
}

void AgentBatchR::finishCallbackWork()
{
    autil::ScopedLock lock(_lock);
    finishCallbackWorkLocked();
}

void AgentBatchR::publishReadyBatch(const CallbackBatchPtr &batch)
{
    batch->readyTimeUs = autil::TimeUtility::currentTime();
    batch->batch->readyTimeUs = batch->readyTimeUs;
    const auto formationTimeUs =
        std::max<int64_t>(0, batch->readyTimeUs - batch->firstRequestTimeUs);
    _batchFormationTimeQueue.Push(formationTimeUs);
    batch->batchId = makeBatchId();
    batch->batch->batchId = batch->batchId;
    batch->batch->scheduleBatchId = batch->scheduleBatchId;
    _readyBatches.emplace(batch->batchId, batch);
    _queuedBatchRequestCount += batch->requestIdSet.size();
    updateReadyWithoutRunningLocked(batch->readyTimeUs);
}

void AgentBatchR::flushExpiredCallbackBatches(
    int64_t nowTimeUs, std::vector<CallbackBatchPtr> &readyBatches,
    std::vector<CallbackBatchPtr> &timedOutBatches)
{
    for (auto it = _callbackBatches.begin(); _callbackBatches.end() != it;) {
        auto &batch = it->second;
        if (batch->deadlineTimeUs > nowTimeUs) {
            ++it;
            continue;
        }
        const auto batchId = it->first;
        auto expiredBatch = batch;
        ++it;
        makeCallbackBatchReady(batchId, expiredBatch, true, nowTimeUs);
        timedOutBatches.emplace_back(expiredBatch);
        auto readyBatch = takeReadyCallback(expiredBatch);
        if (readyBatch) {
            readyBatches.emplace_back(std::move(readyBatch));
        }
    }
}

void AgentBatchR::logBatchTimeout(const CallbackBatchPtr &batch)
{
    const auto missingRequestCount = std::max<int64_t>(
        0, batch->expectedBatchSize -
               static_cast<int64_t>(batch->timeoutReceivedRequestCount));
    NAVI_LOG(WARN,
             "[navi-batch-timeout] schedule_batch_id=%ld local_batch_id=%ld "
             "expected_batch_size=%ld received_request_count=%zu "
             "missing_request_count=%ld first_arrival_time_us=%ld "
             "last_arrival_time_us=%ld arrival_span_us=%ld timeout_time_us=%ld "
             "batch_timeout_ms=%ld",
             batch->scheduleBatchId, batch->batchId, batch->expectedBatchSize,
             batch->timeoutReceivedRequestCount, missingRequestCount,
             batch->firstRequestTimeUs, batch->lastRequestTimeUs,
             std::max<int64_t>(0, batch->lastRequestTimeUs - batch->firstRequestTimeUs),
             batch->timeoutDetectedTimeUs,
             (batch->deadlineTimeUs - batch->firstRequestTimeUs) / 1000);
}

void AgentBatchR::recycleClosedBatches(int64_t nowTimeUs)
{
    for (auto it = _closedBatches.begin(); _closedBatches.end() != it;) {
        if (it->second.expireTimeUs <= nowTimeUs) {
            it = _closedBatches.erase(it);
        } else {
            ++it;
        }
    }
}

void AgentBatchR::makeCallbackBatchReady(int64_t groupBatchId, CallbackBatchPtr batch,
                                         bool timedOut, int64_t nowTimeUs)
{
    auto it = _callbackBatches.find(groupBatchId);
    if (_callbackBatches.end() == it || it->second != batch) {
        return;
    }
    _callbackBatches.erase(it);
    if (timedOut) {
        batch->timeoutDetectedTimeUs = nowTimeUs;
        batch->timeoutReceivedRequestCount = batch->requestIdSet.size();
        _timeoutBatchCount.fetch_add(1, std::memory_order_relaxed);
        const auto missing =
            std::max<int64_t>(0, batch->expectedBatchSize -
                                     static_cast<int64_t>(batch->requestIdSet.size()));
        _timeoutMissingRequestCount.fetch_add(missing, std::memory_order_relaxed);
    }
    batch->batch->timedOut = timedOut;
    _closedBatches[groupBatchId] =
        ClosedBatch{nowTimeUs + _closedBatchKeepMs * 1000, false};
    auto requestCount = batch->requestIdSet.size();
    _batchWaitingRequestCount -= requestCount;
    publishReadyBatch(batch);
}

int64_t AgentBatchR::makeBatchId()
{
    const auto current = _nextBatchId.load(std::memory_order_relaxed);
    const auto next = current >= std::numeric_limits<int64_t>::max() ? 1 : current + 1;
    _nextBatchId.store(next, std::memory_order_release);
    return next;
}

bool AgentBatchR::runCallbackBatch(const CallbackBatchPtr &callbackBatch)
{
    auto callback = std::move(callbackBatch->callback);
    auto batch = callbackBatch->batch;
    batch->callbackDispatchTimeUs = autil::TimeUtility::currentTime();
    if (callbackBatch->timeoutDetectedTimeUs > 0) {
        NAVI_LOG(INFO,
                 "[navi-batch-timeout-dispatch] schedule_batch_id=%ld local_batch_id=%ld "
                 "timeout_time_us=%ld callback_dispatch_time_us=%ld "
                 "timeout_to_callback_us=%ld",
                 callbackBatch->scheduleBatchId, callbackBatch->batchId,
                 callbackBatch->timeoutDetectedTimeUs, batch->callbackDispatchTimeUs,
                 std::max<int64_t>(0, batch->callbackDispatchTimeUs -
                                          callbackBatch->timeoutDetectedTimeUs));
    }
    try {
        callback(batch);
    } catch (...) {
        _callbackFailureCount.fetch_add(1, std::memory_order_relaxed);
        ExecutionState state;
        const auto aborted = abortBatch(callbackBatch->batchId, state);
        if (aborted && callbackBatch->acceptedFailureCallback) {
            try {
                callbackBatch->acceptedFailureCallback(state);
            } catch (...) {
                NAVI_LOG(ERROR,
                         "accepted failure callback threw, batchId [%ld], "
                         "scheduleBatchId [%ld], requestCount [%d]",
                         state.batchId, state.scheduleBatchId, state.batchRequestCount);
            }
        }
        callbackBatch->batch.reset();
        return false;
    }
    callbackBatch->batch.reset();
    return true;
}

void AgentBatchR::makeExecutionState(ExecutionState &state) const
{
    refreshExecutionUtilizationLocked();
    state = ExecutionState();
    state.batchWaitingRequests = _batchWaitingRequestCount;
    state.queuedBatchRequests = _queuedBatchRequestCount;
    state.runningRequests = _runningRequestCount;
    state.eventTimeUs = autil::TimeUtility::currentTime();
}

AgentBatchR::ExecutionUtilizationSnapshot AgentBatchR::getExecutionUtilizationSnapshot()
    const
{
    autil::ScopedLock lock(_lock);
    autil::ScopedLock backendLock(_backendExecutionLock);
    ExecutionUtilizationSnapshot snapshot;
    snapshot.busyEma = _executionUtilizationEstimator.value();
    snapshot.busyWindow = _executionUtilizationWindow.value();
    snapshot.windowBusyUs = _executionUtilizationWindow.busyUs();
    snapshot.windowElapsedUs = _executionUtilizationWindow.elapsedUs();
    snapshot.windowSampleMonoUs = _executionUtilizationWindow.sampleMonoUs();
    snapshot.activeCount = _executionUtilizationEstimator.active();
    snapshot.sampleBusyUs = _executionUtilizationEstimator.sampleBusyUs();
    snapshot.sampleElapsedUs = _executionUtilizationEstimator.sampleElapsedUs();
    snapshot.runningRequests = _utilizationRunningRequests;
    const auto total =
        _batchWaitingRequestCount + _queuedBatchRequestCount + _runningRequestCount;
    snapshot.queuedRequests = std::max<int32_t>(0, total - _utilizationRunningRequests);
    snapshot.utilization = _executionUtilization->load(std::memory_order_relaxed);
    snapshot.executionCapacity =
        _backendExecutionCapacity.load(std::memory_order_relaxed);
    snapshot.executionCapacityDenominator =
        snapshot.executionCapacity > 0
            ? std::max(snapshot.executionCapacity,
                       ExecutionUtilizationController::MIN_EXECUTION_CAPACITY)
            : 0.0;
    snapshot.gpuReadyQueuedRequests =
        _backendWaitingRequests.load(std::memory_order_relaxed);
    return snapshot;
}

bool AgentBatchR::beginBatchExecution(int64_t batchId)
{
    autil::ScopedLock lock(_lock);
    return _readyBatches.find(batchId) != _readyBatches.end();
}

void AgentBatchR::beginUtilizationExecutionLocked(CallbackBatch &batch)
{
    if (batch.executionBusy) return;
    batch.executionBusy = true;
    _utilizationRunningRequests += static_cast<int32_t>(batch.requestIdSet.size());
}

void AgentBatchR::finishUtilizationExecutionLocked(CallbackBatch &batch)
{
    if (!batch.executionBusy) return;
    batch.executionBusy = false;
    _utilizationRunningRequests -= static_cast<int32_t>(batch.requestIdSet.size());
}

void AgentBatchR::refreshExecutionUtilizationLocked() const
{
    _executionUtilization->store(
        ExecutionUtilizationController::effectiveUtilization(
            _backendExecutionUtilization->load(std::memory_order_relaxed),
            _backendExecutionCapacity.load(std::memory_order_relaxed),
            _backendWaitingRequests.load(std::memory_order_relaxed)),
        std::memory_order_relaxed);
}

void AgentBatchR::updateBackendWaitingRequests(int32_t requestCount,
                                               int32_t runningRequests)
{
    if (requestCount < 0 || runningRequests < 0) return;
    _backendWaitingRequests.store(requestCount, std::memory_order_relaxed);
    refreshExecutionUtilizationLocked();
}

void AgentBatchR::updateReadyWithoutRunningLocked(int64_t nowTimeUs)
{
    const bool readyWithoutRunning = !_readyBatches.empty() && _runningBatches.empty();
    if (readyWithoutRunning) {
        if (0 == _readyWithoutRunningStartTimeUs) {
            _readyWithoutRunningStartTimeUs = nowTimeUs;
            ++_readyWithoutRunningGapCount;
        }
        return;
    }
    if (0 == _readyWithoutRunningStartTimeUs) {
        return;
    }
    const auto durationUs =
        std::max<int64_t>(0, nowTimeUs - _readyWithoutRunningStartTimeUs);
    _readyWithoutRunningTotalTimeUs += durationUs;
    _readyWithoutRunningMaxTimeUs = std::max(_readyWithoutRunningMaxTimeUs, durationUs);
    _readyWithoutRunningStartTimeUs = 0;
}

REGISTER_RESOURCE(AgentBatchR);

}
