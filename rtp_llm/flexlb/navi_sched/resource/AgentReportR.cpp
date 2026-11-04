#include "aios/navi_sched/resource/AgentReportR.h"
#include "aios/navi_sched/data/AgentCacheToMasterData.h"
#include "aios/navi_sched/data/AgentControlToMasterData.h"
#include "aios/navi_sched/data/AgentLatencyToMasterData.h"
#include "aios/navi_sched/data/AgentRequestEventToMasterData.h"
#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/data/NaviSchedAgentEvent.h"
#include "aios/navi_sched/data/SummaryUtil.h"
#include "aios/navi_sched/resource/LatencyModelR.h"
#include "aios/navi_sched/resource/AgentBatchR.h"
#include <autil/TimeUtility.h>
#include <autil/HashAlgorithm.h>
#include <algorithm>
#include <cerrno>
#include <limits>
#include <unordered_set>

namespace navi_sched {
namespace {

AgentLatencyToMasterDataPtr makeLatencyModelData(
    const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot)
{
    if (!snapshot || !snapshot->valid()) {
        return nullptr;
    }
    auto latency = std::make_shared<AgentLatencyToMasterData>();
    latency->formula = snapshot->formula;
    latency->binaryVersion = snapshot->binaryVersion;
    latency->modelRevision = snapshot->revision;
    latency->parameters = snapshot->parameters;
    latency->optimizerState = NonLinearPrefillModelR::makeOptimizerStateData(
        snapshot->optimizerState, snapshot->parameters.size());
    latency->residualVersion = snapshot->residualVersion;
    latency->residualParameters = snapshot->residualParameters;
    latency->residualLastUpdateTimeUs = snapshot->residualLastUpdateTimeUs;
    latency->zeroPoint =
        NonLinearPrefillModelR::calculateZeroPoint(snapshot->parameters);
    latency->sessionSampleCount = snapshot->sessionSampleCount;
    return latency;
}

}

const std::string AgentReportBridgeR::RESOURCE_ID = "navi_sched.agent_report_bridge_r";

AgentReportBridgeR::AgentReportBridgeR()
    : navi::ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL)
{
}

void AgentReportBridgeR::setAgentReportR(
    const std::shared_ptr<AgentReportR> &agentReportR)
{
    autil::ScopedLock lock(_lock);
    _agentReportR = agentReportR;
}

std::shared_ptr<AgentReportR> AgentReportBridgeR::getAgentReportR() const
{
    autil::ScopedLock lock(_lock);
    return _agentReportR.lock();
}

const std::string AgentReportR::RESOURCE_ID = "navi_sched.agent_report_r";

void AgentReportR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

bool AgentReportR::config(navi::ResourceConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "server_ip", _serverInfo.ip, _serverInfo.ip);
    NAVI_JSONIZE(ctx, "server_port", _serverInfo.port, _serverInfo.port);
    bool latencyValidationEnabled =
        _latencyValidationEnabled.load(std::memory_order_relaxed);
    NAVI_JSONIZE(ctx, "latency_validation_enabled", latencyValidationEnabled,
                 latencyValidationEnabled);
    _latencyValidationEnabled.store(latencyValidationEnabled,
                                    std::memory_order_relaxed);
    NAVI_JSONIZE(ctx, "pd_separated", _pdSeparated, _pdSeparated);
    NAVI_JSONIZE(ctx, "unified_latency_prediction", _unifiedLatencyPrediction,
                 _unifiedLatencyPrediction);
    NAVI_JSONIZE(ctx, "workload_publish_timing_enabled",
                 _workloadPublishTimingEnabled,
                 _workloadPublishTimingEnabled);
    NAVI_JSONIZE(ctx, "latency_model_fingerprint", _latencyModelFingerprint,
                 _latencyModelFingerprint);
    return _serverInfo.port >= 0 && _latencyModelFingerprint.size() <= 4096;
}

navi::ErrorCode AgentReportR::init(navi::ResourceInitContext &)
{
    initLogger(_logger, "AgentReportR");
    _agentStartTimeUs = autil::TimeUtility::currentTime();
    if (_unifiedLatencyPrediction) {
        NaviSchedWorkloadSnapshot workload;
        workload.revision = 1;
        workload.eventTimeUs = autil::TimeUtility::currentTime();
        workload.prefillCapacitySplit = !_pdSeparated;
        std::atomic_store_explicit(
            &_workloadSnapshot,
            std::make_shared<const NaviSchedWorkloadSnapshot>(std::move(workload)),
            std::memory_order_release);
    }
    setHandshake(makeHandshakeUpdate());
    _agentReportBridgeR->setAgentReportR(
        std::dynamic_pointer_cast<AgentReportR>(shared_from_this()));
    return navi::EC_NONE;
}

AgentReportR::~AgentReportR() { finishAll(); }

NaviSchedServerInfo AgentReportR::getServerInfo() const
{
    autil::ScopedLock lock(_mutex);
    return _serverInfo;
}

void AgentReportR::setLatencyModelSnapshot(
    const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot)
{
    {
        autil::ScopedWriteLock lock(_latencySnapshotLock);
        _latencySnapshot = snapshot;
    }
    setHandshake(makeHandshakeUpdate());
}

std::shared_ptr<const NaviSchedLatencyModelSnapshot>
AgentReportR::getLatencyModelSnapshot() const
{
    autil::ScopedReadLock lock(_latencySnapshotLock);
    return _latencySnapshot;
}

AgentSchedulerConfigR::Snapshot AgentReportR::getSchedulerConfigSnapshot() const
{
    return _agentSchedulerConfigR ? _agentSchedulerConfigR->get()
                                  : AgentSchedulerConfigR::Snapshot();
}

bool AgentReportR::setInflightTracker(
    const std::shared_ptr<ConcurrentRequestLatencyTracker> &tracker)
{
    if (!tracker) {
        return false;
    }
    std::atomic_store_explicit(&_inflightTracker, tracker,
                               std::memory_order_release);
    return setInflightStats(tracker->snapshot());
}

bool AgentReportR::setInflightStats(
    const RequestLatencyTracker::Snapshot &snapshot)
{
    if (!snapshot.valid()) {
        return false;
    }
    auto next = std::make_shared<const RequestLatencyTracker::Snapshot>(snapshot);
    while (true) {
        auto current = std::atomic_load_explicit(
            &_inflightStats, std::memory_order_acquire);
        if (current && current->revision > snapshot.revision) {
            return true;
        }
        if (current && current->revision == snapshot.revision) {
            return current->requestCount == snapshot.requestCount &&
                   current->totalPredictedLatencyMs ==
                       snapshot.totalPredictedLatencyMs;
        }
        if (std::atomic_compare_exchange_weak_explicit(
                &_inflightStats, &current, next, std::memory_order_release,
                std::memory_order_acquire)) {
            return true;
        }
    }
}

RequestLatencyTracker::Snapshot AgentReportR::getInflightStats() const
{
    auto snapshot = std::atomic_load_explicit(
        &_inflightStats, std::memory_order_acquire);
    return snapshot ? *snapshot : RequestLatencyTracker::Snapshot();
}

bool AgentReportR::getLiveInflightStats(
    RequestLatencyTracker::Snapshot &snapshot) const
{
    auto tracker = std::atomic_load_explicit(
        &_inflightTracker, std::memory_order_acquire);
    if (!tracker) {
        return false;
    }
    snapshot = tracker->snapshot();
    return true;
}

size_t AgentReportR::reportLatencyModelUpdated(
    const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
    bool recomputeWaitingInflight)
{
    if (!snapshot || !snapshot->valid()) {
        return 0;
    }
    if (snapshot->onlineFitUpdate || recomputeWaitingInflight) {
        auto tracker = std::atomic_load_explicit(
            &_inflightTracker, std::memory_order_acquire);
        if (!tracker) {
            _inflightRecomputeFailureCount.fetch_add(1,
                                                     std::memory_order_relaxed);
            NAVI_KERNEL_LOG(ERROR,
                            "recompute inflight latency without tracker, "
                            "modelRevision [%ld]",
                            snapshot->revision);
            return 0;
        }
        RequestLatencyTracker::Snapshot inflight;
        size_t recomputedRequestCount = 0;
        const auto predictionTimeUs = autil::TimeUtility::currentTime();
        const auto residualConfig = getSchedulerConfigSnapshot()
                                        .config.residualRuntimeConfig();
        const auto predictor = [&snapshot, predictionTimeUs, residualConfig](
                                   const NaviSchedLatencyFeatures &features) {
            const auto base = NonLinearPrefillModelR::predict(
                features, snapshot->parameters);
            return LatencyResidualModel::evaluate(
                       base, snapshot->residualParameters,
                       snapshot->residualLastUpdateTimeUs,
                       predictionTimeUs, residualConfig)
                .predictionMs;
        };
        if (!tracker->recomputeWaiting(
                predictor, inflight, recomputedRequestCount) ||
            !setInflightStats(inflight)) {
            _inflightRecomputeFailureCount.fetch_add(1,
                                                     std::memory_order_relaxed);
            NAVI_KERNEL_LOG(ERROR,
                            "recompute inflight latency after model change failed, "
                            "modelRevision [%ld], inflightRevision [%lu], "
                            "inflightRequests [%d]",
                            snapshot->revision, inflight.revision,
                            inflight.requestCount);
            return 0;
        }
        _inflightRecomputeCount.fetch_add(1, std::memory_order_relaxed);
        _inflightRecomputedRequestCount.fetch_add(
            recomputedRequestCount, std::memory_order_relaxed);
    }
    setLatencyModelSnapshot(snapshot);
    auto update = makeUpdate(NaviSchedAgentEvent::LATENCY_MODEL_UPDATED, 0);
    attachInflightStats(update);
    update->latency = makeLatencyModelData(snapshot);
    update->latencyModelTrusted = snapshot->overloadPredictionTrusted;
    update->latencyModelWarmStarted = snapshot->warmStarted;
    return publish(update);
}

AgentReportR::QueuedPublishPtr AgentReportR::enqueueWorkloadLocked(
    NaviSchedWorkloadSnapshot &snapshot, AgentUpdateToMasterDataPtr update,
    std::vector<AgentUpdateToMasterDataPtr> trailingUpdates)
{
    const auto publishTrailing = [&]() -> QueuedPublishPtr {
        if (trailingUpdates.empty()) return nullptr;
        std::vector<SubscriberPtr> subscribers;
        subscribers.reserve(_subscribers.size());
        for (const auto &pair : _subscribers) subscribers.emplace_back(pair.second);
        return enqueuePublishLocked(std::move(trailingUpdates), std::move(subscribers), true);
    };
    const auto current = std::atomic_load_explicit(&_workloadSnapshot, std::memory_order_acquire);
    if (!update && current && snapshot.admittedBatch.empty() && current->sameWorkload(snapshot)) {
        snapshot.revision = current->revision;
        return publishTrailing();
    }
    if (_stopping || (current && current->revision == std::numeric_limits<uint64_t>::max())) {
        snapshot.revision = 0;
        return publishTrailing();
    }
    snapshot.revision = current ? current->revision + 1 : 1;
    if (!update) update = makeUpdate(NaviSchedAgentEvent::WORKLOAD_UPDATED, snapshot.eventTimeUs);
    update->workload = std::make_shared<NaviSchedWorkloadSnapshot>(snapshot);
    auto storedSnapshot = snapshot;
    storedSnapshot.admittedBatch = NaviSchedLocalBatchInfo();
    std::vector<SubscriberPtr> subscribers;
    subscribers.reserve(_subscribers.size());
    for (const auto &pair : _subscribers) subscribers.emplace_back(pair.second);
    std::atomic_store_explicit(&_workloadSnapshot,
        std::make_shared<const NaviSchedWorkloadSnapshot>(std::move(storedSnapshot)),
        std::memory_order_release);
    trailingUpdates.insert(trailingUpdates.begin(), std::move(update));
    auto task = enqueuePublishLocked(std::move(trailingUpdates), std::move(subscribers), true);
    if (!task) {
        snapshot.revision = 0;
        return nullptr;
    }
    return task;
}

void AgentReportR::finishWorkloadPublish(const QueuedPublishPtr &task) const
{
    if (!task) return;
    const auto beginUs = _workloadPublishTimingEnabled ? autil::TimeUtility::currentTime() : 0;
    drainPublishQueue();
    waitPublish(task);
    if (_workloadPublishTimingEnabled) {
        _workloadPublishTimeQueue.Push(autil::TimeUtility::currentTime() - beginUs);
    }
}

NaviSchedWorkloadSnapshot AgentReportR::observedWorkloadLocked() const
{
    auto snapshot = _observedWorkload;
    const auto nowUs = autil::TimeUtility::currentTime();
    if (snapshot.eventTimeUs > 0 && snapshot.runningRequestCount > 0) {
        snapshot.runningElapsedTimeMs += std::max<int64_t>(0, nowUs - snapshot.eventTimeUs) /
                                        1000.0 * snapshot.runningCapacityPrefillRequestCount;
    }
    snapshot.revision = 1;
    snapshot.eventTimeUs = nowUs;
    snapshot.prefillCapacitySplit = true;
    snapshot.admittedBatch = {};
    return snapshot;
}

uint64_t AgentReportR::reportWorkloadUpdated(
    const NaviSchedWorkloadSnapshot &snapshot, const std::vector<int64_t> &coveredNativeBatchIds)
{
    auto observed = snapshot;
    observed.revision = 1;
    if (observed.eventTimeUs <= 0) observed.eventTimeUs = autil::TimeUtility::currentTime();
    if (!observed.valid()) return 0;
    NaviSchedWorkloadSnapshot published = observed;
    QueuedPublishPtr task;
    AgentWorkloadOwnership::RetiredBatches retired;
    auto covered = coveredNativeBatchIds;
    if (observed.admittedBatch.valid()) covered.emplace_back(observed.admittedBatch.localBatchId);
    {
        autil::ScopedLock lock(_mutex);
        if (_stopping) return 0;
        if (!_pdSeparated && _unifiedLatencyPrediction) {
            if (!_nativeWorkload.combine(observed, published, covered) ||
                !_nativeWorkload.handoff(covered, retired)) return 0;
            _observedWorkload = observed;
            _observedWorkload.admittedBatch = {};
        }
        task = enqueueWorkloadLocked(published);
    }
    retired.clear();
    finishWorkloadPublish(task);
    return published.revision;
}

bool AgentReportR::reportNativeBatchReady(const NaviSchedBatchPtr &batch)
{
    if (_pdSeparated || !_unifiedLatencyPrediction) return true;
    if (!batch) return false;
    const auto prepared = AgentWorkloadOwnership::prepareBatch(AgentBatchR::getRequestStats(*batch));
    if (!prepared) return false;
    NaviSchedWorkloadSnapshot combined;
    QueuedPublishPtr task;
    AgentWorkloadOwnership::RetiredBatches retired;
    {
        autil::ScopedLock lock(_mutex);
        if (_stopping || !_nativeWorkload.addBatch(batch->batchId, prepared)) return false;
        if (!_nativeWorkload.combine(observedWorkloadLocked(), combined)) {
            _nativeWorkload.handoff({batch->batchId}, retired);
            return false;
        }
        combined.admittedBatch = batch->makeLocalBatchInfo();
        task = enqueueWorkloadLocked(combined);
    }
    finishWorkloadPublish(task);
    return combined.revision > 0;
}

bool AgentReportR::reportNativeRequestReceived(
    const NaviSchedRequestStats &request, int32_t batchWaitingRequests,
    int32_t queuedBatchRequests, int32_t runningRequests, int64_t eventTimeUs)
{
    const auto prepared = AgentWorkloadOwnership::prepareRequest(request);
    if (!prepared) return false;
    auto update = makeUpdate(NaviSchedAgentEvent::REQUEST_RECEIVED, eventTimeUs);
    attachInflightStats(update);
    update->requestEvent->requestId = request.requestId;
    update->requestEvent->batchWaitingRequests = batchWaitingRequests;
    update->requestEvent->queuedBatchRequests = queuedBatchRequests;
    update->requestEvent->runningRequests = runningRequests;
    NaviSchedWorkloadSnapshot combined;
    QueuedPublishPtr task;
    {
        autil::ScopedLock lock(_mutex);
        if (_stopping || !_nativeWorkload.addUnbatchedRequest(request.requestId, prepared)) return false;
        if (!_nativeWorkload.combine(observedWorkloadLocked(), combined)) {
            _nativeWorkload.removeUnbatchedRequests({request.requestId});
            return false;
        }
        task = enqueueWorkloadLocked(combined, std::move(update));
    }
    finishWorkloadPublish(task);
    if (combined.revision == 0) return false;
    _publishedPromptTokenCount.fetch_add(request.totalTokenCount, std::memory_order_relaxed);
    return true;
}

bool AgentReportR::reportNativeUnbatchedRequestsRemoved(const std::vector<uint64_t> &requestIds)
{
    if (_pdSeparated || !_unifiedLatencyPrediction) return true;
    NaviSchedWorkloadSnapshot combined;
    QueuedPublishPtr task;
    {
        autil::ScopedLock lock(_mutex);
        const auto before = _nativeWorkload.requestCount();
        if (!_nativeWorkload.removeUnbatchedRequests(requestIds)) return false;
        if (_nativeWorkload.requestCount() == before) return true;
        if (!_nativeWorkload.combine(observedWorkloadLocked(), combined)) return false;
        task = enqueueWorkloadLocked(combined);
    }
    finishWorkloadPublish(task);
    return combined.revision > 0;
}

bool AgentReportR::reportNativeRequestsRemoved(
    int64_t localBatchId, const std::vector<uint64_t> &requestIds)
{
    if (_pdSeparated || !_unifiedLatencyPrediction) return true;
    NaviSchedWorkloadSnapshot combined;
    QueuedPublishPtr task;
    {
        autil::ScopedLock lock(_mutex);
        if (!_nativeWorkload.hasBatch(localBatchId)) return true;
        if (!_nativeWorkload.removeRequests(localBatchId, requestIds) ||
            !_nativeWorkload.combine(observedWorkloadLocked(), combined)) return false;
        task = enqueueWorkloadLocked(combined);
    }
    finishWorkloadPublish(task);
    return combined.revision > 0;
}

bool AgentReportR::reportNativeBatchRemoved(int64_t localBatchId)
{
    if (_pdSeparated || !_unifiedLatencyPrediction) return true;
    NaviSchedWorkloadSnapshot combined;
    QueuedPublishPtr task;
    AgentWorkloadOwnership::RetiredBatches retired;
    {
        autil::ScopedLock lock(_mutex);
        if (!_nativeWorkload.hasBatch(localBatchId)) return true;
        if (!_nativeWorkload.handoff({localBatchId}, retired) ||
            !_nativeWorkload.combine(observedWorkloadLocked(), combined)) return false;
        task = enqueueWorkloadLocked(combined);
    }
    retired.clear();
    finishWorkloadPublish(task);
    return combined.revision > 0;
}

bool AgentReportR::reportNativeBatchCompleted(int64_t localBatchId, int64_t pendingBatchId)
{
    if (pendingBatchId <= 0) {
        reportNativeBatchRemoved(localBatchId);
        return true;
    }
    QueuedPublishPtr task;
    AgentWorkloadOwnership::RetiredBatches retired;
    bool pendingCompleted = true;
    bool workloadCompleted = true;
    {
        autil::ScopedLock cacheLock(_cacheReportLock);
        std::vector<AgentUpdateToMasterDataPtr> updates;
        if (pendingBatchId > 0) {
            AgentPendingCacheToMasterData pending;
            bool changed = false;
            pendingCompleted = _agentPendingCacheR->retire(pendingBatchId, pending, changed) && changed;
            if (pendingCompleted && changed) {
                auto update = makeUpdate(NaviSchedAgentEvent::PENDING_CACHE_UPDATED, 0);
                update->pendingCache = std::make_shared<AgentPendingCacheToMasterData>(std::move(pending));
                updates.emplace_back(std::move(update));
            }
        }
        autil::ScopedLock lock(_mutex);
        if (!_pdSeparated && _unifiedLatencyPrediction && _nativeWorkload.hasBatch(localBatchId)) {
            NaviSchedWorkloadSnapshot combined;
            workloadCompleted = _nativeWorkload.handoff({localBatchId}, retired) &&
                                _nativeWorkload.combine(observedWorkloadLocked(), combined);
            if (workloadCompleted) {
                task = enqueueWorkloadLocked(combined, {}, std::move(updates));
                workloadCompleted = combined.revision > 0;
            }
        }
        if (!updates.empty()) {
            std::vector<SubscriberPtr> subscribers;
            for (const auto &pair : _subscribers) subscribers.emplace_back(pair.second);
            task = enqueuePublishLocked(std::move(updates), std::move(subscribers), true);
        }
    }
    retired.clear();
    finishWorkloadPublish(task);
    if (!workloadCompleted) {
        NAVI_KERNEL_LOG(WARN, "complete native workload report failed, batchId [%ld]", localBatchId);
    }
    return pendingCompleted;
}

size_t AgentReportR::reportSchedulerConfigUpdated(
    uint64_t requestGeneration, uint64_t appliedGeneration,
    const std::string &configHash, const std::string &configJson,
    const std::string &error) const
{
    auto update = makeUpdate(NaviSchedAgentEvent::SCHEDULER_CONFIG_UPDATED, 0);
    update->schedulerConfigRequestGeneration = requestGeneration;
    update->schedulerConfigGeneration = appliedGeneration;
    update->schedulerConfigHash = configHash;
    update->schedulerConfigJson = configJson;
    update->schedulerConfigError = error;
    return publish(update);
}

size_t AgentReportR::reportRequestReceived(uint64_t requestId,
                                           int64_t promptTokenCount,
                                           int32_t inflightRequests,
                                           int32_t batchWaitingRequests,
                                           int32_t queuedBatchRequests,
                                           int32_t runningRequests,
                                           int64_t eventTimeUs) const
{
    if (promptTokenCount < 0 || inflightRequests < 0 || batchWaitingRequests < 0 ||
        queuedBatchRequests < 0 || runningRequests < 0) {
        return 0;
    }
    auto update = makeUpdate(NaviSchedAgentEvent::REQUEST_RECEIVED, eventTimeUs);
    attachInflightStats(update);
    update->requestEvent->requestId = requestId;
    update->requestEvent->batchWaitingRequests = batchWaitingRequests;
    update->requestEvent->queuedBatchRequests = queuedBatchRequests;
    update->requestEvent->runningRequests = runningRequests;
    const auto publishCount = publish(update);
    _publishedPromptTokenCount.fetch_add(
        static_cast<uint64_t>(promptTokenCount), std::memory_order_relaxed);
    return publishCount;
}

size_t AgentReportR::reportExecutionLoad(
    int32_t batchWaitingRequests, int32_t queuedBatchRequests, int32_t runningRequests,
    double queueTotalPredictedLatencyMs, double queueDecayHeadRemainingMs,
    uint64_t batchTimeoutCount, uint64_t abnormalBatchFinishCount,
    int64_t eventTimeUs, const NaviSchedLocalBatchInfo &localBatch) const
{
    if (batchWaitingRequests < 0 || queuedBatchRequests < 0 || runningRequests < 0 ||
        !std::isfinite(queueTotalPredictedLatencyMs) ||
        queueTotalPredictedLatencyMs < 0.0 ||
        !std::isfinite(queueDecayHeadRemainingMs) ||
        queueDecayHeadRemainingMs < 0.0 ||
        queueDecayHeadRemainingMs > queueTotalPredictedLatencyMs) {
        return 0;
    }
    auto update = makeUpdate(NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED, eventTimeUs);
    attachInflightStats(update);
    update->requestEvent->batchWaitingRequests = batchWaitingRequests;
    update->requestEvent->queuedBatchRequests = queuedBatchRequests;
    update->requestEvent->runningRequests = runningRequests;
    update->requestEvent->batchTimeoutCount = batchTimeoutCount;
    update->requestEvent->abnormalBatchFinishCount = abnormalBatchFinishCount;
    update->requestEvent->queueTotalPredictedLatencyMs = queueTotalPredictedLatencyMs;
    update->requestEvent->queueDecayHeadRemainingMs =
        queueDecayHeadRemainingMs;
    update->requestEvent->localBatch = localBatch;
    return publish(update);
}

size_t AgentReportR::reportBatchFinished(
    int32_t inflightRequests, int32_t batchWaitingRequests, int32_t queuedBatchRequests,
    int32_t runningRequests, double latencyMs, double queueTotalPredictedLatencyMs,
    double queueDecayHeadRemainingMs,
    const NaviSchedCacheUpdate &cacheUpdate,
    int32_t completedRequestCount,
    int64_t cacheHitTokens, int64_t cacheMissTokens, uint64_t batchTimeoutCount,
    uint64_t abnormalBatchFinishCount, bool abnormalBatchFinished,
    int64_t eventTimeUs, const NaviSchedLocalBatchInfo &localBatch,
    double agentPredictedExecutionLatencyMs, double backendQueueLatencyMs,
    const NaviSchedLatencyValidationStats &validationStats) const
{
    if (inflightRequests < 0 || batchWaitingRequests < 0 || queuedBatchRequests < 0 ||
        runningRequests < 0 || latencyMs < 0.0 ||
        !std::isfinite(queueTotalPredictedLatencyMs) ||
        queueTotalPredictedLatencyMs < 0.0 ||
        !std::isfinite(queueDecayHeadRemainingMs) ||
        queueDecayHeadRemainingMs < 0.0 ||
        queueDecayHeadRemainingMs > queueTotalPredictedLatencyMs ||
        completedRequestCount <= 0 || cacheHitTokens < 0 || cacheMissTokens < 0) {
        return 0;
    }
    const auto latencyValidationEnabled =
        _latencyValidationEnabled.load(std::memory_order_relaxed);
    const bool latencyValidationValid =
        latencyValidationEnabled && localBatch.valid() &&
        std::isfinite(agentPredictedExecutionLatencyMs) &&
        agentPredictedExecutionLatencyMs >= 0.0 &&
        std::isfinite(backendQueueLatencyMs) && backendQueueLatencyMs >= 0.0;
    if (latencyValidationEnabled && !latencyValidationValid) {
        _suppressedLatencyValidationBatchCount.fetch_add(
            1, std::memory_order_relaxed);
    }
    auto update = makeUpdate(NaviSchedAgentEvent::BATCH_FINISHED, eventTimeUs);
    attachInflightStats(update);
    auto &requestEvent = *update->requestEvent;
    requestEvent.batchWaitingRequests = batchWaitingRequests;
    requestEvent.queuedBatchRequests = queuedBatchRequests;
    requestEvent.runningRequests = runningRequests;
    requestEvent.completedRequestCount = completedRequestCount;
    requestEvent.cacheHitTokens = cacheHitTokens;
    requestEvent.cacheMissTokens = cacheMissTokens;
    requestEvent.batchTimeoutCount = batchTimeoutCount;
    requestEvent.abnormalBatchFinishCount = abnormalBatchFinishCount;
    requestEvent.abnormalBatchFinished = abnormalBatchFinished;
    requestEvent.latencyMs = latencyMs;
    requestEvent.queueTotalPredictedLatencyMs = queueTotalPredictedLatencyMs;
    requestEvent.queueDecayHeadRemainingMs = queueDecayHeadRemainingMs;
    requestEvent.localBatch = localBatch;
    if (latencyValidationValid) {
        requestEvent.latencyValidation = true;
        requestEvent.agentPredictedExecutionLatencyMs =
            agentPredictedExecutionLatencyMs;
        requestEvent.backendQueueLatencyMs = backendQueueLatencyMs;
        requestEvent.latencyValidationStats = validationStats;
        autil::ScopedWriteLock lock(_latencyValidationStatsLock);
        const auto nowTimeUs = autil::TimeUtility::currentTime();
        if (validationStats.sampleCount > 0 &&
            validationStats.sampleCount != _latencyValidationStats.sampleCount) {
            _latencyValidationLastUpdateTimeUs = nowTimeUs;
        }
        _latencyValidationBucketLastUpdateTimeUs.resize(
            validationStats.executionLatencyBuckets.size(), 0);
        for (size_t bucketId = 0;
             bucketId < validationStats.executionLatencyBuckets.size(); ++bucketId) {
            const auto previousCount =
                bucketId < _latencyValidationStats.executionLatencyBuckets.size()
                    ? _latencyValidationStats.executionLatencyBuckets[bucketId].sampleCount
                    : 0;
            const auto currentCount =
                validationStats.executionLatencyBuckets[bucketId].sampleCount;
            if (currentCount > 0 && currentCount != previousCount) {
                _latencyValidationBucketLastUpdateTimeUs[bucketId] = nowTimeUs;
            }
        }
        _latencyValidationStats = validationStats;
    }
    if (cacheUpdate.totalBlocks >= 0 ||
        cacheUpdate.blockSize >= 0 || cacheUpdate.linearStep >= 0 ||
        BlockHashUtil::size(cacheUpdate.addedBlockHashes) > 0 ||
        BlockHashUtil::size(cacheUpdate.removedBlockHashes) > 0 ||
        BlockHashUtil::size(cacheUpdate.addedFullReadyBlockHashes) > 0 ||
        BlockHashUtil::size(cacheUpdate.removedFullReadyBlockHashes) > 0 ||
        BlockHashUtil::size(cacheUpdate.addedTailReadyBlockHashes) > 0 ||
        BlockHashUtil::size(cacheUpdate.removedTailReadyBlockHashes) > 0) {
        update->cache = std::make_shared<AgentCacheToMasterData>();
        update->cache->totalBlocks = cacheUpdate.totalBlocks;
        update->cache->fullSnapshot = cacheUpdate.fullSnapshot;
        update->cache->addedBlockHashes = cacheUpdate.addedBlockHashes;
        update->cache->removedBlockHashes = cacheUpdate.removedBlockHashes;
        update->cache->addedFullReadyBlockHashes = cacheUpdate.addedFullReadyBlockHashes;
        update->cache->removedFullReadyBlockHashes = cacheUpdate.removedFullReadyBlockHashes;
        update->cache->addedTailReadyBlockHashes = cacheUpdate.addedTailReadyBlockHashes;
        update->cache->removedTailReadyBlockHashes = cacheUpdate.removedTailReadyBlockHashes;
        update->cache->blockSize = cacheUpdate.blockSize;
        update->cache->linearStep = cacheUpdate.linearStep;
        update->cache->hybridTailRequired = cacheUpdate.hybridTailRequired;
    }
    return publish(update);
}

bool AgentReportR::latencyValidationEnabled() const
{
    return _latencyValidationEnabled.load(std::memory_order_relaxed);
}

void AgentReportR::setLatencyValidationEnabled(bool enabled)
{
    _latencyValidationEnabled.store(enabled, std::memory_order_relaxed);
}

NaviSchedLatencyValidationStats AgentReportR::getLatencyValidationStats() const
{
    autil::ScopedReadLock lock(_latencyValidationStatsLock);
    return _latencyValidationStats;
}

AgentReportR::LatencyValidationDiagnostics
AgentReportR::getLatencyValidationDiagnostics() const
{
    autil::ScopedReadLock lock(_latencyValidationStatsLock);
    LatencyValidationDiagnostics diagnostics;
    diagnostics.stats = _latencyValidationStats;
    diagnostics.lastUpdateTimeUs = _latencyValidationLastUpdateTimeUs;
    diagnostics.bucketLastUpdateTimeUs =
        _latencyValidationBucketLastUpdateTimeUs;
    return diagnostics;
}

NaviSchedWorkloadSnapshotConstPtr AgentReportR::getWorkloadSnapshot() const
{
    return std::atomic_load_explicit(&_workloadSnapshot,
                                     std::memory_order_acquire);
}

AgentReportR::WorkloadPublishWindowStats
AgentReportR::takeWorkloadPublishWindowStats() const
{
    WorkloadPublishWindowStats stats;
    int64_t publishTimeUs = 0;
    while (_workloadPublishTimeQueue.Pop(&publishTimeUs)) {
        ++stats.sampleCount;
        stats.totalTimeUs += publishTimeUs;
        stats.maxTimeUs = std::max(stats.maxTimeUs, publishTimeUs);
    }
    return stats;
}

size_t AgentReportR::reportCacheUpdated(const NaviSchedCacheUpdate &update) const
{
    QueuedPublishPtr cachePublish;
    {
        autil::ScopedLock cacheReportLock(_cacheReportLock);
        AgentCacheR::State state;
        if (!_agentCacheR->apply(update, state)) {
            return 0;
        }
        auto report = makeUpdate(NaviSchedAgentEvent::CACHE_UPDATED,
                                 update.eventTimeUs);
        report->cache = std::make_shared<AgentCacheToMasterData>();
        report->cache->totalBlocks = update.totalBlocks;
        report->cache->fullSnapshot = update.fullSnapshot;
        report->cache->addedBlockHashes = update.addedBlockHashes;
        report->cache->removedBlockHashes = update.removedBlockHashes;
        report->cache->addedFullReadyBlockHashes = update.addedFullReadyBlockHashes;
        report->cache->removedFullReadyBlockHashes = update.removedFullReadyBlockHashes;
        report->cache->addedTailReadyBlockHashes = update.addedTailReadyBlockHashes;
        report->cache->removedTailReadyBlockHashes = update.removedTailReadyBlockHashes;
        report->cache->blockSize = update.blockSize;
        report->cache->linearStep = update.linearStep;
        report->cache->hybridTailRequired = update.hybridTailRequired;
        report->cache->revision = state.revision;
        report->cache->cachedBlockCount = state.cachedBlockCount;
        report->cache->hashChecksum = state.hashChecksum;
        report->cache->fullReadyBlockCount = state.fullReadyBlockCount;
        report->cache->fullReadyHashChecksum = state.fullReadyHashChecksum;
        report->cache->tailReadyBlockCount = state.tailReadyBlockCount;
        report->cache->tailReadyHashChecksum = state.tailReadyHashChecksum;
        report->cache->rtpCachedBlockCount = state.rtpCachedBlockCount;
        cachePublish = enqueuePublish(report);
    }
    drainPublishQueue();
    return waitPublish(cachePublish);
}

bool AgentReportR::reportPendingCacheReady(
    const NaviSchedPendingCacheReservation &reservation) const
{
    QueuedPublishPtr publishTask;
    {
        autil::ScopedLock cacheReportLock(_cacheReportLock);
        AgentPendingCacheToMasterData pendingUpdate;
        bool changed = false;
        if (!_agentPendingCacheR->reserve(reservation, pendingUpdate, changed)) {
            return false;
        }
        if (!changed) {
            return true;
        }
        auto report = makeUpdate(NaviSchedAgentEvent::PENDING_CACHE_UPDATED, 0);
        report->pendingCache =
            std::make_shared<AgentPendingCacheToMasterData>(std::move(pendingUpdate));
        publishTask = enqueuePublish(report);
    }
    drainPublishQueue();
    waitPublish(publishTask);
    return true;
}

bool AgentReportR::reportPendingCacheCompleted(int64_t localBatchId) const
{
    QueuedPublishPtr publishTask;
    {
        autil::ScopedLock cacheReportLock(_cacheReportLock);
        NaviSchedPendingCacheReservation reservation;
        if (!_agentPendingCacheR->getReservation(localBatchId, reservation)) {
            return false;
        }
        if (!retirePendingCacheLocked(localBatchId, &publishTask)) {
            return false;
        }
    }
    drainPublishQueue();
    waitPublish(publishTask);
    return true;
}

bool AgentReportR::reportPendingCacheAborted(int64_t localBatchId) const
{
    QueuedPublishPtr publishTask;
    {
        autil::ScopedLock cacheReportLock(_cacheReportLock);
        if (!retirePendingCacheLocked(localBatchId, &publishTask)) {
            return false;
        }
    }
    drainPublishQueue();
    waitPublish(publishTask);
    return true;
}

bool AgentReportR::replacePendingCache(
    int64_t localBatchId,
    const NaviSchedPendingCacheReservation *remainingReservation) const
{
    if (localBatchId <= 0 ||
        (remainingReservation &&
         remainingReservation->batchId != localBatchId)) {
        return false;
    }
    QueuedPublishPtr publishTask;
    {
        autil::ScopedLock cacheReportLock(_cacheReportLock);
        AgentPendingCacheToMasterData pendingUpdate;
        bool changed = false;
        const auto updated = remainingReservation
                                 ? _agentPendingCacheR->retain(
                                       *remainingReservation, pendingUpdate, changed)
                                 : _agentPendingCacheR->retire(
                                       localBatchId, pendingUpdate, changed);
        if (!updated) {
            return false;
        }
        if (!changed) {
            return true;
        }
        auto report = makeUpdate(NaviSchedAgentEvent::PENDING_CACHE_UPDATED, 0);
        report->pendingCache =
            std::make_shared<AgentPendingCacheToMasterData>(std::move(pendingUpdate));
        publishTask = enqueuePublish(report);
    }
    drainPublishQueue();
    waitPublish(publishTask);
    return true;
}

size_t AgentReportR::reportBatchRequestsAborted(
    int32_t inflightRequests, int32_t batchWaitingRequests,
    int32_t queuedBatchRequests, int32_t runningRequests,
    double queueTotalPredictedLatencyMs, double queueDecayHeadRemainingMs,
    uint64_t batchTimeoutCount, uint64_t abnormalBatchFinishCount,
    int32_t abortedRequestCount, int64_t eventTimeUs,
    const NaviSchedLocalBatchInfo &localBatch) const
{
    if (inflightRequests < 0 || batchWaitingRequests < 0 ||
        queuedBatchRequests < 0 || runningRequests < 0 ||
        !std::isfinite(queueTotalPredictedLatencyMs) ||
        queueTotalPredictedLatencyMs < 0.0 ||
        !std::isfinite(queueDecayHeadRemainingMs) ||
        queueDecayHeadRemainingMs < 0.0 ||
        queueDecayHeadRemainingMs > queueTotalPredictedLatencyMs ||
        abortedRequestCount <= 0 || !localBatch.valid() ||
        localBatch.requestIds.empty() ||
        localBatch.requestCount != abortedRequestCount) {
        return 0;
    }
    auto update =
        makeUpdate(NaviSchedAgentEvent::BATCH_REQUESTS_ABORTED, eventTimeUs);
    attachInflightStats(update);
    auto &requestEvent = *update->requestEvent;
    requestEvent.batchWaitingRequests = batchWaitingRequests;
    requestEvent.queuedBatchRequests = queuedBatchRequests;
    requestEvent.runningRequests = runningRequests;
    requestEvent.completedRequestCount = abortedRequestCount;
    requestEvent.batchTimeoutCount = batchTimeoutCount;
    requestEvent.abnormalBatchFinishCount = abnormalBatchFinishCount;
    requestEvent.abnormalBatchFinished = true;
    requestEvent.queueTotalPredictedLatencyMs = queueTotalPredictedLatencyMs;
    requestEvent.queueDecayHeadRemainingMs = queueDecayHeadRemainingMs;
    requestEvent.localBatch = localBatch;
    return publish(update);
}

bool AgentReportR::reportPendingCacheSnapshot(
    const std::string &collectorId) const
{
    if (collectorId.empty()) {
        return false;
    }
    QueuedPublishPtr publishTask;
    {
        autil::ScopedLock cacheReportLock(_cacheReportLock);
        autil::ScopedLock lock(_mutex);
        auto it = _subscribers.find(collectorId);
        if (_subscribers.end() == it || it->second->asyncPipe.expired()) {
            return false;
        }
        auto report = makeUpdate(NaviSchedAgentEvent::PENDING_CACHE_UPDATED, 0);
        report->pendingCache =
            std::make_shared<AgentPendingCacheToMasterData>(
                _agentPendingCacheR->makeSnapshot());
        publishTask = enqueuePublishLocked({std::move(report)}, {it->second}, true);
    }
    drainPublishQueue();
    return 1 == waitPublish(publishTask);
}

bool AgentReportR::retirePendingCacheLocked(
    int64_t localBatchId, QueuedPublishPtr *publishTask) const
{
    AgentPendingCacheToMasterData pendingUpdate;
    bool changed = false;
    if (!_agentPendingCacheR->retire(localBatchId, pendingUpdate, changed)) {
        return false;
    }
    if (!changed) {
        return true;
    }
    auto report = makeUpdate(NaviSchedAgentEvent::PENDING_CACHE_UPDATED, 0);
    report->pendingCache =
        std::make_shared<AgentPendingCacheToMasterData>(std::move(pendingUpdate));
    auto task = enqueuePublish(report);
    if (publishTask) {
        *publishTask = std::move(task);
    }
    return true;
}

NaviSchedRequestStats AgentReportR::makeRequestStats(
    const NaviSchedBatchRequest &request) const
{
    return makeRequestAudit(request).stats;
}

AgentCacheR::RequestAudit AgentReportR::makeRequestAudit(
    const NaviSchedBatchRequest &request) const
{
    return _agentCacheR->makeRequestAudit(request, _agentPendingCacheR);
}

size_t AgentReportR::beginStop(int64_t eventTimeUs)
{
    auto update = makeUpdate(NaviSchedAgentEvent::AGENT_STOPPING, eventTimeUs);
    update->control = std::make_shared<AgentControlToMasterData>();
    QueuedPublishPtr publishTask;
    {
        autil::ScopedLock lock(_mutex);
        if (_stopping) {
            return 0;
        }
        _stopping = true;
        std::vector<SubscriberPtr> subscribers;
        subscribers.reserve(_subscribers.size());
        for (auto it = _subscribers.begin(); it != _subscribers.end();) {
            if (it->second->asyncPipe.expired()) {
                it = _subscribers.erase(it);
            } else {
                _pendingStopAcks.emplace(it->first);
                subscribers.emplace_back(it->second);
                ++it;
            }
        }
        publishTask = enqueuePublishLocked(
            {std::move(update)}, std::move(subscribers), true, true);
    }
    drainPublishQueue();
    auto count = waitPublish(publishTask);
    if (0 == count) {
        autil::ScopedLock lock(_mutex);
        _pendingStopAcks.clear();
        _mutex.broadcast();
    }
    return count;
}

bool AgentReportR::waitForStopAcks(int64_t timeoutUs)
{
    autil::ScopedLock lock(_mutex);
    if (!_stopping || _pendingStopAcks.empty()) {
        return true;
    }
    auto deadlineUs = autil::TimeUtility::currentTime() + timeoutUs;
    while (!_pendingStopAcks.empty()) {
        auto remainingUs = deadlineUs - autil::TimeUtility::currentTime();
        if (remainingUs <= 0 || ETIMEDOUT == _mutex.wait(remainingUs)) {
            return false;
        }
    }
    return true;
}

bool AgentReportR::confirmStop(const std::string &collectorId)
{
    autil::ScopedLock lock(_mutex);
    if (!_stopping || 0 == _pendingStopAcks.erase(collectorId)) {
        return false;
    }
    if (_pendingStopAcks.empty()) {
        _mutex.broadcast();
    }
    return true;
}

bool AgentReportR::setHandshake(const AgentUpdateToMasterDataPtr &handshake)
{
    if (!handshake || NaviSchedAgentEvent::AGENT_HANDSHAKE != handshake->event ||
        !handshake->control) {
        return false;
    }
    autil::ScopedLock lock(_mutex);
    if (_stopping) {
        return false;
    }
    _handshake = handshake;
    return true;
}

bool AgentReportR::subscribe(const std::string &collectorId,
                             const navi::AsyncPipePtr &asyncPipe, bool orderedReports)
{
    if (collectorId.empty() || !asyncPipe) {
        return false;
    }
    SubscriberPtr subscriber;
    QueuedPublishPtr publishTask;
    {
        autil::ScopedLock cacheReportLock(_cacheReportLock);
        autil::ScopedLock lock(_mutex);
        if (_stopping || !_handshake) {
            return false;
        }
        auto it = _subscribers.find(collectorId);
        if (_subscribers.end() != it && !it->second->asyncPipe.expired()) {
            return false;
        }
        auto handshake = makeUpdate(NaviSchedAgentEvent::AGENT_HANDSHAKE, 0);
        handshake->control = _handshake->control;
        attachInflightStats(handshake);
        handshake->latency = _handshake->latency;
        handshake->latencyModelFingerprint =
            _handshake->latencyModelFingerprint;
        handshake->latencyModelTrusted = _handshake->latencyModelTrusted;
        handshake->latencyModelWarmStarted =
            _handshake->latencyModelWarmStarted;
        handshake->schedulerConfigGeneration =
            _handshake->schedulerConfigGeneration;
        handshake->schedulerConfigHash = _handshake->schedulerConfigHash;
        handshake->schedulerConfigJson = _handshake->schedulerConfigJson;
        auto workload = getWorkloadSnapshot();
        if (workload) {
            handshake->workload =
                std::make_shared<NaviSchedWorkloadSnapshot>(*workload);
        }
        auto cacheSnapshot = _agentCacheR->makeSnapshot();
        auto cacheUpdate =
            makeUpdate(NaviSchedAgentEvent::CACHE_UPDATED, cacheSnapshot.eventTimeUs);
        cacheUpdate->cache = std::make_shared<AgentCacheToMasterData>();
        cacheUpdate->cache->totalBlocks = cacheSnapshot.totalBlocks;
        cacheUpdate->cache->fullSnapshot = true;
        cacheUpdate->cache->addedBlockHashes = cacheSnapshot.addedBlockHashes;
        cacheUpdate->cache->addedFullReadyBlockHashes =
            cacheSnapshot.addedFullReadyBlockHashes;
        cacheUpdate->cache->addedTailReadyBlockHashes =
            cacheSnapshot.addedTailReadyBlockHashes;
        cacheUpdate->cache->blockSize = cacheSnapshot.blockSize;
        cacheUpdate->cache->linearStep = cacheSnapshot.linearStep;
        cacheUpdate->cache->hybridTailRequired = cacheSnapshot.hybridTailRequired;
        cacheUpdate->cache->revision = cacheSnapshot.revision;
        cacheUpdate->cache->cachedBlockCount = cacheSnapshot.cachedBlockCount;
        cacheUpdate->cache->hashChecksum = cacheSnapshot.hashChecksum;
        cacheUpdate->cache->fullReadyBlockCount = cacheSnapshot.fullReadyBlockCount;
        cacheUpdate->cache->fullReadyHashChecksum =
            cacheSnapshot.fullReadyHashChecksum;
        cacheUpdate->cache->tailReadyBlockCount = cacheSnapshot.tailReadyBlockCount;
        cacheUpdate->cache->tailReadyHashChecksum =
            cacheSnapshot.tailReadyHashChecksum;
        cacheUpdate->cache->rtpCachedBlockCount = cacheSnapshot.rtpCachedBlockCount;
        auto pendingUpdate =
            makeUpdate(NaviSchedAgentEvent::PENDING_CACHE_UPDATED, 0);
        pendingUpdate->pendingCache =
            std::make_shared<AgentPendingCacheToMasterData>(
                _agentPendingCacheR->makeSnapshot());
        subscriber = std::make_shared<Subscriber>();
        subscriber->collectorId = collectorId;
        subscriber->asyncPipe = asyncPipe;
        subscriber->orderedReports = orderedReports;
        publishTask = enqueuePublishLocked(
            {std::move(handshake), std::move(cacheUpdate), std::move(pendingUpdate)},
            {subscriber}, false);
        _subscribers[collectorId] = subscriber;
    }
    drainPublishQueue();
    if (1 == waitPublish(publishTask)) {
        return true;
    }
    removeSubscriber(subscriber);
    return false;
}

bool AgentReportR::unsubscribe(const std::string &collectorId,
                               const navi::AsyncPipePtr &asyncPipe)
{
    autil::ScopedLock lock(_mutex);
    auto it = _subscribers.find(collectorId);
    if (_subscribers.end() == it) {
        return false;
    }
    auto current = it->second->asyncPipe.lock();
    if (current && current != asyncPipe) {
        return false;
    }
    return removeSubscriberLocked(collectorId);
}

size_t AgentReportR::publish(const AgentUpdateToMasterDataPtr &update) const
{
    if (!update || !update->valid()) {
        return 0;
    }
    const bool trackRequestTiming =
        NaviSchedAgentEvent::REQUEST_RECEIVED == update->event;
    const bool trackWorkloadTiming =
        _workloadPublishTimingEnabled &&
        NaviSchedAgentEvent::WORKLOAD_UPDATED == update->event;
    const bool trackPublishTiming = trackRequestTiming || trackWorkloadTiming;
    const auto publishBeginTimeUs = trackPublishTiming
                                        ? autil::TimeUtility::currentTime()
                                        : 0;
    auto publishTask = enqueuePublish(update);
    if (!publishTask) {
        return 0;
    }
    const auto subscribersReadyTimeUs = trackRequestTiming
                                            ? autil::TimeUtility::currentTime()
                                            : 0;
    drainPublishQueue();
    const auto publishCount = waitPublish(publishTask);
    const auto &subscribers = publishTask->subscribers;
    if (trackPublishTiming) {
        const auto publishDoneTimeUs = autil::TimeUtility::currentTime();
        const auto publishTimeUs = publishDoneTimeUs - publishBeginTimeUs;
        if (trackWorkloadTiming) {
            _workloadPublishTimeQueue.Push(publishTimeUs);
        }
        if (!trackRequestTiming) {
            return publishCount;
        }
        constexpr int64_t SLOW_REPORT_THRESHOLD_US = 50 * 1000;
        if (publishTimeUs >= SLOW_REPORT_THRESHOLD_US) {
            NAVI_LOG(WARN,
                     "[navi-request-report-slow] request_id=%lu subscribers=%zu "
                     "subscriber_snapshot_us=%ld max_set_data_us=%ld total_us=%ld",
                     update->requestEvent ? update->requestEvent->requestId : 0,
                     subscribers.size(),
                     subscribersReadyTimeUs - publishBeginTimeUs,
                     publishTask->maxSetDataTimeUs, publishTimeUs);
        }
    }
    return publishCount;
}

void AgentReportR::recordPublishedEvent(
    const AgentUpdateToMasterDataPtr &update, bool noSubscribers) const
{
    _publishedEventCount.fetch_add(1, std::memory_order_relaxed);
    if (NaviSchedAgentEvent::REQUEST_RECEIVED == update->event) {
        _publishedRequestReceivedCount.fetch_add(1, std::memory_order_relaxed);
    } else if ((NaviSchedAgentEvent::BATCH_FINISHED == update->event ||
                NaviSchedAgentEvent::BATCH_REQUESTS_ABORTED == update->event) &&
               update->requestEvent) {
        if (NaviSchedAgentEvent::BATCH_FINISHED == update->event) {
            _publishedFinishedBatchCount.fetch_add(1, std::memory_order_relaxed);
        }
        if (NaviSchedAgentEvent::BATCH_FINISHED == update->event &&
            update->requestEvent->abnormalBatchFinished) {
            _publishedAbnormalFinishedBatchCount.fetch_add(
                1, std::memory_order_relaxed);
        }
        auto &counter = update->requestEvent->abnormalBatchFinished
                            ? _publishedAbnormalFinishedRequestCount
                            : _publishedNormalFinishedRequestCount;
        counter.fetch_add(update->requestEvent->completedRequestCount,
                          std::memory_order_relaxed);
    }
    if (update->workload && NaviSchedAgentEvent::AGENT_HANDSHAKE != update->event) {
        _publishedWorkloadUpdateCount.fetch_add(1, std::memory_order_relaxed);
    }
    if (noSubscribers) {
        _noSubscriberEventCount.fetch_add(1, std::memory_order_relaxed);
    }
}

AgentReportR::Stats AgentReportR::getStats() const
{
    Stats stats;
    stats.publishedEventCount = _publishedEventCount.load(std::memory_order_relaxed);
    stats.publishedRequestReceivedCount =
        _publishedRequestReceivedCount.load(std::memory_order_relaxed);
    stats.publishedPromptTokenCount =
        _publishedPromptTokenCount.load(std::memory_order_relaxed);
    stats.publishedFinishedBatchCount =
        _publishedFinishedBatchCount.load(std::memory_order_relaxed);
    stats.publishedAbnormalFinishedBatchCount =
        _publishedAbnormalFinishedBatchCount.load(std::memory_order_relaxed);
    stats.publishedNormalFinishedRequestCount =
        _publishedNormalFinishedRequestCount.load(std::memory_order_relaxed);
    stats.publishedAbnormalFinishedRequestCount =
        _publishedAbnormalFinishedRequestCount.load(std::memory_order_relaxed);
    stats.deliveredMessageCount = _deliveredMessageCount.load(std::memory_order_relaxed);
    stats.combinedMessageSavings = _combinedMessageSavings.load(std::memory_order_relaxed);
    stats.deliverySuccessCount = _deliverySuccessCount.load(std::memory_order_relaxed);
    stats.deliveryFailureCount = _deliveryFailureCount.load(std::memory_order_relaxed);
    stats.noSubscriberEventCount =
        _noSubscriberEventCount.load(std::memory_order_relaxed);
    stats.publishedWorkloadUpdateCount =
        _publishedWorkloadUpdateCount.load(std::memory_order_relaxed);
    stats.suppressedLatencyValidationBatchCount =
        _suppressedLatencyValidationBatchCount.load(std::memory_order_relaxed);
    return stats;
}

bool AgentReportR::finish(const std::string &collectorId)
{
    navi::AsyncPipePtr asyncPipe;
    {
        autil::ScopedLock lock(_mutex);
        auto it = _subscribers.find(collectorId);
        if (_subscribers.end() == it) {
            return false;
        }
        asyncPipe = it->second->asyncPipe.lock();
        removeSubscriberLocked(collectorId);
    }
    if (asyncPipe) {
        asyncPipe->setEof();
    }
    return true;
}

AgentReportR::WorkloadOwnershipStats AgentReportR::getWorkloadOwnershipStats() const
{
    autil::ScopedLock lock(_mutex);
    return {_nativeWorkload.batchCount(), _nativeWorkload.requestCount(),
            _nativeWorkload.unbatchedRequestCount(), _observedWorkload.eventTimeUs};
}

void AgentReportR::finishAll()
{
    std::map<std::string, SubscriberPtr> subscribers;
    {
        autil::ScopedLock lock(_mutex);
        subscribers.swap(_subscribers);
        _pendingStopAcks.clear();
        _mutex.broadcast();
    }
    for (const auto &pair : subscribers) {
        auto asyncPipe = pair.second->asyncPipe.lock();
        if (asyncPipe) {
            asyncPipe->setEof();
        }
    }
}

size_t AgentReportR::getSubscriberCount() const
{
    autil::ScopedLock lock(_mutex);
    size_t count = 0;
    for (const auto &pair : _subscribers) {
        if (!pair.second->asyncPipe.expired()) {
            ++count;
        }
    }
    return count;
}

size_t AgentReportR::getCachedBlockCount() const
{
    return _agentCacheR->getCachedBlockCount();
}

AgentCacheR::State AgentReportR::getCacheState() const
{
    return _agentCacheR->getState();
}

AgentPendingCacheR::State AgentReportR::getPendingCacheState() const
{
    return _agentPendingCacheR->getState();
}

AgentUpdateToMasterDataPtr AgentReportR::makeUpdate(NaviSchedAgentEvent::Type event,
                                                    int64_t eventTimeUs) const
{
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = event;
    update->eventTimeUs =
        eventTimeUs > 0 ? eventTimeUs : autil::TimeUtility::currentTime();
    if (const auto source = _executionUtilizationSource.load(std::memory_order_acquire)) {
        update->executionUtilization = source->load(std::memory_order_relaxed);
    }
    if (const auto source = _executionBusyUtilizationSource.load(std::memory_order_acquire)) {
        update->executionBusyUtilization = source->load(std::memory_order_relaxed);
    }
    return update;
}

bool AgentReportR::setExecutionUtilizationSource(
    const std::shared_ptr<const std::atomic<double>> &source,
    const std::shared_ptr<const std::atomic<double>> &busySource)
{
    if (!source) return false;
    autil::ScopedLock lock(_mutex);
    if (_executionUtilizationOwner && _executionUtilizationOwner != source) return false;
    if (_executionBusyUtilizationOwner && _executionBusyUtilizationOwner != busySource) return false;
    _executionBusyUtilizationOwner = busySource;
    _executionBusyUtilizationSource.store(busySource.get(), std::memory_order_release);
    _executionUtilizationOwner = source;
    _executionUtilizationSource.store(source.get(), std::memory_order_release);
    return true;
}

void AgentReportR::attachInflightStats(
    const AgentUpdateToMasterDataPtr &update) const
{
    if (!update) {
        return;
    }
    const auto snapshot = getInflightStats();
    update->requestEvent = std::make_shared<AgentRequestEventToMasterData>();
    update->requestEvent->inflightStatsRevision = snapshot.revision;
    update->requestEvent->inflightRequests = snapshot.requestCount;
    update->requestEvent->inflightPredictedLatencySumMs =
        snapshot.totalPredictedLatencyMs;
}

AgentUpdateToMasterDataPtr AgentReportR::makeHandshakeUpdate() const
{
    auto update = makeUpdate(NaviSchedAgentEvent::AGENT_HANDSHAKE, 0);
    attachInflightStats(update);
    update->control = std::make_shared<AgentControlToMasterData>();
    update->control->serverInfo = getServerInfo();
    update->control->agentStartTimeUs = _agentStartTimeUs;
    update->control->pdSeparated = _pdSeparated;
    update->control->unifiedLatencyPrediction = _unifiedLatencyPrediction;
    update->latencyModelFingerprint = _latencyModelFingerprint;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> snapshot;
    {
        autil::ScopedReadLock lock(_latencySnapshotLock);
        snapshot = _latencySnapshot;
    }
    if (snapshot) {
        update->latency = makeLatencyModelData(snapshot);
        update->latencyModelTrusted = snapshot->overloadPredictionTrusted;
        update->latencyModelWarmStarted = snapshot->warmStarted;
    }
    auto workload = getWorkloadSnapshot();
    if (workload) {
        update->workload =
            std::make_shared<NaviSchedWorkloadSnapshot>(*workload);
    }
    if (_agentSchedulerConfigR) {
        const auto config = _agentSchedulerConfigR->get();
        update->schedulerConfigGeneration = config.generation;
        update->schedulerConfigHash =
            config.generation > 0 ? config.configHash : "";
        update->schedulerConfigJson = config.generation > 0 ? config.configJson : "";
    }
    return update;
}

navi::SummaryErrorCode AgentReportR::getSummary(const std::string &query,
                                                navi::SummaryBuilder &builder) const
{
    AgentUpdateToMasterDataPtr handshake;
    std::vector<std::string> collectorIds;
    {
        autil::ScopedLock lock(_mutex);
        handshake = _handshake;
        collectorIds.reserve(_subscribers.size());
        for (const auto &pair : _subscribers) {
            if (!pair.second->asyncPipe.expired()) {
                collectorIds.emplace_back(pair.first);
            }
        }
    }
    auto overview = builder.newTable("NaviSchedAgentOverview");
    TABLE_ADD_COLUMN_VALUE(overview, "subscriber_count", collectorIds.size());
    TABLE_ADD_COLUMN_VALUE(
        overview, "latency_formula_id",
        handshake && handshake->latency ? handshake->latency->formula : 0);
    TABLE_ADD_COLUMN_VALUE(
        overview, "latency_formula_name",
        getNaviSchedLatencyFormulaName(
            handshake && handshake->latency ? handshake->latency->formula : 0));
    TABLE_ADD_COLUMN_VALUE(
        overview, "latency_model_revision",
        handshake && handshake->latency ? handshake->latency->modelRevision : 0);
    TABLE_ADD_COLUMN_VALUE(overview, "latency_model_fingerprint",
                           handshake ? handshake->latencyModelFingerprint : "");
    TABLE_ADD_COLUMN_VALUE(overview, "latency_model_trusted",
                           handshake && handshake->latencyModelTrusted);
    TABLE_ADD_COLUMN_VALUE(overview, "latency_model_warm_started",
                           handshake && handshake->latencyModelWarmStarted);
    TABLE_ADD_COLUMN_VALUE(
        overview, "server_ip",
        handshake && handshake->control ? handshake->control->serverInfo.ip : "");
    TABLE_ADD_COLUMN_VALUE(
        overview, "server_port",
        handshake && handshake->control ? handshake->control->serverInfo.port : 0);
    if (SummaryUtil::getSummaryLevel(query) < 1) {
        return navi::SEC_OK;
    }
    auto collectorTable = builder.newTable("NaviSchedAgentCollectors");
    for (const auto &collectorId : collectorIds) {
        TABLE_ADD_COLUMN_VALUE(collectorTable, "collector_id", collectorId);
    }
    return navi::SEC_OK;
}

AgentReportR::QueuedPublishPtr AgentReportR::enqueuePublish(
    const AgentUpdateToMasterDataPtr &update, bool trackStats) const
{
    if (!update || !update->valid()) {
        return nullptr;
    }
    autil::ScopedLock lock(_mutex);
    if (_stopping) {
        return nullptr;
    }
    std::vector<SubscriberPtr> subscribers;
    subscribers.reserve(_subscribers.size());
    for (const auto &pair : _subscribers) {
        subscribers.emplace_back(pair.second);
    }
    return enqueuePublishLocked({update}, std::move(subscribers), trackStats);
}

AgentReportR::QueuedPublishPtr AgentReportR::enqueuePublishLocked(
    std::vector<AgentUpdateToMasterDataPtr> updates,
    std::vector<SubscriberPtr> subscribers, bool trackStats,
    bool allowStopping) const
{
    if (updates.empty() || (_stopping && !allowStopping)) {
        return nullptr;
    }
    auto publishTask = std::make_shared<QueuedPublish>();
    publishTask->updates = std::move(updates);
    publishTask->subscribers = std::move(subscribers);
    publishTask->trackStats = trackStats;
    _publishQueue.Push(publishTask);
    return publishTask;
}

void AgentReportR::drainPublishQueue() const
{
    bool expected = false;
    if (!_publishDraining.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
        return;
    }
    while (true) {
        QueuedPublishPtr publishTask;
        while (_publishQueue.Pop(&publishTask)) {
            deliverPublish(publishTask);
        }
        _publishDraining.store(false, std::memory_order_release);
        if (_publishQueue.Empty()) {
            return;
        }
        expected = false;
        if (!_publishDraining.compare_exchange_strong(
                expected, true, std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return;
        }
    }
}

size_t AgentReportR::waitPublish(const QueuedPublishPtr &publishTask) const
{
    if (!publishTask) {
        return 0;
    }
    autil::ScopedLock lock(publishTask->completionCond);
    while (!publishTask->completed) {
        publishTask->completionCond.wait();
    }
    return publishTask->publishCount;
}

void AgentReportR::deliverPublish(const QueuedPublishPtr &publishTask) const
{
    if (!publishTask) {
        return;
    }
    if (publishTask->trackStats) {
        for (const auto &update : publishTask->updates) {
            recordPublishedEvent(update, publishTask->subscribers.empty());
        }
    }
    AgentUpdateToMasterDataPtr message;
    size_t publishCount = 0;
    int64_t maxSetDataTimeUs = 0;
    for (const auto &subscriber : publishTask->subscribers) {
        bool delivered = true;
        auto asyncPipe = subscriber->asyncPipe.lock();
        if (!asyncPipe) {
            delivered = false;
            if (publishTask->trackStats) {
                _deliveryFailureCount.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            const auto send = [&](const AgentUpdateToMasterDataPtr &value, size_t eventCount) {
                const bool trackRequestTiming = publishTask->trackStats &&
                    (value->event == NaviSchedAgentEvent::REQUEST_RECEIVED ||
                     std::any_of(value->updates.begin(), value->updates.end(), [](const auto &event) {
                         return event->event == NaviSchedAgentEvent::REQUEST_RECEIVED;
                     }));
                const auto beginUs = trackRequestTiming ? autil::TimeUtility::currentTime() : 0;
                const auto result = asyncPipe->setData(value);
                if (trackRequestTiming) {
                    maxSetDataTimeUs = std::max(maxSetDataTimeUs, autil::TimeUtility::currentTime() - beginUs);
                }
                if (result != navi::EC_NONE) {
                    if (publishTask->trackStats) _deliveryFailureCount.fetch_add(1, std::memory_order_relaxed);
                    return false;
                }
                _deliveredMessageCount.fetch_add(1, std::memory_order_relaxed);
                _combinedMessageSavings.fetch_add(eventCount - 1, std::memory_order_relaxed);
                if (publishTask->trackStats) {
                    _deliverySuccessCount.fetch_add(eventCount, std::memory_order_relaxed);
                }
                return true;
            };
            if (subscriber->orderedReports) {
                if (!message) message = AgentUpdateToMasterData::makeBatch(publishTask->updates);
                delivered = message && send(message, publishTask->updates.size());
            } else {
                for (const auto &update : publishTask->updates) {
                    if (!send(update, 1)) {
                        delivered = false;
                        break;
                    }
                }
            }
        }
        if (delivered) {
            ++publishCount;
        } else {
            removeSubscriber(subscriber);
        }
    }
    {
        autil::ScopedLock lock(publishTask->completionCond);
        publishTask->publishCount = publishCount;
        publishTask->maxSetDataTimeUs = maxSetDataTimeUs;
        publishTask->completed = true;
        publishTask->completionCond.broadcast();
    }
}

bool AgentReportR::removeSubscriberLocked(
    const std::string &collectorId, const SubscriberPtr *expected) const
{
    auto it = _subscribers.find(collectorId);
    if (_subscribers.end() == it || (expected && it->second != *expected)) {
        return false;
    }
    _subscribers.erase(it);
    if (_pendingStopAcks.erase(collectorId) > 0 && _pendingStopAcks.empty()) {
        _mutex.broadcast();
    }
    return true;
}

bool AgentReportR::removeSubscriber(const SubscriberPtr &subscriber) const
{
    if (!subscriber) {
        return false;
    }
    autil::ScopedLock lock(_mutex);
    return removeSubscriberLocked(subscriber->collectorId, &subscriber);
}

REGISTER_RESOURCE(AgentReportR);
REGISTER_RESOURCE(AgentReportBridgeR);

}
