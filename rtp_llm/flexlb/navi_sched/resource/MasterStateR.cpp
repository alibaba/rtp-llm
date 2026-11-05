#include "aios/navi_sched/resource/MasterStateR.h"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

#include "aios/navi_sched/data/BlockHashUtil.h"
#include "aios/navi_sched/data/NaviSchedAgentEvent.h"
#include "aios/navi_sched/data/NaviSchedRequestTrace.h"
#include "aios/navi_sched/data/SummaryUtil.h"
#include "aios/navi_sched/resource/LatencyModelR.h"
#include "aios/navi_sched/resource/AgentSchedulerConfigR.h"
#include "aios/navi_sched/resource/LatencyResidualModel.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/scheduler/CostSchedulerColdStartStrategy.h"
#include "autil/TimeUtility.h"
#include "autil/HashAlgorithm.h"
#include "autil/Scope.h"
#include "autil/legacy/jsonizable.h"
#include "navi/log/NaviLogger.h"

namespace navi_sched {

struct MasterDispatchHandle::State {
    std::weak_ptr<MasterStateR::NodeState> node;
    std::weak_ptr<MasterStateR::NodeState::PendingBatchReservation> reservation;

    struct FailureTrace {
        FailureTrace(const MasterStateR::NodeState &node, const char *source)
            : node(node), source(source)
        {
        }
        void recordBefore(const MasterBatchReservation &batch, size_t requestIndex)
        {
            sampled = traceRequestBatch(batchId);
            if (!sampled) {
                return;
            }
            const auto &entry = batch._requests[requestIndex];
            requestId = entry.request.requestId;
            requestStateBefore = static_cast<int>(entry.state);
            dispatchAcknowledgedBefore = entry.dispatchAcknowledged;
            waitingBefore = batch.getWaitingRequestCount();
            predictionBeforeMs = batch.getPredictedLatencyMs();
        }
        void recordAfter(const MasterBatchReservation &batch, bool released)
        {
            if (!sampled) {
                return;
            }
            batchReleased = released;
            waitingAfter = batch.getWaitingRequestCount();
            predictionAfterMs = batch.getPredictedLatencyMs();
            completedTimeUs = autil::TimeUtility::currentTime();
        }
        ~FailureTrace()
        {
            const auto unlockedUs = autil::TimeUtility::monotonicTimeUs();
            if (attempted && sampled) {
                const auto info = std::atomic_load_explicit(&node._infoSnapshot,
                                                            std::memory_order_acquire);
                NAVI_KERNEL_LOG(
                    INFO,
                    "[navi-request-stage] stage=master_dispatch_release source=%s "
                    "agent=%s request_id=%lu schedule_batch_id=%ld event_time_us=%ld "
                    "changed=%d batch_released=%d request_state_before=%d "
                    "dispatch_acknowledged_before=%d waiting_before=%ld "
                    "waiting_after=%ld "
                    "prediction_before_ms=%.9f prediction_after_ms=%.9f "
                    "runtime_wait_us=%ld runtime_hold_us=%ld",
                    source, info->agentId.c_str(), requestId, batchId, completedTimeUs,
                    int(changed), int(batchReleased), requestStateBefore,
                    int(dispatchAcknowledgedBefore), waitingBefore, waitingAfter,
                    predictionBeforeMs, predictionAfterMs, lockAcquiredUs - lockBeginUs,
                    unlockedUs - lockAcquiredUs);
            }
            if (unlockedUs - lockBeginUs < 50'000) {
                return;
            }
            const auto info =
                std::atomic_load_explicit(&node._infoSnapshot, std::memory_order_acquire);
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-master-failure-function-timing] function=failRequestIndexesLocked "
                "source=%s agent=%s batch_id=%ld requests=1 attempted=%d changed=%d "
                "runtime_wait_us=%ld runtime_hold_us=%ld copy_us=%ld mark_failed_us=%ld "
                "prediction_us=%ld pending_update_us=%ld commit_us=%ld "
                "reference_index_built=%d reference_hashes=%lu reference_slots=%lu",
                source, info->agentId.c_str(), batchId, int(attempted), int(changed),
                lockAcquiredUs - lockBeginUs, unlockedUs - lockAcquiredUs, timing.copyUs,
                timing.markFailedUs, timing.predictionUs, timing.pendingUpdateUs,
                timing.commitUs, int(indexBuilt), referenceHashes, referenceSlots);
        }

        const MasterStateR::NodeState &node;
        const char *source;
        const int64_t lockBeginUs = autil::TimeUtility::monotonicTimeUs();
        int64_t lockAcquiredUs = 0;
        int64_t batchId = 0;
        bool attempted = false;
        bool changed = false;
        bool sampled = false;
        bool batchReleased = false;
        uint64_t requestId = 0;
        int requestStateBefore = -1;
        bool dispatchAcknowledgedBefore = false;
        int64_t waitingBefore = 0;
        int64_t waitingAfter = 0;
        double predictionBeforeMs = 0.0;
        double predictionAfterMs = 0.0;
        int64_t completedTimeUs = 0;
        bool indexBuilt = false;
        size_t referenceHashes = 0;
        size_t referenceSlots = 0;
        MasterStateR::NodeState::FailureTiming timing;
    };
};

bool MasterDispatchHandle::registerWaiter(const std::shared_ptr<navi::AsyncPipe> &pipe,
                                          bool &acknowledged) const
{
    acknowledged = false;
    if (!_state || !pipe) {
        return false;
    }
    auto node = _state->node.lock();
    auto reservation = _state->reservation.lock();
    if (!node || !reservation) {
        acknowledged = true;
        return true;
    }
    autil::ScopedLock lock(node->_runtimeLock);
    if (!node->_acceptReservations || !reservation->attached) {
        acknowledged = true;
        return true;
    }
    const auto &entry = reservation->batch._requests[_requestIndex];
    if (entry.state != MasterBatchReservation::RequestState::WAITING ||
        entry.dispatchAcknowledged) {
        acknowledged = true;
        return true;
    }
    auto [it, inserted] =
        reservation->dispatchWaiters.emplace(entry.request.requestId, pipe);
    if (!inserted) {
        auto current = it->second.lock();
        if (current && current != pipe) {
            return false;
        }
        it->second = pipe;
    }
    return true;
}

bool MasterDispatchHandle::cancel(const std::shared_ptr<navi::AsyncPipe> &pipe) const
{
    if (!_state || !pipe) {
        return false;
    }
    auto node = _state->node.lock();
    auto reservation = _state->reservation.lock();
    if (!node || !reservation) {
        return false;
    }
    // Declared before the lock so slow-path logging happens after unlocking.
    State::FailureTrace trace(*node, "cancel");
    autil::ScopedLock lock(node->_runtimeLock);
    trace.lockAcquiredUs = autil::TimeUtility::monotonicTimeUs();
    trace.batchId = reservation->batch.getBatchId();
    if (!node->_acceptReservations || !reservation->attached) {
        return false;
    }
    const auto requestId = reservation->batch._requests[_requestIndex].request.requestId;
    auto it = reservation->dispatchWaiters.find(requestId);
    if (it == reservation->dispatchWaiters.end()) {
        return false;
    }
    auto current = it->second.lock();
    if (current && current != pipe) {
        return false;
    }
    bool released = false;
    const bool hadReferences = reservation->batch.hasPendingCacheReferences();
    trace.attempted = true;
    trace.recordBefore(reservation->batch, _requestIndex);
    trace.changed = node->failRequestIndexesLocked(*reservation, {_requestIndex},
                                                   released, &trace.timing);
    trace.recordAfter(reservation->batch, released);
    trace.indexBuilt = !hadReferences && reservation->batch.hasPendingCacheReferences();
    trace.referenceHashes = reservation->batch._pendingReferences.size();
    trace.referenceSlots = reservation->batch._requestBlockIndexes
                               ? reservation->batch._requestBlockIndexes->size()
                               : 0;
    return trace.changed;
}

bool MasterDispatchHandle::abandon() const
{
    if (!_state) {
        return false;
    }
    auto node = _state->node.lock();
    auto reservation = _state->reservation.lock();
    if (!node || !reservation) {
        return false;
    }
    State::FailureTrace trace(*node, "abandon");
    autil::ScopedLock lock(node->_runtimeLock);
    trace.lockAcquiredUs = autil::TimeUtility::monotonicTimeUs();
    trace.batchId = reservation->batch.getBatchId();
    if (!node->_acceptReservations || !reservation->attached) {
        return false;
    }
    const auto &entry = reservation->batch._requests[_requestIndex];
    if (entry.state != MasterBatchReservation::RequestState::WAITING ||
        entry.dispatchAcknowledged) {
        return false;
    }
    const auto requestId = entry.request.requestId;
    auto it = reservation->dispatchWaiters.find(requestId);
    if (it != reservation->dispatchWaiters.end() && !it->second.expired()) {
        return false;
    }
    bool released = false;
    const bool hadReferences = reservation->batch.hasPendingCacheReferences();
    trace.attempted = true;
    trace.recordBefore(reservation->batch, _requestIndex);
    trace.changed = node->failRequestIndexesLocked(*reservation, {_requestIndex},
                                                   released, &trace.timing);
    trace.recordAfter(reservation->batch, released);
    trace.indexBuilt = !hadReferences && reservation->batch.hasPendingCacheReferences();
    trace.referenceHashes = reservation->batch._pendingReferences.size();
    trace.referenceSlots = reservation->batch._requestBlockIndexes
                               ? reservation->batch._requestBlockIndexes->size()
                               : 0;
    return trace.changed;
}

void MasterDispatchHandle::fail(const std::vector<MasterDispatchHandle> &handles)
{
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    struct FailedBatch {
        std::shared_ptr<State> state;
        std::vector<size_t> indexes;
    };
    std::unordered_map<const State *, FailedBatch> batches;
    for (const auto &handle : handles) {
        if (handle._state) {
            auto &batch = batches[handle._state.get()];
            batch.state = handle._state;
            batch.indexes.push_back(handle._requestIndex);
        }
    }
    const auto groupedUs = autil::TimeUtility::monotonicTimeUs();
    for (const auto &[key, batch] : batches) {
        (void)key;
        auto node = batch.state->node.lock();
        auto reservation = batch.state->reservation.lock();
        if (!node || !reservation) {
            continue;
        }
        bool released = false;
        bool changed = false;
        int64_t batchId = 0;
        MasterStateR::NodeState::FailureTiming timing;
        const auto lockBeginUs = autil::TimeUtility::monotonicTimeUs();
        int64_t lockAcquiredUs = 0;
        {
            autil::ScopedLock lock(node->_runtimeLock);
            lockAcquiredUs = autil::TimeUtility::monotonicTimeUs();
            if (node->_acceptReservations && reservation->attached) {
                batchId = reservation->batch.getBatchId();
                changed = node->failRequestIndexesLocked(*reservation, batch.indexes,
                                                         released, &timing);
            }
        }
        const auto unlockedUs = autil::TimeUtility::monotonicTimeUs();
        if (unlockedUs - lockBeginUs >= 50'000) {
            const auto info = std::atomic_load_explicit(&node->_infoSnapshot,
                                                        std::memory_order_acquire);
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-master-failure-function-timing] function=failRequestIndexesLocked "
                "agent=%s batch_id=%ld requests=%lu runtime_wait_us=%ld "
                "runtime_hold_us=%ld "
                "copy_us=%ld mark_failed_us=%ld prediction_us=%ld pending_update_us=%ld "
                "commit_us=%ld",
                info->agentId.c_str(), batchId, batch.indexes.size(),
                lockAcquiredUs - lockBeginUs, unlockedUs - lockAcquiredUs, timing.copyUs,
                timing.markFailedUs, timing.predictionUs, timing.pendingUpdateUs,
                timing.commitUs);
        }
        if (changed) {
            auto info = std::atomic_load_explicit(&node->_infoSnapshot,
                                                  std::memory_order_acquire);
            NAVI_KERNEL_LOG(WARN,
                            "[navi-master-dispatch-lifecycle] phase=bulk_failed agent=%s "
                            "batch_id=%ld request_count=%lu batch_released=%d",
                            info->agentId.c_str(), batchId, batch.indexes.size(),
                            int(released));
        }
    }
    const auto endUs = autil::TimeUtility::monotonicTimeUs();
    if (endUs - beginUs >= 50'000) {
        NAVI_KERNEL_LOG(
            INFO,
            "[navi-master-failure-function-timing] function=MasterDispatchHandle::fail "
            "handles=%lu batches=%lu group_us=%ld total_us=%ld",
            handles.size(), batches.size(), groupedUs - beginUs, endUs - beginUs);
    }
}

std::string MasterStateR::getHost(const std::string &address)
{
    if (address.empty()) {
        return "";
    }
    auto separator = address.rfind(':');
    return std::string::npos == separator ? address : address.substr(0, separator);
}

NaviSchedScheduleRequest MasterStateR::makeScheduleRequest(
    const ClientRequestToMasterData &request)
{
    NaviSchedScheduleRequest output;
    output.version = request.version;
    output.requestId = request.requestId;
    output.bizName = request.bizName;
    output.promptTokenCount = request.promptTokenCount;
    output.explicitCacheCheckpointPosition = request.explicitCacheCheckpointPosition;
    output.blockSize = request.blockSize;
    output.blockHashes = request.blockHashes;
    output.timeoutMs = request.timeoutMs;
    output.routeOnly = request.routeOnly;
    return output;
}

NaviSchedScheduleItem MasterStateR::makeScheduleItem(
    const ClientRequestToMasterData &request, int64_t waitTimeMs)
{
    NaviSchedScheduleItem item;
    item.request = makeScheduleRequest(request);
    item.waitTimeMs = waitTimeMs;
    item.remainingTimeoutMs = std::max<int64_t>(0, request.timeoutMs - item.waitTimeMs);
    item.timeoutTriggered = item.waitTimeMs >= request.timeoutMs;
    return item;
}

bool MasterStateR::validLatencyModel(const AgentLatencyToMasterDataPtr &latency)
{
    if (!latency || latency->modelRevision <= 0) {
        return !latency || (latency->formula == NAVI_SCHED_LATENCY_FORMULA_UNKNOWN &&
                            latency->parameters.empty());
    }
    NaviSchedLatencyModelSnapshot snapshot;
    snapshot.formula = latency->formula;
    snapshot.binaryVersion = latency->binaryVersion;
    snapshot.revision = latency->modelRevision;
    snapshot.parameters = latency->parameters;
    if (!NonLinearPrefillModelR::makeOptimizerState(latency->optimizerState,
                                                    snapshot.optimizerState)) {
        return false;
    }
    snapshot.residualVersion = latency->residualVersion;
    snapshot.residualParameters = latency->residualParameters;
    snapshot.residualLastUpdateTimeUs = latency->residualLastUpdateTimeUs;
    return snapshot.valid();
}

namespace {

void applyLatencyModelInfo(
    NaviSchedNodeInfo &info,
    const std::shared_ptr<const MasterStateR::LatencyInfo> &latencyInfo)
{
    if (!latencyInfo) {
        return;
    }
    info.latencyFormula = latencyInfo->formula;
    info.latencyBinaryVersion = latencyInfo->binaryVersion;
    info.latencyModelRevision = latencyInfo->revision;
    info.latencyParameters = latencyInfo->parameters;
    info.latencyResidualVersion = latencyInfo->residualVersion;
    info.latencyResidualParameters = latencyInfo->residualParameters;
    info.latencyResidualLastUpdateTimeUs = latencyInfo->residualLastUpdateTimeUs;
    info.latencyModelTrusted = latencyInfo->overloadPredictionTrusted;
    info.latencyModelWarmStarted = latencyInfo->warmStarted;
    info.latencyModelConsecutiveRejectedCount = latencyInfo->consecutiveRejectedCount;
    info.latencyModelLastRejectReason = latencyInfo->lastRejectReason;
}

bool latencyModelMatchesPdMode(const AgentLatencyToMasterDataPtr &latency,
                               bool pdSeparated)
{
    if (!latency || latency->modelRevision <= 0) {
        return true;
    }
    return NonLinearPrefillModelR::usesCapacitySplitParameters(latency->parameters) ==
           !pdSeparated;
}

bool workloadMatchesPdMode(const NaviSchedWorkloadSnapshotPtr &workload, bool pdSeparated)
{
    return !workload || workload->prefillCapacitySplit == !pdSeparated;
}

} // namespace

const std::string MasterStateR::RESOURCE_ID = "navi_sched.master_state_r";

MasterStateR::MasterStateR() {}

NaviSchedNodeInfo MasterStateR::NodeState::getInfo() const
{
    auto published =
        std::atomic_load_explicit(&_publishedSnapshot, std::memory_order_acquire);
    auto info = published && published->info ? *published->info : NaviSchedNodeInfo();
    fillDynamicInfo(info);
    if (published) {
        applyLatencyModelInfo(info, published->latencyModel);
    }
    return info;
}

bool MasterStateR::NodeState::getScheduleSnapshot(const std::string &bizName,
                                                  NaviSchedNodeSnapshot &snapshot) const
{
    snapshot.runtimeLockWaitUs = 0;
    snapshot.runtimeLockHoldUs = 0;
    auto published =
        std::atomic_load_explicit(&_publishedSnapshot, std::memory_order_acquire);
    if (!published || !published->info || !published->info->online ||
        published->info->bizName != bizName || published->info->serverAddress.empty()) {
        return false;
    }
    snapshot.info = published->info;
    snapshot.latencyModel = published->latencyModel;
    if (!published->executionGeometryValid) {
        return false;
    }
    snapshot.prefillExecutionPolicy = published->prefillExecutionPolicy;
    snapshot.residualRuntimeConfig = getResidualRuntimeConfig();
    {
        const auto lockBeginUs = autil::TimeUtility::monotonicTimeUs();
        autil::ScopedLock lock(_runtimeLock);
        const auto lockAcquiredUs = autil::TimeUtility::monotonicTimeUs();
        snapshot.runtimeLockWaitUs = lockAcquiredUs - lockBeginUs;
        autil::ScopeGuard recordLockHold([&snapshot, lockAcquiredUs]() {
            snapshot.runtimeLockHoldUs =
                autil::TimeUtility::monotonicTimeUs() - lockAcquiredUs;
        });
        if (!_acceptReservations || !_cachePolicyReady.load(std::memory_order_acquire) ||
            !_cacheStateReady.load(std::memory_order_acquire) ||
            !_pendingCacheR->getStats(_info.agentIdHash).agentSnapshotReady) {
            return false;
        }
        snapshot.load = _runtimeState.load;
        snapshot.agentStartTimeUs = _runtimeState.agentStartTimeUs;
        snapshot.peerCacheReadAvailable = _runtimeState.peerCacheReadAvailable;
        snapshot.peerCacheReadRevision = _runtimeState.peerCacheReadRevision;
        int64_t effectiveInflightRequests = snapshot.load.inflightRequests;
        long double effectivePredictedLatencySumMs =
            snapshot.load.inflightPredictedLatencySumMs;
        if (effectiveInflightRequests < 0 ||
            !std::isfinite(effectivePredictedLatencySumMs) ||
            effectivePredictedLatencySumMs < 0.0) {
            return false;
        }
        for (const auto &reservationPtr : _pendingBatchReservations) {
            const auto &reservation = *reservationPtr;
            snapshot.load.queueModel.setBaseMs(
                snapshot.load.queueModel.baseMs(snapshot.load.queueTotalPredictedLatencyMs) +
                    reservation.batch.getPredictedLatencyMs(),
                snapshot.load.queueTotalPredictedLatencyMs);
            const auto unreported = reservation.batch.getUnreportedInflightStats();
            if (!unreported.valid()) {
                return false;
            }
            effectiveInflightRequests += unreported.requestCount;
            effectivePredictedLatencySumMs += unreported.totalPredictedLatencyMs;
        }
        if (effectiveInflightRequests > INT_MAX ||
            !std::isfinite(effectivePredictedLatencySumMs) ||
            effectivePredictedLatencySumMs >
                static_cast<long double>(std::numeric_limits<double>::max())) {
            return false;
        }
        snapshot.load.inflightRequests = static_cast<int32_t>(effectiveInflightRequests);
        snapshot.load.inflightPredictedLatencySumMs =
            static_cast<double>(effectivePredictedLatencySumMs);
        snapshot.workload = _runtimeState.workload;
        snapshot.unifiedLatencyPrediction = _runtimeState.unifiedLatencyPrediction;
        if (snapshot.unifiedLatencyPrediction) {
            if (!snapshot.workload.valid()) {
                return false;
            }
            if (!snapshot.reportedWorkloadFeatures.appendPrefillWorkload(
                    _runtimeState.workload)) {
                return false;
            }
            if (_runtimeState.workload.prefillCapacitySplit) {
                snapshot.maxRunningRequestCount =
                    _runtimeState.workload.maxRunningRequestCount;
                auto workloadSlots = _runtimeState.workload.runningCapacitySlotCount;
                int32_t reservedSlots = 0;
                if (!snapshot.workloadFeatures.append(
                        snapshot.reportedWorkloadFeatures)) {
                    return false;
                }
                for (const auto &reservationPtr : _pendingBatchReservations) {
                    const auto &reservation = *reservationPtr;
                    if (!reservation.batch.appendWaitingCapacityFeatures(
                            snapshot.workloadFeatures, snapshot.maxRunningRequestCount,
                            workloadSlots) ||
                        !reservation.batch.appendWaitingCapacityFeatures(
                            snapshot.reservedWorkloadFeatures,
                            snapshot.maxRunningRequestCount, reservedSlots)) {
                        return false;
                    }
                }
                snapshot.workloadRunningCapacitySlotCount = workloadSlots;
                snapshot.reservedRunningCapacitySlotCount = reservedSlots;
            } else {
                for (const auto &reservationPtr : _pendingBatchReservations) {
                    const auto &reservation = *reservationPtr;
                    if (!snapshot.reservedWorkloadFeatures.append(
                            reservation.batch.getWaitingFeatures())) {
                        return false;
                    }
                }
                if (!snapshot.workloadFeatures.append(
                        snapshot.reportedWorkloadFeatures) ||
                    !snapshot.workloadFeatures.append(
                        snapshot.reservedWorkloadFeatures)) {
                    return false;
                }
            }
        }
        snapshot.workloadReportReceiveTimeUs = _runtimeState.workloadReportReceiveTimeUs;
    }
    return true;
}

void MasterStateR::NodeState::rebuildInfoSnapshot()
{
    auto info = std::make_shared<NaviSchedNodeInfo>(_info);
    info->serverAddress = info->serverInfo.getAddress(getHost(info->collectorAddress));
    auto latencyModel = getLatencyInfo();
    applyLatencyModelInfo(*info, latencyModel);
    auto published = std::make_shared<PublishedSnapshot>();
    published->info = info;
    published->latencyModel = std::move(latencyModel);
    if (published->latencyModel) {
        const auto version = published->latencyModel->binaryVersion;
        const auto geometry = NonLinearPrefillModelR::usesGeometryForVersion(version);
        const auto projected =
            NonLinearPrefillModelR::usesProjectedGeometryForVersion(version);
        published->executionGeometryValid =
            geometry == info->executionGeometryEnabled &&
            projected == info->projectedExecutionGeometryEnabled;
        if (geometry && published->executionGeometryValid) {
            published->executionGeometryValid =
                info->pdSeparated && !info->unifiedLatencyPrediction &&
                TPUPrefillExecutionPolicy::validate(info->prefillExecutionPolicy);
            if (published->executionGeometryValid) {
                published->prefillExecutionPolicy = info->prefillExecutionPolicy;
            }
        }
    }
    std::shared_ptr<const PublishedSnapshot> immutablePublished = published;
    std::atomic_store_explicit(&_publishedSnapshot, immutablePublished,
                               std::memory_order_release);
    std::shared_ptr<const NaviSchedNodeInfo> immutableInfo = info;
    std::atomic_store_explicit(&_infoSnapshot, std::move(immutableInfo),
                               std::memory_order_release);
}

double MasterStateR::NodeState::getExecutionUtilization() const
{
    return _executionUtilization.load(std::memory_order_relaxed);
}

double MasterStateR::NodeState::getExecutionBusyUtilization() const
{
    return _executionBusyUtilization.load(std::memory_order_relaxed);
}

void MasterStateR::NodeState::fillDynamicInfo(NaviSchedNodeInfo &info) const
{
    auto runtime = getRuntimeState();
    info.peerCacheReadAvailable = runtime.peerCacheReadAvailable;
    info.peerCacheReadRevision = runtime.peerCacheReadRevision;
    info.peerCacheReadAgentStartTimeUs = runtime.agentStartTimeUs;
    info.totalBlocks = runtime.load.totalBlocks;
    info.inflightStatsRevision = runtime.load.inflightStatsRevision;
    info.inflightRequests = runtime.load.inflightRequests;
    info.inflightPredictedLatencySumMs = runtime.load.inflightPredictedLatencySumMs;
    info.batchWaitingRequests = runtime.load.batchWaitingRequests;
    info.queuedBatchRequests = runtime.load.queuedBatchRequests;
    info.runningRequests = runtime.load.runningRequests;
    info.unifiedLatencyPrediction = runtime.unifiedLatencyPrediction;
    info.batchTimeoutCount = runtime.load.batchTimeoutCount;
    info.abnormalBatchFinishCount = runtime.load.abnormalBatchFinishCount;
    info.lastRequestLatencyMs = runtime.lastRequestLatencyMs;
    info.queueTotalPredictedLatencyMs = runtime.load.queueTotalPredictedLatencyMs;
    info.queueDecayHeadRemainingMs = runtime.load.queueDecayHeadRemainingMs;
    info.queueModel = runtime.load.queueModel;
    info.queueReportReceiveTimeUs = runtime.load.queueReportReceiveTimeUs;
    info.workload = runtime.workload;
    info.workloadReportReceiveTimeUs = runtime.workloadReportReceiveTimeUs;
    info.workloadStaleUpdateCount = runtime.workloadStaleUpdateCount;
    info.lastEventType = NaviSchedAgentEvent::UNKNOWN == runtime.lastEvent
                             ? std::string()
                             : NaviSchedAgentEvent::getName(runtime.lastEvent);
    info.lastEventTimeUs = runtime.lastEventTimeUs;
    info.lastRequestEventType =
        NaviSchedAgentEvent::UNKNOWN == runtime.lastRequestEvent
            ? std::string()
            : NaviSchedAgentEvent::getName(runtime.lastRequestEvent);
    info.lastRequestEventTimeUs = runtime.lastRequestEventTimeUs;
    info.lastReceiveTimeUs = runtime.lastReceiveTimeUs;
    info.latencyValidation = runtime.latencyValidation;
}

MasterStateR::NodeState::RuntimeState MasterStateR::NodeState::getRuntimeState() const
{
    autil::ScopedLock lock(_runtimeLock);
    return _runtimeState;
}

NaviSchedNodeLoad MasterStateR::NodeState::getScheduleLoad(int64_t currentTimeUs) const
{
    autil::ScopedLock lock(_runtimeLock);
    return getScheduleLoadLocked(currentTimeUs);
}

NaviSchedNodeLoad MasterStateR::NodeState::getScheduleLoadLocked(
    int64_t currentTimeUs) const
{
    auto load = _runtimeState.load;
    const auto elapsedMs =
        load.queueReportReceiveTimeUs > 0
            ? std::max<int64_t>(0, currentTimeUs - load.queueReportReceiveTimeUs) / 1000.0
            : 0.0;
    const auto decayMs = std::min(load.queueDecayHeadRemainingMs, elapsedMs);
    load.queueModel.setBaseMs(
        load.queueModel.decayedBaseMs(load.queueTotalPredictedLatencyMs, decayMs),
        load.queueTotalPredictedLatencyMs);
    load.queueDecayHeadRemainingMs =
        std::max(0.0, load.queueDecayHeadRemainingMs - decayMs);
    load.queueReportReceiveTimeUs = currentTimeUs;
    for (const auto &reservationPtr : _pendingBatchReservations) {
        const auto &reservation = *reservationPtr;
        load.queueModel.setBaseMs(
            load.queueModel.baseMs(load.queueTotalPredictedLatencyMs) +
                reservation.batch.getPredictedLatencyMs(),
            load.queueTotalPredictedLatencyMs);
    }
    return load;
}

bool MasterStateR::NodeState::reserveBatch(
    int64_t batchId, int64_t scheduleTimeUs, double predictedLatencyMs,
    std::vector<MasterBatchReservation::Request> requests,
    const NaviSchedCachePolicy *cachePolicy, bool *pendingCacheTracked,
    ReservationPtr *reserved, ReservationTiming *timing)
{
    ReservationTiming localTiming;
    if (!timing) timing = &localTiming;
    *timing = {};
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    const auto beginCpuUs = autil::TimeUtility::cpuThreadTimeUs();
    const auto requestCountBeforeMove = requests.size();
    autil::ScopeGuard finishTiming([&, beginUs]() {
        timing->totalUs = autil::TimeUtility::monotonicTimeUs() - beginUs;
        timing->totalCpuUs = autil::TimeUtility::cpuThreadTimeUs() - beginCpuUs;
        if (timing->totalUs >= 50'000) {
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-master-reservation-function-timing] "
                "function=NodeState::reserveBatch "
                "agent=%s batch_id=%ld requests=%lu total_us=%ld init_us=%ld "
                "runtime_wait_us=%ld runtime_hold_us=%ld pending_publish_us=%ld "
                "total_cpu_us=%ld init_cpu_us=%ld runtime_hold_cpu_us=%ld "
                "pending_publish_cpu_us=%ld",
                _info.agentId.c_str(), batchId, requestCountBeforeMove, timing->totalUs,
                timing->initUs, timing->runtimeWaitUs, timing->runtimeHoldUs,
                timing->pendingPublishUs, timing->totalCpuUs, timing->initCpuUs,
                timing->runtimeHoldCpuUs, timing->pendingPublishCpuUs);
        }
    });
    if (reserved) {
        reserved->reset();
    }
    if (pendingCacheTracked) {
        *pendingCacheTracked = false;
    }
    NaviSchedCachePolicy externalPolicy;
    const bool hasExternalPolicy =
        getCachePolicy(externalPolicy, NaviSchedCacheKeyType::V6D);
    auto reservation = std::make_shared<PendingBatchReservation>();
    if (!reservation->batch.init(batchId, scheduleTimeUs, predictedLatencyMs,
                                 std::move(requests), cachePolicy)) {
        timing->failureReason = "invalid_batch_definition";
        return false;
    }
    if (hasExternalPolicy &&
        !reservation->batch.setPendingCachePolicy(NaviSchedCacheKeyType::V6D,
                                                   externalPolicy)) {
        timing->failureReason = "invalid_external_cache_policy";
        return false;
    }
    const auto initEndUs = autil::TimeUtility::monotonicTimeUs();
    const auto initEndCpuUs = autil::TimeUtility::cpuThreadTimeUs();
    timing->initUs = initEndUs - beginUs;
    timing->initCpuUs = initEndCpuUs - beginCpuUs;
    autil::ScopedLock lock(_runtimeLock);
    const auto acquiredUs = autil::TimeUtility::monotonicTimeUs();
    const auto acquiredCpuUs = autil::TimeUtility::cpuThreadTimeUs();
    timing->runtimeWaitUs = acquiredUs - initEndUs;
    autil::ScopeGuard recordHold([&, acquiredUs]() {
        timing->runtimeHoldUs = autil::TimeUtility::monotonicTimeUs() - acquiredUs;
        timing->runtimeHoldCpuUs = autil::TimeUtility::cpuThreadTimeUs() - acquiredCpuUs;
    });
    if (!_acceptReservations) {
        timing->failureReason = "reservations_disabled";
        return false;
    }
    if (_runtimeState.unifiedLatencyPrediction &&
        !reservation->batch.setPredictedLatencyMs(0.0)) {
        timing->failureReason = "invalid_unified_prediction";
        return false;
    }
    auto existing = _reservationsById.find(batchId);
    if (existing != _reservationsById.end()) {
        if (!existing->second->batch.sameDefinition(reservation->batch)) {
            timing->failureReason = "conflicting_batch_definition";
            return false;
        }
        if (pendingCacheTracked) {
            *pendingCacheTracked = existing->second->batch.tracksPendingCache();
        }
        if (reserved) {
            *reserved = existing->second;
        }
        return true;
    }
    const auto pendingBeginUs = autil::TimeUtility::monotonicTimeUs();
    const auto pendingBeginCpuUs = autil::TimeUtility::cpuThreadTimeUs();
    const auto pendingUpdated = updateReservationPendingCacheLocked(reservation->batch);
    timing->pendingPublishUs = autil::TimeUtility::monotonicTimeUs() - pendingBeginUs;
    timing->pendingPublishCpuUs =
        autil::TimeUtility::cpuThreadTimeUs() - pendingBeginCpuUs;
    if (!pendingUpdated) {
        timing->failureReason = "pending_cache_update_failed";
        return false;
    }
    const auto requestCount = reservation->batch.getRequestCount();
    if (pendingCacheTracked) {
        *pendingCacheTracked = reservation->batch.tracksPendingCache();
    }
    reservation->position =
        _pendingBatchReservations.insert(_pendingBatchReservations.end(), reservation);
    reservation->attached = true;
    _reservationsById.emplace(batchId, reservation);
    if (reserved) {
        *reserved = reservation;
    }
    _dispatchedRequestCount.fetch_add(requestCount, std::memory_order_relaxed);
    _dispatchedBatchCount.fetch_add(1, std::memory_order_relaxed);
    return true;
}

MasterStateR::NodeState::PendingBatchReservation *
MasterStateR::NodeState::findReservationLocked(int64_t batchId)
{
    const auto it = _reservationsById.find(batchId);
    return it == _reservationsById.end() ? nullptr : it->second.get();
}

void MasterStateR::NodeState::eraseReservationLocked(PendingBatchReservation &reservation)
{
    reservation.attached = false;
    const auto batchId = reservation.batch.getBatchId();
    _pendingBatchReservations.erase(reservation.position);
    _reservationsById.erase(batchId);
}

void MasterStateR::NodeState::clearReservationsLocked()
{
    for (const auto &reservation : _pendingBatchReservations) {
        reservation->attached = false;
    }
    _pendingBatchReservations.clear();
    _reservationsById.clear();
}

bool MasterStateR::NodeState::updateReservationPendingCacheLocked(
    const MasterBatchReservation &reservation,
    const MasterBatchReservation::Transition *transition)
{
    if (reservation.tracksPendingCache()) {
        if (transition) {
            if (transition->cacheChanged) {
                if (transition->retireCache) {
                    _pendingCacheR->retire(_info.agentIdHash, reservation.getBatchId());
                } else if (!_pendingCacheR->removeScheduledBlocks(
                               _info.agentIdHash, reservation.getBatchId(),
                               transition->removedFullReadyBlockHashes,
                               transition->removedTailReadyBlockHashes)) {
                    return false;
                }
            }
        } else {
            const auto &pending = reservation.getPendingCacheReservation();
            if (pending.fullReadyBlockHashes.empty()) {
                _pendingCacheR->retire(_info.agentIdHash, reservation.getBatchId());
            } else if (!_pendingCacheR->updateScheduled(_info.agentIdHash, pending)) {
                return false;
            }
        }
    }
    // V6D has different publication boundaries: never reuse BLOCK_POOL deltas.
    for (const auto &entry : reservation.getPendingCachePolicies()) {
        if (entry.first == NaviSchedCacheKeyType::BLOCK_POOL) continue;
        NaviSchedPendingCacheReservation pending;
        if (!reservation.makePendingCacheReservation(entry.first, pending)) return false;
        if (pending.fullReadyBlockHashes.empty()) {
            _pendingCacheR->retire(_info.agentIdHash, reservation.getBatchId(), entry.first);
        } else if (!_pendingCacheR->updateScheduled(_info.agentIdHash, pending, entry.first)) {
            return false;
        }
    }
    return true;
}

bool MasterStateR::NodeState::refreshReservationPredictionLocked(
    MasterBatchReservation &batch, bool recomputeCacheHits)
{
    if (recomputeCacheHits && batch.tracksPendingCache() &&
        batch.getWaitingRequestCount() > 0) {
        autil::ScopedReadLock cacheLock(_cacheLock);
        const auto batchId = batch.getBatchId();
        if (!batch.recomputeWaitingFeatures(
                [this, batchId](const MasterBatchReservation::Request &request) {
                    const auto local =
                        _pendingCacheR->getMatchedBlockCountExcludingSchedule(
                            _info.agentIdHash, batchId, *request.blockHashes,
                            _cachedBlockHashes, _fullReadyBlockHashes,
                            _tailReadyBlockHashes, _hybridTailRequired,
                            request.blockHashes->size(),
                            _cachePolicyReady.load(std::memory_order_relaxed)
                                ? _cachePolicy.fullAttentionQueryDropTokens /
                                      _cachePolicy.blockSize
                                : 0);
                    return local + matchExternalSuffix(*request.blockHashes,
                                                       request.totalTokenCount,
                                                       request.blockSize, local, batchId);
                })) {
            return false;
        }
    }
    if (batch.getWaitingRequestCount() == 0) {
        return batch.setPredictedLatencyMs(0.0);
    }
    if (_runtimeState.unifiedLatencyPrediction) {
        return batch.setPredictedLatencyMs(0.0);
    }
    const auto latency = getLatencyInfo();
    if (!latency || latency->formula != NAVI_SCHED_LATENCY_FORMULA_NONLINEAR_PREFILL ||
        !NonLinearPrefillModelR::supportsBinaryVersion(latency->binaryVersion) ||
        !NonLinearPrefillModelR::validateParametersForVersion(latency->parameters,
                                                              latency->binaryVersion) ||
        latency->residualVersion != LatencyResidualModel::VERSION ||
        !LatencyResidualModel::validateParameters(latency->residualParameters)) {
        return batch.setPredictedLatencyMs(0.0);
    }
    auto features = batch.getWaitingFeatures();
    double basePrediction;
    if (NonLinearPrefillModelR::usesGeometryForVersion(latency->binaryVersion)) {
        const auto published =
            std::atomic_load_explicit(&_publishedSnapshot, std::memory_order_acquire);
        if (!published || !published->executionGeometryValid ||
            !published->prefillExecutionPolicy || !published->info->pdSeparated ||
            !published->info->executionGeometryEnabled ||
            ((NonLinearPrefillModelR::usesProjectedGeometryForVersion(
                 latency->binaryVersion)) !=
             published->info->projectedExecutionGeometryEnabled)) {
            return false;
        }
        const auto &policy = *published->prefillExecutionPolicy;
        std::vector<TPUPrefillRequestInput> inputs;
        NaviSchedLatencyFeatures logical;
        if (!logical.appendBatchCount(1)) {
            return false;
        }
        size_t maxSteps = 0;
        autil::ScopedReadLock cacheLock(_cacheLock);
        if (!_cachePolicyReady.load(std::memory_order_relaxed)) {
            return false;
        }
        for (const auto &request : batch.getWaitingRequests()) {
            if (request.blockSize <= 0 || request.blockSize != _cachePolicy.blockSize ||
                request.totalTokenCount <= 0) {
                return false;
            }
            const auto &hashes = BlockHashUtil::get(request.blockHashes);
            const auto local = _pendingCacheR->getMatchBreakdown(
                _info.agentIdHash, hashes, _cachedBlockHashes, _fullReadyBlockHashes,
                _tailReadyBlockHashes, _hybridTailRequired,
                BlockHashUtil::getMatchableBlockCount(
                    request.totalTokenCount, request.blockSize, request.blockHashes),
                _cachePolicy.fullAttentionQueryDropTokens / _cachePolicy.blockSize,
                NaviSchedCacheKeyType::BLOCK_POOL, 0, 1, batch.getBatchId());
            const auto suffix =
                matchExternalSuffix(hashes, request.totalTokenCount, request.blockSize,
                                    local.matchedBlockCount, batch.getBatchId());
            const auto maximumBlocks = request.totalTokenCount / request.blockSize;
            if (local.matchedBlockCount < 0 || suffix < 0 ||
                local.matchedBlockCount > maximumBlocks - suffix ||
                local.effectiveFullMatchedBlockCount < 0 ||
                local.effectiveFullMatchedBlockCount > maximumBlocks) {
                return false;
            }
            TPUPrefillRequestInput input;
            input.promptTokens = request.totalTokenCount;
            input.computedTokens = (local.matchedBlockCount + suffix) * request.blockSize;
            input.externalReadTokens = suffix * request.blockSize;
            // Only the effective local Full boundary can create a shared junction.
            if (local.effectiveFullMatchedBlockCount > local.matchedBlockCount) {
                input.sharedPrefixBoundary =
                    local.effectiveFullMatchedBlockCount * request.blockSize;
            }
            if (request.explicitCacheCheckpointPosition > 0) {
                const auto boundary = request.explicitCacheCheckpointPosition /
                                      policy.schedulerBlockSize *
                                      policy.schedulerBlockSize;
                if (boundary > 0) {
                    input.sharedPrefixBoundary = boundary;
                }
            }
            const auto remaining = input.promptTokens - input.computedTokens;
            if (remaining <= 0 || static_cast<uint64_t>(remaining) >
                                      std::numeric_limits<size_t>::max() - maxSteps) {
                return false;
            }
            maxSteps += static_cast<size_t>(remaining);
            logical.appendRequest(input.computedTokens, remaining);
            inputs.push_back(input);
        }
        TPUPrefillExecutionProjection projection;
        const auto projector = NonLinearPrefillModelR::usesProjectedGeometryForVersion(
                                   latency->binaryVersion)
                                   ? TPUPrefillExecutionPolicy::projectLatencyFeatures
                                   : TPUPrefillExecutionPolicy::projectCacheResolvedBatch;
        if (!projector(inputs, policy, maxSteps, projection, nullptr) ||
            !logical.setExecutionGeometry(projection.stepCount,
                                          projection.globalPaddedTokenSum)) {
            return false;
        }
        if (NonLinearPrefillModelR::usesCacheSourceForVersion(latency->binaryVersion)) {
            int64_t reads = 0, maximum = 0;
            for (const auto &input : inputs) {
                reads += input.externalReadTokens > 0;
                maximum = std::max(maximum, input.externalReadTokens);
            }
            if (!logical.setExecutionCacheSource(reads, maximum)) {
                return false;
            }
        }
        features = std::move(logical);
        basePrediction = NonLinearPrefillModelR::predictForVersion(
            features, latency->parameters, latency->binaryVersion);
    } else {
        basePrediction = NonLinearPrefillModelR::predict(
            features, NonLinearPrefillModelR::parametersForLegacyPrediction(
                          latency->parameters, latency->binaryVersion));
    }
    return batch.setPredictedLatencyMs(
        LatencyResidualModel::evaluate(basePrediction, latency->residualParameters,
                                       latency->residualLastUpdateTimeUs,
                                       autil::TimeUtility::currentTime(),
                                       getResidualRuntimeConfig())
            .predictionMs);
}

void MasterStateR::NodeCacheState::clear()
{
    _cachedBlockHashes.clear();
    _fullReadyBlockHashes.clear();
    _tailReadyBlockHashes.clear();
    _cachePolicy = NaviSchedCachePolicy();
    _cachePolicyReady.store(false, std::memory_order_release);
    _cacheStateReady.store(false, std::memory_order_release);
    _hybridTailRequired = false;
    _cachePolicyMismatchCount.store(0, std::memory_order_relaxed);
    _cacheRevision = 0;
    _agentCachedBlockCount = -1;
    _agentCacheHashChecksum = 0;
    _agentFullReadyBlockCount = -1;
    _agentFullReadyHashChecksum = 0;
    _agentTailReadyBlockCount = -1;
    _agentTailReadyHashChecksum = 0;
    _masterCacheHashChecksum = 0;
    _masterFullReadyHashChecksum = 0;
    _masterTailReadyHashChecksum = 0;
    _rtpCachedBlockCount = -1;
    _cacheRevisionGapCount = 0;
    _cacheStateMismatchCount = 0;
    _rtpCacheMirrorSampleCount = 0;
    _rtpCacheMirrorMismatchCount = 0;
    _pendingCacheResyncTriggerCount.store(0, std::memory_order_relaxed);
}

MasterStateR::NodeCacheState &MasterStateR::NodeState::cacheForType(
    NaviSchedCacheKeyType type)
{
    if (!validCacheKeyType(type)) {
        throw std::invalid_argument("invalid cache key type");
    }
    return type == NaviSchedCacheKeyType::V6D ? _v6dCacheState : *this;
}

const MasterStateR::NodeCacheState &MasterStateR::NodeState::cacheForType(
    NaviSchedCacheKeyType type) const
{
    if (!validCacheKeyType(type)) {
        throw std::invalid_argument("invalid cache key type");
    }
    return type == NaviSchedCacheKeyType::V6D ? _v6dCacheState : *this;
}

void MasterStateR::NodeState::clearCacheStateLocked()
{
    NodeCacheState::clear();
    autil::ScopedWriteLock lock(_v6dCacheState._cacheLock);
    _v6dCacheState.clear();
}

void MasterStateR::NodeState::collectDispatchWaitersLocked(
    PendingBatchReservation &reservation, const std::vector<uint64_t> &requestIds,
    std::vector<navi::AsyncPipePtr> &pipes)
{
    for (auto requestId : requestIds) {
        auto it = reservation.dispatchWaiters.find(requestId);
        if (it == reservation.dispatchWaiters.end()) {
            continue;
        }
        if (auto pipe = it->second.lock()) {
            pipes.emplace_back(std::move(pipe));
        }
        reservation.dispatchWaiters.erase(it);
    }
}

bool MasterStateR::NodeState::registerDispatchWaiter(int64_t batchId, uint64_t requestId,
                                                     const navi::AsyncPipePtr &pipe,
                                                     bool &acknowledged)
{
    acknowledged = false;
    if (batchId <= 0 || requestId == 0 || !pipe) {
        return false;
    }
    autil::ScopedLock lock(_runtimeLock);
    auto reservation = findReservationLocked(batchId);
    if (!reservation) {
        acknowledged = true;
        return true;
    }
    if (!reservation->batch.containsRequest(requestId)) {
        return false;
    }
    if ((reservation->batch.getRequestState(requestId) !=
             MasterBatchReservation::RequestState::WAITING &&
         !reservation->batch.isAwaitingBatchReady(requestId)) ||
        reservation->batch.isDispatchAcknowledged(requestId)) {
        acknowledged = true;
        return true;
    }
    auto [waiterIt, inserted] = reservation->dispatchWaiters.emplace(requestId, pipe);
    if (inserted) {
        return true;
    }
    auto current = waiterIt->second.lock();
    if (current && current != pipe) {
        return false;
    }
    waiterIt->second = pipe;
    return true;
}

bool MasterStateR::NodeState::unregisterDispatchWaiter(int64_t batchId,
                                                       uint64_t requestId,
                                                       const navi::AsyncPipePtr &pipe)
{
    if (batchId <= 0 || requestId == 0 || !pipe) {
        return false;
    }
    autil::ScopedLock lock(_runtimeLock);
    auto reservation = findReservationLocked(batchId);
    if (!reservation) {
        return false;
    }
    auto waiterIt = reservation->dispatchWaiters.find(requestId);
    if (waiterIt == reservation->dispatchWaiters.end()) {
        return false;
    }
    auto current = waiterIt->second.lock();
    if (current && current != pipe) {
        return false;
    }
    reservation->dispatchWaiters.erase(waiterIt);
    return true;
}

bool MasterStateR::NodeState::failRequestLocked(PendingBatchReservation &reservation,
                                                uint64_t requestId, bool &batchReleased)
{
    return failRequestsLocked(reservation, {requestId}, batchReleased);
}

bool MasterStateR::NodeState::failRequestsLocked(PendingBatchReservation &reservation,
                                                 const std::vector<uint64_t> &requestIds,
                                                 bool &batchReleased)
{
    std::vector<size_t> indexes;
    indexes.reserve(requestIds.size());
    for (const auto requestId : requestIds) {
        const auto index = reservation.batch.findRequest(requestId);
        if (index == reservation.batch.getRequestCount()) {
            batchReleased = false;
            return false;
        }
        indexes.push_back(index);
    }
    return failRequestIndexesLocked(reservation, indexes, batchReleased);
}

bool MasterStateR::NodeState::failRequestIndexesLocked(
    PendingBatchReservation &reservation, const std::vector<size_t> &requestIndexes,
    bool &batchReleased, FailureTiming *timing)
{
    const auto stamp = [timing]() {
        return timing ? autil::TimeUtility::monotonicTimeUs() : int64_t(0);
    };
    auto beginUs = stamp();
    batchReleased = false;
    auto candidate = reservation.batch;
    auto endUs = stamp();
    if (timing) timing->copyUs = endUs - beginUs;
    beginUs = endUs;
    MasterBatchReservation::Transition transition;
    const auto marked = candidate.markFailedIndexes(requestIndexes, transition);
    endUs = stamp();
    if (timing) timing->markFailedUs = endUs - beginUs;
    if (!marked || !transition.changed) {
        return false;
    }
    beginUs = endUs;
    const auto predicted = refreshReservationPredictionLocked(candidate, false);
    endUs = stamp();
    if (timing) timing->predictionUs = endUs - beginUs;
    if (!predicted) return false;
    beginUs = endUs;
    const auto updated = updateReservationPendingCacheLocked(candidate, &transition);
    endUs = stamp();
    if (timing) timing->pendingUpdateUs = endUs - beginUs;
    if (!updated) {
        return false;
    }
    beginUs = endUs;
    autil::ScopeGuard recordCommit([&]() {
        if (timing) timing->commitUs = stamp() - beginUs;
    });
    reservation.batch = std::move(candidate);
    for (auto index : requestIndexes) {
        reservation.dispatchWaiters.erase(
            reservation.batch._requests[index].request.requestId);
    }
    if (!transition.resolved) {
        return true;
    }
    eraseReservationLocked(reservation);
    batchReleased = true;
    return true;
}

bool MasterStateR::NodeState::cancelDispatchWaiter(int64_t batchId, uint64_t requestId,
                                                   const navi::AsyncPipePtr &pipe,
                                                   bool &batchReleased)
{
    batchReleased = false;
    if (batchId <= 0 || requestId == 0 || !pipe) {
        return false;
    }
    autil::ScopedLock lock(_runtimeLock);
    auto reservation = findReservationLocked(batchId);
    if (!reservation) {
        return false;
    }
    auto waiterIt = reservation->dispatchWaiters.find(requestId);
    if (waiterIt == reservation->dispatchWaiters.end()) {
        return false;
    }
    auto current = waiterIt->second.lock();
    if (current && current != pipe) {
        return false;
    }
    return failRequestLocked(*reservation, requestId, batchReleased);
}

bool MasterStateR::NodeState::reportDispatchFailed(int64_t batchId, uint64_t requestId,
                                                   bool &batchReleased)
{
    batchReleased = false;
    if (batchId <= 0 || requestId == 0) {
        return false;
    }
    autil::ScopedLock lock(_runtimeLock);
    auto reservation = findReservationLocked(batchId);
    if (!reservation) {
        return false;
    }
    return failRequestLocked(*reservation, requestId, batchReleased);
}

bool MasterStateR::NodeState::reportDispatchFailures(
    int64_t batchId, const std::vector<uint64_t> &requestIds, bool &batchReleased)
{
    batchReleased = false;
    if (batchId <= 0 || requestIds.empty()) {
        return false;
    }
    autil::ScopedLock lock(_runtimeLock);
    auto reservation = findReservationLocked(batchId);
    return reservation && failRequestsLocked(*reservation, requestIds, batchReleased);
}

bool MasterStateR::NodeState::abandonUnownedDispatch(int64_t batchId, uint64_t requestId,
                                                     bool &batchReleased)
{
    batchReleased = false;
    if (batchId <= 0 || requestId == 0) {
        return false;
    }
    autil::ScopedLock lock(_runtimeLock);
    auto reservation = findReservationLocked(batchId);
    if (!reservation || (reservation->batch.getRequestState(requestId) !=
                             MasterBatchReservation::RequestState::WAITING &&
                         !reservation->batch.isAwaitingBatchReady(requestId))) {
        return false;
    }
    if (reservation->batch.isDispatchAcknowledged(requestId)) {
        return false;
    }
    auto waiterIt = reservation->dispatchWaiters.find(requestId);
    if (waiterIt != reservation->dispatchWaiters.end()) {
        if (!waiterIt->second.expired()) {
            return false;
        }
        reservation->dispatchWaiters.erase(waiterIt);
    }
    return failRequestLocked(*reservation, requestId, batchReleased);
}

MasterStateR::NodeDispatchStats MasterStateR::NodeState::getDispatchStats() const
{
    NodeDispatchStats stats;
    stats.requestCount = _dispatchedRequestCount.load(std::memory_order_relaxed);
    stats.batchCount = _dispatchedBatchCount.load(std::memory_order_relaxed);
    return stats;
}

MasterStateR::ReservationStats MasterStateR::NodeState::getReservationStats() const
{
    ReservationStats stats;
    autil::ScopedLock lock(_runtimeLock);
    stats.batchCount = _pendingBatchReservations.size();
    for (const auto &reservationPtr : _pendingBatchReservations) {
        const auto &reservation = *reservationPtr;
        stats.requestCount += reservation.batch.getRequestCount();
        stats.waitingRequestCount += reservation.batch.getWaitingRequestCount();
        stats.dispatchedRequestCount += reservation.batch.getDispatchedRequestCount();
        stats.dispatchWaiterCount += reservation.dispatchWaiters.size();
        for (const auto &[requestId, waiter] : reservation.dispatchWaiters) {
            (void)requestId;
            stats.liveDispatchWaiterCount += waiter.expired() ? 0 : 1;
        }
    }
    return stats;
}

size_t MasterStateR::NodeState::getCachedBlockCount(
    NaviSchedCacheKeyType cacheKeyType) const
{
    if (!validCacheKeyType(cacheKeyType)) {
        return 0;
    }
    const auto &mirror = cacheForType(cacheKeyType);
    autil::ScopedReadLock lock(mirror._cacheLock);
    return mirror._cachedBlockHashes.size();
}

MasterStateR::CacheSyncStats MasterStateR::NodeState::getCacheSyncStats(
    NaviSchedCacheKeyType cacheKeyType) const
{
    if (!validCacheKeyType(cacheKeyType)) {
        return {};
    }
    const auto &mirror = cacheForType(cacheKeyType);
    const auto pendingCacheStats =
        _pendingCacheR->getStats(_info.agentIdHash, cacheKeyType);
    autil::ScopedReadLock lock(mirror._cacheLock);
    return {mirror._cacheRevision,
            mirror._agentCachedBlockCount,
            mirror._agentCacheHashChecksum,
            mirror._agentFullReadyBlockCount,
            mirror._agentFullReadyHashChecksum,
            mirror._agentTailReadyBlockCount,
            mirror._agentTailReadyHashChecksum,
            mirror._cachedBlockHashes.size(),
            mirror._masterCacheHashChecksum,
            mirror._fullReadyBlockHashes.size(),
            mirror._masterFullReadyHashChecksum,
            mirror._tailReadyBlockHashes.size(),
            mirror._masterTailReadyHashChecksum,
            mirror._rtpCachedBlockCount,
            mirror._cacheRevisionGapCount,
            mirror._cacheStateMismatchCount,
            mirror._rtpCacheMirrorSampleCount,
            mirror._rtpCacheMirrorMismatchCount,
            mirror._cacheStateReady.load(std::memory_order_relaxed),
            pendingCacheStats.batchCount,
            pendingCacheStats.uniqueBlockCount,
            pendingCacheStats.uniqueTailReadyBlockCount,
            mirror._cachePolicy.blockSize,
            mirror._cachePolicy.linearStep,
            mirror._cachePolicyReady.load(std::memory_order_relaxed),
            mirror._hybridTailRequired,
            mirror._cachePolicyMismatchCount.load(std::memory_order_relaxed),
            pendingCacheStats.scheduledBatchCount,
            pendingCacheStats.agentBatchCount,
            pendingCacheStats.agentRevision,
            pendingCacheStats.agentChecksum,
            pendingCacheStats.agentRevisionGapCount,
            pendingCacheStats.agentStateMismatchCount,
            mirror._pendingCacheResyncTriggerCount.load(std::memory_order_relaxed),
            pendingCacheStats.agentSnapshotReady,
            mirror._cachePolicy.fullAttentionQueryDropTokens};
}

int64_t MasterStateR::NodeState::getMatchedBlockCount(
    const NaviSchedBlockHashes &blockHashes) const
{
    autil::ScopedReadLock lock(_cacheLock);
    return matchBlockCount(blockHashes);
}

int64_t MasterStateR::NodeState::getMatchedBlockCountForRequest(
    const NaviSchedScheduleRequest &request) const
{
    autil::ScopedReadLock lock(_cacheLock);
    if (!_cachePolicyReady.load(std::memory_order_relaxed) ||
        request.blockSize != _cachePolicy.blockSize) {
        _cachePolicyMismatchCount.fetch_add(1, std::memory_order_relaxed);
        return 0;
    }
    const auto local = _pendingCacheR->getMatchedBlockCount(
        _info.agentIdHash, BlockHashUtil::get(request.blockHashes), _cachedBlockHashes,
        _fullReadyBlockHashes, _tailReadyBlockHashes, _hybridTailRequired,
        BlockHashUtil::getMatchableBlockCount(request.promptTokenCount, request.blockSize,
                                              request.blockHashes),
        _cachePolicy.fullAttentionQueryDropTokens / _cachePolicy.blockSize);
    return local + matchExternalSuffix(BlockHashUtil::get(request.blockHashes),
                                       request.promptTokenCount, request.blockSize,
                                       local);
}

bool MasterStateR::NodeState::getExecutionCacheInputs(
    const NaviSchedScheduleRequest &request, int64_t &computedTokens,
    int64_t &sharedPrefixBoundary) const
{
    computedTokens = 0;
    sharedPrefixBoundary = 0;
    std::vector<NaviSchedScheduleItem> requests(1);
    requests.front().request = request;
    std::vector<TPUPrefillRequestInput> inputs;
    if (!getExecutionCacheInputs(requests, inputs)) {
        return false;
    }
    computedTokens = inputs.front().computedTokens;
    sharedPrefixBoundary = inputs.front().sharedPrefixBoundary;
    return true;
}

bool MasterStateR::NodeState::getExecutionCacheInputs(
    const std::vector<NaviSchedScheduleItem> &requests,
    std::vector<TPUPrefillRequestInput> &inputs, int64_t *externalSuffixBlockCounts,
    TPUPrefillCacheObservationBatch *observation) const
{
    if (observation) {
        observation->pool.reset();
        observation->fullPrefixBlocksBeforeQueryDrop.clear();
    }
    if (requests.empty()) {
        inputs.clear();
        return true;
    }
    if (requests.size() > std::numeric_limits<size_t>::max() / 2) {
        inputs.clear();
        return false;
    }
    const auto publishedPolicy =
        std::atomic_load_explicit(&_publishedSnapshot, std::memory_order_acquire);
    const auto count = requests.size();
    std::unique_ptr<int64_t[]> counts(new int64_t[count * 2]);
    auto *boundaries = counts.get() + count;
    TPUPrefillCacheObservation poolObservation;
    if (observation) {
        observation->fullPrefixBlocksBeforeQueryDrop.resize(count);
    }
    std::vector<int64_t> ownedExternal(count, 0);
    auto *external =
        externalSuffixBlockCounts ? externalSuffixBlockCounts : ownedExternal.data();
    if (!matchCacheBatch(requests, counts.get(), boundaries, external,
                         observation ? &poolObservation : nullptr,
                         observation ? observation->fullPrefixBlocksBeforeQueryDrop.data()
                                     : nullptr)) {
        if (observation) {
            observation->fullPrefixBlocksBeforeQueryDrop.clear();
        }
        inputs.clear();
        return false;
    }
    inputs.resize(count);
    for (size_t i = 0; i < count; ++i) {
        if (requests[i].request.promptTokenCount <= 0) {
            if (observation) {
                observation->fullPrefixBlocksBeforeQueryDrop.clear();
            }
            inputs.clear();
            return false;
        }
        inputs[i] = {requests[i].request.promptTokenCount,
                     counts[i] * requests[i].request.blockSize, boundaries[i],
                     external[i] * requests[i].request.blockSize};
        const auto position = requests[i].request.explicitCacheCheckpointPosition;
        if (position < 0 || position > inputs[i].promptTokens) {
            inputs.clear();
            return false;
        }
        if (position > 0) {
            if (!publishedPolicy || !publishedPolicy->prefillExecutionPolicy ||
                publishedPolicy->prefillExecutionPolicy->schedulerBlockSize <= 0) {
                inputs.clear();
                return false;
            }
            const auto blockSize =
                publishedPolicy->prefillExecutionPolicy->schedulerBlockSize;
            const auto boundary = position / blockSize * blockSize;
            // The runtime ignores sub-block directives; they do not erase a cache
            // junction.
            if (boundary > 0) {
                inputs[i].sharedPrefixBoundary = boundary;
            }
        }
    }
    if (observation) {
        observation->pool = poolObservation;
    }
    return true;
}

int64_t MasterStateR::NodeState::matchExternalSuffix(
    const NaviSchedBlockHashes &blockHashes, int64_t promptTokens, int64_t blockSize,
    int64_t localBlocks, int64_t excludedScheduleBatchId,
    MasterPendingCacheR::MatchBreakdown *details) const
{
    const auto &external = _v6dCacheState;
    autil::ScopedReadLock lock(external._cacheLock);
    const auto &policy = external._cachePolicy;
    if (!external._cachePolicyReady.load(std::memory_order_relaxed) ||
        !policy.valid(NaviSchedCacheKeyType::V6D) || blockSize != policy.blockSize ||
        localBlocks < 0 || localBlocks % policy.linearStep != 0) {
        return 0;
    }
    const auto limit = static_cast<size_t>(
        std::max<int64_t>(0, promptTokens - policy.fullAttentionQueryDropTokens) /
        blockSize);
    if (details) {
        *details = _pendingCacheR->getMatchBreakdown(
            _info.agentIdHash, blockHashes, external._cachedBlockHashes,
            external._fullReadyBlockHashes, external._tailReadyBlockHashes,
            external._hybridTailRequired, limit, 0, NaviSchedCacheKeyType::V6D,
            static_cast<size_t>(localBlocks), static_cast<size_t>(policy.linearStep));
        return details->matchedBlockCount;
    }
    return _pendingCacheR->getMatchedBlockCountExcludingSchedule(
        _info.agentIdHash, excludedScheduleBatchId, blockHashes,
        external._cachedBlockHashes, external._fullReadyBlockHashes,
        external._tailReadyBlockHashes, external._hybridTailRequired, limit, 0,
        NaviSchedCacheKeyType::V6D, static_cast<size_t>(localBlocks),
        static_cast<size_t>(policy.linearStep));
}

int64_t MasterStateR::NodeState::matchBlockCount(
    const NaviSchedBlockHashes &blockHashes) const
{
    return _pendingCacheR->getMatchedBlockCount(
        _info.agentIdHash, blockHashes, _cachedBlockHashes, _fullReadyBlockHashes,
        _tailReadyBlockHashes, _hybridTailRequired, std::numeric_limits<size_t>::max(),
        _cachePolicyReady.load(std::memory_order_relaxed)
            ? _cachePolicy.fullAttentionQueryDropTokens / _cachePolicy.blockSize
            : 0);
}

void MasterStateR::NodeState::getMatchedBlockCounts(
    const std::vector<NaviSchedScheduleItem> &requests, int64_t *matchedBlockCounts,
    int64_t *externalSuffixBlockCounts) const
{
    matchCacheBatch(requests, matchedBlockCounts, nullptr, externalSuffixBlockCounts);
}

void MasterStateR::NodeState::getMatchedBlockCounts(
    const std::vector<NaviSchedScheduleItem> &requests, int64_t *matchedBlockCounts,
    const std::vector<size_t> &commonPrefixes) const
{
    matchCacheBatch(requests, matchedBlockCounts, nullptr, nullptr, nullptr, nullptr,
                    &commonPrefixes);
}

bool MasterStateR::NodeState::getReadyExternalCacheKeyStates(
    const NaviSchedBlockHashes &keys, std::vector<uint8_t> &states,
    NaviSchedCachePolicy &policy) const
{
    const auto &external = _v6dCacheState;
    autil::ScopedReadLock lock(external._cacheLock);
    if (!external._cachePolicyReady.load(std::memory_order_relaxed) ||
        !external._cacheStateReady.load(std::memory_order_relaxed) ||
        !external._cachePolicy.valid(NaviSchedCacheKeyType::V6D)) {
        return false;
    }
    policy = external._cachePolicy;
    const auto &full = external._hybridTailRequired ? external._fullReadyBlockHashes
                                                    : external._cachedBlockHashes;
    states.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        states[i] = (full.count(keys[i]) ? 1 : 0) |
                    (external._tailReadyBlockHashes.count(keys[i]) ? 2 : 0);
    }
    return true;
}

bool MasterStateR::NodeState::getReadyExternalCacheCounts(
    const std::vector<NaviSchedScheduleItem> &requests, std::vector<int64_t> &counts,
    NaviSchedCachePolicy &policy) const
{
    const auto &external = _v6dCacheState;
    autil::ScopedReadLock lock(external._cacheLock);
    if (!external._cachePolicyReady.load(std::memory_order_relaxed) ||
        !external._cacheStateReady.load(std::memory_order_relaxed) ||
        !external._cachePolicy.valid(NaviSchedCacheKeyType::V6D)) {
        return false;
    }
    policy = external._cachePolicy;
    counts.resize(requests.size());
    std::vector<size_t> limits(requests.size());
    for (size_t i = 0; i < requests.size(); ++i) {
        limits[i] = std::max<int64_t>(0, requests[i].request.promptTokenCount -
                                             policy.fullAttentionQueryDropTokens) /
                    policy.blockSize;
    }
    // Peer reads may use published READY objects only, never planned publications.
    return _pendingCacheR->getMatchedBlockCounts(
               _info.agentIdHash, requests, policy.blockSize, external._cachedBlockHashes,
               external._fullReadyBlockHashes, external._tailReadyBlockHashes,
               external._hybridTailRequired, counts.data(), 0, NaviSchedCacheKeyType::V6D,
               nullptr, nullptr, nullptr, limits.data(), policy.linearStep, false) == 0;
}

bool MasterStateR::NodeState::matchCacheBatch(
    const std::vector<NaviSchedScheduleItem> &requests, int64_t *matchedBlockCounts,
    int64_t *sharedPrefixBoundaries, int64_t *externalSuffixBlockCounts,
    TPUPrefillCacheObservation *observation, int64_t *fullPrefixBlockCounts,
    const std::vector<size_t> *commonPrefixes) const
{
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    int64_t cacheAcquiredUs = beginUs;
    size_t physicalBlocks = 0;
    MasterPendingCacheR::BatchMatchTiming timing;
    autil::ScopeGuard recordMatchTiming([&]() {
    const auto endUs = autil::TimeUtility::monotonicTimeUs();
    if (endUs - beginUs >= 10'000) {
        // Emit after releasing both cache locks, at most once per second process-wide.
        static std::atomic<int64_t> lastLogUs{0};
        auto previousLogUs = lastLogUs.load(std::memory_order_relaxed);
        if (endUs - previousLogUs >= 1'000'000 &&
            lastLogUs.compare_exchange_strong(previousLogUs, endUs,
                                              std::memory_order_relaxed)) {
            const auto info =
                std::atomic_load_explicit(&_infoSnapshot, std::memory_order_acquire);
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-master-kv-match-timing] agent=%s requests=%lu "
                "first_request_id=%lu total_us=%ld physical_lock_wait_us=%ld "
                "physical_lock_hold_us=%ld pending_node_lookup_us=%ld "
                "pending_lock_wait_us=%ld match_us=%ld physical_blocks=%lu "
                "scheduled_batches=%lu agent_batches=%lu scheduled_blocks=%lu "
                "agent_blocks=%lu "
                "prefix_comparisons=%lu reused_full_prefix_blocks=%lu "
                "full_prefix_probes=%lu tail_probes=%lu",
                info ? info->agentId.c_str() : "unknown", requests.size(),
                requests.empty() ? uint64_t(0) : requests.front().request.requestId,
                endUs - beginUs, cacheAcquiredUs - beginUs, endUs - cacheAcquiredUs,
                timing.nodeLookupUs, timing.lockWaitUs, timing.matchUs, physicalBlocks,
                timing.scheduledBatches, timing.agentBatches, timing.scheduledBlocks,
                timing.agentBlocks, timing.prefixComparisonCount,
                timing.reusedFullPrefixBlockCount, timing.fullPrefixProbeCount,
                timing.tailProbeCount);
        }
    }
    });
    if (externalSuffixBlockCounts) {
        std::fill(externalSuffixBlockCounts, externalSuffixBlockCounts + requests.size(),
                  0);
    }
    autil::ScopedReadLock lock(_cacheLock);
    cacheAcquiredUs = autil::TimeUtility::monotonicTimeUs();
    physicalBlocks = _cachedBlockHashes.size();
    const bool ready = _cachePolicyReady.load(std::memory_order_relaxed);
    if (observation) {
        *observation = {_cacheRevision,
                        _cacheStateReady.load(std::memory_order_relaxed),
                        static_cast<uint64_t>(_cachedBlockHashes.size()),
                        static_cast<uint64_t>(_fullReadyBlockHashes.size()),
                        static_cast<uint64_t>(_tailReadyBlockHashes.size()),
                        _masterCacheHashChecksum,
                        _masterFullReadyHashChecksum,
                        _masterTailReadyHashChecksum};
    }
    // Geometry adds an output to the existing batch observation, not another read.
    if (sharedPrefixBoundaries) {
        if (!ready) {
            return false;
        }
    }
    const auto count = requests.size();
    const auto mismatchCount = _pendingCacheR->getMatchedBlockCounts(
        _info.agentIdHash, requests, ready ? _cachePolicy.blockSize : -1,
        _cachedBlockHashes, _fullReadyBlockHashes, _tailReadyBlockHashes,
        _hybridTailRequired, matchedBlockCounts,
        ready ? _cachePolicy.fullAttentionQueryDropTokens / _cachePolicy.blockSize : 0,
        NaviSchedCacheKeyType::BLOCK_POOL, fullPrefixBlockCounts,
        _hybridTailRequired ? sharedPrefixBoundaries : nullptr,
        nullptr, nullptr, 1, true, &timing, commonPrefixes);
    if (mismatchCount > 0) {
        if (sharedPrefixBoundaries) {
            return false;
        }
        _cachePolicyMismatchCount.fetch_add(mismatchCount, std::memory_order_relaxed);
    }
    if (!matchedBlockCounts || requests.empty()) {
        return ready && mismatchCount == 0;
    }
    if (sharedPrefixBoundaries && !_hybridTailRequired) {
        std::fill(sharedPrefixBoundaries, sharedPrefixBoundaries + count, 0);
    } else if (sharedPrefixBoundaries) {
        for (size_t i = 0; i < count; ++i) {
            const auto &request = requests[i].request;
            const auto full = sharedPrefixBoundaries[i];
            // Marconi's boundary is local; external hits do not change it.
            sharedPrefixBoundaries[i] =
                full > matchedBlockCounts[i] ? full * request.blockSize : 0;
        }
    }
    const auto &external = _v6dCacheState;
    {
        autil::ScopedReadLock externalLock(external._cacheLock);
        const auto &policy = external._cachePolicy;
        if (external._cachePolicyReady.load(std::memory_order_relaxed) &&
            policy.valid(NaviSchedCacheKeyType::V6D) &&
            policy.blockSize == _cachePolicy.blockSize) {
            std::vector<size_t> limits(count);
            std::vector<int64_t> suffixStorage;
            if (!externalSuffixBlockCounts) {
                suffixStorage.resize(count, 0);
            }
            auto *suffix = externalSuffixBlockCounts ? externalSuffixBlockCounts
                                                     : suffixStorage.data();
            for (size_t i = 0; i < count; ++i) {
                limits[i] = static_cast<size_t>(
                    std::max<int64_t>(0, requests[i].request.promptTokenCount -
                                             policy.fullAttentionQueryDropTokens) /
                    policy.blockSize);
            }
            _pendingCacheR->getMatchedBlockCounts(
                _info.agentIdHash, requests, policy.blockSize,
                external._cachedBlockHashes, external._fullReadyBlockHashes,
                external._tailReadyBlockHashes, external._hybridTailRequired, suffix, 0,
                NaviSchedCacheKeyType::V6D, nullptr, nullptr, matchedBlockCounts,
                limits.data(), static_cast<size_t>(policy.linearStep));
            for (size_t i = 0; i < count; ++i) {
                matchedBlockCounts[i] += suffix[i];
            }
        }
    }
    // Both matchers clip to the prompt, and the external suffix starts at the
    // local hit. Their sum is already bounded without another request scan.
    return ready && mismatchCount == 0;
}

bool MasterStateR::NodeState::getCachePolicy(NaviSchedCachePolicy &policy,
                                             NaviSchedCacheKeyType cacheKeyType) const
{
    if (!validCacheKeyType(cacheKeyType)) {
        return false;
    }
    const auto &mirror = cacheForType(cacheKeyType);
    autil::ScopedReadLock lock(mirror._cacheLock);
    if (!mirror._cachePolicyReady.load(std::memory_order_relaxed)) {
        return false;
    }
    policy = mirror._cachePolicy;
    return true;
}

MasterStateR::CacheMatchBreakdown MasterStateR::NodeState::getCacheMatchBreakdown(
    const NaviSchedScheduleRequest &request, NaviSchedCacheKeyType cacheKeyType) const
{
    if (!validCacheKeyType(cacheKeyType)) {
        return {};
    }
    const auto &mirror = cacheForType(cacheKeyType);
    CacheMatchBreakdown breakdown;
    autil::ScopedReadLock lock(mirror._cacheLock);
    if (!mirror._cachePolicyReady.load(std::memory_order_relaxed) ||
        request.blockSize != mirror._cachePolicy.blockSize) {
        return breakdown;
    }
    const bool externalType = cacheKeyType == NaviSchedCacheKeyType::V6D;
    const auto queryLimit =
        externalType
            ? static_cast<size_t>(
                  std::max<int64_t>(
                      0, request.promptTokenCount -
                             mirror._cachePolicy.fullAttentionQueryDropTokens) /
                  request.blockSize)
            : BlockHashUtil::getMatchableBlockCount(
                  request.promptTokenCount, request.blockSize, request.blockHashes);
    const auto matchBreakdown = _pendingCacheR->getMatchBreakdown(
        _info.agentIdHash, BlockHashUtil::get(request.blockHashes),
        mirror._cachedBlockHashes, mirror._fullReadyBlockHashes,
        mirror._tailReadyBlockHashes, mirror._hybridTailRequired, queryLimit,
        externalType ? 0
                     : mirror._cachePolicy.fullAttentionQueryDropTokens /
                           mirror._cachePolicy.blockSize,
        cacheKeyType, 0, externalType ? mirror._cachePolicy.linearStep : 1);
    const auto pendingStats = _pendingCacheR->getStats(_info.agentIdHash, cacheKeyType);
    breakdown.staticMatchedBlockCount = matchBreakdown.staticMatchedBlockCount;
    breakdown.matchedBlockCount = matchBreakdown.matchedBlockCount;
    breakdown.firstPendingBlockSourceBatchId =
        matchBreakdown.firstPendingBlockSourceBatchId;
    breakdown.pendingTailSourceBatchId = matchBreakdown.pendingTailSourceBatchId;
    breakdown.cacheRevision = mirror._cacheRevision;
    breakdown.pendingBatchCount = pendingStats.batchCount;
    breakdown.pendingBlockCount = pendingStats.uniqueBlockCount;
    breakdown.valid = true;
    if (!externalType) {
        const auto local = breakdown.matchedBlockCount;
        MasterPendingCacheR::MatchBreakdown external;
        breakdown.matchedBlockCount += matchExternalSuffix(
            BlockHashUtil::get(request.blockHashes), request.promptTokenCount,
            request.blockSize, local, 0, &external);
        breakdown.staticMatchedBlockCount += external.staticMatchedBlockCount;
        if (breakdown.firstPendingBlockSourceBatchId == 0) {
            breakdown.firstPendingBlockSourceBatchId =
                external.firstPendingBlockSourceBatchId;
        }
        if (breakdown.pendingTailSourceBatchId == 0) {
            breakdown.pendingTailSourceBatchId = external.pendingTailSourceBatchId;
        }
    }
    return breakdown;
}

std::shared_ptr<const MasterStateR::LatencyInfo> MasterStateR::NodeState::getLatencyInfo()
    const
{
    auto slot = std::atomic_load_explicit(&_latencyModelSlot, std::memory_order_acquire);
    return slot ? slot->getModel() : nullptr;
}

void MasterStateR::NodeState::setLatencyModelSlot(
    const MasterLatencyModelR::LiveModelSlotPtr &slot)
{
    std::atomic_store_explicit(&_latencyModelSlot, slot, std::memory_order_release);
    autil::ScopedWriteLock lock(_infoLock);
    rebuildInfoSnapshot();
}

void MasterStateR::NodeState::clearLatencyModelSlot()
{
    std::atomic_store_explicit(&_latencyModelSlot,
                               MasterLatencyModelR::LiveModelSlotPtr(),
                               std::memory_order_release);
    autil::ScopedWriteLock lock(_infoLock);
    rebuildInfoSnapshot();
}

LatencyResidualModel::RuntimeConfig MasterStateR::NodeState::getResidualRuntimeConfig()
    const
{
    const auto config =
        std::atomic_load_explicit(&_residualRuntimeConfig, std::memory_order_acquire);
    return config ? *config : LatencyResidualModel::RuntimeConfig();
}

void MasterStateR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

navi::ErrorCode MasterStateR::init(navi::ResourceInitContext &ctx)
{
    return navi::EC_NONE;
}

MasterStateR::NodeStatePtr MasterStateR::registerAgent(const std::string &nodeId,
                                                       const std::string &bizName,
                                                       const std::string &address,
                                                       int64_t totalBlocks)
{
    if (nodeId.empty() || bizName.empty()) {
        return nullptr;
    }
    auto agentIdHash = autil::HashAlgorithm::hashString64(nodeId.data(), nodeId.size());
    auto state = getOrCreateNode(agentIdHash, nodeId);
    updateNodeBiz(agentIdHash, state, bizName);
    {
        autil::ScopedWriteLock lock(state->_infoLock);
        auto &node = state->_info;
        bool changed = false;
        if (!address.empty()) {
            changed |= node.collectorAddress != address;
            node.collectorAddress = address;
        }
        if (totalBlocks > 0) {
            autil::ScopedLock runtimeLock(state->_runtimeLock);
            state->_runtimeState.load.totalBlocks = totalBlocks;
        }
        if (changed) {
            state->rebuildInfoSnapshot();
        }
    }
    refreshBizNode(state);
    return state;
}

MasterStateR::NodeStatePtr MasterStateR::createAgent(const std::string &nodeId,
                                                     const std::string &bizName,
                                                     const std::string &address,
                                                     int64_t totalBlocks)
{
    if (nodeId.empty() || bizName.empty()) {
        return nullptr;
    }
    auto agentIdHash = autil::HashAlgorithm::hashString64(nodeId.data(), nodeId.size());
    auto state = std::make_shared<NodeState>();
    state->_pendingCacheR = _pendingCacheR;
    state->_info.agentIdHash = agentIdHash;
    state->_info.agentId = nodeId;
    state->_info.bizName = bizName;
    state->_info.collectorAddress = address;
    state->_runtimeState.load.totalBlocks = totalBlocks;
    state->rebuildInfoSnapshot();

    std::vector<navi::AsyncPipePtr> dispatchWaiters;
    {
        autil::ScopedWriteLock nodeIndexLock(_nodeIndexLock);
        auto deactivate = [&dispatchWaiters](const NodeStatePtr &oldState) {
            if (!oldState) {
                return;
            }
            oldState->clearLatencyModelSlot();
            autil::ScopedLock runtimeLock(oldState->_runtimeLock);
            oldState->_acceptReservations = false;
            for (const auto &reservationPtr : oldState->_pendingBatchReservations) {
                const auto &reservation = *reservationPtr;
                for (const auto &[requestId, waiter] : reservation.dispatchWaiters) {
                    (void)requestId;
                    if (auto pipe = waiter.lock()) {
                        dispatchWaiters.emplace_back(std::move(pipe));
                    }
                }
            }
            oldState->clearReservationsLocked();
        };
        auto pendingIt = _pendingNodes.find(agentIdHash);
        if (_pendingNodes.end() != pendingIt) {
            deactivate(pendingIt->second);
            _pendingNodes.erase(pendingIt);
        }
        for (auto bizIt = _readyNodesByBiz.begin(); bizIt != _readyNodesByBiz.end();) {
            auto nodeIt = bizIt->second->find(agentIdHash);
            if (bizIt->second->end() == nodeIt) {
                ++bizIt;
                continue;
            }
            deactivate(nodeIt->second);
            auto readyNodes = std::make_shared<NodeStateMap>(*bizIt->second);
            readyNodes->erase(agentIdHash);
            if (readyNodes->empty()) {
                bizIt = _readyNodesByBiz.erase(bizIt);
            } else {
                bizIt->second = std::move(readyNodes);
                ++bizIt;
            }
        }
        _pendingCacheR->remove(agentIdHash);
        _pendingNodes[agentIdHash] = state;
    }
    for (const auto &pipe : dispatchWaiters) {
        pipe->setEof();
    }
    return state;
}

bool MasterStateR::removeAgent(const NodeStatePtr &state)
{
    if (!state) {
        return false;
    }
    auto info =
        std::atomic_load_explicit(&state->_infoSnapshot, std::memory_order_acquire);
    if (!info) {
        return false;
    }
    std::vector<navi::AsyncPipePtr> dispatchWaiters;
    bool removed = false;
    {
        autil::ScopedWriteLock nodeIndexLock(_nodeIndexLock);
        auto pendingIt = _pendingNodes.find(info->agentIdHash);
        if (_pendingNodes.end() != pendingIt && pendingIt->second == state) {
            _pendingNodes.erase(pendingIt);
            removed = true;
        }
        auto bizIt = _readyNodesByBiz.find(info->bizName);
        if (_readyNodesByBiz.end() != bizIt) {
            auto nodeIt = bizIt->second->find(info->agentIdHash);
            if (bizIt->second->end() != nodeIt && nodeIt->second == state) {
                removeBizNode(info->bizName, info->agentIdHash, state);
                removed = true;
            }
        }
        if (removed) {
            autil::ScopedLock runtimeLock(state->_runtimeLock);
            state->_acceptReservations = false;
            for (const auto &reservationPtr : state->_pendingBatchReservations) {
                const auto &reservation = *reservationPtr;
                for (const auto &[requestId, waiter] : reservation.dispatchWaiters) {
                    (void)requestId;
                    if (auto pipe = waiter.lock()) {
                        dispatchWaiters.emplace_back(std::move(pipe));
                    }
                }
            }
            state->clearReservationsLocked();
            state->_pendingCacheR->remove(info->agentIdHash);
        }
    }
    for (const auto &pipe : dispatchWaiters) {
        pipe->setEof();
    }
    if (removed) {
        state->clearLatencyModelSlot();
    }
    return removed;
}

bool MasterStateR::openAgentCollector(const std::string &nodeId,
                                      const std::string &collectGraphId)
{
    if (nodeId.empty()) {
        return false;
    }
    autil::ScopedWriteLock lock(_collectorLock);
    _agentCollectors[nodeId] = collectGraphId;
    return true;
}

bool MasterStateR::closeAgentCollector(const std::string &nodeId,
                                       const std::string &collectorId)
{
    autil::ScopedWriteLock lock(_collectorLock);
    auto it = _agentCollectors.find(nodeId);
    if (_agentCollectors.end() == it || it->second != collectorId) {
        return false;
    }
    _agentCollectors.erase(it);
    return true;
}

bool MasterStateR::stopAgentCollector(const std::string &nodeId,
                                      const std::string &collectorId)
{
    NodeStatePtr state;
    {
        autil::ScopedWriteLock lock(_collectorLock);
        auto it = _agentCollectors.find(nodeId);
        if (_agentCollectors.end() == it || it->second != collectorId) {
            return false;
        }
        auto stateIt = _collectorNodeStates.find(collectorId);
        if (_collectorNodeStates.end() != stateIt) {
            state = stateIt->second.lock();
        }
        _agentCollectors.erase(it);
    }
    auto update = std::make_shared<AgentUpdateToMasterData>();
    update->event = NaviSchedAgentEvent::AGENT_STOPPING;
    update->eventTimeUs = autil::TimeUtility::currentTime();
    update->control = std::make_shared<AgentControlToMasterData>();
    if (state) {
        reportAgent(state, update);
    } else {
        reportAgent(nodeId, update);
    }
    return true;
}

bool MasterStateR::bindAgentCollector(const std::string &collectorId,
                                      const std::string &nodeId,
                                      const NodeStatePtr &state)
{
    if (collectorId.empty() || nodeId.empty()) {
        return false;
    }
    autil::ScopedWriteLock lock(_collectorLock);
    _collectorNodeIds[collectorId] = nodeId;
    if (state) {
        _collectorNodeStates[collectorId] = state;
    } else {
        _collectorNodeStates.erase(collectorId);
    }
    _agentCollectors[nodeId] = collectorId;
    return true;
}

bool MasterStateR::unbindAgentCollector(const std::string &collectorId)
{
    autil::ScopedWriteLock lock(_collectorLock);
    _collectorNodeStates.erase(collectorId);
    return _collectorNodeIds.erase(collectorId) > 0;
}

bool MasterStateR::getAgentCollectorNodeId(const std::string &collectorId,
                                           std::string &nodeId) const
{
    autil::ScopedReadLock lock(_collectorLock);
    auto it = _collectorNodeIds.find(collectorId);
    if (_collectorNodeIds.end() == it) {
        return false;
    }
    nodeId = it->second;
    return true;
}

bool MasterStateR::resetAgentInstance(const std::string &nodeId)
{
    if (nodeId.empty()) {
        return false;
    }
    auto agentIdHash = autil::HashAlgorithm::hashString64(nodeId.data(), nodeId.size());
    auto state = std::const_pointer_cast<NodeState>(getNodeState(agentIdHash));
    return resetAgentState(state);
}

bool MasterStateR::resetAgentState(const NodeStatePtr &state,
                                   MasterPendingCacheR::Stats *clearedPendingStats)
{
    if (!state) {
        return false;
    }
    auto info =
        std::atomic_load_explicit(&state->_infoSnapshot, std::memory_order_acquire);
    if (!info) {
        return false;
    }
    std::vector<navi::AsyncPipePtr> dispatchWaiters;
    {
        autil::ScopedReadLock nodeIndexLock(_nodeIndexLock);
        if (!isCurrentNodeStateLocked(info->agentIdHash, state)) {
            return false;
        }
        // Remove the live model while the node identity is still verified so a
        // reset cannot leave the scheduler's model pool serving a dead node.
        if (_masterLatencyModelR) {
            _masterLatencyModelR->removeAgent(info->agentId);
        }
        {
            autil::ScopedWriteLock infoLock(state->_infoLock);
            auto &node = state->_info;
            node.online = false;
            node.serverInfo = NaviSchedServerInfo();
            state->rebuildInfoSnapshot();
        }
        {
            autil::ScopedLock runtimeLock(state->_runtimeLock);
            state->_acceptReservations = false;
            state->_runtimeState = NodeState::RuntimeState();
            state->_executionUtilization.store(-1.0, std::memory_order_relaxed);
            state->_executionBusyUtilization.store(-1.0, std::memory_order_relaxed);
            for (const auto &reservationPtr : state->_pendingBatchReservations) {
                const auto &reservation = *reservationPtr;
                for (const auto &[requestId, waiter] : reservation.dispatchWaiters) {
                    (void)requestId;
                    if (auto pipe = waiter.lock()) {
                        dispatchWaiters.emplace_back(std::move(pipe));
                    }
                }
            }
            state->clearReservationsLocked();
            autil::ScopedWriteLock cacheLock(state->_cacheLock);
            state->clearCacheStateLocked();
            if (clearedPendingStats) {
                *clearedPendingStats = state->_pendingCacheR->getStats(info->agentIdHash);
            }
            state->_pendingCacheR->clear(info->agentIdHash);
        }
    }
    state->clearLatencyModelSlot();
    for (const auto &pipe : dispatchWaiters) {
        pipe->setEof();
    }
    refreshBizNode(state);
    return true;
}

bool MasterStateR::reportAgent(const std::string &agentId,
                               const AgentUpdateToMasterDataConstPtr &update)
{
    if (agentId.empty()) {
        return false;
    }
    auto agentIdHash = autil::HashAlgorithm::hashString64(agentId.data(), agentId.size());
    auto state = getOrCreateNode(agentIdHash, agentId);
    return reportAgent(state, update);
}

bool MasterStateR::reportAgent(const NodeStatePtr &state,
                               const AgentUpdateToMasterDataConstPtr &update,
                               bool *resyncRequired, bool *pendingCacheResyncRequired,
                               MasterPendingCacheR::ApplyResult *pendingCacheApplyResult,
                               const std::string &latencySourceId,
                               MasterLatencyModelR::RecordResult *latencyRecordResult,
                               AgentUpdateTiming *timing)
{
    if (timing) {
        *timing = {};
        timing->beginMonoUs = autil::TimeUtility::monotonicTimeUs();
    }
    autil::ScopeGuard finishUpdateTiming([timing]() {
        if (timing) {
            timing->endMonoUs = autil::TimeUtility::monotonicTimeUs();
        }
    });
    if (resyncRequired) {
        *resyncRequired = false;
    }
    if (pendingCacheResyncRequired) {
        *pendingCacheResyncRequired = false;
    }
    if (pendingCacheApplyResult) {
        *pendingCacheApplyResult = {};
    }
    if (latencyRecordResult) {
        *latencyRecordResult = {};
    }
    if (!state || !update || !update->valid()) {
        return false;
    }
    const auto infoSnapshot =
        std::atomic_load_explicit(&state->_infoSnapshot, std::memory_order_acquire);
    if (!infoSnapshot) {
        return false;
    }
    const auto agentId = infoSnapshot->agentId;
    const auto agentIdHash = infoSnapshot->agentIdHash;
    auto event = update->event;
    AgentSchedulerConfig acknowledgedSchedulerConfig;
    std::string acknowledgedSchedulerConfigJson;
    std::string schedulerConfigError;
    const auto hasSchedulerConfig = update->schedulerConfigGeneration > 0;
    if (hasSchedulerConfig && !AgentSchedulerConfigR::parse(update->schedulerConfigJson,
                                                            acknowledgedSchedulerConfig,
                                                            schedulerConfigError)) {
        NAVI_KERNEL_LOG(ERROR,
                        "reject invalid Agent scheduler config ACK, agentId [%s], "
                        "generation [%lu], error [%s]",
                        agentId.c_str(), update->schedulerConfigGeneration,
                        schedulerConfigError.c_str());
        return false;
    }
    if (hasSchedulerConfig) {
        acknowledgedSchedulerConfigJson =
            AgentSchedulerConfigR::toJson(acknowledgedSchedulerConfig);
        if (AgentSchedulerConfigR::hashJson(acknowledgedSchedulerConfigJson) !=
            update->schedulerConfigHash) {
            NAVI_KERNEL_LOG(ERROR,
                            "reject Agent scheduler config ACK hash mismatch, "
                            "agentId [%s], generation [%lu], hash [%s]",
                            agentId.c_str(), update->schedulerConfigGeneration,
                            update->schedulerConfigHash.c_str());
            return false;
        }
    }
    const auto latencyEvent = NaviSchedAgentEvent::AGENT_HANDSHAKE == event ||
                              NaviSchedAgentEvent::LATENCY_MODEL_UPDATED == event;
    const auto invalidLatencyModel = latencyEvent && !validLatencyModel(update->latency);
    const auto pdSeparated = NaviSchedAgentEvent::AGENT_HANDSHAKE == event
                                 ? update->control->pdSeparated
                                 : infoSnapshot->pdSeparated;
    if (!latencyModelMatchesPdMode(update->latency, pdSeparated)) {
        NAVI_KERNEL_LOG(ERROR,
                        "reject latency model incompatible with PD mode, "
                        "agentId [%s], pdSeparated [%d], parameterCount [%lu]",
                        agentId.c_str(), pdSeparated, update->latency->parameters.size());
        return false;
    }
    if (!workloadMatchesPdMode(update->workload, pdSeparated)) {
        NAVI_KERNEL_LOG(ERROR,
                        "reject workload incompatible with PD mode, agentId [%s], "
                        "pdSeparated [%d], prefillCapacitySplit [%d]",
                        agentId.c_str(), pdSeparated,
                        update->workload->prefillCapacitySplit);
        return false;
    }
    MasterLatencyModelR::RecordResult localLatencyRecordResult;
    auto &modelResult =
        latencyRecordResult ? *latencyRecordResult : localLatencyRecordResult;
    const auto sourceId = latencySourceId.empty() ? agentId : latencySourceId;
    const auto fingerprint = !update->latencyModelFingerprint.empty()
                                 ? update->latencyModelFingerprint
                                 : infoSnapshot->latencyModelFingerprint;
    const auto geometryEnabled = update->control
                                     ? update->control->executionGeometryEnabled
                                     : infoSnapshot->executionGeometryEnabled;
    const auto projectedGeometryEnabled =
        update->control ? update->control->projectedExecutionGeometryEnabled
                        : infoSnapshot->projectedExecutionGeometryEnabled;
    const auto &prefillPolicy = update->control ? update->control->prefillExecutionPolicy
                                                : infoSnapshot->prefillExecutionPolicy;
    if (latencyEvent && update->latency &&
        (NonLinearPrefillModelR::usesGeometryForVersion(update->latency->binaryVersion) !=
             geometryEnabled ||
         (NonLinearPrefillModelR::usesProjectedGeometryForVersion(
             update->latency->binaryVersion)) != projectedGeometryEnabled)) {
        NAVI_KERNEL_LOG(
            ERROR, "reject Agent/Master execution geometry mode mismatch, agentId [%s]",
            agentId.c_str());
        return false;
    }
    if (latencyEvent && update->latency && geometryEnabled &&
        (!pdSeparated || !TPUPrefillExecutionPolicy::validate(prefillPolicy))) {
        NAVI_KERNEL_LOG(ERROR,
                        "reject geometry latency model without valid typed runtime "
                        "policy, agentId [%s]",
                        agentId.c_str());
        return false;
    }
    const auto recordLatencyModel = [&]() {
        if (invalidLatencyModel) {
            if (update->latency) {
                bool ignored = false;
                _masterLatencyModelR->record(agentId, fingerprint, *update->latency,
                                             false, update->eventTimeUs, sourceId,
                                             &ignored, false, &modelResult, false);
                if (auto slot =
                        _masterLatencyModelR->getLiveModelSlot(agentId, sourceId)) {
                    state->setLatencyModelSlot(slot);
                }
            }
            NAVI_KERNEL_LOG(
                ERROR,
                "reject invalid latency model, agentId [%s], sourceId [%s], "
                "formula [%d], binaryVersion [%ld], revision [%ld], "
                "parameterCount [%lu], rejectReason [%s], currentRevision [%ld], "
                "currentRejectCount [%u]",
                agentId.c_str(), sourceId.c_str(),
                update->latency ? update->latency->formula : 0,
                update->latency ? update->latency->binaryVersion : 0,
                update->latency ? update->latency->modelRevision : 0,
                update->latency ? update->latency->parameters.size() : 0,
                MasterLatencyModelR::getRejectReasonName(modelResult.rejectReason),
                modelResult.model ? modelResult.model->revision : 0,
                modelResult.consecutiveRejectedCount);
            return false;
        }
        if (!update->latency || update->latency->modelRevision <= 0) {
            return true;
        }
        const auto runtime = state->getRuntimeState();
        const auto agentStartTimeUs = NaviSchedAgentEvent::AGENT_HANDSHAKE == event
                                          ? update->control->agentStartTimeUs
                                          : runtime.agentStartTimeUs;
        const auto rampFinished = CostSchedulerColdStartStrategy::isRampFinished(
            update->eventTimeUs, agentStartTimeUs, update->latency->sessionSampleCount);
        const auto agentTrusted = update->latencyModelTrusted;
        bool liveUpdateAccepted = false;
        _masterLatencyModelR->record(
            agentId, fingerprint, *update->latency, agentTrusted && rampFinished,
            update->eventTimeUs, sourceId, &liveUpdateAccepted,
            update->version >= 2 && update->latencyModelWarmStarted, &modelResult,
            rampFinished);
        if (auto slot = _masterLatencyModelR->getLiveModelSlot(agentId, sourceId)) {
            state->setLatencyModelSlot(slot);
        }
        if (modelResult.rejectReason ==
                MasterLatencyModelR::RejectReason::INVALID_MODEL ||
            modelResult.rejectReason ==
                MasterLatencyModelR::RejectReason::SOURCE_MISMATCH) {
            NAVI_KERNEL_LOG(
                ERROR,
                "reject latency model update, agentId [%s], sourceId [%s], "
                "reason [%s], currentRevision [%ld], currentTrusted [%d], "
                "currentBinaryVersion [%ld], currentParameterCount [%lu], "
                "currentRejectCount [%u], incomingRevision [%ld], "
                "incomingTrusted [%d]",
                agentId.c_str(), sourceId.c_str(),
                MasterLatencyModelR::getRejectReasonName(modelResult.rejectReason),
                modelResult.model ? modelResult.model->revision : 0,
                modelResult.model
                    ? static_cast<int>(modelResult.model->overloadPredictionTrusted)
                    : 0,
                modelResult.model ? modelResult.model->binaryVersion : 0,
                modelResult.model ? modelResult.model->parameters.size() : 0,
                modelResult.consecutiveRejectedCount, update->latency->modelRevision,
                static_cast<int>(agentTrusted));
            return false;
        }
        if (!liveUpdateAccepted) {
            NAVI_KERNEL_LOG(
                WARN,
                "reject latency model update, agentId [%s], sourceId [%s], "
                "reason [%s], currentRevision [%ld], currentTrusted [%d], "
                "incomingRevision [%ld], incomingTrusted [%d], "
                "consecutiveRejectedCount [%u], forcedUntrusted [%d]",
                agentId.c_str(), sourceId.c_str(),
                MasterLatencyModelR::getRejectReasonName(modelResult.rejectReason),
                modelResult.model ? modelResult.model->revision : 0,
                modelResult.model
                    ? static_cast<int>(modelResult.model->overloadPredictionTrusted)
                    : 0,
                update->latency->modelRevision,
                static_cast<int>(agentTrusted && rampFinished),
                modelResult.consecutiveRejectedCount,
                static_cast<int>(modelResult.forcedUntrusted));
        }
        return true;
    };
    const auto rejectInvalidLatencyModel =
        NaviSchedAgentEvent::LATENCY_MODEL_UPDATED == event && invalidLatencyModel;
    if ((NaviSchedAgentEvent::AGENT_HANDSHAKE == event || rejectInvalidLatencyModel) &&
        !recordLatencyModel()) {
        return false;
    }
    if (NaviSchedAgentEvent::LATENCY_MODEL_UPDATED == event && update->latency) {
        const auto activeSlot = _masterLatencyModelR->getLiveModelSlot(agentId);
        const auto sourceSlot = _masterLatencyModelR->getLiveModelSlot(agentId, sourceId);
        if (activeSlot && !sourceSlot) {
            NAVI_KERNEL_LOG(
                ERROR,
                "reject latency model update before runtime commit, agentId [%s], "
                "sourceId [%s], reason [source_mismatch]",
                agentId.c_str(), sourceId.c_str());
            return false;
        }
    }
    const auto &request = update->requestEvent;
    const auto &cache = update->cache;
    const auto &pendingCache = update->pendingCache;
    auto &cacheState = state->cacheForType(cache ? cache->cacheKeyType
                                                 : NaviSchedCacheKeyType::BLOCK_POOL);
    auto &pendingCacheState = state->cacheForType(
        pendingCache ? pendingCache->cacheKeyType : NaviSchedCacheKeyType::BLOCK_POOL);
    const auto &workload = update->workload;
    bool stateDiverged = false;
    bool cachePolicyInitialized = false;
    std::optional<NaviSchedCachePolicy> incomingCachePolicy;
    std::optional<NaviSchedCachePolicy> conflictingCachePolicy;
    if (cache) {
        const bool hasPolicyFields = cache->blockSize >= 0 || cache->linearStep >= 0 ||
                                     cache->hybridTailRequired >= 0;
        if (hasPolicyFields) {
            if (cache->blockSize <= 0 || cache->linearStep <= 0 ||
                cache->hybridTailRequired < 0 ||
                update->fullAttentionQueryDropTokens < 0 ||
                (cache->cacheKeyType == NaviSchedCacheKeyType::V6D
                     ? update->fullAttentionQueryDropTokens > 1
                     : update->fullAttentionQueryDropTokens % cache->blockSize != 0)) {
                NAVI_KERNEL_LOG(ERROR,
                                "reject incomplete agent cache policy, agentId [%s], "
                                "blockSize [%ld], linearStep [%ld], "
                                "hybridTailRequired [%d]",
                                agentId.c_str(), cache->blockSize, cache->linearStep,
                                static_cast<int>(cache->hybridTailRequired));
                return false;
            }
            incomingCachePolicy.emplace(NaviSchedCachePolicy{
                cache->blockSize, cache->linearStep, cache->hybridTailRequired != 0,
                update->mambaReplayStopShift, update->mambaRetentionInterval,
                update->mambaAlign, update->fullAttentionQueryDropTokens});
            {
                autil::ScopedReadLock cacheLock(cacheState._cacheLock);
                if (cacheState._cachePolicyReady.load(std::memory_order_relaxed) &&
                    (cacheState._cachePolicy.blockSize !=
                         incomingCachePolicy->blockSize ||
                     cacheState._cachePolicy.linearStep !=
                         incomingCachePolicy->linearStep ||
                     cacheState._cachePolicy.hybridTailRequired !=
                         incomingCachePolicy->hybridTailRequired ||
                     cacheState._cachePolicy.mambaReplayStopShift !=
                         incomingCachePolicy->mambaReplayStopShift ||
                     cacheState._cachePolicy.mambaRetentionInterval !=
                         incomingCachePolicy->mambaRetentionInterval ||
                     cacheState._cachePolicy.mambaAlign !=
                         incomingCachePolicy->mambaAlign ||
                     cacheState._cachePolicy.fullAttentionQueryDropTokens !=
                         incomingCachePolicy->fullAttentionQueryDropTokens)) {
                    conflictingCachePolicy = cacheState._cachePolicy;
                }
            }
            if (conflictingCachePolicy) {
                NAVI_KERNEL_LOG(ERROR,
                                "reject agent cache policy change, agentId [%s], "
                                "effective [%ld,%ld,%d], incoming [%ld,%ld,%d]",
                                agentId.c_str(), conflictingCachePolicy->blockSize,
                                conflictingCachePolicy->linearStep,
                                conflictingCachePolicy->hybridTailRequired ? 1 : 0,
                                incomingCachePolicy->blockSize,
                                incomingCachePolicy->linearStep,
                                incomingCachePolicy->hybridTailRequired ? 1 : 0);
                return false;
            }
        }
    }
    if (pendingCache) {
        bool current = false;
        MasterPendingCacheR::ApplyResult pendingResult;
        {
            autil::ScopedReadLock nodeIndexLock(_nodeIndexLock);
            current = isCurrentNodeStateLocked(agentIdHash, state);
            if (current) {
                pendingResult =
                    state->_pendingCacheR->applyAgentUpdate(agentIdHash, *pendingCache);
                if (pendingCacheApplyResult) {
                    *pendingCacheApplyResult = pendingResult;
                }
            }
        }
        if (!current) {
            NAVI_KERNEL_LOG(INFO,
                            "ignore stale Agent pending cache update, agentId [%s], "
                            "revision [%lu], fullSnapshot [%d]",
                            agentId.c_str(), pendingCache->revision,
                            static_cast<int>(pendingCache->fullSnapshot));
        } else if (!pendingResult.applied) {
            pendingCacheState._pendingCacheResyncTriggerCount.fetch_add(
                1, std::memory_order_relaxed);
            if (pendingCacheResyncRequired) {
                *pendingCacheResyncRequired = true;
            }
            NAVI_KERNEL_LOG(
                WARN,
                "reject Agent pending cache update without touching real cache, "
                "agentId [%s], reason [%s], fullSnapshot [%d], "
                "snapshotReadyBefore [%d], masterRevisionBefore [%lu], "
                "expectedRevision [%lu], receivedRevision [%lu], "
                "masterChecksumBefore [%lu], candidateChecksum [%lu], "
                "receivedChecksum [%lu], candidateStateComputed [%d], "
                "masterBatchCountBefore [%ld], candidateBatchCount [%ld], "
                "receivedBatchCount [%ld], masterBlockCountBefore [%ld], "
                "candidateBlockCount [%ld], receivedBlockCount [%ld], "
                "masterTailBlockCountBefore [%ld], candidateTailBlockCount [%ld], "
                "receivedTailBlockCount [%ld], revisionGap [%d], stateMismatch [%d], "
                "scheduledBatches [%lu], agentBatches [%lu], "
                "agentSnapshotReady [%d], gapCount [%lu], mismatchCount [%lu], "
                "resyncTriggerCount [%lu]",
                agentId.c_str(), pendingResult.reason.c_str(),
                static_cast<int>(pendingCache->fullSnapshot),
                static_cast<int>(pendingResult.snapshotReadyBefore),
                pendingResult.masterRevisionBefore, pendingResult.expectedRevision,
                pendingResult.receivedRevision, pendingResult.masterChecksumBefore,
                pendingResult.candidateChecksum, pendingResult.receivedChecksum,
                static_cast<int>(pendingResult.candidateStateComputed),
                pendingResult.masterBatchCountBefore, pendingResult.candidateBatchCount,
                pendingResult.receivedBatchCount, pendingResult.masterBlockCountBefore,
                pendingResult.candidateBlockCount, pendingResult.receivedBlockCount,
                pendingResult.masterTailBlockCountBefore,
                pendingResult.candidateTailBlockCount,
                pendingResult.receivedTailBlockCount,
                static_cast<int>(pendingResult.revisionGap),
                static_cast<int>(pendingResult.stateMismatch),
                pendingResult.stats.scheduledBatchCount,
                pendingResult.stats.agentBatchCount,
                static_cast<int>(pendingResult.stats.agentSnapshotReady),
                pendingResult.stats.agentRevisionGapCount,
                pendingResult.stats.agentStateMismatchCount,
                pendingCacheState._pendingCacheResyncTriggerCount.load(
                    std::memory_order_relaxed));
        }
    }
    auto requestEvent = NaviSchedAgentEvent::REQUEST_RECEIVED == event ||
                        NaviSchedAgentEvent::BATCH_FINISHED == event ||
                        NaviSchedAgentEvent::BATCH_REQUESTS_ABORTED == event;
    auto executionLoadEvent = NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED == event ||
                              NaviSchedAgentEvent::BATCH_FINISHED == event ||
                              NaviSchedAgentEvent::BATCH_REQUESTS_ABORTED == event;
    auto receiveTimeUs = autil::TimeUtility::currentTime();
    decltype(state->_pendingBatchReservations) handshakeReservations;
    std::vector<navi::AsyncPipePtr> terminalDispatchPipes;
    bool reservationStateValid = true;
    bool admissionObserved = false;
    int64_t admittedScheduleBatchId = 0;
    int64_t admittedLocalBatchId = 0;
    uint64_t admittedWorkloadRevision = 0;
    std::unique_ptr<autil::ScopedReadLock> handshakeNodeIndexLock;
    bool currentHandshakeState = false;
    bool handshakePendingCleared = false;
    MasterPendingCacheR::Stats clearedHandshakePendingStats;
    bool staleInflightSnapshotIgnored = false;
    uint64_t staleInflightIncomingRevision = 0;
    uint64_t staleInflightCurrentRevision = 0;
    bool inflightSnapshotConflict = false;
    uint64_t conflictingInflightRevision = 0;
    int32_t conflictingInflightCurrentCount = 0;
    int32_t conflictingInflightIncomingCount = 0;
    double conflictingInflightCurrentLatencySumMs = 0.0;
    double conflictingInflightIncomingLatencySumMs = 0.0;
    bool reservationPendingCacheRestoreFailed = false;
    int64_t reservationPendingCacheRestoreBatchId = 0;
    if (NaviSchedAgentEvent::AGENT_HANDSHAKE == event) {
        handshakeNodeIndexLock = std::make_unique<autil::ScopedReadLock>(_nodeIndexLock);
        currentHandshakeState = isCurrentNodeStateLocked(agentIdHash, state);
    }
    {
        if (timing) {
            timing->runtimeLockBeginMonoUs = autil::TimeUtility::monotonicTimeUs();
        }
        autil::ScopeGuard finishRuntimeTiming([timing]() {
            if (timing) {
                timing->runtimeLockReleasedMonoUs = autil::TimeUtility::monotonicTimeUs();
            }
        });
        autil::ScopedLock runtimeLock(state->_runtimeLock);
        if (timing) {
            timing->runtimeLockAcquiredMonoUs = autil::TimeUtility::monotonicTimeUs();
        }
        auto &runtime = state->_runtimeState;
        auto acknowledgeLocalBatch = [&](const NaviSchedLocalBatchInfo &localBatch) {
            auto reservation = state->findReservationLocked(localBatch.scheduleBatchId);
            if (!reservation) {
                return true;
            }
            std::vector<uint64_t> requestIds;
            if (!reservation->batch.acknowledgeDispatch(localBatch, requestIds)) {
                return false;
            }
            NodeState::collectDispatchWaitersLocked(*reservation, requestIds,
                                                    terminalDispatchPipes);
            return true;
        };
        auto handoffLocalBatch = [&](const NaviSchedLocalBatchInfo &localBatch) {
            auto reservation = state->findReservationLocked(localBatch.scheduleBatchId);
            if (!reservation) {
                return true;
            }
            auto candidate = reservation->batch;
            MasterBatchReservation::Transition transition;
            if (!candidate.markAdmitted(localBatch, transition)) {
                return false;
            }
            if (transition.changed) {
                if (!state->refreshReservationPredictionLocked(candidate, false)) {
                    return false;
                }
                reservation->batch = std::move(candidate);
                NodeState::collectDispatchWaitersLocked(
                    *reservation, transition.dispatchedRequestIds, terminalDispatchPipes);
                admissionObserved = true;
                admittedScheduleBatchId = localBatch.scheduleBatchId;
                admittedLocalBatchId = localBatch.localBatchId;
            }
            return true;
        };
        auto completeLocalBatch = [&](const NaviSchedLocalBatchInfo &localBatch,
                                      bool successful) {
            auto reservation = state->findReservationLocked(localBatch.scheduleBatchId);
            if (!reservation) {
                return true;
            }
            auto candidate = reservation->batch;
            MasterBatchReservation::Transition admitted;
            if (!candidate.markAdmitted(localBatch, admitted)) {
                return false;
            }
            MasterBatchReservation::Transition transition;
            if (!candidate.markCompleted(localBatch, successful, transition)) {
                return false;
            }
            if (transition.changed) {
                if (!state->refreshReservationPredictionLocked(candidate, false) ||
                    (transition.cacheChanged &&
                     !state->updateReservationPendingCacheLocked(candidate,
                                                                 &transition))) {
                    return false;
                }
                reservation->batch = std::move(candidate);
            }
            if (!transition.resolved) {
                return true;
            }
            for (const auto &[requestId, waiter] : reservation->dispatchWaiters) {
                (void)requestId;
                if (auto pipe = waiter.lock()) {
                    terminalDispatchPipes.emplace_back(std::move(pipe));
                }
            }
            state->eraseReservationLocked(*reservation);
            return true;
        };
        auto failLocalBatchRequests = [&](const NaviSchedLocalBatchInfo &localBatch) {
            if (!localBatch.valid() || localBatch.requestIds.empty()) {
                return false;
            }
            auto reservation = state->findReservationLocked(localBatch.scheduleBatchId);
            if (!reservation) {
                return true;
            }
            auto candidate = reservation->batch;
            std::vector<uint64_t> changedRequestIds;
            changedRequestIds.reserve(localBatch.requestIds.size());
            for (auto requestId : localBatch.requestIds) {
                const auto requestState = candidate.getRequestState(requestId);
                if (requestState != MasterBatchReservation::RequestState::FAILED &&
                    requestState != MasterBatchReservation::RequestState::COMPLETED) {
                    changedRequestIds.emplace_back(requestId);
                }
            }
            MasterBatchReservation::Transition transition;
            if (!candidate.markFailed(localBatch.requestIds, transition)) {
                return false;
            }
            if (changedRequestIds.empty()) {
                return true;
            }
            if (!state->refreshReservationPredictionLocked(candidate, false) ||
                !state->updateReservationPendingCacheLocked(candidate, &transition)) {
                return false;
            }
            reservation->batch = std::move(candidate);
            for (auto requestId : changedRequestIds) {
                auto waiterIt = reservation->dispatchWaiters.find(requestId);
                if (waiterIt == reservation->dispatchWaiters.end()) {
                    continue;
                }
                if (auto pipe = waiterIt->second.lock()) {
                    terminalDispatchPipes.emplace_back(std::move(pipe));
                }
                reservation->dispatchWaiters.erase(waiterIt);
            }
            if (!reservation->batch.isResolved()) {
                return true;
            }
            for (const auto &[requestId, waiter] : reservation->dispatchWaiters) {
                (void)requestId;
                if (auto pipe = waiter.lock()) {
                    terminalDispatchPipes.emplace_back(std::move(pipe));
                }
            }
            state->eraseReservationLocked(*reservation);
            return true;
        };
        auto applyInflightSnapshot = [&]() {
            if (!request) {
                return true;
            }
            auto &load = runtime.load;
            const auto incomingRevision = request->inflightStatsRevision;
            if (NaviSchedAgentEvent::AGENT_HANDSHAKE != event &&
                incomingRevision < load.inflightStatsRevision) {
                staleInflightSnapshotIgnored = true;
                staleInflightIncomingRevision = incomingRevision;
                staleInflightCurrentRevision = load.inflightStatsRevision;
                return true;
            }
            if (NaviSchedAgentEvent::AGENT_HANDSHAKE != event &&
                incomingRevision == load.inflightStatsRevision) {
                if (load.inflightRequests == request->inflightRequests &&
                    load.inflightPredictedLatencySumMs ==
                        request->inflightPredictedLatencySumMs) {
                    return true;
                }
                inflightSnapshotConflict = true;
                conflictingInflightRevision = incomingRevision;
                conflictingInflightCurrentCount = load.inflightRequests;
                conflictingInflightIncomingCount = request->inflightRequests;
                conflictingInflightCurrentLatencySumMs =
                    load.inflightPredictedLatencySumMs;
                conflictingInflightIncomingLatencySumMs =
                    request->inflightPredictedLatencySumMs;
                return false;
            }
            load.inflightStatsRevision = incomingRevision;
            load.inflightRequests = request->inflightRequests;
            load.inflightPredictedLatencySumMs = request->inflightPredictedLatencySumMs;
            return true;
        };
        auto handoffRequestInflight = [&](uint64_t requestId) {
            if (requestId == 0) {
                return false;
            }
            for (auto &reservationPtr : state->_pendingBatchReservations) {
                auto &reservation = *reservationPtr;
                if (!reservation.batch.containsRequest(requestId)) {
                    continue;
                }
                bool changed = false;
                if (workload && workload->revision > runtime.workload.revision &&
                    runtime.unifiedLatencyPrediction) {
                    if (!reservation.batch.markAcceptedWorkloadReported(requestId))
                        return false;
                }
                return reservation.batch.markInflightReported(requestId, changed);
            }
            return true;
        };
        auto handoffLocalBatchInflight = [&](const NaviSchedLocalBatchInfo &localBatch) {
            auto reservation = state->findReservationLocked(localBatch.scheduleBatchId);
            if (!reservation) {
                return true;
            }
            std::vector<uint64_t> changedRequestIds;
            return reservation->batch.markInflightReported(localBatch, changedRequestIds);
        };
        const bool requestsAborted = NaviSchedAgentEvent::BATCH_REQUESTS_ABORTED == event;
        const bool batchFinished = NaviSchedAgentEvent::BATCH_FINISHED == event;
        const auto requestLocalBatch =
            request && request->localBatch.valid() ? &request->localBatch : nullptr;
        const auto reservationLocalBatch =
            requestLocalBatch
                ? requestLocalBatch
                : (workload && workload->admittedBatch.valid() ? &workload->admittedBatch
                                                               : nullptr);
        std::optional<NodeState::PendingBatchReservation> previousReservation;
        NodeState::ReservationPtr previousReservationOwner;
        auto previousNextReservation = state->_pendingBatchReservations.end();
        if (reservationLocalBatch) {
            auto existing =
                state->_reservationsById.find(reservationLocalBatch->scheduleBatchId);
            if (existing != state->_reservationsById.end()) {
                previousReservationOwner = existing->second;
                previousReservation = *previousReservationOwner;
                previousNextReservation = std::next(previousReservationOwner->position);
            }
        }
        const bool mayChangeReservation =
            NaviSchedAgentEvent::REQUEST_RECEIVED == event ||
            (requestLocalBatch && executionLoadEvent) ||
            (workload && workload->admittedBatch.valid()) || requestsAborted;
        std::optional<NodeState::RuntimeState> previousRuntime;
        if (mayChangeReservation) {
            previousRuntime.emplace(runtime);
        }
        if (NaviSchedAgentEvent::AGENT_HANDSHAKE == event) {
            runtime.unifiedLatencyPrediction = update->control->unifiedLatencyPrediction;
        }
        if (request) {
            reservationStateValid = applyInflightSnapshot();
        }
        if (reservationStateValid && NaviSchedAgentEvent::BATCH_FINISHED == event &&
            request->latencyValidation && !request->abnormalBatchFinished) {
            runtime.latencyValidation = request->latencyValidationStats;
        }
        if (reservationStateValid && NaviSchedAgentEvent::REQUEST_RECEIVED == event) {
            reservationStateValid = handoffRequestInflight(request->requestId);
        }
        if (reservationStateValid && requestLocalBatch && executionLoadEvent) {
            reservationStateValid = handoffLocalBatchInflight(*requestLocalBatch);
        }
        if (reservationStateValid && !requestsAborted && requestLocalBatch &&
            executionLoadEvent) {
            reservationStateValid = acknowledgeLocalBatch(*requestLocalBatch);
        }
        if (reservationStateValid && requestEvent) {
            runtime.load.batchWaitingRequests = request->batchWaitingRequests;
            runtime.load.queuedBatchRequests = request->queuedBatchRequests;
            runtime.load.runningRequests = request->runningRequests;
            if (NaviSchedAgentEvent::BATCH_FINISHED == event &&
                !request->abnormalBatchFinished) {
                runtime.lastRequestLatencyMs = request->latencyMs;
            }
            runtime.lastRequestEvent = event;
            runtime.lastRequestEventTimeUs = update->eventTimeUs;
        }
        if (reservationStateValid && executionLoadEvent) {
            runtime.load.batchWaitingRequests = request->batchWaitingRequests;
            runtime.load.queuedBatchRequests = request->queuedBatchRequests;
            runtime.load.runningRequests = request->runningRequests;
            runtime.load.batchTimeoutCount = request->batchTimeoutCount;
            runtime.load.abnormalBatchFinishCount = request->abnormalBatchFinishCount;
            runtime.load.queueTotalPredictedLatencyMs =
                request->queueTotalPredictedLatencyMs;
            runtime.load.queueDecayHeadRemainingMs = request->queueDecayHeadRemainingMs;
            runtime.load.queueModel = request->queueModel;
            runtime.load.queueReportReceiveTimeUs = receiveTimeUs;
        }
        if (reservationStateValid && requestLocalBatch &&
            NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED == event &&
            !runtime.unifiedLatencyPrediction) {
            reservationStateValid = handoffLocalBatch(*requestLocalBatch);
        }
        if (NaviSchedAgentEvent::AGENT_HANDSHAKE == event) {
            for (const auto &reservation : state->_pendingBatchReservations) {
                reservation->attached = false;
            }
            handshakeReservations.swap(state->_pendingBatchReservations);
            state->_reservationsById.clear();
            if (currentHandshakeState) {
                state->_acceptReservations = true;
                autil::ScopedWriteLock cacheLock(state->_cacheLock);
                state->clearCacheStateLocked();
                clearedHandshakePendingStats =
                    state->_pendingCacheR->getStats(agentIdHash);
                state->_pendingCacheR->clear(agentIdHash);
                handshakePendingCleared = true;
            }
            state->_executionUtilization.store(-1.0, std::memory_order_relaxed);
            state->_executionBusyUtilization.store(-1.0, std::memory_order_relaxed);
            runtime.workload = workload ? *workload : NaviSchedWorkloadSnapshot();
            runtime.workload.admittedBatch = NaviSchedLocalBatchInfo();
            runtime.workloadReportReceiveTimeUs = workload ? receiveTimeUs : 0;
            runtime.workloadStaleUpdateCount = 0;
            if (runtime.agentStartTimeUs != update->control->agentStartTimeUs) {
                runtime.peerCacheReadAvailable = false;
                runtime.peerCacheReadRevision = 0;
            }
            runtime.agentStartTimeUs = update->control->agentStartTimeUs;
        } else if (workload) {
            if (workload->admittedBatch.valid()) {
                reservationStateValid = acknowledgeLocalBatch(workload->admittedBatch);
            }
            if (reservationStateValid && workload->revision > runtime.workload.revision) {
                runtime.workload = *workload;
                runtime.workload.admittedBatch = NaviSchedLocalBatchInfo();
                runtime.workloadReportReceiveTimeUs = receiveTimeUs;
                if (workload->admittedBatch.valid()) {
                    reservationStateValid = handoffLocalBatch(workload->admittedBatch);
                    if (reservationStateValid) {
                        admittedWorkloadRevision = workload->revision;
                    }
                }
            } else {
                ++runtime.workloadStaleUpdateCount;
            }
        }
        if (reservationStateValid && requestsAborted) {
            reservationStateValid = failLocalBatchRequests(request->localBatch);
        } else if (reservationStateValid && batchFinished && requestLocalBatch) {
            reservationStateValid =
                completeLocalBatch(*requestLocalBatch, !request->abnormalBatchFinished);
        }
        if (reservationStateValid) {
            if (update->executionUtilization >= 0.0) {
                state->_executionUtilization.store(update->executionUtilization,
                                                   std::memory_order_relaxed);
            }
            if (update->executionBusyUtilization >= 0.0) {
                state->_executionBusyUtilization.store(update->executionBusyUtilization,
                                                       std::memory_order_relaxed);
            }
            if (update->peerCacheReadAgentStartTimeUs == runtime.agentStartTimeUs &&
                runtime.agentStartTimeUs > 0 &&
                update->peerCacheReadRevision > runtime.peerCacheReadRevision) {
                const auto changed =
                    runtime.peerCacheReadAvailable != update->peerCacheReadAvailable;
                runtime.peerCacheReadAvailable = update->peerCacheReadAvailable;
                runtime.peerCacheReadRevision = update->peerCacheReadRevision;
                if (changed) {
                    NAVI_KERNEL_LOG(
                        INFO, "[navi-peer-cache-read] agent=%s available=%d revision=%lu",
                        agentId.c_str(), int(runtime.peerCacheReadAvailable),
                        runtime.peerCacheReadRevision);
                }
            }
            runtime.lastEvent = event;
            runtime.lastEventTimeUs = update->eventTimeUs;
            runtime.lastReceiveTimeUs = receiveTimeUs;
        } else {
            if (previousRuntime) {
                runtime = std::move(*previousRuntime);
            }
            if (previousReservation) {
                auto current =
                    state->findReservationLocked(previousReservation->batch.getBatchId());
                if (current) {
                    *current = *previousReservation;
                } else {
                    *previousReservationOwner = *previousReservation;
                    previousReservationOwner->position =
                        state->_pendingBatchReservations.insert(previousNextReservation,
                                                                previousReservationOwner);
                    previousReservationOwner->attached = true;
                    state->_reservationsById.emplace(
                        previousReservationOwner->batch.getBatchId(),
                        previousReservationOwner);
                }
                if (!state->updateReservationPendingCacheLocked(
                        previousReservation->batch)) {
                    reservationPendingCacheRestoreFailed = true;
                    reservationPendingCacheRestoreBatchId =
                        previousReservation->batch.getBatchId();
                }
            }
            terminalDispatchPipes.clear();
            admissionObserved = false;
            admittedScheduleBatchId = 0;
            admittedLocalBatchId = 0;
            admittedWorkloadRevision = 0;
        }
    }
    if (staleInflightSnapshotIgnored) {
        NAVI_KERNEL_LOG(
            DEBUG,
            "ignore stale Agent inflight snapshot, agentId [%s], eventType [%s], "
            "incomingRevision [%lu], currentRevision [%lu]",
            agentId.c_str(), NaviSchedAgentEvent::getName(event).c_str(),
            staleInflightIncomingRevision, staleInflightCurrentRevision);
    }
    if (inflightSnapshotConflict) {
        NAVI_KERNEL_LOG(ERROR,
                        "reject inconsistent Agent inflight snapshot at same revision, "
                        "agentId [%s], eventType [%s], revision [%lu], "
                        "currentCount [%d], incomingCount [%d], "
                        "currentLatencySumMs [%.6f], incomingLatencySumMs [%.6f]",
                        agentId.c_str(), NaviSchedAgentEvent::getName(event).c_str(),
                        conflictingInflightRevision, conflictingInflightCurrentCount,
                        conflictingInflightIncomingCount,
                        conflictingInflightCurrentLatencySumMs,
                        conflictingInflightIncomingLatencySumMs);
    }
    if (reservationPendingCacheRestoreFailed) {
        NAVI_KERNEL_LOG(ERROR,
                        "restore Master pending cache after reservation failure failed, "
                        "agentId [%s], batchId [%ld], eventType [%s]",
                        agentId.c_str(), reservationPendingCacheRestoreBatchId,
                        NaviSchedAgentEvent::getName(event).c_str());
    }
    const bool reservationTransitionFailed = !reservationStateValid;
    if (reservationTransitionFailed) {
        NAVI_KERNEL_LOG(ERROR,
                        "invalid Master batch reservation transition, agentId [%s], "
                        "eventType [%s]",
                        agentId.c_str(), NaviSchedAgentEvent::getName(event).c_str());
        if (resyncRequired) {
            *resyncRequired = true;
        }
    }
    handshakeNodeIndexLock.reset();
    for (const auto &reservationPtr : handshakeReservations) {
        const auto &reservation = *reservationPtr;
        for (const auto &[requestId, waiter] : reservation.dispatchWaiters) {
            (void)requestId;
            if (auto pipe = waiter.lock()) {
                pipe->setEof();
            }
        }
    }
    for (const auto &pipe : terminalDispatchPipes) {
        pipe->setEof();
    }
    if (!handshakeReservations.empty()) {
        NAVI_KERNEL_LOG(INFO,
                        "[navi-master-workload-reservation] phase=agent_handshake "
                        "agent=%s cleared_reservations=%lu workload_revision=%lu",
                        agentId.c_str(), handshakeReservations.size(),
                        workload ? workload->revision : 0);
    }
    if (admissionObserved) {
        NAVI_KERNEL_LOG(INFO,
                        "[navi-master-workload-reservation] phase=admitted "
                        "agent=%s schedule_batch_id=%ld local_batch_id=%ld "
                        "workload_revision=%lu",
                        agentId.c_str(), admittedScheduleBatchId, admittedLocalBatchId,
                        admittedWorkloadRevision);
    }
    if (NaviSchedAgentEvent::BATCH_FINISHED == event && request &&
        request->localBatch.valid()) {
        NAVI_KERNEL_LOG(
            INFO,
            "[navi-master-batch-completion] event_time_us=%ld receive_time_us=%ld "
            "agent=%s schedule_batch_id=%ld local_batch_id=%ld "
            "request_count=%d abnormal=%d reported_latency_ms=%.9f "
            "latency_validation=%d agent_prediction_ms=%.9f "
            "backend_queue_ms=%.9f reservation_transition_valid=%d",
            update->eventTimeUs, receiveTimeUs, agentId.c_str(),
            request->localBatch.scheduleBatchId, request->localBatch.localBatchId,
            request->completedRequestCount, request->abnormalBatchFinished ? 1 : 0,
            request->latencyMs, request->latencyValidation ? 1 : 0,
            request->agentPredictedExecutionLatencyMs, request->backendQueueLatencyMs,
            reservationStateValid ? 1 : 0);
    }
    if (handshakePendingCleared) {
        NAVI_KERNEL_LOG(INFO,
                        "[navi-master-pending-cache-lifecycle] phase=agent_handshake "
                        "agent=%s cleared_batches=%lu cleared_blocks=%lu",
                        agentId.c_str(), clearedHandshakePendingStats.batchCount,
                        clearedHandshakePendingStats.uniqueBlockCount);
    }
    bool becameOnline = false;
    bool becameOffline = false;
    if (NaviSchedAgentEvent::AGENT_HANDSHAKE == event ||
        NaviSchedAgentEvent::AGENT_STOPPING == event ||
        NaviSchedAgentEvent::SCHEDULER_CONFIG_UPDATED == event) {
        autil::ScopedWriteLock infoLock(state->_infoLock);
        auto &node = state->_info;
        bool rebuildInfo = false;
        if (NaviSchedAgentEvent::AGENT_HANDSHAKE == event) {
            becameOnline = !node.online;
            node.online = true;
            rebuildInfo = true;
        } else if (NaviSchedAgentEvent::AGENT_STOPPING == event) {
            becameOffline = node.online;
            node.online = false;
            rebuildInfo = true;
        }
        if (NaviSchedAgentEvent::AGENT_HANDSHAKE == event) {
            node.serverInfo = update->control->serverInfo;
            node.pdSeparated = update->control->pdSeparated;
            node.unifiedLatencyPrediction = update->control->unifiedLatencyPrediction;
            node.latencyModelFingerprint = update->latencyModelFingerprint;
            node.executionGeometryEnabled = update->control->executionGeometryEnabled;
            node.projectedExecutionGeometryEnabled =
                update->control->projectedExecutionGeometryEnabled;
            node.prefillExecutionPolicy = update->control->prefillExecutionPolicy;
            node.peerCacheEnabled = update->control->peerCacheEnabled;
            rebuildInfo = true;
        } else if (NaviSchedAgentEvent::SCHEDULER_CONFIG_UPDATED == event) {
            node.schedulerConfigGeneration = update->schedulerConfigGeneration;
            node.schedulerConfigHash = update->schedulerConfigHash;
            node.schedulerConfigJson = acknowledgedSchedulerConfigJson;
            rebuildInfo = true;
        }
        if (NaviSchedAgentEvent::AGENT_HANDSHAKE == event && hasSchedulerConfig) {
            node.schedulerConfigGeneration = update->schedulerConfigGeneration;
            node.schedulerConfigHash = update->schedulerConfigHash;
            node.schedulerConfigJson = acknowledgedSchedulerConfigJson;
            rebuildInfo = true;
        }
        if (hasSchedulerConfig) {
            std::atomic_store_explicit(
                &state->_residualRuntimeConfig,
                std::make_shared<const LatencyResidualModel::RuntimeConfig>(
                    acknowledgedSchedulerConfig.residualRuntimeConfig()),
                std::memory_order_release);
        }
        if (rebuildInfo) {
            state->rebuildInfoSnapshot();
        }
    }
    if (NaviSchedAgentEvent::AGENT_HANDSHAKE == event ||
        NaviSchedAgentEvent::AGENT_STOPPING == event) {
        refreshBizNode(state);
    }
    if (incomingCachePolicy) {
        bool cachePolicyChangedConcurrently = false;
        {
            autil::ScopedWriteLock cacheLock(cacheState._cacheLock);
            cachePolicyChangedConcurrently =
                cacheState._cachePolicyReady.load(std::memory_order_relaxed) &&
                (cacheState._cachePolicy.blockSize != incomingCachePolicy->blockSize ||
                 cacheState._cachePolicy.linearStep != incomingCachePolicy->linearStep ||
                 cacheState._cachePolicy.hybridTailRequired !=
                     incomingCachePolicy->hybridTailRequired ||
                 cacheState._cachePolicy.mambaReplayStopShift !=
                     incomingCachePolicy->mambaReplayStopShift ||
                 cacheState._cachePolicy.mambaRetentionInterval !=
                     incomingCachePolicy->mambaRetentionInterval ||
                 cacheState._cachePolicy.mambaAlign != incomingCachePolicy->mambaAlign ||
                 cacheState._cachePolicy.fullAttentionQueryDropTokens !=
                     incomingCachePolicy->fullAttentionQueryDropTokens);
            if (!cachePolicyChangedConcurrently &&
                !cacheState._cachePolicyReady.load(std::memory_order_relaxed)) {
                cacheState._cachePolicy = *incomingCachePolicy;
                cacheState._cachePolicyReady.store(true, std::memory_order_release);
                cacheState._hybridTailRequired = incomingCachePolicy->hybridTailRequired;
                cachePolicyInitialized = true;
            }
        }
        if (cachePolicyChangedConcurrently) {
            NAVI_KERNEL_LOG(ERROR,
                            "reject concurrently changed agent cache policy, "
                            "agentId [%s]",
                            agentId.c_str());
            return false;
        }
    }
    if (cachePolicyInitialized) {
        NAVI_KERNEL_LOG(INFO,
                        "effective agent cache policy, agentId [%s], blockSize [%ld], "
                        "linearStep [%ld], hybridTailRequired [%d], "
                        "mambaAlign [%d], mambaRetentionInterval [%ld], "
                        "mambaReplayStopShift [%d]",
                        agentId.c_str(), cache->blockSize, cache->linearStep,
                        static_cast<int>(cache->hybridTailRequired),
                        static_cast<int>(incomingCachePolicy->mambaAlign),
                        incomingCachePolicy->mambaRetentionInterval,
                        static_cast<int>(incomingCachePolicy->mambaReplayStopShift));
    }
    if (cache) {
        bool cacheStateMismatch = false;
        bool cacheRevisionGap = false;
        bool staleCacheSnapshotIgnored = false;
        bool rtpCacheMirrorMismatch = false;
        CacheSyncStats cacheSyncStats;
        {
            autil::ScopedWriteLock cacheLock(cacheState._cacheLock);
            auto clearMirroredCache = [&]() {
                cacheState._cachedBlockHashes.clear();
                cacheState._fullReadyBlockHashes.clear();
                cacheState._tailReadyBlockHashes.clear();
                cacheState._masterCacheHashChecksum = 0;
                cacheState._masterFullReadyHashChecksum = 0;
                cacheState._masterTailReadyHashChecksum = 0;
            };
            auto hashesMatch = [](const NaviSchedBlockHashesConstPtr &hashes,
                                  const MasterPendingCacheR::PhysicalBlockSet &expected) {
                const auto &values = BlockHashUtil::get(hashes);
                return values.size() == expected.size() &&
                       std::all_of(values.begin(), values.end(),
                                   [&expected](uint64_t hash) {
                                       return expected.end() != expected.find(hash);
                                   });
            };
            const bool cacheStateReady =
                cacheState._cacheStateReady.load(std::memory_order_relaxed);
            staleCacheSnapshotIgnored = cache->fullSnapshot && cacheStateReady &&
                                        cache->revision < cacheState._cacheRevision;
            const bool sameRevisionStateMismatch =
                cache->fullSnapshot && cacheStateReady &&
                cache->revision == cacheState._cacheRevision &&
                (!BlockHashUtil::get(cache->removedBlockHashes).empty() ||
                 !BlockHashUtil::get(cache->removedFullReadyBlockHashes).empty() ||
                 !BlockHashUtil::get(cache->removedTailReadyBlockHashes).empty() ||
                 cache->cachedBlockCount !=
                     static_cast<int64_t>(cacheState._cachedBlockHashes.size()) ||
                 cache->hashChecksum != cacheState._masterCacheHashChecksum ||
                 cache->fullReadyBlockCount !=
                     static_cast<int64_t>(cacheState._fullReadyBlockHashes.size()) ||
                 cache->fullReadyHashChecksum !=
                     cacheState._masterFullReadyHashChecksum ||
                 cache->tailReadyBlockCount !=
                     static_cast<int64_t>(cacheState._tailReadyBlockHashes.size()) ||
                 cache->tailReadyHashChecksum !=
                     cacheState._masterTailReadyHashChecksum ||
                 !hashesMatch(cache->addedBlockHashes, cacheState._cachedBlockHashes) ||
                 !hashesMatch(cache->addedFullReadyBlockHashes,
                              cacheState._fullReadyBlockHashes) ||
                 !hashesMatch(cache->addedTailReadyBlockHashes,
                              cacheState._tailReadyBlockHashes));
            if (staleCacheSnapshotIgnored) {
            } else if (sameRevisionStateMismatch) {
                cacheStateMismatch = true;
                ++cacheState._cacheStateMismatchCount;
                clearMirroredCache();
                cacheState._cacheStateReady.store(false, std::memory_order_release);
            } else if (cache->cachedBlockCount >= 0) {
                if (cache->hybridTailRequired >= 0) {
                    cacheState._hybridTailRequired = cache->hybridTailRequired != 0;
                }
                cacheState._agentCachedBlockCount = cache->cachedBlockCount;
                cacheState._agentCacheHashChecksum = cache->hashChecksum;
                cacheState._agentFullReadyBlockCount = cache->fullReadyBlockCount;
                cacheState._agentFullReadyHashChecksum = cache->fullReadyHashChecksum;
                cacheState._agentTailReadyBlockCount = cache->tailReadyBlockCount;
                cacheState._agentTailReadyHashChecksum = cache->tailReadyHashChecksum;
                const bool completeState =
                    cache->fullReadyBlockCount >= 0 && cache->tailReadyBlockCount >= 0;
                cacheRevisionGap =
                    !cache->fullSnapshot &&
                    (!cacheState._cacheStateReady.load(std::memory_order_relaxed) ||
                     cache->revision != cacheState._cacheRevision + 1);
                if (cacheRevisionGap) {
                    ++cacheState._cacheRevisionGapCount;
                    clearMirroredCache();
                    cacheState._cacheStateReady.store(false, std::memory_order_release);
                } else if (!completeState) {
                    cacheStateMismatch = true;
                } else {
                    if (cache->fullSnapshot) {
                        clearMirroredCache();
                    }
                    for (auto hash : BlockHashUtil::get(cache->removedBlockHashes)) {
                        if (cacheState._cachedBlockHashes.erase(hash) > 0) {
                            cacheState._masterCacheHashChecksum ^= hash;
                        }
                    }
                    for (auto hash : BlockHashUtil::get(cache->addedBlockHashes)) {
                        if (cacheState._cachedBlockHashes.emplace(hash).second) {
                            cacheState._masterCacheHashChecksum ^= hash;
                        }
                    }
                    for (auto hash :
                         BlockHashUtil::get(cache->removedFullReadyBlockHashes)) {
                        if (cacheState._fullReadyBlockHashes.erase(hash) > 0) {
                            cacheState._masterFullReadyHashChecksum ^= hash;
                        }
                    }
                    for (auto hash :
                         BlockHashUtil::get(cache->addedFullReadyBlockHashes)) {
                        if (cacheState._fullReadyBlockHashes.emplace(hash).second) {
                            cacheState._masterFullReadyHashChecksum ^= hash;
                        }
                    }
                    for (auto hash :
                         BlockHashUtil::get(cache->removedTailReadyBlockHashes)) {
                        if (cacheState._tailReadyBlockHashes.erase(hash) > 0) {
                            cacheState._masterTailReadyHashChecksum ^= hash;
                        }
                    }
                    for (auto hash :
                         BlockHashUtil::get(cache->addedTailReadyBlockHashes)) {
                        if (cacheState._tailReadyBlockHashes.emplace(hash).second) {
                            cacheState._masterTailReadyHashChecksum ^= hash;
                        }
                    }
                    cacheStateMismatch =
                        cache->cachedBlockCount !=
                            static_cast<int64_t>(cacheState._cachedBlockHashes.size()) ||
                        cache->hashChecksum != cacheState._masterCacheHashChecksum ||
                        cache->fullReadyBlockCount !=
                            static_cast<int64_t>(
                                cacheState._fullReadyBlockHashes.size()) ||
                        cache->fullReadyHashChecksum !=
                            cacheState._masterFullReadyHashChecksum ||
                        cache->tailReadyBlockCount !=
                            static_cast<int64_t>(
                                cacheState._tailReadyBlockHashes.size()) ||
                        cache->tailReadyHashChecksum !=
                            cacheState._masterTailReadyHashChecksum;
                    if (!cacheStateMismatch) {
                        cacheState._cacheRevision = cache->revision;
                        cacheState._cacheStateReady.store(true,
                                                          std::memory_order_release);
                    }
                }
                if (cacheStateMismatch) {
                    ++cacheState._cacheStateMismatchCount;
                    clearMirroredCache();
                    cacheState._cacheStateReady.store(false, std::memory_order_release);
                }
                if (cache->rtpCachedBlockCount >= 0) {
                    ++cacheState._rtpCacheMirrorSampleCount;
                    cacheState._rtpCachedBlockCount = cache->rtpCachedBlockCount;
                    if (cache->rtpCachedBlockCount != cacheState._agentCachedBlockCount) {
                        rtpCacheMirrorMismatch = true;
                        ++cacheState._rtpCacheMirrorMismatchCount;
                    }
                }
            }
            cacheSyncStats = {
                cacheState._cacheRevision,
                cacheState._agentCachedBlockCount,
                cacheState._agentCacheHashChecksum,
                cacheState._agentFullReadyBlockCount,
                cacheState._agentFullReadyHashChecksum,
                cacheState._agentTailReadyBlockCount,
                cacheState._agentTailReadyHashChecksum,
                cacheState._cachedBlockHashes.size(),
                cacheState._masterCacheHashChecksum,
                cacheState._fullReadyBlockHashes.size(),
                cacheState._masterFullReadyHashChecksum,
                cacheState._tailReadyBlockHashes.size(),
                cacheState._masterTailReadyHashChecksum,
                cacheState._rtpCachedBlockCount,
                cacheState._cacheRevisionGapCount,
                cacheState._cacheStateMismatchCount,
                cacheState._rtpCacheMirrorSampleCount,
                cacheState._rtpCacheMirrorMismatchCount,
                cacheState._cacheStateReady.load(std::memory_order_relaxed)};
        }
        if (staleCacheSnapshotIgnored) {
            NAVI_KERNEL_LOG(INFO,
                            "ignore stale agent cache full snapshot, agentId [%s], "
                            "masterRevision [%lu], receivedRevision [%lu]",
                            agentId.c_str(), cacheSyncStats.agentRevision,
                            cache->revision);
        }
        const auto rtpMismatchCount = cacheSyncStats.rtpMirrorMismatchCount;
        const bool logRtpMismatch =
            rtpCacheMirrorMismatch &&
            (rtpMismatchCount <= 10 || (rtpMismatchCount & (rtpMismatchCount - 1)) == 0);
        if (cacheRevisionGap || cacheStateMismatch || logRtpMismatch) {
            NAVI_KERNEL_LOG(
                WARN,
                "agent cache state diverged, agentId [%s], revision [%lu], "
                "revisionGap [%d], agentBlocks [%ld], masterBlocks [%lu], "
                "rtpBlocks [%ld], rtpMirrorMismatch [%d], "
                "agentChecksum [%lu], masterChecksum [%lu], gapCount [%lu], "
                "mismatchCount [%lu], rtpMirrorSamples [%lu], "
                "rtpMirrorMismatches [%lu]",
                agentId.c_str(), cache->revision, static_cast<int>(cacheRevisionGap),
                cacheSyncStats.agentCachedBlockCount,
                cacheSyncStats.masterCachedBlockCount, cacheSyncStats.rtpCachedBlockCount,
                static_cast<int>(rtpCacheMirrorMismatch),
                cacheSyncStats.agentHashChecksum, cacheSyncStats.masterHashChecksum,
                cacheSyncStats.revisionGapCount, cacheSyncStats.stateMismatchCount,
                cacheSyncStats.rtpMirrorSampleCount,
                cacheSyncStats.rtpMirrorMismatchCount);
        }
        stateDiverged = stateDiverged || cacheRevisionGap || cacheStateMismatch;
        if (!staleCacheSnapshotIgnored && cache->totalBlocks >= 0 &&
            cache->cacheKeyType == NaviSchedCacheKeyType::BLOCK_POOL) {
            autil::ScopedLock runtimeLock(state->_runtimeLock);
            state->_runtimeState.load.totalBlocks = cache->totalBlocks;
        }
    }
    if (reservationTransitionFailed) {
        return false;
    }
    if (NaviSchedAgentEvent::LATENCY_MODEL_UPDATED == event && !recordLatencyModel()) {
        return false;
    }
    if (becameOnline) {
        NAVI_KERNEL_LOG(INFO, "agent online, agentId [%s], eventType [%s]",
                        agentId.c_str(), NaviSchedAgentEvent::getName(event).c_str());
    } else if (becameOffline) {
        NAVI_KERNEL_LOG(INFO, "agent offline, agentId [%s], eventType [%s]",
                        agentId.c_str(), NaviSchedAgentEvent::getName(event).c_str());
    }
    switch (event) {
        case NaviSchedAgentEvent::AGENT_HANDSHAKE:
            _agentHandshakeCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case NaviSchedAgentEvent::AGENT_STOPPING:
            _agentStoppingCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case NaviSchedAgentEvent::REQUEST_RECEIVED:
            _agentRequestReceivedCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case NaviSchedAgentEvent::EXECUTION_LOAD_UPDATED:
            _agentExecutionLoadCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case NaviSchedAgentEvent::BATCH_FINISHED: {
            _agentBatchFinishedCount.fetch_add(1, std::memory_order_relaxed);
            const auto completedRequestCount = request->completedRequestCount;
            if (request->abnormalBatchFinished) {
                _agentRequestAbnormalFinishedCount.fetch_add(completedRequestCount,
                                                             std::memory_order_relaxed);
                break;
            }
            _agentRequestFinishedCount.fetch_add(completedRequestCount,
                                                 std::memory_order_relaxed);
            _agentRequestCacheHitTokens.fetch_add(request->cacheHitTokens,
                                                  std::memory_order_relaxed);
            _agentRequestCacheMissTokens.fetch_add(request->cacheMissTokens,
                                                   std::memory_order_relaxed);
            _agentRequestTotalLatencyUs.fetch_add(
                static_cast<int64_t>(
                    std::llround(request->latencyMs * 1000.0 * completedRequestCount)),
                std::memory_order_relaxed);
            break;
        }
        case NaviSchedAgentEvent::BATCH_REQUESTS_ABORTED:
            _agentRequestAbnormalFinishedCount.fetch_add(request->completedRequestCount,
                                                         std::memory_order_relaxed);
            break;
        case NaviSchedAgentEvent::CACHE_UPDATED:
            _agentCacheUpdateCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case NaviSchedAgentEvent::PENDING_CACHE_UPDATED:
            _agentPendingCacheUpdateCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case NaviSchedAgentEvent::LATENCY_MODEL_UPDATED:
            _agentLatencyModelUpdateCount.fetch_add(1, std::memory_order_relaxed);
            break;
        case NaviSchedAgentEvent::WORKLOAD_UPDATED:
            _agentWorkloadUpdateCount.fetch_add(1, std::memory_order_relaxed);
            break;
        default:
            break;
    }
    _agentReportCount.fetch_add(1, std::memory_order_relaxed);
    if (NaviSchedAgentEvent::AGENT_STOPPING == event) {
        MasterPendingCacheR::Stats pendingStats;
        if (resetAgentState(state, &pendingStats)) {
            NAVI_KERNEL_LOG(INFO,
                            "[navi-master-pending-cache-lifecycle] phase=agent_stopping "
                            "agent=%s cleared_batches=%lu cleared_blocks=%lu",
                            agentId.c_str(), pendingStats.batchCount,
                            pendingStats.uniqueBlockCount);
        } else {
            NAVI_KERNEL_LOG(INFO, "ignore stale agent stopping state reset, agentId [%s]",
                            agentId.c_str());
        }
    }
    if (stateDiverged) {
        if (resyncRequired) {
            *resyncRequired = true;
        }
        return false;
    }
    return true;
}

bool MasterStateR::receiveClientRequest(const ClientRequestToMasterData &request)
{
    if (!request.valid()) {
        return false;
    }
    _clientRequestCount.fetch_add(1, std::memory_order_relaxed);
    if (_metricsReporterR) {
        const auto &reporter = _metricsReporterR->getReporter();
        if (reporter) {
            reporter->report(1, "master.client_request_received_qps",
                             kmonitor::MetricType::QPS, nullptr,
                             kmonitor::MetricLevel::FATAL);
        }
    }
    return true;
}

autil::legacy::json::JsonMap MasterStateR::NodeState::makePendingReplayState() const
{
    return capturePendingReplaySnapshot().toJson();
}

MasterStateR::NodeState::PendingReplaySnapshot
MasterStateR::NodeState::capturePendingReplaySnapshot() const
{
    PendingReplaySnapshot snapshot;
    autil::ScopedLock runtimeLock(_runtimeLock);
    autil::ScopedReadLock cacheLock(_cacheLock);
    snapshot.captureTimeUs = autil::TimeUtility::currentTime();
    snapshot.agentId = _info.agentId;
    snapshot.agentIdHash = _info.agentIdHash;
    const auto copyHashes = [](const auto &hashes, auto &output) {
        output.reserve(hashes.size());
        for (const auto hash : hashes) output.push_back(hash);
    };
    copyHashes(_cachedBlockHashes, snapshot.cachedBlockHashes);
    copyHashes(_fullReadyBlockHashes, snapshot.fullReadyBlockHashes);
    copyHashes(_tailReadyBlockHashes, snapshot.tailReadyBlockHashes);
    snapshot.hybridTailRequired = _hybridTailRequired;
    snapshot.blockSize = _cachePolicyReady.load(std::memory_order_relaxed)
                             ? _cachePolicy.blockSize
                             : int64_t(-1);
    snapshot.reservations.reserve(_pendingBatchReservations.size());
    for (const auto &reservationPtr : _pendingBatchReservations) {
        const auto &reservation = *reservationPtr;
        snapshot.reservations.emplace_back(reservation.batch);
    }
    if (_pendingCacheR) {
        snapshot.hasPendingCache = true;
        snapshot.pendingCache = _pendingCacheR->captureReplaySnapshot(_info.agentIdHash);
    }
    return snapshot;
}

autil::legacy::json::JsonMap MasterStateR::NodeState::PendingReplaySnapshot::toJson()
    const
{
    using autil::legacy::ToJson;
    autil::legacy::json::JsonMap result;
    result["capture_time_us"] = ToJson(captureTimeUs);
    result["agent_id"] = ToJson(agentId);
    result["agent_id_hash"] = ToJson(agentIdHash);
    result["cached_block_hashes"] = ToJson(cachedBlockHashes);
    result["full_ready_block_hashes"] = ToJson(fullReadyBlockHashes);
    result["tail_ready_block_hashes"] = ToJson(tailReadyBlockHashes);
    result["hybrid_tail_required"] = ToJson(hybridTailRequired);
    result["block_size"] = ToJson(blockSize);
    std::vector<autil::legacy::json::JsonMap> records;
    records.reserve(reservations.size());
    for (const auto &reservation : reservations) {
        records.emplace_back(reservation.makeReplayState());
    }
    result["reservations"] = ToJson(records);
    if (hasPendingCache) result["pending_cache"] = ToJson(pendingCache.toJson());
    return result;
}

std::shared_ptr<NaviSchedScheduleReplay> MasterStateR::makeScheduleReplay(
    const NaviSchedScheduleContext &context) const
{
    auto replay = std::make_shared<NaviSchedScheduleReplay>();
    replay->scheduleTimeUs = context._scheduleTimeUs;
    replay->requests = context._requests;
    replay->bizName = replay->requests.empty() ? std::string()
                                               : replay->requests.front().request.bizName;
    for (auto &request : replay->requests) {
        request.request.bizName.clear();
    }
    replay->nodes.reserve(context._scheduleNodes.size());
    replay->nodeMatches.resize(context._requests.size());
    for (auto &matches : replay->nodeMatches) {
        matches.reserve(context._scheduleNodes.size());
    }
    std::vector<int64_t> matchedBlockCounts(context._requests.size());
    std::vector<autil::legacy::json::JsonMap> pendingStates;
    for (size_t nodeIndex = 0; nodeIndex < context._scheduleNodes.size(); ++nodeIndex) {
        const auto &scheduleNode = context._scheduleNodes[nodeIndex];
        const auto &node = scheduleNode._node;
        bool capturedMatches = false;
        if (const auto state = std::dynamic_pointer_cast<const NodeState>(node)) {
            const auto snapshot = state->capturePendingReplaySnapshot();
            pendingStates.emplace_back(snapshot.toJson());
            MasterPendingCacheR frozenPending;
            if (snapshot.pendingCache.present &&
                !frozenPending.restoreReplayState(snapshot.agentIdHash,
                                                  snapshot.pendingCache.toJson())) {
                NAVI_KERNEL_LOG(
                    ERROR, "restore captured replay pending state failed, agent [%s]",
                    snapshot.agentId.c_str());
                return nullptr;
            }
            const MasterPendingCacheR::PhysicalBlockSet cached(
                snapshot.cachedBlockHashes.begin(), snapshot.cachedBlockHashes.end());
            const MasterPendingCacheR::PhysicalBlockSet fullReady(
                snapshot.fullReadyBlockHashes.begin(),
                snapshot.fullReadyBlockHashes.end());
            const MasterPendingCacheR::PhysicalBlockSet tailReady(
                snapshot.tailReadyBlockHashes.begin(),
                snapshot.tailReadyBlockHashes.end());
            frozenPending.getMatchedBlockCounts(snapshot.agentIdHash, context._requests,
                                                snapshot.blockSize, cached, fullReady,
                                                tailReady, snapshot.hybridTailRequired,
                                                matchedBlockCounts.data());
            capturedMatches = true;
        } else {
            pendingStates.emplace_back();
        }
        NaviSchedNodeInfo nodeInfo =
            scheduleNode._info ? *scheduleNode._info : NaviSchedNodeInfo();
        nodeInfo.totalBlocks = scheduleNode._totalBlocks;
        nodeInfo.inflightRequests = scheduleNode._inflightRequests;
        nodeInfo.inflightPredictedLatencySumMs =
            scheduleNode._inflightPredictedLatencySumMs;
        nodeInfo.batchWaitingRequests = scheduleNode._batchWaitingRequests;
        nodeInfo.queuedBatchRequests = scheduleNode._queuedBatchRequests;
        nodeInfo.runningRequests = scheduleNode._runningRequests;
        nodeInfo.batchTimeoutCount = scheduleNode._batchTimeoutCount;
        nodeInfo.abnormalBatchFinishCount = scheduleNode._abnormalBatchFinishCount;
        nodeInfo.queueModel = scheduleNode._queueModel;
        nodeInfo.queueModel.setBaseMs(
            scheduleNode.getEstimatedQueueBaseMsAt(context._scheduleTimeUs),
            nodeInfo.queueTotalPredictedLatencyMs);
        const auto elapsedMs =
            scheduleNode._queueReportReceiveTimeUs > 0
                ? std::max<int64_t>(0, context._scheduleTimeUs -
                                           scheduleNode._queueReportReceiveTimeUs) /
                      1000.0 * context._timeCompression
                : 0.0;
        nodeInfo.queueDecayHeadRemainingMs = std::max(
            0.0, scheduleNode._queueDecayHeadRemainingMs -
                     std::min(scheduleNode._queueDecayHeadRemainingMs, elapsedMs));
        nodeInfo.queueReportReceiveTimeUs = context._scheduleTimeUs;
        nodeInfo.unifiedLatencyPrediction = scheduleNode._unifiedLatencyPrediction;
        nodeInfo.workload = scheduleNode._workload;
        nodeInfo.reservedWorkloadFeatures = scheduleNode._reservedWorkloadFeatures;
        nodeInfo.workloadFeatureSnapshotValid = true;
        nodeInfo.workloadFeatures = scheduleNode._workloadFeatures;
        nodeInfo.maxRunningRequestCount = scheduleNode._maxRunningRequestCount;
        nodeInfo.workloadRunningCapacitySlotCount =
            scheduleNode._workloadRunningCapacitySlotCount;
        nodeInfo.reservedRunningCapacitySlotCount =
            scheduleNode._reservedRunningCapacitySlotCount;
        nodeInfo.workloadReportReceiveTimeUs = scheduleNode._workloadReportReceiveTimeUs;
        if (scheduleNode._latencyModel) {
            nodeInfo.latencyFormula = scheduleNode._latencyModel->formula;
            nodeInfo.latencyBinaryVersion = scheduleNode._latencyModel->binaryVersion;
            nodeInfo.latencyModelRevision = scheduleNode._latencyModel->revision;
            nodeInfo.latencyParameters = scheduleNode._latencyModel->parameters;
            nodeInfo.latencyResidualVersion = scheduleNode._latencyModel->residualVersion;
            nodeInfo.latencyResidualParameters =
                scheduleNode._latencyModel->residualParameters;
            nodeInfo.latencyResidualLastUpdateTimeUs =
                scheduleNode._latencyModel->residualLastUpdateTimeUs;
            nodeInfo.latencyModelTrusted =
                scheduleNode._latencyModel->overloadPredictionTrusted;
            nodeInfo.latencyModelWarmStarted = scheduleNode._latencyModel->warmStarted;
        }
        nodeInfo.bizName.clear();
        replay->nodes.emplace_back(std::move(nodeInfo));
        if (node && !capturedMatches) {
            if (!context.getCacheHitBlockCounts(nodeIndex, matchedBlockCounts)) {
                return nullptr;
            }
        }
        for (size_t requestIndex = 0; requestIndex < context._requests.size();
             ++requestIndex) {
            NaviSchedReplayNodeMatch match;
            match.matchedBlockCount = node ? matchedBlockCounts[requestIndex] : 0;
            replay->nodeMatches[requestIndex].emplace_back(std::move(match));
        }
    }
    replay->schedulerInfo["pending_state"] = autil::legacy::ToJson(pendingStates);
    return replay;
}

void MasterStateR::snapshotScheduleNodes(const std::string &bizName,
                                         NaviSchedScheduleContext &context) const
{
    context._scheduleNodes.clear();
    context._snapshotRuntimeLockWaitUs = 0;
    context._snapshotRuntimeLockHoldUs = 0;
    NodeStateMapConstPtr states;
    {
        autil::ScopedReadLock lock(_nodeIndexLock);
        auto bizIt = _readyNodesByBiz.find(bizName);
        if (_readyNodesByBiz.end() != bizIt) {
            states = bizIt->second;
        }
    }
    if (!states) {
        return;
    }
    context._scheduleNodes.reserve(states->size());
    for (const auto &[agentIdHash, state] : *states) {
        (void)agentIdHash;
        NaviSchedNodeSnapshot snapshot;
        const auto accepted = state->getScheduleSnapshot(bizName, snapshot);
        context._snapshotRuntimeLockWaitUs += snapshot.runtimeLockWaitUs;
        context._snapshotRuntimeLockHoldUs += snapshot.runtimeLockHoldUs;
        if (accepted) {
            context._scheduleNodes.push_back(
                NaviSchedScheduleNode(state, std::move(snapshot), context._scheduleTimeUs,
                                      context._timeCompression));
        }
    }
}

void MasterStateR::reserveScheduleBatches(NaviSchedScheduleContext &context)
{
    context._dispatchHandles.assign(context.requestCount(), MasterDispatchHandle());
    std::vector<std::vector<size_t>> nodeRequestIndexes(context.nodeCount());
    for (size_t requestIndex = 0; requestIndex < context.requestCount(); ++requestIndex) {
        const auto nodeIndex = context._selectedNodeIndexes[requestIndex];
        if (nodeIndex < nodeRequestIndexes.size()) {
            nodeRequestIndexes[nodeIndex].emplace_back(requestIndex);
        }
    }
    for (size_t nodeIndex = 0; nodeIndex < nodeRequestIndexes.size(); ++nodeIndex) {
        if (!nodeRequestIndexes[nodeIndex].empty()) {
            reserveScheduleNodeBatch(context, nodeIndex, nodeRequestIndexes[nodeIndex]);
        }
    }
}

void MasterStateR::reserveScheduleNodeBatch(NaviSchedScheduleContext &context,
                                            size_t nodeIndex,
                                            const std::vector<size_t> &requestIndexes)
{
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    const auto beginCpuUs = autil::TimeUtility::cpuThreadTimeUs();
    int64_t requestBuildUs = 0, registryWaitUs = 0, auditUs = 0;
    int64_t policyUs = 0, estimateUs = 0, reserveUs = 0, handlesUs = 0;
    int64_t estimateCpuUs = 0, reserveCpuUs = 0;
    NodeState::ReservationTiming reservationTiming;
    autil::ScopeGuard recordTiming([&]() {
        const auto totalUs = autil::TimeUtility::monotonicTimeUs() - beginUs;
        if (totalUs >= 50'000 || context._scheduleId % 100 == 0) {
            const auto totalCpuUs = autil::TimeUtility::cpuThreadTimeUs() - beginCpuUs;
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-master-reserve-node-timing] function=reserveScheduleNodeBatch "
                "schedule_id=%lu agent=%s requests=%lu total_us=%ld "
                "request_build_us=%ld registry_wait_us=%ld audit_us=%ld policy_us=%ld "
                "estimate_us=%ld reserve_us=%ld handles_us=%ld "
                "total_cpu_us=%ld estimate_cpu_us=%ld reserve_cpu_us=%ld "
                "init_us=%ld init_cpu_us=%ld pending_publish_us=%ld "
                "pending_publish_cpu_us=%ld runtime_wait_us=%ld runtime_hold_cpu_us=%ld",
                context._scheduleId,
                context._scheduleNodes[nodeIndex].getAgentId().c_str(),
                requestIndexes.size(), totalUs, requestBuildUs, registryWaitUs, auditUs,
                policyUs, estimateUs, reserveUs, handlesUs, totalCpuUs, estimateCpuUs,
                reserveCpuUs, reservationTiming.initUs, reservationTiming.initCpuUs,
                reservationTiming.pendingPublishUs, reservationTiming.pendingPublishCpuUs,
                reservationTiming.runtimeWaitUs, reservationTiming.runtimeHoldCpuUs);
        }
    });
    auto rejectReservation = [&context, nodeIndex](
                                 const std::vector<size_t> &requestIndexes,
                                 const char *reason) {
        const auto batchId = context._decisions[requestIndexes.front()].batchId;
        if (context.isCollectingReplay()) {
            context._replay
                ->schedulerInfo["reservation_failure_" + std::to_string(batchId)] =
                autil::legacy::json::JsonMap{
                    {"reason", std::string(reason)},
                    {"agent", context._scheduleNodes[nodeIndex].getAgentId()},
                    {"batch_id", batchId},
                    {"request_count", uint64_t(requestIndexes.size())},
                };
        }
        NAVI_KERNEL_LOG(WARN,
                        "[navi-master-reservation-failed] schedule_id=%lu agent=%s "
                        "batch_id=%ld requests=%lu reason=%s",
                        context._scheduleId,
                        context._scheduleNodes[nodeIndex].getAgentId().c_str(), batchId,
                        requestIndexes.size(), reason);
        for (auto requestIndex : requestIndexes) {
            auto &decision = context._decisions[requestIndex];
            decision.reason = "reservation_failed";
            decision.batchId = 0;
            decision.expectedBatchSize = 0;
            decision.predictedCacheHitTokenCount = 0;
            decision.predictedQueueLatencyMs = 0.0;
            decision.predictedExecutionLatencyMs = 0.0;
            decision.target = NaviSchedTarget();
            context._selectedNodeIndexes[requestIndex] =
                NaviSchedScheduleContext::INVALID_NODE_INDEX;
        }
    };
    const auto routeOnly = context._requests[requestIndexes.front()].request.routeOnly;
    const auto mixedIngress = std::any_of(
        requestIndexes.begin(), requestIndexes.end(), [&](size_t requestIndex) {
            return context._requests[requestIndex].request.routeOnly != routeOnly;
        });
    if (mixedIngress) {
        NAVI_KERNEL_LOG(ERROR, "mixed route-only and forwarded requests in one batch");
        rejectReservation(requestIndexes, "mixed_ingress_batch");
        return;
    }
    if (routeOnly) return;
    std::vector<MasterBatchReservation::Request> reservationRequests;
    reservationRequests.reserve(requestIndexes.size());
    for (auto requestIndex : requestIndexes) {
        const auto &request = context._requests[requestIndex].request;
        reservationRequests.push_back(
            {request.requestId, request.promptTokenCount, request.blockSize,
             context._decisions[requestIndex].predictedCacheHitTokenCount,
             request.blockHashes,
             context._decisions[requestIndex].predictedQueueLatencyMs +
                 context._decisions[requestIndex].predictedExecutionLatencyMs,
             request.explicitCacheCheckpointPosition});
    }
    const auto &decision = context._decisions[requestIndexes.front()];
    auto state =
        std::const_pointer_cast<NodeState>(std::dynamic_pointer_cast<const NodeState>(
            context._scheduleNodes[nodeIndex]._node));
    if (!state) {
        rejectReservation(requestIndexes, "invalid_node_state");
        return;
    }
    auto stageBeginUs = autil::TimeUtility::monotonicTimeUs();
    requestBuildUs = stageBeginUs - beginUs;
    autil::ScopedReadLock nodeIndexLock(_nodeIndexLock);
    auto stageEndUs = autil::TimeUtility::monotonicTimeUs();
    registryWaitUs = stageEndUs - stageBeginUs;
    stageBeginUs = stageEndUs;
    if (!isCurrentNodeStateLocked(state->_info.agentIdHash, state)) {
        rejectReservation(requestIndexes, "node_state_replaced");
        return;
    }
    if (context.isCollectingReplay()) {
        for (auto requestIndex : requestIndexes) {
            const auto &request = context._requests[requestIndex].request;
            const auto &requestDecision = context._decisions[requestIndex];
            if (request.blockSize <= 0 ||
                requestDecision.predictedCacheHitTokenCount <= 0) {
                continue;
            }
            const auto breakdown = state->getCacheMatchBreakdown(request);
            const auto decisionHitBlocks =
                requestDecision.predictedCacheHitTokenCount / request.blockSize;
            const auto pendingHitBlocks =
                breakdown.matchedBlockCount - breakdown.staticMatchedBlockCount;
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-master-cache-decision-audit] request_id=%lu "
                "schedule_time_us=%ld batch_id=%ld agent=%s block_size=%ld "
                "decision_hit_blocks=%ld static_hit_blocks=%ld "
                "pending_hit_blocks=%ld recomputed_hit_blocks=%ld "
                "pending_first_source_batch_id=%ld "
                "pending_tail_source_batch_id=%ld "
                "decision_matches_recomputed=%d cache_revision=%lu "
                "pending_batches=%lu pending_blocks=%lu cache_ready=%d",
                request.requestId, context._scheduleTimeUs, requestDecision.batchId,
                requestDecision.target.agentAddress.c_str(), request.blockSize,
                decisionHitBlocks, breakdown.staticMatchedBlockCount, pendingHitBlocks,
                breakdown.matchedBlockCount, breakdown.firstPendingBlockSourceBatchId,
                breakdown.pendingTailSourceBatchId,
                static_cast<int>(breakdown.valid &&
                                 decisionHitBlocks == breakdown.matchedBlockCount),
                breakdown.cacheRevision, breakdown.pendingBatchCount,
                breakdown.pendingBlockCount, static_cast<int>(breakdown.valid));
        }
    }
    stageEndUs = autil::TimeUtility::monotonicTimeUs();
    auditUs = stageEndUs - stageBeginUs;
    stageBeginUs = stageEndUs;
    NaviSchedCachePolicy cachePolicy;
    if (!state->getCachePolicy(cachePolicy)) {
        NAVI_KERNEL_LOG(ERROR,
                        "missing cache policy while reserving batch, "
                        "agentId [%s], batchId [%ld]",
                        state->_info.agentId.c_str(), decision.batchId);
        rejectReservation(requestIndexes, "missing_cache_policy");
        return;
    }
    bool pendingCacheTracked = false;
    NodeState::ReservationPtr reserved;
    stageEndUs = autil::TimeUtility::monotonicTimeUs();
    policyUs = stageEndUs - stageBeginUs;
    stageBeginUs = stageEndUs;
    const auto estimateBeginCpuUs = autil::TimeUtility::cpuThreadTimeUs();
    const auto predictedLatencyMs =
        context.estimateBatchExecutionLatencyMs(nodeIndex, requestIndexes);
    const auto estimateEndCpuUs = autil::TimeUtility::cpuThreadTimeUs();
    stageEndUs = autil::TimeUtility::monotonicTimeUs();
    estimateUs = stageEndUs - stageBeginUs;
    estimateCpuUs = estimateEndCpuUs - estimateBeginCpuUs;
    stageBeginUs = stageEndUs;
    if (!state->reserveBatch(decision.batchId, context._scheduleTimeUs,
                             predictedLatencyMs, std::move(reservationRequests),
                             &cachePolicy, &pendingCacheTracked,
                             &reserved, &reservationTiming)) {
        NAVI_KERNEL_LOG(ERROR,
                        "failed to reserve scheduled batch, agentId [%s], "
                        "batchId [%ld], requestCount [%lu]",
                        state->_info.agentId.c_str(), decision.batchId,
                        requestIndexes.size());
        rejectReservation(requestIndexes, reservationTiming.failureReason
                                              ? reservationTiming.failureReason
                                              : "unknown_reservation_failure");
        return;
    }
    stageEndUs = autil::TimeUtility::monotonicTimeUs();
    reserveCpuUs = autil::TimeUtility::cpuThreadTimeUs() - estimateEndCpuUs;
    reserveUs = stageEndUs - stageBeginUs;
    stageBeginUs = stageEndUs;
    auto handleState = std::make_shared<MasterDispatchHandle::State>();
    handleState->node = state;
    handleState->reservation = reserved;
    for (size_t index = 0; index < requestIndexes.size(); ++index) {
        auto &handle = context._dispatchHandles[requestIndexes[index]];
        handle._state = handleState;
        handle._requestIndex = index;
    }
    if (pendingCacheTracked) {
        const auto pendingStats =
            state->_pendingCacheR->getStats(state->_info.agentIdHash);
        NAVI_KERNEL_LOG(INFO,
                        "[navi-master-pending-cache-lifecycle] phase=reserved "
                        "agent=%s batch_id=%ld schedule_time_us=%ld "
                        "request_count=%lu pending_batches=%lu "
                        "pending_blocks=%lu",
                        state->_info.agentId.c_str(), decision.batchId,
                        context._scheduleTimeUs, requestIndexes.size(),
                        pendingStats.batchCount, pendingStats.uniqueBlockCount);
    }
    handlesUs = autil::TimeUtility::monotonicTimeUs() - stageBeginUs;
}

MasterStateR::NodeStatePtr MasterStateR::findNodeState(
    const NaviSchedTarget &target) const
{
    for (const auto &node : getNodeStates()) {
        auto info =
            std::atomic_load_explicit(&node->_infoSnapshot, std::memory_order_acquire);
        if (!info || info->bizName != target.bizName ||
            info->collectorAddress != target.agentAddress) {
            continue;
        }
        return std::const_pointer_cast<NodeState>(node);
    }
    return nullptr;
}

bool MasterStateR::registerDispatchWaiter(const NaviSchedTarget &target, int64_t batchId,
                                          uint64_t requestId,
                                          const navi::AsyncPipePtr &pipe,
                                          bool &acknowledged)
{
    acknowledged = false;
    if (requestId == 0 || batchId <= 0 || target.bizName.empty() ||
        target.agentAddress.empty() || !pipe) {
        return false;
    }
    auto state = findNodeState(target);
    if (!state) {
        acknowledged = true;
        return true;
    }
    return state->registerDispatchWaiter(batchId, requestId, pipe, acknowledged);
}

bool MasterStateR::unregisterDispatchWaiter(const NaviSchedTarget &target,
                                            int64_t batchId, uint64_t requestId,
                                            const navi::AsyncPipePtr &pipe)
{
    auto state = findNodeState(target);
    return state && state->unregisterDispatchWaiter(batchId, requestId, pipe);
}

bool MasterStateR::cancelDispatchWaiter(const NaviSchedTarget &target, int64_t batchId,
                                        uint64_t requestId,
                                        const navi::AsyncPipePtr &pipe)
{
    if (requestId == 0 || batchId <= 0 || target.bizName.empty() ||
        target.agentAddress.empty() || !pipe) {
        return false;
    }
    auto state = findNodeState(target);
    if (!state) {
        return false;
    }
    bool batchReleased = false;
    return state->cancelDispatchWaiter(batchId, requestId, pipe, batchReleased);
}

bool MasterStateR::reportDispatchFailed(const NaviSchedTarget &target, int64_t batchId,
                                        uint64_t requestId)
{
    if (requestId == 0 || batchId <= 0 || target.bizName.empty() ||
        target.agentAddress.empty()) {
        return false;
    }
    auto state = findNodeState(target);
    if (!state) {
        return false;
    }
    auto info =
        std::atomic_load_explicit(&state->_infoSnapshot, std::memory_order_acquire);
    if (!info) {
        return false;
    }
    bool batchReleased = false;
    if (!state->reportDispatchFailed(batchId, requestId, batchReleased)) {
        return false;
    }
    NAVI_KERNEL_LOG(WARN,
                    "[navi-master-dispatch-lifecycle] phase=failed agent=%s "
                    "batch_id=%ld request_id=%lu batch_released=%d",
                    info->agentId.c_str(), batchId, requestId,
                    static_cast<int>(batchReleased));
    return true;
}

bool MasterStateR::reportDispatchFailures(const NaviSchedTarget &target, int64_t batchId,
                                          const std::vector<uint64_t> &requestIds)
{
    if (requestIds.empty() || batchId <= 0 || target.bizName.empty() ||
        target.agentAddress.empty()) {
        return false;
    }
    auto state = findNodeState(target);
    if (!state) {
        return false;
    }
    const auto info =
        std::atomic_load_explicit(&state->_infoSnapshot, std::memory_order_acquire);
    bool batchReleased = false;
    if (!info || !state->reportDispatchFailures(batchId, requestIds, batchReleased)) {
        return false;
    }
    NAVI_KERNEL_LOG(WARN,
                    "[navi-master-dispatch-lifecycle] phase=bulk_failed agent=%s "
                    "batch_id=%ld request_count=%lu batch_released=%d",
                    info->agentId.c_str(), batchId, requestIds.size(),
                    static_cast<int>(batchReleased));
    return true;
}

bool MasterStateR::reportDispatchAbandoned(const NaviSchedTarget &target, int64_t batchId,
                                           uint64_t requestId)
{
    if (requestId == 0 || batchId <= 0 || target.bizName.empty() ||
        target.agentAddress.empty()) {
        return false;
    }
    auto state = findNodeState(target);
    if (!state) {
        return false;
    }
    auto info =
        std::atomic_load_explicit(&state->_infoSnapshot, std::memory_order_acquire);
    if (!info) {
        return false;
    }
    bool batchReleased = false;
    if (!state->abandonUnownedDispatch(batchId, requestId, batchReleased)) {
        return false;
    }
    NAVI_KERNEL_LOG(WARN,
                    "[navi-master-dispatch-lifecycle] phase=abandoned agent=%s "
                    "batch_id=%ld request_id=%lu batch_released=%d",
                    info->agentId.c_str(), batchId, requestId,
                    static_cast<int>(batchReleased));
    return true;
}

std::map<std::string, NaviSchedNodeInfo> MasterStateR::getNodes() const
{
    auto states = getNodeStates();
    std::map<std::string, NaviSchedNodeInfo> nodes;
    for (const auto &state : states) {
        auto info = state->getInfo();
        nodes[info.agentId] = std::move(info);
    }
    return nodes;
}

MasterStateR::NodeStateConstPtr MasterStateR::getNodeState(uint64_t agentIdHash) const
{
    autil::ScopedReadLock lock(_nodeIndexLock);
    auto pendingIt = _pendingNodes.find(agentIdHash);
    if (_pendingNodes.end() != pendingIt) {
        return pendingIt->second;
    }
    for (const auto &bizNodes : _readyNodesByBiz) {
        auto nodeIt = bizNodes.second->find(agentIdHash);
        if (bizNodes.second->end() != nodeIt) {
            return nodeIt->second;
        }
    }
    return nullptr;
}

MasterStateR::AgentRequestStats MasterStateR::getAgentRequestStats() const
{
    AgentRequestStats stats;
    stats.reportCount = _agentReportCount.load(std::memory_order_relaxed);
    stats.handshakeCount = _agentHandshakeCount.load(std::memory_order_relaxed);
    stats.stoppingCount = _agentStoppingCount.load(std::memory_order_relaxed);
    stats.receivedCount = _agentRequestReceivedCount.load(std::memory_order_relaxed);
    stats.executionLoadCount = _agentExecutionLoadCount.load(std::memory_order_relaxed);
    stats.batchFinishedCount = _agentBatchFinishedCount.load(std::memory_order_relaxed);
    stats.finishedCount = _agentRequestFinishedCount.load(std::memory_order_relaxed);
    stats.abnormalFinishedCount =
        _agentRequestAbnormalFinishedCount.load(std::memory_order_relaxed);
    stats.cacheUpdateCount = _agentCacheUpdateCount.load(std::memory_order_relaxed);
    stats.pendingCacheUpdateCount =
        _agentPendingCacheUpdateCount.load(std::memory_order_relaxed);
    stats.latencyModelUpdateCount =
        _agentLatencyModelUpdateCount.load(std::memory_order_relaxed);
    stats.workloadUpdateCount = _agentWorkloadUpdateCount.load(std::memory_order_relaxed);
    stats.cacheHitTokens = _agentRequestCacheHitTokens.load(std::memory_order_relaxed);
    stats.cacheMissTokens = _agentRequestCacheMissTokens.load(std::memory_order_relaxed);
    stats.totalLatencyUs = _agentRequestTotalLatencyUs.load(std::memory_order_relaxed);
    return stats;
}

void MasterStateR::recordScheduleRound(size_t requestCount)
{
    if (0 == requestCount) {
        return;
    }
    if (_metricsReporterR) {
        const auto &reporter = _metricsReporterR->getReporter();
        if (reporter) {
            reporter->report(requestCount, "master.schedule_round_request_count",
                             kmonitor::MetricType::GAUGE, nullptr,
                             kmonitor::MetricLevel::FATAL);
        }
    }
    autil::ScopedLock lock(_scheduleRoundStatsLock);
    auto &stats = _scheduleRoundStats;
    ++stats.roundCount;
    stats.requestCount += requestCount;
    stats.minRequestCount = 1 == stats.roundCount
                                ? requestCount
                                : std::min(stats.minRequestCount, requestCount);
    stats.maxRequestCount = std::max(stats.maxRequestCount, requestCount);
    ++stats.requestCountHistogram[requestCount];
}

MasterStateR::ScheduleRoundStats MasterStateR::getScheduleRoundStats() const
{
    autil::ScopedLock lock(_scheduleRoundStatsLock);
    return _scheduleRoundStats;
}

uint64_t MasterStateR::getClientRequestCount() const
{
    return _clientRequestCount.load(std::memory_order_relaxed);
}

uint64_t MasterStateR::getAgentReportCount() const
{
    return _agentReportCount.load(std::memory_order_relaxed);
}

std::map<std::string, std::string> MasterStateR::getAgentCollectors() const
{
    autil::ScopedReadLock lock(_collectorLock);
    return _agentCollectors;
}

MasterStateR::NodeStatePtr MasterStateR::getOrCreateNode(uint64_t agentIdHash,
                                                         const std::string &agentId)
{
    {
        autil::ScopedReadLock lock(_nodeIndexLock);
        auto pendingIt = _pendingNodes.find(agentIdHash);
        if (_pendingNodes.end() != pendingIt) {
            return pendingIt->second;
        }
        for (const auto &bizNodes : _readyNodesByBiz) {
            auto nodeIt = bizNodes.second->find(agentIdHash);
            if (bizNodes.second->end() != nodeIt) {
                return nodeIt->second;
            }
        }
    }
    autil::ScopedWriteLock lock(_nodeIndexLock);
    auto pendingIt = _pendingNodes.find(agentIdHash);
    if (_pendingNodes.end() != pendingIt) {
        return pendingIt->second;
    }
    for (const auto &bizNodes : _readyNodesByBiz) {
        auto nodeIt = bizNodes.second->find(agentIdHash);
        if (bizNodes.second->end() != nodeIt) {
            return nodeIt->second;
        }
    }
    auto state = std::make_shared<NodeState>();
    state->_pendingCacheR = _pendingCacheR;
    state->_info.agentIdHash = agentIdHash;
    state->_info.agentId = agentId;
    state->rebuildInfoSnapshot();
    _pendingNodes[agentIdHash] = state;
    return state;
}

bool MasterStateR::isCurrentNodeStateLocked(uint64_t agentIdHash,
                                            const NodeStatePtr &state) const
{
    auto pendingIt = _pendingNodes.find(agentIdHash);
    if (_pendingNodes.end() != pendingIt && pendingIt->second == state) {
        return true;
    }
    for (const auto &bizNodes : _readyNodesByBiz) {
        auto nodeIt = bizNodes.second->find(agentIdHash);
        if (bizNodes.second->end() != nodeIt && nodeIt->second == state) {
            return true;
        }
    }
    return false;
}

void MasterStateR::updateNodeBiz(uint64_t agentIdHash, const NodeStatePtr &state,
                                 const std::string &bizName)
{
    if (!state || bizName.empty()) {
        return;
    }
    autil::ScopedWriteLock nodeIndexLock(_nodeIndexLock);
    autil::ScopedWriteLock infoLock(state->_infoLock);
    auto &oldBizName = state->_info.bizName;
    if (oldBizName == bizName) {
        return;
    }
    bool current = false;
    auto pendingIt = _pendingNodes.find(agentIdHash);
    if (_pendingNodes.end() != pendingIt && pendingIt->second == state) {
        current = true;
    }
    if (!oldBizName.empty()) {
        auto oldBizIt = _readyNodesByBiz.find(oldBizName);
        if (_readyNodesByBiz.end() != oldBizIt) {
            auto oldNodeIt = oldBizIt->second->find(agentIdHash);
            current = current || (oldBizIt->second->end() != oldNodeIt &&
                                  oldNodeIt->second == state);
        }
        removeBizNode(oldBizName, agentIdHash, state);
    }
    if (!current) {
        return;
    }
    oldBizName = bizName;
    state->rebuildInfoSnapshot();
    auto info =
        std::atomic_load_explicit(&state->_infoSnapshot, std::memory_order_acquire);
    if (info && info->online && !info->serverAddress.empty()) {
        _pendingNodes.erase(agentIdHash);
        addBizNode(bizName, agentIdHash, state);
    } else {
        _pendingNodes[agentIdHash] = state;
    }
}

void MasterStateR::refreshBizNode(const NodeStatePtr &state)
{
    if (!state) {
        return;
    }
    auto info =
        std::atomic_load_explicit(&state->_infoSnapshot, std::memory_order_acquire);
    if (!info) {
        return;
    }
    autil::ScopedWriteLock nodeIndexLock(_nodeIndexLock);
    auto pendingIt = _pendingNodes.find(info->agentIdHash);
    bool current = _pendingNodes.end() != pendingIt && pendingIt->second == state;
    auto bizIt = _readyNodesByBiz.find(info->bizName);
    if (_readyNodesByBiz.end() != bizIt) {
        auto nodeIt = bizIt->second->find(info->agentIdHash);
        current = current || (bizIt->second->end() != nodeIt && nodeIt->second == state);
    }
    if (!current) {
        return;
    }
    if (!info->bizName.empty() && info->online && !info->serverAddress.empty()) {
        _pendingNodes.erase(info->agentIdHash);
        addBizNode(info->bizName, info->agentIdHash, state);
    } else {
        removeBizNode(info->bizName, info->agentIdHash, state);
        _pendingNodes[info->agentIdHash] = state;
    }
}

void MasterStateR::addBizNode(const std::string &bizName, uint64_t agentIdHash,
                              const NodeStatePtr &state)
{
    auto bizIt = _readyNodesByBiz.find(bizName);
    if (_readyNodesByBiz.end() != bizIt) {
        auto nodeIt = bizIt->second->find(agentIdHash);
        if (bizIt->second->end() != nodeIt && nodeIt->second == state) {
            return;
        }
    }
    auto bizNodes = _readyNodesByBiz.end() == bizIt
                        ? std::make_shared<NodeStateMap>()
                        : std::make_shared<NodeStateMap>(*bizIt->second);
    (*bizNodes)[agentIdHash] = state;
    _readyNodesByBiz[bizName] = std::move(bizNodes);
}

void MasterStateR::removeBizNode(const std::string &bizName, uint64_t agentIdHash,
                                 const NodeStatePtr &state)
{
    auto bizIt = _readyNodesByBiz.find(bizName);
    if (_readyNodesByBiz.end() == bizIt) {
        return;
    }
    auto nodeIt = bizIt->second->find(agentIdHash);
    if (bizIt->second->end() == nodeIt || nodeIt->second != state) {
        return;
    }
    auto bizNodes = std::make_shared<NodeStateMap>(*bizIt->second);
    bizNodes->erase(agentIdHash);
    if (bizNodes->empty()) {
        _readyNodesByBiz.erase(bizIt);
    } else {
        bizIt->second = std::move(bizNodes);
    }
}

std::vector<MasterStateR::NodeStateConstPtr> MasterStateR::getNodeStates() const
{
    std::vector<NodeStateConstPtr> states;
    snapshotNodeStates(states, nullptr);
    return states;
}

void MasterStateR::snapshotNodeStates(
    std::vector<NodeStateConstPtr> &states,
    std::unordered_set<const NodeState *> *readyStates) const
{
    states.clear();
    if (readyStates) {
        readyStates->clear();
    }
    autil::ScopedReadLock lock(_nodeIndexLock);
    size_t stateCount = _pendingNodes.size();
    for (const auto &bizNodes : _readyNodesByBiz) {
        stateCount += bizNodes.second->size();
    }
    states.reserve(stateCount);
    if (readyStates) {
        readyStates->reserve(stateCount - _pendingNodes.size());
    }
    for (const auto &pair : _pendingNodes) {
        states.emplace_back(pair.second);
    }
    for (const auto &bizNodes : _readyNodesByBiz) {
        for (const auto &node : *bizNodes.second) {
            states.emplace_back(node.second);
            if (readyStates) {
                readyStates->emplace(node.second.get());
            }
        }
    }
}

navi::SummaryErrorCode MasterStateR::getSummary(const std::string &query,
                                                navi::SummaryBuilder &builder) const
{
    auto level = SummaryUtil::getSummaryLevel(query);
    std::vector<NodeStateConstPtr> states;
    std::unordered_set<const NodeState *> readyStates;
    snapshotNodeStates(states, &readyStates);
    std::map<std::string, NaviSchedNodeInfo> nodes;
    std::map<std::string, size_t> cachedBlockCounts;
    std::map<std::string, CacheSyncStats> cacheSyncStats;
    std::map<std::string, ReservationStats> reservationStats;
    std::map<std::string, bool> nodeReady;
    std::map<std::string, double> executionUtilizations;
    std::map<std::string, double> executionBusyUtilizations;
    for (const auto &state : states) {
        auto node = state->getInfo();
        cachedBlockCounts[node.agentId] = state->getCachedBlockCount();
        cacheSyncStats[node.agentId] = state->getCacheSyncStats();
        reservationStats[node.agentId] = state->getReservationStats();
        nodeReady[node.agentId] = readyStates.end() != readyStates.find(state.get());
        executionUtilizations[node.agentId] = state->getExecutionUtilization();
        executionBusyUtilizations[node.agentId] = state->getExecutionBusyUtilization();
        nodes[node.agentId] = std::move(node);
    }
    auto collectors = getAgentCollectors();
    auto overview = builder.newTable("NaviSchedMasterOverview");
    TABLE_ADD_COLUMN_VALUE(overview, "node_count", nodes.size());
    TABLE_ADD_COLUMN_VALUE(overview, "ready_node_count", readyStates.size());
    TABLE_ADD_COLUMN_VALUE(overview, "pending_node_count",
                           states.size() - readyStates.size());
    TABLE_ADD_COLUMN_VALUE(overview, "collector_count", collectors.size());
    TABLE_ADD_COLUMN_VALUE(overview, "client_request_count", getClientRequestCount());
    TABLE_ADD_COLUMN_VALUE(overview, "agent_report_count", getAgentReportCount());
    if (level < 1) {
        return navi::SEC_OK;
    }
    auto nodeTable = builder.newTable("NaviSchedMasterNodes");
    for (const auto &pair : nodes) {
        const auto &node = pair.second;
        TABLE_ADD_COLUMN_VALUE(nodeTable, "agent_id_hash", node.agentIdHash);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "agent_id", node.agentId);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "biz_name", node.bizName);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "collector_address", node.collectorAddress);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "server_ip", node.serverInfo.ip);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "server_port", node.serverInfo.port);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "server_address", node.serverAddress);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "scheduling_state",
                               nodeReady.at(pair.first) ? "ready" : "pending");
        TABLE_ADD_COLUMN_VALUE(nodeTable, "online", node.online);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "peer_cache_enabled", node.peerCacheEnabled);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "peer_cache_read_available",
                               node.peerCacheReadAvailable);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "peer_cache_read_revision",
                               node.peerCacheReadRevision);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "peer_cache_read_agent_start_time_us",
                               node.peerCacheReadAgentStartTimeUs);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "total_blocks", node.totalBlocks);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "inflight_requests", node.inflightRequests);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "batch_waiting_requests",
                               node.batchWaitingRequests);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "queued_batch_requests",
                               node.queuedBatchRequests);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "running_requests", node.runningRequests);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "execution_busy_utilization",
                               executionBusyUtilizations.at(node.agentId));
        TABLE_ADD_COLUMN_VALUE(nodeTable, "execution_utilization",
                               executionUtilizations.at(node.agentId));
        TABLE_ADD_COLUMN_VALUE(nodeTable, "batch_timeout_count", node.batchTimeoutCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "abnormal_batch_finish_count",
                               node.abnormalBatchFinishCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "last_request_latency_ms",
                               node.lastRequestLatencyMs);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "queue_total_predicted_latency_ms",
                               node.queueTotalPredictedLatencyMs);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "last_event_type", node.lastEventType);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "last_event_time_us", node.lastEventTimeUs);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "last_request_event_type",
                               node.lastRequestEventType);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "last_request_event_time_us",
                               node.lastRequestEventTimeUs);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "last_receive_time_us", node.lastReceiveTimeUs);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "cached_block_count",
                               cachedBlockCounts[node.agentId]);
        const auto &cacheStats = cacheSyncStats.at(node.agentId);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "agent_cache_revision",
                               cacheStats.agentRevision);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "agent_cached_block_count",
                               cacheStats.agentCachedBlockCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "agent_cache_hash_checksum",
                               cacheStats.agentHashChecksum);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "master_cache_hash_checksum",
                               cacheStats.masterHashChecksum);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "agent_full_ready_block_count",
                               cacheStats.agentFullReadyBlockCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "master_full_ready_block_count",
                               cacheStats.masterFullReadyBlockCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "agent_tail_ready_block_count",
                               cacheStats.agentTailReadyBlockCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "master_tail_ready_block_count",
                               cacheStats.masterTailReadyBlockCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "rtp_cached_block_count",
                               cacheStats.rtpCachedBlockCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "cache_revision_gap_count",
                               cacheStats.revisionGapCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "cache_state_mismatch_count",
                               cacheStats.stateMismatchCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "rtp_cache_mirror_sample_count",
                               cacheStats.rtpMirrorSampleCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "rtp_cache_mirror_mismatch_count",
                               cacheStats.rtpMirrorMismatchCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_batch_count",
                               cacheStats.pendingCacheBatchCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_block_count",
                               cacheStats.pendingCacheBlockCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_tail_ready_block_count",
                               cacheStats.pendingCacheTailReadyBlockCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_scheduled_batch_count",
                               cacheStats.pendingCacheScheduledBatchCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_agent_batch_count",
                               cacheStats.pendingCacheAgentBatchCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_agent_revision",
                               cacheStats.pendingCacheAgentRevision);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_agent_checksum",
                               cacheStats.pendingCacheAgentChecksum);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_revision_gap_count",
                               cacheStats.pendingCacheRevisionGapCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_state_mismatch_count",
                               cacheStats.pendingCacheStateMismatchCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_resync_trigger_count",
                               cacheStats.pendingCacheResyncTriggerCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "pending_cache_snapshot_ready",
                               cacheStats.pendingCacheSnapshotReady);
        const auto &reservation = reservationStats.at(node.agentId);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "reservation_batch_count",
                               reservation.batchCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "reservation_request_count",
                               reservation.requestCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "reservation_waiting_request_count",
                               reservation.waitingRequestCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "reservation_dispatched_request_count",
                               reservation.dispatchedRequestCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "dispatch_waiter_count",
                               reservation.dispatchWaiterCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "live_dispatch_waiter_count",
                               reservation.liveDispatchWaiterCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "cache_policy_ready",
                               cacheStats.cachePolicyReady);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "cache_block_size", cacheStats.cacheBlockSize);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "cache_linear_step",
                               cacheStats.cacheLinearStep);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "full_attention_query_drop_tokens",
                               cacheStats.fullAttentionQueryDropTokens);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "hybrid_tail_required",
                               cacheStats.hybridTailRequired);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "cache_policy_mismatch_count",
                               cacheStats.cachePolicyMismatchCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "latency_formula_id", node.latencyFormula);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "latency_formula_name",
                               getNaviSchedLatencyFormulaName(node.latencyFormula));
        TABLE_ADD_COLUMN_VALUE(nodeTable, "latency_binary_version",
                               node.latencyBinaryVersion);
        TABLE_ADD_COLUMN_VALUE(
            nodeTable, "execution_geometry_enabled",
            NonLinearPrefillModelR::usesGeometryForVersion(node.latencyBinaryVersion) &&
                node.executionGeometryEnabled);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "latency_model_revision",
                               node.latencyModelRevision);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "latency_parameter_count",
                               node.latencyParameters.size());
        TABLE_ADD_COLUMN_VALUE(nodeTable, "latency_model_fingerprint",
                               node.latencyModelFingerprint);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "latency_model_trusted",
                               node.latencyModelTrusted);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "latency_model_warm_started",
                               node.latencyModelWarmStarted);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "latency_model_consecutive_rejected_count",
                               node.latencyModelConsecutiveRejectedCount);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "latency_model_last_reject_reason",
                               node.latencyModelLastRejectReason);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "scheduler_config_generation",
                               node.schedulerConfigGeneration);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "scheduler_config_hash",
                               node.schedulerConfigHash);
        TABLE_ADD_COLUMN_VALUE(nodeTable, "scheduler_config_json",
                               node.schedulerConfigJson);
    }
    auto collectorTable = builder.newTable("NaviSchedMasterCollectors");
    for (const auto &pair : collectors) {
        TABLE_ADD_COLUMN_VALUE(collectorTable, "node_id", pair.first);
        TABLE_ADD_COLUMN_VALUE(collectorTable, "collector", pair.second);
    }
    if (level < 2) {
        return navi::SEC_OK;
    }
    auto parameterTable = builder.newTable("NaviSchedMasterLatencyParameters");
    for (const auto &pair : nodes) {
        const auto &parameters = pair.second.latencyParameters;
        auto count =
            level >= 3 ? parameters.size() : std::min<size_t>(parameters.size(), 64);
        for (size_t index = 0; index < count; ++index) {
            TABLE_ADD_COLUMN_VALUE(parameterTable, "node_id", pair.first);
            TABLE_ADD_COLUMN_VALUE(parameterTable, "index", index);
            TABLE_ADD_COLUMN_VALUE(parameterTable, "parameter", parameters[index]);
        }
    }
    return navi::SEC_OK;
}

REGISTER_RESOURCE(MasterStateR);

}
