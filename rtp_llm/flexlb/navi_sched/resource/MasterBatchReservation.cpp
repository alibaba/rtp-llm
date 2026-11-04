#include "aios/navi_sched/resource/MasterBatchReservation.h"

#include "aios/navi_sched/resource/PendingCacheUtil.h"
#include "absl/container/flat_hash_map.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include "autil/legacy/jsonizable.h"

namespace navi_sched {

autil::legacy::json::JsonMap MasterBatchReservation::makeReplayState() const
{
    using autil::legacy::ToJson;
    autil::legacy::json::JsonMap state;
    state["batch_id"] = ToJson(_batchId);
    state["schedule_time_us"] = ToJson(_scheduleTimeUs);
    state["predicted_latency_ms"] = ToJson(_predictedLatencyMs);
    state["cache_tracked"] = ToJson(_cacheTracked);
    state["block_size"] = ToJson(_cachePolicy.blockSize);
    state["linear_step"] = ToJson(_cachePolicy.linearStep);
    state["hybrid_tail_required"] = ToJson(_cachePolicy.hybridTailRequired);
    state["prefix_tree_built"] = ToJson(hasPendingCacheReferences());
    state["pending_rebuild_count"] = ToJson(_pendingRebuildCount);
    state["pending_rebuild_visited_nodes"] = ToJson(_pendingRebuildVisitedNodes);
    std::vector<autil::legacy::json::JsonMap> requests;
    for (const auto &entry : _requests) {
        autil::legacy::json::JsonMap request;
        request["request_id"] = ToJson(entry.request.requestId);
        request["total_token_count"] = ToJson(entry.request.totalTokenCount);
        request["block_size"] = ToJson(entry.request.blockSize);
        request["cache_hit_token_count"] = ToJson(entry.request.predictedCacheHitTokenCount);
        request["predicted_latency_ms"] = ToJson(entry.request.predictedLatencyMs);
        request["has_block_hashes"] = ToJson(bool(entry.request.blockHashes));
        request["block_hashes"] = ToJson(entry.request.blockHashes
                                              ? *entry.request.blockHashes
                                              : NaviSchedBlockHashes());
        request["state"] = ToJson(static_cast<uint32_t>(entry.state));
        request["dispatch_acknowledged"] = ToJson(entry.dispatchAcknowledged);
        request["awaiting_batch_ready"] = ToJson(entry.awaitingBatchReady);
        request["unreported"] = ToJson(entry.unreportedInflight);
        request["unreported_latency_ms"] =
            ToJson(entry.unreportedInflight ? entry.unreportedLatencyMs : 0.0);
        requests.emplace_back(std::move(request));
    }
    state["requests"] = ToJson(requests);
    return state;
}

bool MasterBatchReservation::restoreReplayState(
    const autil::legacy::json::JsonMap &state)
{
    using autil::legacy::FromJson;
    try {
        MasterBatchReservation candidate;
        int64_t batchId = 0, scheduleTimeUs = 0;
        double latency = 0;
        bool tracked = false, treeBuilt = false;
        NaviSchedCachePolicy policy;
        FromJson(batchId, state.at("batch_id"));
        FromJson(scheduleTimeUs, state.at("schedule_time_us"));
        FromJson(latency, state.at("predicted_latency_ms"));
        FromJson(tracked, state.at("cache_tracked"));
        FromJson(treeBuilt, state.at("prefix_tree_built"));
        FromJson(policy.blockSize, state.at("block_size"));
        FromJson(policy.linearStep, state.at("linear_step"));
        FromJson(policy.hybridTailRequired, state.at("hybrid_tail_required"));
        std::vector<autil::legacy::json::JsonMap> records;
        FromJson(records, state.at("requests"));
        std::vector<Request> requests;
        for (const auto &record : records) {
            Request request;
            bool hasHashes = false;
            FromJson(request.requestId, record.at("request_id"));
            FromJson(request.totalTokenCount, record.at("total_token_count"));
            FromJson(request.blockSize, record.at("block_size"));
            FromJson(request.predictedCacheHitTokenCount, record.at("cache_hit_token_count"));
            FromJson(request.predictedLatencyMs, record.at("predicted_latency_ms"));
            FromJson(hasHashes, record.at("has_block_hashes"));
            if (hasHashes) {
                auto hashes = std::make_shared<NaviSchedBlockHashes>();
                FromJson(*hashes, record.at("block_hashes"));
                request.blockHashes = std::move(hashes);
            }
            requests.emplace_back(std::move(request));
        }
        if (!candidate.init(batchId, scheduleTimeUs, latency, std::move(requests),
                            tracked ? &policy : nullptr) ||
            (treeBuilt && !candidate.buildPendingCacheReferences())) {
            return false;
        }
        candidate._waitingRequestCount = 0;
        candidate.clearUnreportedStats();
        bool removed = false;
        Transition restoredTransition;
        for (size_t i = 0; i < records.size(); ++i) {
            const auto &record = records[i];
            auto &entry = candidate._requests.writable(i);
            uint32_t requestState = 0;
            bool unreported = false;
            double unreportedLatency = 0;
            FromJson(requestState, record.at("state"));
            if (requestState > static_cast<uint32_t>(RequestState::COMPLETED)) return false;
            FromJson(entry.dispatchAcknowledged, record.at("dispatch_acknowledged"));
            const auto awaiting = record.find("awaiting_batch_ready");
            if (awaiting != record.end()) FromJson(entry.awaitingBatchReady, awaiting->second);
            FromJson(unreported, record.at("unreported"));
            FromJson(unreportedLatency, record.at("unreported_latency_ms"));
            entry.unreportedInflight = false;
            const auto restoredState = static_cast<RequestState>(requestState);
            if (restoredState == RequestState::FAILED || restoredState == RequestState::COMPLETED) {
                if (!candidate.removeCacheRequestReferences(i, restoredTransition)) return false;
                removed = true;
            }
            entry.state = restoredState;
            switch (entry.state) {
            case RequestState::WAITING:
                ++candidate._waitingRequestCount;
                candidate._acknowledgedWaitingRequestCount += entry.dispatchAcknowledged;
                break;
            case RequestState::DISPATCHED: ++candidate._dispatchedRequestCount; break;
            case RequestState::FAILED: ++candidate._failedRequestCount; break;
            case RequestState::COMPLETED: ++candidate._completedRequestCount; break;
            }
            if (unreported && !candidate.addUnreportedRequest(i, unreportedLatency)) return false;
        }
        if ((removed && !candidate.rebuildPendingCacheReservation()) ||
            !candidate.rebuildWaitingFeatures(nullptr)) return false;
        if (candidate.isResolved() && !treeBuilt) {
            candidate._pendingBlockHashes.reset();
            candidate._pendingReferences.clear();
            candidate._requestBlockIndexes.reset();
        }
        FromJson(candidate._pendingRebuildCount, state.at("pending_rebuild_count"));
        FromJson(candidate._pendingRebuildVisitedNodes, state.at("pending_rebuild_visited_nodes"));
        *this = std::move(candidate);
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

bool MasterBatchReservation::init(int64_t batchId, int64_t scheduleTimeUs,
                                  double predictedLatencyMs,
                                  std::vector<Request> requests,
                                  const NaviSchedCachePolicy *cachePolicy)
{
    if (_batchId != 0 || batchId <= 0 || scheduleTimeUs <= 0 ||
        !std::isfinite(predictedLatencyMs) || predictedLatencyMs < 0.0 ||
        requests.empty()) {
        return false;
    }
    if (cachePolicy && !cachePolicy->valid()) {
        return false;
    }
    auto requestIndexes = std::make_shared<std::unordered_map<uint64_t, size_t>>();
    requestIndexes->reserve(requests.size());
    std::vector<PendingCacheReferencedRequest> referencedRequests;
    referencedRequests.reserve(requests.size());
    bool cacheTrackable = cachePolicy != nullptr;
    for (const auto &request : requests) {
        if (request.requestId == 0 || request.totalTokenCount < 0 ||
            request.blockSize <= 0 || request.predictedCacheHitTokenCount < 0 ||
            request.predictedCacheHitTokenCount > request.totalTokenCount ||
            !std::isfinite(request.predictedLatencyMs) ||
            request.predictedLatencyMs < 0.0 ||
            !requestIndexes->emplace(request.requestId, requestIndexes->size()).second) {
            return false;
        }
        if (cachePolicy) {
            if (!request.blockHashes || request.blockHashes->empty()) {
                if (request.predictedCacheHitTokenCount != 0) {
                    return false;
                }
                if (request.totalTokenCount / request.blockSize != 0) {
                    cacheTrackable = false;
                } else if (request.blockHashes) {
                    referencedRequests.push_back({request.blockHashes.get(),
                                                  request.totalTokenCount,
                                                  request.blockSize});
                }
                continue;
            }
            if (request.blockSize != cachePolicy->blockSize ||
                request.blockHashes->size() !=
                    static_cast<size_t>(request.totalTokenCount / request.blockSize)) {
                return false;
            }
            referencedRequests.push_back(
                {request.blockHashes.get(), request.totalTokenCount, request.blockSize});
        }
    }

    MasterBatchReservation candidate;
    NaviSchedPendingCacheReservation pendingCacheReservation;
    if (cacheTrackable &&
        PendingCacheUtil::makeReservation(batchId, referencedRequests, *cachePolicy,
                                          pendingCacheReservation)) {
        candidate._cachePolicy = *cachePolicy;
        candidate._cacheTracked = true;
        candidate._pendingCacheReservation =
            std::make_shared<const NaviSchedPendingCacheReservation>(std::move(pendingCacheReservation));
    }

    candidate._batchId = batchId;
    candidate._scheduleTimeUs = scheduleTimeUs;
    candidate._predictedLatencyMs = predictedLatencyMs;
    candidate._waitingRequestCount = requests.size();
    candidate._requestIndexes = std::move(requestIndexes);
    candidate._requests.reserve(requests.size());
    for (auto &request : requests) {
        const auto latencyMs = request.predictedLatencyMs;
        candidate._requests.push_back(
            {std::move(request), RequestState::WAITING, false, INVALID_BLOCK_OFFSET});
        if (!candidate.addUnreportedRequest(candidate._requests.size() - 1, latencyMs)) {
            return false;
        }
    }
    if (!candidate.rebuildWaitingFeatures(nullptr)) {
        return false;
    }
    *this = std::move(candidate);
    return true;
}

bool MasterBatchReservation::markFailed(uint64_t requestId, Transition &transition)
{
    resetTransition(transition);
    const auto requestIndex = findRequest(requestId);
    if (requestIndex == _requests.size()) {
        return false;
    }
    auto &request = _requests.writable(requestIndex);
    if (request.state == RequestState::FAILED ||
        request.state == RequestState::COMPLETED) {
        return true;
    }
    const auto previousState = request.state;
    if (request.unreportedInflight && !removeUnreportedRequest(requestIndex)) {
        return false;
    }
    if (!removeCacheRequestReferences(requestIndex, transition)) {
        return false;
    }
    request.state = RequestState::FAILED;
    request.awaitingBatchReady = false;
    if (previousState == RequestState::WAITING) {
        if (!subtractWaitingRequest(request.request)) {
            return false;
        }
        if (request.dispatchAcknowledged) {
            request.dispatchAcknowledged = false;
            --_acknowledgedWaitingRequestCount;
        }
        --_waitingRequestCount;
    } else {
        --_dispatchedRequestCount;
    }
    ++_failedRequestCount;
    transition.changed = true;
    transition.resolved = isResolved();
    return true;
}

bool MasterBatchReservation::markFailed(
    const std::vector<uint64_t> &requestIds, Transition &transition)
{
    resetTransition(transition);
    if (requestIds.empty()) {
        return false;
    }
    std::vector<size_t> indexes;
    indexes.reserve(requestIds.size());
    for (auto requestId : requestIds) {
        const auto index = findRequest(requestId);
        if (index == _requests.size()) {
            return false;
        }
        indexes.push_back(index);
    }
    return markFailedIndexes(indexes, transition);
}

bool MasterBatchReservation::markFailedIndexes(
    const std::vector<size_t> &indexes, Transition &transition)
{
    resetTransition(transition);
    if (indexes.empty() || std::any_of(indexes.begin(), indexes.end(),
            [this](size_t index) { return index >= _requests.size(); })) {
        return false;
    }
    const auto activeCount = _waitingRequestCount + _dispatchedRequestCount;
    bool fullFailure = activeCount > 0 && indexes.size() >= activeCount;
    if (fullFailure) {
        size_t position = 0;
        const bool canonical = indexes.size() == _requests.size() &&
            std::all_of(indexes.begin(), indexes.end(),
                        [&position](size_t index) { return index == position++; });
        if (!canonical) {
            std::vector<bool> selected(_requests.size(), false);
            size_t selectedActiveCount = 0;
            for (auto index : indexes) {
                if (selected[index]) continue;
                selected[index] = true;
                const auto state = _requests[index].state;
                selectedActiveCount += state == RequestState::WAITING ||
                                       state == RequestState::DISPATCHED;
            }
            fullFailure = selectedActiveCount == activeCount;
        }
    }
    if (fullFailure) {
        for (size_t index = 0; index < _requests.size(); ++index) {
            const auto state = _requests[index].state;
            if (state != RequestState::WAITING && state != RequestState::DISPATCHED) continue;
            auto &entry = _requests.writable(index);
            if (state == RequestState::WAITING) entry.dispatchAcknowledged = false;
            entry.unreportedInflight = false;
            entry.state = RequestState::FAILED;
            entry.awaitingBatchReady = false;
        }
        _failedRequestCount += activeCount;
        _waitingRequestCount = 0;
        _dispatchedRequestCount = 0;
        _acknowledgedWaitingRequestCount = 0;
        _waitingFeatures = NaviSchedLatencyFeatures();
        clearUnreportedStats();
        _pendingCacheReservation.reset();
        _pendingCacheDirty = false;
        _pendingBlockHashes.reset();
        _pendingReferences.clear();
        _requestBlockIndexes.reset();
        transition.changed = true;
        transition.fullBatchFastPath = true;
        transition.cacheChanged = _cacheTracked;
        transition.retireCache = _cacheTracked;
        transition.resolved = true;
        return true;
    }
    for (auto index : indexes) {
        auto &entry = _requests.writable(index);
        if (entry.state == RequestState::FAILED || entry.state == RequestState::COMPLETED) {
            continue;
        }
        if ((entry.unreportedInflight && !removeUnreportedRequest(index)) ||
            !removeCacheRequestReferences(index, transition)) {
            return false;
        }
        if (entry.state == RequestState::WAITING) {
            if (!subtractWaitingRequest(entry.request)) {
                return false;
            }
            --_waitingRequestCount;
            if (entry.dispatchAcknowledged) {
                entry.dispatchAcknowledged = false;
                --_acknowledgedWaitingRequestCount;
            }
        } else {
            --_dispatchedRequestCount;
        }
        entry.state = RequestState::FAILED;
        entry.awaitingBatchReady = false;
        ++_failedRequestCount;
        transition.changed = true;
    }
    transition.resolved = isResolved();
    return true;
}

bool MasterBatchReservation::markAdmitted(const NaviSchedLocalBatchInfo &batch,
                                          Transition &transition)
{
    resetTransition(transition);
    if (!validLocalBatch(batch)) {
        return false;
    }
    std::vector<size_t> requestIndexes;
    if (!collectRequestIndexes(batch, requestIndexes)) {
        return false;
    }
    transition.fullBatchFastPath = batch.requestIds.empty();
    transition.dispatchedRequestIds.reserve(requestIndexes.size());
    for (auto requestIndex : requestIndexes) {
        auto &request = _requests.writable(requestIndex);
        if (request.state == RequestState::DISPATCHED && request.awaitingBatchReady) {
            request.awaitingBatchReady = false;
            transition.dispatchedRequestIds.emplace_back(request.request.requestId);
            continue;
        }
        if (request.state != RequestState::WAITING) {
            continue;
        }
        if (!subtractWaitingRequest(request.request)) {
            return false;
        }
        if (request.dispatchAcknowledged) {
            request.dispatchAcknowledged = false;
            --_acknowledgedWaitingRequestCount;
        }
        request.state = RequestState::DISPATCHED;
        --_waitingRequestCount;
        ++_dispatchedRequestCount;
        transition.dispatchedRequestIds.emplace_back(request.request.requestId);
    }
    if (transition.dispatchedRequestIds.empty()) {
        transition.resolved = isResolved();
        return true;
    }
    transition.changed = true;
    transition.resolved = isResolved();
    return true;
}

bool MasterBatchReservation::markCompleted(const NaviSchedLocalBatchInfo &batch,
                                           bool successful, Transition &transition)
{
    resetTransition(transition);
    if (!validLocalBatch(batch)) {
        return false;
    }
    std::vector<size_t> requestIndexes;
    if (!collectRequestIndexes(batch, requestIndexes)) {
        return false;
    }
    for (auto requestIndex : requestIndexes) {
        if (_requests[requestIndex].state == RequestState::WAITING) {
            return false;
        }
    }

    const bool fullFastPath = batch.requestIds.empty() &&
                              requestIndexes.size() == _requests.size() &&
                              _failedRequestCount == 0;
    if (fullFastPath) {
        const auto completedNow = _dispatchedRequestCount;
        for (size_t index = 0; index < _requests.size(); ++index) {
            auto &request = _requests.writable(index);
            request.unreportedInflight = false;
            if (request.state == RequestState::DISPATCHED) {
                request.state =
                    successful ? RequestState::COMPLETED : RequestState::FAILED;
            }
        }
        if (successful) {
            _completedRequestCount += completedNow;
        } else {
            _failedRequestCount += completedNow;
        }
        _dispatchedRequestCount = 0;
        _pendingCacheReservation.reset();
        _pendingCacheDirty = false;
        clearUnreportedStats();
        _pendingBlockHashes.reset();
        _pendingReferences.clear();
        _requestBlockIndexes.reset();
        transition.changed = completedNow > 0;
        transition.fullBatchFastPath = true;
        transition.cacheChanged = transition.changed && _cacheTracked;
        transition.retireCache = transition.cacheChanged;
        transition.resolved = true;
        return true;
    }

    bool changed = false;
    for (auto requestIndex : requestIndexes) {
        auto &request = _requests.writable(requestIndex);
        if (request.state != RequestState::DISPATCHED) {
            continue;
        }
        if (request.unreportedInflight && !removeUnreportedRequest(requestIndex)) {
            return false;
        }
        if (!removeCacheRequestReferences(requestIndex, transition)) {
            return false;
        }
        request.state = successful ? RequestState::COMPLETED : RequestState::FAILED;
        request.awaitingBatchReady = false;
        --_dispatchedRequestCount;
        if (successful) {
            ++_completedRequestCount;
        } else {
            ++_failedRequestCount;
        }
        changed = true;
    }
    transition.changed = changed;
    transition.resolved = isResolved();
    return true;
}

bool MasterBatchReservation::acknowledgeDispatch(
    const NaviSchedLocalBatchInfo &batch, std::vector<uint64_t> &acknowledgedRequestIds)
{
    acknowledgedRequestIds.clear();
    if (!validLocalBatch(batch)) {
        return false;
    }
    std::vector<size_t> requestIndexes;
    if (!collectRequestIndexes(batch, requestIndexes)) {
        return false;
    }
    acknowledgedRequestIds.reserve(requestIndexes.size());
    for (auto requestIndex : requestIndexes) {
        auto &request = _requests.writable(requestIndex);
        if (request.state == RequestState::DISPATCHED && request.awaitingBatchReady) {
            request.awaitingBatchReady = false;
            acknowledgedRequestIds.emplace_back(request.request.requestId);
            continue;
        }
        if (request.state == RequestState::WAITING && !request.dispatchAcknowledged) {
            request.dispatchAcknowledged = true;
            ++_acknowledgedWaitingRequestCount;
            acknowledgedRequestIds.emplace_back(request.request.requestId);
        }
    }
    return true;
}

bool MasterBatchReservation::markAcceptedWorkloadReported(uint64_t requestId)
{
    const auto index = findRequest(requestId);
    if (index == _requests.size()) return false;
    if (_requests[index].state != RequestState::WAITING) return true;
    auto &request = _requests.writable(index);
    if (!subtractWaitingRequest(request.request)) return false;
    if (request.dispatchAcknowledged) {
        request.dispatchAcknowledged = false;
        --_acknowledgedWaitingRequestCount;
    }
    request.state = RequestState::DISPATCHED;
    request.awaitingBatchReady = true;
    --_waitingRequestCount;
    ++_dispatchedRequestCount;
    return true;
}

bool MasterBatchReservation::isAwaitingBatchReady(uint64_t requestId) const
{
    const auto index = findRequest(requestId);
    return index != _requests.size() && _requests[index].state == RequestState::DISPATCHED &&
           _requests[index].awaitingBatchReady;
}

bool MasterBatchReservation::markInflightReported(uint64_t requestId, bool &changed)
{
    changed = false;
    const auto requestIndex = findRequest(requestId);
    if (requestIndex == _requests.size()) {
        return false;
    }
    const auto state = _requests[requestIndex].state;
    if (state == RequestState::FAILED || state == RequestState::COMPLETED ||
        !_requests[requestIndex].unreportedInflight) {
        return true;
    }
    changed = removeUnreportedRequest(requestIndex);
    return changed;
}

bool MasterBatchReservation::markInflightReported(
    const NaviSchedLocalBatchInfo &batch, std::vector<uint64_t> &changedRequestIds)
{
    changedRequestIds.clear();
    if (!validLocalBatch(batch)) {
        return false;
    }
    std::vector<size_t> requestIndexes;
    if (!collectRequestIndexes(batch, requestIndexes)) {
        return false;
    }
    changedRequestIds.reserve(requestIndexes.size());
    for (auto requestIndex : requestIndexes) {
        const auto requestId = _requests[requestIndex].request.requestId;
        if (!_requests[requestIndex].unreportedInflight) {
            continue;
        }
        if (!removeUnreportedRequest(requestIndex)) {
            return false;
        }
        changedRequestIds.emplace_back(requestId);
    }
    return true;
}

bool MasterBatchReservation::recomputeWaitingFeatures(const CacheMatch &cacheMatch)
{
    if (!cacheMatch) {
        return false;
    }
    return rebuildWaitingFeatures(&cacheMatch);
}

bool MasterBatchReservation::appendWaitingCapacityFeatures(
    NaviSchedLatencyFeatures &features, int32_t maxRunningRequestCount,
    int32_t &runningCapacitySlotCount) const
{
    if (maxRunningRequestCount < 0 || runningCapacitySlotCount < 0 ||
        runningCapacitySlotCount > maxRunningRequestCount) {
        return false;
    }
    if (_waitingRequestCount == 0) {
        return true;
    }
    NaviSchedLatencyFeatures running;
    NaviSchedLatencyFeatures waiting;
    auto updatedRunningCapacitySlotCount = runningCapacitySlotCount;
    const bool reuseWaitingFeatures =
        runningCapacitySlotCount == maxRunningRequestCount &&
        _waitingRequestCount == _requests.size();
    if (!reuseWaitingFeatures) {
        for (const auto &entry : _requests) {
            if (RequestState::WAITING != entry.state) {
                continue;
            }
            const auto hit = std::min(entry.request.totalTokenCount,
                                      entry.request.predictedCacheHitTokenCount);
            const auto miss = entry.request.totalTokenCount - hit;
            if (updatedRunningCapacitySlotCount < maxRunningRequestCount) {
                running.appendRequest(hit, miss);
                ++updatedRunningCapacitySlotCount;
            } else {
                waiting.appendRequest(hit, miss);
            }
        }
    }
    const auto &waitingFeatures = reuseWaitingFeatures ? _waitingFeatures : waiting;
    const auto requestCount = running.getBatchSize() + waitingFeatures.getBatchSize();
    if (0 == requestCount) {
        return true;
    }
    NaviSchedLatencyFeatures aggregate;
    if (!aggregate.appendCapacitySplitAggregate(
            1, running.getBatchSize(), running.getCacheHitTokenCount(),
            running.getCacheMissTokenCount(), running.getCacheMissTokenSquareSum(),
            running.getCacheHitMissTokenProductSum(), waitingFeatures.getBatchSize(),
            waitingFeatures.getCacheHitTokenCount(), waitingFeatures.getCacheMissTokenCount(),
            waitingFeatures.getCacheMissTokenSquareSum(),
            waitingFeatures.getCacheHitMissTokenProductSum()) ||
        !aggregate.appendPrecedingRequestCount(requestCount)) {
        return false;
    }
    auto updatedFeatures = features;
    if (!updatedFeatures.append(aggregate)) {
        return false;
    }
    features = std::move(updatedFeatures);
    runningCapacitySlotCount = updatedRunningCapacitySlotCount;
    return true;
}

bool MasterBatchReservation::setPredictedLatencyMs(double predictedLatencyMs)
{
    if (!std::isfinite(predictedLatencyMs) || predictedLatencyMs < 0.0 ||
        (_waitingRequestCount == 0 && predictedLatencyMs != 0.0)) {
        return false;
    }
    _predictedLatencyMs = predictedLatencyMs;
    return true;
}

bool MasterBatchReservation::sameDefinition(const MasterBatchReservation &other) const
{
    if (_batchId != other._batchId || _scheduleTimeUs != other._scheduleTimeUs ||
        _predictedLatencyMs != other._predictedLatencyMs ||
        _cacheTracked != other._cacheTracked ||
        _requests.size() != other._requests.size()) {
        return false;
    }
    if (_cacheTracked &&
        (_cachePolicy.blockSize != other._cachePolicy.blockSize ||
         _cachePolicy.linearStep != other._cachePolicy.linearStep ||
         _cachePolicy.hybridTailRequired != other._cachePolicy.hybridTailRequired)) {
        return false;
    }
    for (size_t index = 0; index < _requests.size(); ++index) {
        const auto &lhs = _requests[index].request;
        const auto &rhs = other._requests[index].request;
        if (lhs.requestId != rhs.requestId ||
            lhs.totalTokenCount != rhs.totalTokenCount ||
            lhs.blockSize != rhs.blockSize ||
            lhs.predictedCacheHitTokenCount != rhs.predictedCacheHitTokenCount ||
            lhs.predictedLatencyMs != rhs.predictedLatencyMs ||
            ((!lhs.blockHashes || lhs.blockHashes->empty()) !=
             (!rhs.blockHashes || rhs.blockHashes->empty())) ||
            (lhs.blockHashes && rhs.blockHashes &&
             *lhs.blockHashes != *rhs.blockHashes)) {
            return false;
        }
    }
    return true;
}

int64_t MasterBatchReservation::getBatchId() const { return _batchId; }

int64_t MasterBatchReservation::getScheduleTimeUs() const { return _scheduleTimeUs; }

double MasterBatchReservation::getPredictedLatencyMs() const
{
    return _predictedLatencyMs;
}

size_t MasterBatchReservation::getRequestCount() const { return _requests.size(); }

size_t MasterBatchReservation::getWaitingRequestCount() const
{
    return _waitingRequestCount;
}

size_t MasterBatchReservation::getDispatchedRequestCount() const
{
    return _dispatchedRequestCount;
}

size_t MasterBatchReservation::getFailedRequestCount() const
{
    return _failedRequestCount;
}

size_t MasterBatchReservation::getCompletedRequestCount() const
{
    return _completedRequestCount;
}

size_t MasterBatchReservation::getAcknowledgedWaitingRequestCount() const
{
    return _acknowledgedWaitingRequestCount;
}

bool MasterBatchReservation::isResolved() const
{
    return _waitingRequestCount == 0 && _dispatchedRequestCount == 0;
}

bool MasterBatchReservation::tracksPendingCache() const { return _cacheTracked; }

bool MasterBatchReservation::hasPendingCacheReferences() const { return !_pendingReferences.empty(); }

bool MasterBatchReservation::containsRequest(uint64_t requestId) const
{
    return findRequest(requestId) != _requests.size();
}

bool MasterBatchReservation::isDispatchAcknowledged(uint64_t requestId) const
{
    const auto requestIndex = findRequest(requestId);
    return requestIndex != _requests.size() &&
           _requests[requestIndex].state == RequestState::WAITING &&
           _requests[requestIndex].dispatchAcknowledged;
}

MasterBatchReservation::RequestState MasterBatchReservation::getRequestState(
    uint64_t requestId) const
{
    const auto requestIndex = findRequest(requestId);
    return requestIndex == _requests.size() ? RequestState::FAILED
                                            : _requests[requestIndex].state;
}

const NaviSchedLatencyFeatures &MasterBatchReservation::getWaitingFeatures() const
{
    return _waitingFeatures;
}

const NaviSchedPendingCacheReservation &
MasterBatchReservation::getPendingCacheReservation() const
{
    if (_pendingCacheDirty) {
        const_cast<MasterBatchReservation *>(this)->rebuildPendingCacheReservation();
    }
    static const NaviSchedPendingCacheReservation EMPTY;
    return _pendingCacheReservation ? *_pendingCacheReservation : EMPTY;
}

RequestLatencyTracker::Snapshot MasterBatchReservation::getUnreportedInflightStats() const
{
    auto stats = _unreportedStats;
    stats.totalPredictedLatencyMs = static_cast<double>(_unreportedLatencySumMs);
    return stats;
}

bool MasterBatchReservation::addUnreportedRequest(size_t requestIndex, double latencyMs)
{
    if (!std::isfinite(latencyMs) || latencyMs < 0.0 ||
        _unreportedStats.requestCount >= RequestLatencyTracker::MAX_REQUEST_COUNT) {
        return false;
    }
    const auto sum = _unreportedLatencySumMs + static_cast<long double>(latencyMs);
    if (!std::isfinite(sum) || sum > std::numeric_limits<double>::max()) {
        return false;
    }
    auto &entry = _requests.writable(requestIndex);
    if (entry.unreportedInflight) {
        return false;
    }
    entry.unreportedInflight = true;
    entry.unreportedLatencyMs = latencyMs;
    _unreportedLatencySumMs = sum;
    ++_unreportedStats.requestCount;
    ++_unreportedStats.revision;
    return true;
}

bool MasterBatchReservation::removeUnreportedRequest(size_t requestIndex)
{
    auto &entry = _requests.writable(requestIndex);
    if (!entry.unreportedInflight) {
        return false;
    }
    entry.unreportedInflight = false;
    _unreportedLatencySumMs -= static_cast<long double>(entry.unreportedLatencyMs);
    if (--_unreportedStats.requestCount == 0) {
        _unreportedLatencySumMs = 0.0;
    }
    ++_unreportedStats.revision;
    return true;
}

void MasterBatchReservation::clearUnreportedStats()
{
    if (_unreportedStats.requestCount != 0) {
        _unreportedStats.requestCount = 0;
        _unreportedLatencySumMs = 0.0;
        ++_unreportedStats.revision;
    }
}

bool MasterBatchReservation::validLocalBatch(const NaviSchedLocalBatchInfo &batch) const
{
    if (!batch.valid() || batch.scheduleBatchId != _batchId ||
        batch.requestCount > static_cast<int32_t>(_requests.size())) {
        return false;
    }
    return !batch.requestIds.empty() ||
           batch.requestCount == static_cast<int32_t>(_requests.size());
}

bool MasterBatchReservation::collectRequestIndexes(
    const NaviSchedLocalBatchInfo &batch, std::vector<size_t> &requestIndexes) const
{
    requestIndexes.clear();
    if (batch.requestIds.empty()) {
        requestIndexes.resize(_requests.size());
        for (size_t index = 0; index < _requests.size(); ++index) {
            requestIndexes[index] = index;
        }
        return true;
    }
    requestIndexes.reserve(batch.requestIds.size());
    for (auto requestId : batch.requestIds) {
        const auto requestIndex = findRequest(requestId);
        if (requestIndex == _requests.size()) {
            return false;
        }
        requestIndexes.emplace_back(requestIndex);
    }
    return true;
}

bool MasterBatchReservation::buildPendingCacheReferences()
{
    if (!_cacheTracked || !_pendingReferences.empty()) {
        return true;
    }
    size_t blockCapacity = 0;
    for (const auto &entry : _requests) {
        if (entry.state == RequestState::WAITING ||
            entry.state == RequestState::DISPATCHED) {
            if (!entry.request.blockHashes) {
                continue;
            }
            blockCapacity += entry.request.blockHashes->size();
        }
    }
    if (blockCapacity == 0) {
        return true;
    }
    auto blockHashes = std::make_shared<NaviSchedBlockHashes>();
    blockHashes->reserve(blockCapacity);
    auto requestBlockIndexes = std::make_shared<std::vector<uint32_t>>();
    requestBlockIndexes->reserve(blockCapacity);
    std::vector<BlockReferences> references;
    references.reserve(blockCapacity);
    absl::flat_hash_map<uint64_t, uint32_t> hashIndexes;
    hashIndexes.reserve(blockCapacity);
    for (size_t index = 0; index < _requests.size(); ++index) {
        auto &entry = _requests.writable(index);
        if (entry.state != RequestState::WAITING &&
            entry.state != RequestState::DISPATCHED) {
            continue;
        }
        entry.blockIndexBegin = requestBlockIndexes->size();
        if (!entry.request.blockHashes) {
            continue;
        }
        const auto &hashes = *entry.request.blockHashes;
        for (size_t blockIndex = 0; blockIndex < hashes.size(); ++blockIndex) {
            auto [it, inserted] =
                hashIndexes.emplace(hashes[blockIndex], static_cast<uint32_t>(references.size()));
            if (inserted) {
                if (references.size() >= INVALID_BLOCK_INDEX) {
                    return false;
                }
                blockHashes->push_back(hashes[blockIndex]);
                references.push_back({0, 0});
            }
            requestBlockIndexes->push_back(it->second);
            auto &node = references[it->second];
            ++node.fullRefCount;
            if (isTailReady(entry, blockIndex + 1)) {
                ++node.tailRefCount;
            }
        }
    }
    _pendingReferences.reserve(references.size());
    for (const auto &reference : references) {
        _pendingReferences.push_back(reference);
    }
    _pendingBlockHashes = std::move(blockHashes);
    _requestBlockIndexes = std::move(requestBlockIndexes);
    return true;
}

bool MasterBatchReservation::removeCacheRequestReferences(size_t requestIndex,
                                                          Transition &transition)
{
    if (!_cacheTracked) {
        return true;
    }
    if (requestIndex >= _requests.size() || !buildPendingCacheReferences()) {
        return false;
    }
    auto &entry = _requests.writable(requestIndex);
    const auto blockCount = entry.request.blockHashes ? entry.request.blockHashes->size() : 0;
    for (size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex) {
        const auto nodeIndex = (*_requestBlockIndexes)[entry.blockIndexBegin + blockIndex];
        auto &node = _pendingReferences.writable(nodeIndex);
        const auto hash = (*_pendingBlockHashes)[nodeIndex];
        if (node.fullRefCount == 0) {
            return false;
        }
        if (--node.fullRefCount == 0) {
            transition.removedFullReadyBlockHashes.push_back(hash);
            transition.cacheChanged = true;
        }
        if (isTailReady(entry, blockIndex + 1)) {
            if (node.tailRefCount == 0) {
                return false;
            }
            if (--node.tailRefCount == 0) {
                transition.removedTailReadyBlockHashes.push_back(hash);
                transition.cacheChanged = true;
            }
        }
    }
    entry.blockIndexBegin = INVALID_BLOCK_OFFSET;
    _pendingCacheDirty = _pendingCacheDirty || transition.cacheChanged;
    return true;
}

bool MasterBatchReservation::rebuildPendingCacheReservation()
{
    if (!_cacheTracked) {
        return true;
    }
    ++_pendingRebuildCount;
    _pendingRebuildVisitedNodes += _pendingReferences.size();
    NaviSchedPendingCacheReservation reservation;
    reservation.batchId = _batchId;
    reservation.scheduleBatchId = _batchId;
    const auto nodeCount = _pendingReferences.size();
    reservation.fullReadyBlockHashes.reserve(nodeCount);
    reservation.tailReadyBlockHashes.reserve(nodeCount);
    for (size_t index = 0; index < nodeCount; ++index) {
        const auto &node = _pendingReferences[index];
        if (node.fullRefCount == 0) {
            continue;
        }
        const auto hash = (*_pendingBlockHashes)[index];
        reservation.fullReadyBlockHashes.emplace_back(hash);
        if (node.tailRefCount > 0) {
            reservation.tailReadyBlockHashes.emplace_back(hash);
        }
    }
    std::sort(reservation.fullReadyBlockHashes.begin(), reservation.fullReadyBlockHashes.end());
    std::sort(reservation.tailReadyBlockHashes.begin(), reservation.tailReadyBlockHashes.end());
    _pendingCacheDirty = false;
    if (reservation.fullReadyBlockHashes.empty()) {
        _pendingCacheReservation.reset();
        return true;
    }
    _pendingCacheReservation =
        std::make_shared<const NaviSchedPendingCacheReservation>(std::move(reservation));
    return true;
}

bool MasterBatchReservation::subtractWaitingRequest(const Request &request)
{
    if (_waitingRequestCount == 1) {
        _waitingFeatures = NaviSchedLatencyFeatures();
        _predictedLatencyMs = 0.0;
        return true;
    }
    NaviSchedLatencyFeatures contribution;
    contribution.appendRequest(request.predictedCacheHitTokenCount,
                               request.totalTokenCount - request.predictedCacheHitTokenCount);
    return _waitingFeatures.subtract(contribution);
}

bool MasterBatchReservation::rebuildWaitingFeatures(const CacheMatch *cacheMatch)
{
    NaviSchedLatencyFeatures features;
    if (_waitingRequestCount > 0 && !features.appendBatchCount(1)) {
        return false;
    }
    std::vector<std::pair<size_t, int64_t>> cacheHitUpdates;
    if (cacheMatch) {
        cacheHitUpdates.reserve(_waitingRequestCount);
    }
    size_t waitingCount = 0;
    for (size_t index = 0; index < _requests.size(); ++index) {
        const auto &entry = _requests[index];
        if (entry.state != RequestState::WAITING) {
            continue;
        }
        ++waitingCount;
        auto predictedCacheHitTokenCount = entry.request.predictedCacheHitTokenCount;
        if (cacheMatch) {
            const auto matchedBlockCount = (*cacheMatch)(entry.request);
            if (matchedBlockCount < 0 ||
                matchedBlockCount >
                    static_cast<int64_t>(entry.request.blockHashes
                                             ? entry.request.blockHashes->size()
                                             : 0)) {
                return false;
            }
            predictedCacheHitTokenCount =
                std::min(entry.request.totalTokenCount,
                         matchedBlockCount * entry.request.blockSize);
            cacheHitUpdates.emplace_back(index, predictedCacheHitTokenCount);
        }
        features.appendRequest(
            predictedCacheHitTokenCount,
            entry.request.totalTokenCount - predictedCacheHitTokenCount);
    }
    if (waitingCount != _waitingRequestCount) {
        return false;
    }
    for (const auto &[index, predictedCacheHitTokenCount] : cacheHitUpdates) {
        _requests.writable(index).request.predictedCacheHitTokenCount =
            predictedCacheHitTokenCount;
    }
    _waitingFeatures = std::move(features);
    if (_waitingRequestCount == 0) {
        _predictedLatencyMs = 0.0;
    }
    return true;
}

bool MasterBatchReservation::isTailReady(const RequestEntry &request,
                                         size_t depth) const
{
    if (!_cachePolicy.hybridTailRequired || depth == 0 || !request.request.blockHashes ||
        depth > request.request.blockHashes->size()) {
        return false;
    }
    const auto blockCount = request.request.blockHashes->size();
    const bool blockAligned =
        request.request.totalTokenCount % request.request.blockSize == 0;
    return depth % _cachePolicy.linearStep == 0 || depth == blockCount ||
           (blockAligned && blockCount >= 2 && depth + 1 == blockCount);
}

size_t MasterBatchReservation::findRequest(uint64_t requestId) const
{
    if (!_requestIndexes) {
        return _requests.size();
    }
    const auto it = _requestIndexes->find(requestId);
    return it == _requestIndexes->end() ? _requests.size() : it->second;
}

void MasterBatchReservation::resetTransition(Transition &transition) const
{
    transition = Transition();
    transition.resolved = isResolved();
}

}
