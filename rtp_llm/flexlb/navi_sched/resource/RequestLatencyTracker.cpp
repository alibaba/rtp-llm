#include "rtp_llm/flexlb/navi_sched/resource/RequestLatencyTracker.h"
#include "rtp_llm/flexlb/navi_sched/data/NaviSchedLatencyPrediction.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace navi_sched {

bool RequestLatencyTracker::Snapshot::valid() const
{
    return revision > 0 && requestCount >= 0 && std::isfinite(totalPredictedLatencyMs) &&
           totalPredictedLatencyMs >= 0.0 &&
           (requestCount != 0 || totalPredictedLatencyMs == 0.0);
}

bool RequestLatencyTracker::add(
    uint64_t requestId, double predictedLatencyMs,
    std::shared_ptr<const NaviSchedLatencyPrediction> latencyPrediction)
{
    if (requestId == 0 || !std::isfinite(predictedLatencyMs) ||
        predictedLatencyMs < 0.0 ||
        (latencyPrediction &&
         (!latencyPrediction->recomputable ||
          !std::isfinite(latencyPrediction->fixedLatencyMs))) ||
        _requests.size() >= MAX_REQUEST_COUNT) {
        ++_invalidAddCount;
        return false;
    }
    if (_requests.count(requestId) != 0) {
        ++_duplicateAddCount;
        return false;
    }
    const auto total =
        _totalPredictedLatencyMs + static_cast<long double>(predictedLatencyMs);
    if (!std::isfinite(total) ||
        total > static_cast<long double>(std::numeric_limits<double>::max())) {
        ++_invalidAddCount;
        return false;
    }
    _requests.emplace(requestId,
                      Request{predictedLatencyMs, std::move(latencyPrediction), false});
    _totalPredictedLatencyMs = total;
    ++_revision;
    return true;
}

bool RequestLatencyTracker::remove(uint64_t requestId, double *removedLatencyMs)
{
    const auto it = _requests.find(requestId);
    if (it == _requests.end()) {
        ++_missingRemoveCount;
        return false;
    }
    const auto latencyMs = it->second.predictedLatencyMs;
    _requests.erase(it);
    _totalPredictedLatencyMs -= static_cast<long double>(latencyMs);
    if (_requests.empty()) {
        _totalPredictedLatencyMs = 0.0;
    }
    ++_revision;
    if (removedLatencyMs) {
        *removedLatencyMs = latencyMs;
    }
    return true;
}

bool RequestLatencyTracker::markRunning(const std::vector<uint64_t> &requestIds)
{
    if (requestIds.empty()) {
        ++_invalidStateTransitionCount;
        return false;
    }
    std::unordered_set<uint64_t> uniqueRequestIds;
    uniqueRequestIds.reserve(requestIds.size());
    for (auto requestId : requestIds) {
        const auto it = _requests.find(requestId);
        if (requestId == 0 || !uniqueRequestIds.emplace(requestId).second ||
            it == _requests.end() || it->second.running) {
            ++_invalidStateTransitionCount;
            return false;
        }
    }
    for (auto requestId : requestIds) {
        _requests.find(requestId)->second.running = true;
    }
    return true;
}

bool RequestLatencyTracker::recomputeWaiting(
    const Predictor &predictor, size_t &recomputedRequestCount)
{
    recomputedRequestCount = 0;
    if (!predictor) {
        ++_recomputeFailureCount;
        return false;
    }
    std::vector<double> predictions;
    predictions.reserve(_requests.size());
    long double totalPredictedLatencyMs = 0.0L;
    try {
        for (const auto &[requestId, request] : _requests) {
            (void)requestId;
            auto prediction = request.predictedLatencyMs;
            if (!request.running && request.latencyPrediction) {
                const auto modelPrediction =
                    predictor(request.latencyPrediction->features);
                if (!std::isfinite(modelPrediction)) {
                    ++_recomputeFailureCount;
                    recomputedRequestCount = 0;
                    return false;
                }
                prediction = modelPrediction +
                             request.latencyPrediction->fixedLatencyMs;
                if (!std::isfinite(prediction)) {
                    ++_recomputeFailureCount;
                    recomputedRequestCount = 0;
                    return false;
                }
                if (request.latencyPrediction->hasFloor) {
                    const auto floorPrediction =
                        predictor(request.latencyPrediction->floorFeatures);
                    if (!std::isfinite(floorPrediction)) {
                        ++_recomputeFailureCount;
                        recomputedRequestCount = 0;
                        return false;
                    }
                    prediction = std::max(prediction, floorPrediction);
                }
                prediction = std::max(0.0, prediction);
                ++recomputedRequestCount;
            }
            if (!std::isfinite(prediction)) {
                ++_recomputeFailureCount;
                recomputedRequestCount = 0;
                return false;
            }
            totalPredictedLatencyMs += static_cast<long double>(prediction);
            if (!std::isfinite(totalPredictedLatencyMs) ||
                totalPredictedLatencyMs >
                    static_cast<long double>(std::numeric_limits<double>::max())) {
                ++_recomputeFailureCount;
                recomputedRequestCount = 0;
                return false;
            }
            predictions.emplace_back(prediction);
        }
    } catch (...) {
        ++_recomputeFailureCount;
        recomputedRequestCount = 0;
        return false;
    }
    if (recomputedRequestCount == 0) {
        return true;
    }
    size_t index = 0;
    for (auto &[requestId, request] : _requests) {
        (void)requestId;
        request.predictedLatencyMs = predictions[index++];
    }
    _totalPredictedLatencyMs = totalPredictedLatencyMs;
    ++_revision;
    ++_recomputeCount;
    _recomputedRequestCount += recomputedRequestCount;
    return true;
}

bool RequestLatencyTracker::contains(uint64_t requestId) const
{
    return _requests.count(requestId) != 0;
}

bool RequestLatencyTracker::get(uint64_t requestId, double &predictedLatencyMs) const
{
    const auto it = _requests.find(requestId);
    if (it == _requests.end()) {
        return false;
    }
    predictedLatencyMs = it->second.predictedLatencyMs;
    return true;
}

void RequestLatencyTracker::clear()
{
    if (_requests.empty()) {
        return;
    }
    _requests.clear();
    _totalPredictedLatencyMs = 0.0;
    ++_revision;
}

RequestLatencyTracker::Snapshot RequestLatencyTracker::snapshot() const
{
    Snapshot snapshot;
    snapshot.revision = _revision;
    snapshot.requestCount = static_cast<int32_t>(_requests.size());
    snapshot.totalPredictedLatencyMs = static_cast<double>(_totalPredictedLatencyMs);
    snapshot.invalidAddCount = _invalidAddCount;
    snapshot.duplicateAddCount = _duplicateAddCount;
    snapshot.missingRemoveCount = _missingRemoveCount;
    snapshot.recomputeCount = _recomputeCount;
    snapshot.recomputedRequestCount = _recomputedRequestCount;
    snapshot.recomputeFailureCount = _recomputeFailureCount;
    snapshot.invalidStateTransitionCount = _invalidStateTransitionCount;
    return snapshot;
}

bool ConcurrentRequestLatencyTracker::add(
    uint64_t requestId, double predictedLatencyMs,
    std::shared_ptr<const NaviSchedLatencyPrediction> latencyPrediction,
                                          RequestLatencyTracker::Snapshot &snapshot)
{
    autil::ScopedLock lock(_lock);
    const auto added = _tracker.add(requestId, predictedLatencyMs,
                                    std::move(latencyPrediction));
    snapshot = _tracker.snapshot();
    return added;
}

bool ConcurrentRequestLatencyTracker::markRunning(
    const std::vector<uint64_t> &requestIds,
    RequestLatencyTracker::Snapshot &snapshot)
{
    autil::ScopedLock lock(_lock);
    const auto marked = _tracker.markRunning(requestIds);
    snapshot = _tracker.snapshot();
    return marked;
}

bool ConcurrentRequestLatencyTracker::recomputeWaiting(
    const RequestLatencyTracker::Predictor &predictor,
    RequestLatencyTracker::Snapshot &snapshot, size_t &recomputedRequestCount)
{
    autil::ScopedLock lock(_lock);
    const auto recomputed =
        _tracker.recomputeWaiting(predictor, recomputedRequestCount);
    snapshot = _tracker.snapshot();
    return recomputed;
}

bool ConcurrentRequestLatencyTracker::remove(uint64_t requestId,
                                             RequestLatencyTracker::Snapshot &snapshot,
                                             double *removedLatencyMs)
{
    autil::ScopedLock lock(_lock);
    const auto removed = _tracker.remove(requestId, removedLatencyMs);
    snapshot = _tracker.snapshot();
    return removed;
}

size_t ConcurrentRequestLatencyTracker::remove(const std::vector<uint64_t> &requestIds,
                                               RequestLatencyTracker::Snapshot &snapshot,
                                               std::vector<uint64_t> *missingRequestIds)
{
    autil::ScopedLock lock(_lock);
    if (missingRequestIds) {
        missingRequestIds->clear();
    }
    size_t removedCount = 0;
    for (auto requestId : requestIds) {
        if (_tracker.remove(requestId)) {
            ++removedCount;
        } else if (missingRequestIds) {
            missingRequestIds->emplace_back(requestId);
        }
    }
    snapshot = _tracker.snapshot();
    return removedCount;
}

size_t ConcurrentRequestLatencyTracker::removePresent(
    const std::vector<uint64_t> &requestIds, RequestLatencyTracker::Snapshot &snapshot)
{
    autil::ScopedLock lock(_lock);
    size_t removedCount = 0;
    for (auto requestId : requestIds) {
        if (_tracker.contains(requestId) && _tracker.remove(requestId)) {
            ++removedCount;
        }
    }
    snapshot = _tracker.snapshot();
    return removedCount;
}

bool ConcurrentRequestLatencyTracker::contains(uint64_t requestId) const
{
    autil::ScopedLock lock(_lock);
    return _tracker.contains(requestId);
}

void ConcurrentRequestLatencyTracker::clear(RequestLatencyTracker::Snapshot &snapshot)
{
    autil::ScopedLock lock(_lock);
    _tracker.clear();
    snapshot = _tracker.snapshot();
}

RequestLatencyTracker::Snapshot ConcurrentRequestLatencyTracker::snapshot() const
{
    autil::ScopedLock lock(_lock);
    return _tracker.snapshot();
}

} // namespace navi_sched
