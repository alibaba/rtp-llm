#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedConstants.h"
#include "autil/Lock.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace navi_sched {

class NaviSchedLatencyFeatures;
struct NaviSchedLatencyPrediction;

class RequestLatencyTracker {
public:
    struct Snapshot {
        bool valid() const;

        uint64_t revision = 1;
        int32_t requestCount = 0;
        double totalPredictedLatencyMs = 0.0;
        uint64_t invalidAddCount = 0;
        uint64_t duplicateAddCount = 0;
        uint64_t missingRemoveCount = 0;
        uint64_t recomputeCount = 0;
        uint64_t recomputedRequestCount = 0;
        uint64_t recomputeFailureCount = 0;
        uint64_t invalidStateTransitionCount = 0;
    };

    using Predictor = std::function<double(const NaviSchedLatencyFeatures &)>;

    static constexpr size_t MAX_REQUEST_COUNT =
        NAVI_SCHED_MAX_INFLIGHT_REQUEST_COUNT;

public:
    bool add(uint64_t requestId, double predictedLatencyMs,
             std::shared_ptr<const NaviSchedLatencyPrediction> latencyPrediction = {});
    bool remove(uint64_t requestId, double *removedLatencyMs = nullptr);
    bool markRunning(const std::vector<uint64_t> &requestIds);
    bool recomputeWaiting(const Predictor &predictor, size_t &recomputedRequestCount);
    bool contains(uint64_t requestId) const;
    bool get(uint64_t requestId, double &predictedLatencyMs) const;
    void clear();
    Snapshot snapshot() const;

private:
    struct Request {
        double predictedLatencyMs = 0.0;
        std::shared_ptr<const NaviSchedLatencyPrediction> latencyPrediction;
        bool running = false;
    };

    std::unordered_map<uint64_t, Request> _requests;
    long double _totalPredictedLatencyMs = 0.0;
    uint64_t _revision = 1;
    uint64_t _invalidAddCount = 0;
    uint64_t _duplicateAddCount = 0;
    uint64_t _missingRemoveCount = 0;
    uint64_t _recomputeCount = 0;
    uint64_t _recomputedRequestCount = 0;
    uint64_t _recomputeFailureCount = 0;
    uint64_t _invalidStateTransitionCount = 0;
};

class ConcurrentRequestLatencyTracker {
public:
    bool add(uint64_t requestId, double predictedLatencyMs,
             std::shared_ptr<const NaviSchedLatencyPrediction> latencyPrediction,
             RequestLatencyTracker::Snapshot &snapshot);
    bool add(uint64_t requestId, double predictedLatencyMs,
             RequestLatencyTracker::Snapshot &snapshot)
    {
        return add(requestId, predictedLatencyMs, {}, snapshot);
    }
    bool remove(uint64_t requestId, RequestLatencyTracker::Snapshot &snapshot,
                double *removedLatencyMs = nullptr);
    size_t remove(const std::vector<uint64_t> &requestIds,
                  RequestLatencyTracker::Snapshot &snapshot,
                  std::vector<uint64_t> *missingRequestIds = nullptr);
    size_t removePresent(const std::vector<uint64_t> &requestIds,
                         RequestLatencyTracker::Snapshot &snapshot);
    bool markRunning(const std::vector<uint64_t> &requestIds,
                     RequestLatencyTracker::Snapshot &snapshot);
    bool recomputeWaiting(const RequestLatencyTracker::Predictor &predictor,
                          RequestLatencyTracker::Snapshot &snapshot,
                          size_t &recomputedRequestCount);
    bool contains(uint64_t requestId) const;
    void clear(RequestLatencyTracker::Snapshot &snapshot);
    RequestLatencyTracker::Snapshot snapshot() const;

private:
    mutable autil::ThreadMutex _lock;
    RequestLatencyTracker _tracker;
};

} // namespace navi_sched
