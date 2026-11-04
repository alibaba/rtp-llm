#pragma once

#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "aios/navi_sched/data/NaviSchedLatencySample.h"

namespace navi_sched {

class AgentWorkloadOwnership {
private:
    struct Batch;

public:
    using RetiredBatches = std::vector<std::shared_ptr<Batch>>;
    using PreparedBatch = std::shared_ptr<Batch>;
    AgentWorkloadOwnership() = default;
    AgentWorkloadOwnership(const AgentWorkloadOwnership &) = delete;
    AgentWorkloadOwnership &operator=(const AgentWorkloadOwnership &) = delete;
    static PreparedBatch prepareBatch(const std::vector<NaviSchedRequestStats> &requests);
    static PreparedBatch prepareRequest(const NaviSchedRequestStats &request);
    bool addUnbatchedRequest(uint64_t requestId, const PreparedBatch &request);
    bool removeUnbatchedRequests(const std::vector<uint64_t> &requestIds);
    bool addBatch(int64_t localBatchId, const std::vector<NaviSchedRequestStats> &requests);
    bool addBatch(int64_t localBatchId, const PreparedBatch &batch);
    bool removeRequests(int64_t localBatchId, const std::vector<uint64_t> &requestIds);
    bool handoff(const std::vector<int64_t> &localBatchIds, RetiredBatches &retired);
    bool combine(const NaviSchedWorkloadSnapshot &observed,
                 NaviSchedWorkloadSnapshot &combined,
                 const std::vector<int64_t> &coveredBatchIds = {}) const;
    size_t requestCount() const { return _requestCount; }
    size_t batchCount() const { return _batches.size(); }
    size_t unbatchedRequestCount() const { return _unbatched ? _unbatched->requests.size() : 0; }
    bool hasBatch(int64_t localBatchId) const { return _batches.count(localBatchId) != 0; }

private:
    static PreparedBatch prepareRequests(const NaviSchedRequestStats *requests, size_t count);
    struct PrefillTotals {
        int64_t count = 0;
        int64_t hit = 0;
        int64_t miss = 0;
        unsigned __int128 squares = 0;
        unsigned __int128 products = 0;
        void appendRequest(int64_t hitTokens, int64_t missTokens);
        void append(const PrefillTotals &other);
        bool subtract(const PrefillTotals &other);
        int64_t getBatchSize() const { return count; }
        int64_t getCacheHitTokenCount() const { return hit; }
        int64_t getCacheMissTokenCount() const { return miss; }
        double getCacheMissTokenSquareSum() const { return static_cast<double>(squares); }
        double getCacheHitMissTokenProductSum() const { return static_cast<double>(products); }
    };
    struct RequestContribution {
        int64_t cacheHitTokens;
        int64_t cacheMissTokens;
    };
    using RequestList = std::list<RequestContribution>;
    using BatchRequests = std::unordered_map<uint64_t, RequestList::iterator>;
    using BatchList = std::list<std::shared_ptr<Batch>>;
    struct Batch {
        int64_t localBatchId = 0;
        RequestList requests;
        BatchRequests byRequestId;
        PrefillTotals prefillFeatures;
        int64_t promptTokens = 0;
        BatchList::iterator position;
    };
    bool eraseRequest(Batch &batch, BatchRequests::iterator request);

    BatchList _batchOrder;
    std::unordered_map<int64_t, std::shared_ptr<Batch>> _batches;
    PreparedBatch _unbatched;
    size_t _requestCount = 0;
    PrefillTotals _prefillFeatures;
    int64_t _promptTokens = 0;
};

}
