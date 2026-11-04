#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedCommon.h"
#include "rtp_llm/flexlb/navi_sched/data/AgentPendingCacheToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/NaviSchedLatencySample.h"
#include "rtp_llm/flexlb/navi_sched/resource/RequestLatencyTracker.h"
#include "autil/legacy/json.h"

#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace navi_sched {

class MasterBatchReservation {
public:
    enum class RequestState : uint8_t {
        WAITING,
        DISPATCHED,
        FAILED,
        COMPLETED,
    };

    struct Request {
        uint64_t requestId = 0;
        int64_t totalTokenCount = 0;
        int64_t blockSize = 0;
        int64_t predictedCacheHitTokenCount = 0;
        NaviSchedBlockHashesConstPtr blockHashes;
        double predictedLatencyMs = 0.0;
    };

    struct Transition {
        bool changed = false;
        bool fullBatchFastPath = false;
        bool cacheChanged = false;
        bool retireCache = false;
        bool resolved = false;
        std::vector<uint64_t> dispatchedRequestIds;
        std::vector<uint64_t> removedFullReadyBlockHashes;
        std::vector<uint64_t> removedTailReadyBlockHashes;
    };

    using CacheMatch = std::function<int64_t(const Request &)>;

    bool init(int64_t batchId, int64_t scheduleTimeUs, double predictedLatencyMs,
              std::vector<Request> requests, const NaviSchedCachePolicy *cachePolicy);

    bool markFailed(uint64_t requestId, Transition &transition);
    bool markFailed(const std::vector<uint64_t> &requestIds, Transition &transition);
    bool markFailedIndexes(const std::vector<size_t> &requestIndexes,
                           Transition &transition);
    bool markAdmitted(const NaviSchedLocalBatchInfo &batch, Transition &transition);
    bool markCompleted(const NaviSchedLocalBatchInfo &batch, bool successful,
                       Transition &transition);
    bool acknowledgeDispatch(const NaviSchedLocalBatchInfo &batch,
                             std::vector<uint64_t> &acknowledgedRequestIds);
    bool markInflightReported(uint64_t requestId, bool &changed);
    bool markAcceptedWorkloadReported(uint64_t requestId);
    bool markInflightReported(const NaviSchedLocalBatchInfo &batch,
                              std::vector<uint64_t> &changedRequestIds);
    bool recomputeWaitingFeatures(const CacheMatch &cacheMatch);
    bool appendWaitingCapacityFeatures(NaviSchedLatencyFeatures &features,
                                       int32_t maxRunningRequestCount,
                                       int32_t &runningCapacitySlotCount) const;
    bool setPredictedLatencyMs(double predictedLatencyMs);
    bool sameDefinition(const MasterBatchReservation &other) const;

    int64_t getBatchId() const;
    int64_t getScheduleTimeUs() const;
    double getPredictedLatencyMs() const;
    size_t getRequestCount() const;
    size_t getWaitingRequestCount() const;
    size_t getDispatchedRequestCount() const;
    size_t getFailedRequestCount() const;
    size_t getCompletedRequestCount() const;
    size_t getAcknowledgedWaitingRequestCount() const;
    bool isResolved() const;
    bool tracksPendingCache() const;
    bool hasPendingCacheReferences() const;
    bool containsRequest(uint64_t requestId) const;
    bool isDispatchAcknowledged(uint64_t requestId) const;
    bool isAwaitingBatchReady(uint64_t requestId) const;
    RequestState getRequestState(uint64_t requestId) const;
    const NaviSchedLatencyFeatures &getWaitingFeatures() const;
    const NaviSchedPendingCacheReservation &getPendingCacheReservation() const;
    RequestLatencyTracker::Snapshot getUnreportedInflightStats() const;
    autil::legacy::json::JsonMap makeReplayState() const;
    bool restoreReplayState(const autil::legacy::json::JsonMap &state);

private:
    friend class MasterDispatchHandle;
    friend class MasterStateR;
    static constexpr uint32_t INVALID_BLOCK_INDEX = std::numeric_limits<uint32_t>::max();
    static constexpr size_t INVALID_BLOCK_OFFSET = std::numeric_limits<size_t>::max();

    struct RequestEntry {
        Request request;
        RequestState state = RequestState::WAITING;
        bool dispatchAcknowledged = false;
        size_t blockIndexBegin = INVALID_BLOCK_OFFSET;
        bool unreportedInflight = false;
        double unreportedLatencyMs = 0.0;
        bool awaitingBatchReady = false;
    };

    struct BlockReferences {
        uint64_t fullRefCount = 0;
        uint64_t tailRefCount = 0;
    };

    template <typename T, size_t PAGE_CAPACITY>
    class SharedPages {
    public:
        static constexpr size_t PAGE_SIZE = PAGE_CAPACITY;
        using Page = std::array<T, PAGE_SIZE>;
        static constexpr size_t PAGES_PER_GROUP = 64;
        using PageGroup = std::array<std::shared_ptr<Page>, PAGES_PER_GROUP>;

        class ConstIterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = std::ptrdiff_t;
            using pointer = const T *;
            using reference = const T &;

            ConstIterator(const SharedPages *owner, size_t index)
                : _owner(owner), _index(index) {}
            reference operator*() const { return (*_owner)[_index]; }
            pointer operator->() const { return &**this; }
            ConstIterator &operator++() { ++_index; return *this; }
            ConstIterator operator++(int)
            {
                auto previous = *this;
                ++*this;
                return previous;
            }
            bool operator==(const ConstIterator &other) const
            {
                return _owner == other._owner && _index == other._index;
            }
            bool operator!=(const ConstIterator &other) const { return !(*this == other); }

        private:
            const SharedPages *_owner;
            size_t _index;
        };

        SharedPages() = default;
        SharedPages(const SharedPages &) = default;
        SharedPages &operator=(const SharedPages &) = default;
        SharedPages(SharedPages &&other) noexcept
            : _groups(std::move(other._groups)), _size(std::exchange(other._size, 0)) {}
        SharedPages &operator=(SharedPages &&other) noexcept
        {
            if (this != &other) {
                _groups = std::move(other._groups);
                _size = std::exchange(other._size, 0);
            }
            return *this;
        }

        size_t size() const { return _size; }
        bool empty() const { return _size == 0; }
        ConstIterator begin() const { return {this, 0}; }
        ConstIterator end() const { return {this, _size}; }
        void clear()
        {
            _groups.clear();
            _size = 0;
        }
        void reserve(size_t count)
        {
            const auto pages = count / PAGE_SIZE + (count % PAGE_SIZE != 0);
            _groups.reserve(pages / PAGES_PER_GROUP + (pages % PAGES_PER_GROUP != 0));
        }
        const T &operator[](size_t index) const
        {
            const auto pageIndex = index / PAGE_SIZE;
            return (*(*_groups[pageIndex / PAGES_PER_GROUP])
                         [pageIndex % PAGES_PER_GROUP])[index % PAGE_SIZE];
        }
        T &writable(size_t index)
        {
            auto &page = writablePage(index / PAGE_SIZE);
            if (!page.unique()) {
                page = std::make_shared<Page>(*page);
            }
            return (*page)[index % PAGE_SIZE];
        }
        void push_back(T value)
        {
            if (_size % PAGE_SIZE == 0) {
                const auto pageIndex = _size / PAGE_SIZE;
                if (pageIndex % PAGES_PER_GROUP == 0) {
                    _groups.emplace_back(std::make_shared<PageGroup>());
                }
                writablePage(pageIndex) = std::make_shared<Page>();
            }
            writable(_size) = std::move(value);
            ++_size;
        }

    private:
        std::shared_ptr<Page> &writablePage(size_t pageIndex)
        {
            auto &group = _groups[pageIndex / PAGES_PER_GROUP];
            if (!group.unique()) {
                group = std::make_shared<PageGroup>(*group);
            }
            return (*group)[pageIndex % PAGES_PER_GROUP];
        }

        // Copies share both directory groups and data pages until mutation.
        std::vector<std::shared_ptr<PageGroup>> _groups;
        size_t _size = 0;
    };

    bool validLocalBatch(const NaviSchedLocalBatchInfo &batch) const;
    bool collectRequestIndexes(const NaviSchedLocalBatchInfo &batch,
                               std::vector<size_t> &requestIndexes) const;
    bool buildPendingCacheReferences();
    bool removeCacheRequestReferences(size_t requestIndex, Transition &transition);
    bool rebuildPendingCacheReservation();
    bool rebuildWaitingFeatures(const CacheMatch *cacheMatch);
    bool subtractWaitingRequest(const Request &request);
    bool addUnreportedRequest(size_t requestIndex, double latencyMs);
    bool removeUnreportedRequest(size_t requestIndex);
    void clearUnreportedStats();
    bool isTailReady(const RequestEntry &request, size_t depth) const;
    size_t findRequest(uint64_t requestId) const;
    void resetTransition(Transition &transition) const;

private:
    int64_t _batchId = 0;
    int64_t _scheduleTimeUs = 0;
    double _predictedLatencyMs = 0.0;
    NaviSchedCachePolicy _cachePolicy;
    bool _cacheTracked = false;
    size_t _waitingRequestCount = 0;
    size_t _dispatchedRequestCount = 0;
    size_t _failedRequestCount = 0;
    size_t _completedRequestCount = 0;
    size_t _acknowledgedWaitingRequestCount = 0;
    SharedPages<RequestEntry, 32> _requests;
    // Request order is fixed; transactional copies can share this immutable index.
    std::shared_ptr<const std::unordered_map<uint64_t, size_t>> _requestIndexes;
    NaviSchedLatencyFeatures _waitingFeatures;
    mutable std::shared_ptr<const NaviSchedPendingCacheReservation> _pendingCacheReservation;
    mutable bool _pendingCacheDirty = false;
    RequestLatencyTracker::Snapshot _unreportedStats;
    long double _unreportedLatencySumMs = 0.0;
    // Hash identities and per-request block indexes are immutable across copies.
    std::shared_ptr<const NaviSchedBlockHashes> _pendingBlockHashes;
    std::shared_ptr<const std::vector<uint32_t>> _requestBlockIndexes;
    // Copies share pages until a transaction changes a reference on that page.
    SharedPages<BlockReferences, 256> _pendingReferences;
    uint64_t _pendingRebuildCount = 0;
    uint64_t _pendingRebuildVisitedNodes = 0;
};

} // namespace navi_sched
