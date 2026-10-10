#pragma once
#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace rtp_llm::rdma_transport {
class RdmaDeferredReadQueue {
public:
    enum class Ownership {
        PENDING,
        DEFERRED,
        CALLER
    };
    using Ticket = std::shared_ptr<std::atomic<Ownership>>;

    // Allocate bookkeeping before the NIC can access storage. PENDING entries
    // cannot be reclaimed, even if completion races ahead of the waiting reader.
    Ticket prepare(std::shared_ptr<const std::atomic<bool>> completed,
                   std::shared_ptr<void>                    storage,
                   std::function<void()>                    release_remote);
    void   defer(std::shared_ptr<const std::atomic<bool>> completed,
                 std::shared_ptr<void>                    storage,
                 std::function<void()>                    release_remote);
    void   reclaimCompleted();
    bool   empty() const;
    void   clearAfterTransportShutdown();

private:
    struct Entry {
        Ticket                                   ownership;
        std::shared_ptr<const std::atomic<bool>> completed;
        std::shared_ptr<void>                    storage;
        std::function<void()>                    release_remote;
    };
    mutable std::mutex mutex_;
    std::vector<Entry> reads_;
};

inline RdmaDeferredReadQueue::Ticket RdmaDeferredReadQueue::prepare(std::shared_ptr<const std::atomic<bool>> completed,
                                                                    std::shared_ptr<void>                    storage,
                                                                    std::function<void()> release_remote) {
    auto                        ownership = std::make_shared<std::atomic<Ownership>>(Ownership::PENDING);
    std::lock_guard<std::mutex> lock(mutex_);
    reads_.push_back({ownership, std::move(completed), std::move(storage), std::move(release_remote)});
    return ownership;
}

inline void RdmaDeferredReadQueue::defer(std::shared_ptr<const std::atomic<bool>> completed,
                                         std::shared_ptr<void>                    storage,
                                         std::function<void()>                    release_remote) {
    auto ownership = prepare(std::move(completed), std::move(storage), std::move(release_remote));
    ownership->store(Ownership::DEFERRED, std::memory_order_release);
}

inline void RdmaDeferredReadQueue::reclaimCompleted() {
    static_assert(std::is_nothrow_move_assignable<Entry>::value, "reclaim must not allocate or throw when moving");
    for (;;) {
        Entry ready;
        bool  release_remote = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto                        it = std::find_if(reads_.begin(), reads_.end(), [](const Entry& entry) {
                const auto owner = entry.ownership->load(std::memory_order_acquire);
                return owner == Ownership::CALLER
                       || (owner == Ownership::DEFERRED && entry.completed->load(std::memory_order_acquire));
            });
            if (it == reads_.end()) {
                return;
            }
            release_remote = it->ownership->load(std::memory_order_relaxed) == Ownership::DEFERRED;
            ready          = std::move(*it);
            reads_.erase(it);
        }
        // Neither extraction nor vector::erase allocates. Run user cleanup
        // outside the queue lock, including release of the receive lease.
        ready.storage.reset();
        try {
            if (release_remote && ready.release_remote) {
                ready.release_remote();
            }
        } catch (...) {
            // Cleanup failures do not terminate the ownership reclaimer.
        }
    }
}

inline bool RdmaDeferredReadQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reads_.empty();
}

inline void RdmaDeferredReadQueue::clearAfterTransportShutdown() {
    std::vector<Entry> retired;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        retired.swap(reads_);
    }
    // Teardown proves the local NIC is quiescent, but is not a completion
    // receipt authorizing a remote release. Do not invoke those callbacks.
}

}  // namespace rtp_llm::rdma_transport
