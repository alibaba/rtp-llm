#pragma once

#include <chrono>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/AlignedHostMemory.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTypes.h"

namespace rtp_llm {

class HostStagingBlockPool {
public:
    using Clock = std::chrono::steady_clock;

    static constexpr size_t kAlignment = 4096;

    class HostStagingBlockLease {
    public:
        HostStagingBlockLease(HostStagingBlockPool* pool, size_t block_id): pool_(pool), block_id_(block_id) {}
        HostStagingBlockLease(const HostStagingBlockLease&)            = delete;
        HostStagingBlockLease& operator=(const HostStagingBlockLease&) = delete;
        HostStagingBlockLease(HostStagingBlockLease&& other) noexcept: pool_(other.pool_), block_id_(other.block_id_) {
            other.pool_ = nullptr;
        }
        HostStagingBlockLease& operator=(HostStagingBlockLease&& other) noexcept {
            if (this != &other) {
                if (pool_ != nullptr) {
                    pool_->free(block_id_);
                }
                pool_       = other.pool_;
                block_id_   = other.block_id_;
                other.pool_ = nullptr;
            }
            return *this;
        }
        ~HostStagingBlockLease() {
            if (pool_ != nullptr) {
                pool_->free(block_id_);
            }
        }

        HostBufferView blockBuffer(size_t payload_bytes) const {
            return pool_->blockBuffer(block_id_, payload_bytes);
        }

        bool belongsTo(const HostStagingBlockPool* pool) const {
            return pool_ == pool;
        }

    private:
        HostStagingBlockPool* pool_{nullptr};
        size_t                block_id_{0};
    };

    using HostStagingBlockBatch = std::vector<HostStagingBlockLease>;
    using BatchReadyCallback    = std::function<void(std::optional<HostStagingBlockBatch>)>;

    HostStagingBlockPool(size_t block_count, size_t stride_bytes);

    // Nonblocking all-or-nothing allocation. Each load WorkLoop owns private pools
    // and a local wait queue; it allocates and returns all of its staging leases.
    std::optional<HostStagingBlockBatch> tryMallocBatch(size_t count);

    size_t capacity() const {
        return block_count_;
    }
    size_t strideBytes() const {
        return stride_bytes_;
    }
    size_t backingBytes() const {
        return backing_.backingBytes();
    }

    // Fair, all-or-nothing allocation; existing async waiters are never bypassed.
    // Invokes callback outside mutex_. A null result means expiration, cancellation, or an invalid request.
    // Returns a nonzero waiter ID only when the request is queued.
    size_t requestBatch(size_t count, Clock::time_point deadline, BatchReadyCallback callback);
    void cancelBatchWaiter(size_t waiter_id);

    // Shutdown boundary: every queued waiter is notified even if another
    // waiter's callback throws, and no exception escapes a caller's destructor.
    void cancelAllBatchWaiters() noexcept;

private:
    friend class HostStagingBlockLease;

    struct BatchWaiter {
        size_t             id;
        size_t             count{0};
        Clock::time_point  deadline;
        BatchReadyCallback callback;
    };

    struct ReadyBatch {
        BatchReadyCallback                   callback;
        std::optional<HostStagingBlockBatch> leases;
    };

    HostStagingBlockBatch   allocateBatchLocked(size_t count);
    std::vector<ReadyBatch> collectReadyBatchesLocked();
    static void             dispatchReadyBatches(std::vector<ReadyBatch> ready_batches);

    void free(size_t block_id);

    HostBufferView blockBuffer(size_t block_id, size_t payload_bytes) const;

    size_t                  block_count_{0};
    size_t                  stride_bytes_{0};
    AlignedHostMemory       backing_;
    std::vector<size_t>     free_id_list_;
    std::deque<BatchWaiter> batch_waiters_;
    size_t                  next_waiter_id_{1};
    std::mutex              mutex_;
};

}  // namespace rtp_llm
