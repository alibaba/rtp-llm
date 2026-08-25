#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
    static constexpr size_t kAlignment = 4096;

    class HostStagingBlockLease {
    public:
        HostStagingBlockLease() = default;
        HostStagingBlockLease(HostStagingBlockPool* pool, size_t block_id): pool_(pool), block_id_(block_id) {}
        HostStagingBlockLease(const HostStagingBlockLease&)            = delete;
        HostStagingBlockLease& operator=(const HostStagingBlockLease&) = delete;
        HostStagingBlockLease(HostStagingBlockLease&& other) noexcept:
            pool_(other.pool_), block_id_(other.block_id_) {
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

    private:
        HostStagingBlockPool* pool_{nullptr};
        size_t                block_id_{0};
    };

    using HostStagingBlockBatch = std::vector<HostStagingBlockLease>;
    using BatchReadyCallback = std::function<void(std::optional<HostStagingBlockBatch>)>;
    using BatchWaiterId = uint64_t;

    // try_pin_memory=false is a test seam to force pageable backing.
    HostStagingBlockPool(size_t block_count, size_t stride_bytes, bool try_pin_memory = true);

    std::optional<HostStagingBlockLease> malloc();

    // Fair, all-or-nothing allocation. Existing async waiters are never bypassed.
    std::optional<HostStagingBlockBatch> tryMallocBatch(size_t count);

    // Invokes callback outside mutex_. A null result means cancellation or an invalid request.
    // Returns zero when the callback was satisfied immediately; otherwise returns a waiter id.
    BatchWaiterId requestBatch(size_t count, BatchReadyCallback callback);

    bool cancelBatchWaiter(BatchWaiterId waiter_id);
    void cancelAllBatchWaiters();

    // Exponential backoff with deadline; never sleeps while holding mutex_.
    std::optional<HostStagingBlockLease> mallocWithBackoff(std::chrono::milliseconds timeout);

private:
    friend class HostStagingBlockLease;

    struct BatchWaiter {
        BatchWaiterId      id{0};
        size_t             count{0};
        BatchReadyCallback callback;
    };

    struct ReadyBatch {
        BatchReadyCallback    callback;
        HostStagingBlockBatch leases;
    };

    HostStagingBlockBatch allocateBatchLocked(size_t count);
    std::vector<ReadyBatch> collectReadyBatchesLocked();
    static void dispatchReadyBatches(std::vector<ReadyBatch> ready_batches);

    void reportAcquireTimeout(std::chrono::steady_clock::time_point wait_start);

    void free(size_t block_id);

    HostBufferView blockBuffer(size_t block_id, size_t payload_bytes) const;

    size_t                  block_count_{0};
    size_t                  stride_bytes_{0};
    AlignedHostMemory       backing_;
    std::vector<size_t>     free_id_list_;
    std::deque<BatchWaiter> batch_waiters_;
    BatchWaiterId           next_waiter_id_{1};
    mutable std::mutex      mutex_;
    std::atomic<size_t>     timeout_count_{0};
    std::atomic<int64_t>    last_timeout_log_ns_{0};
};

}  // namespace rtp_llm
