#pragma once

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

    private:
        HostStagingBlockPool* pool_{nullptr};
        size_t                block_id_{0};
    };

    using HostStagingBlockBatch = std::vector<HostStagingBlockLease>;
    using BatchReadyCallback    = std::function<void(std::optional<HostStagingBlockBatch>)>;

    // try_pin_memory=false is a test seam to force pageable backing.
    HostStagingBlockPool(size_t block_count, size_t stride_bytes, bool try_pin_memory = true);

    // Test seam: no production caller; lets tests occupy staging blocks atomically.
    std::optional<HostStagingBlockBatch> tryMallocBatch(size_t count);

    // Fair, all-or-nothing allocation; existing async waiters are never bypassed.
    // Invokes callback outside mutex_. A null result means cancellation or an invalid request.
    void requestBatch(size_t count, BatchReadyCallback callback);

    void cancelAllBatchWaiters();

private:
    friend class HostStagingBlockLease;

    struct BatchWaiter {
        size_t             count{0};
        BatchReadyCallback callback;
    };

    struct ReadyBatch {
        BatchReadyCallback    callback;
        HostStagingBlockBatch leases;
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
    std::mutex              mutex_;
};

}  // namespace rtp_llm
