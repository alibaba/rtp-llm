#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

constexpr std::chrono::milliseconds kInitialBackoff{1};
constexpr std::chrono::milliseconds kMaxBackoff{64};

}  // namespace

HostStagingBlockPool::HostStagingBlockPool(size_t block_count, size_t stride_bytes, bool try_pin_memory):
    block_count_(block_count),
    stride_bytes_(stride_bytes),
    backing_(block_count_ * stride_bytes_, kAlignment, try_pin_memory, "host staging block pool") {
    const size_t total_bytes = block_count_ * stride_bytes_;
    free_id_list_.reserve(block_count_);
    for (size_t block_id = 0; block_id < block_count_; ++block_id) {
        free_id_list_.push_back(block_id);
    }
    RTP_LLM_LOG_INFO("host staging block pool ready: blocks=%zu stride=%zu total_bytes=%zu pinned=%d",
                     block_count_,
                     stride_bytes_,
                     total_bytes,
                     static_cast<int>(backing_.isPinned()));
}

std::optional<HostStagingBlockPool::HostStagingBlockLease> HostStagingBlockPool::malloc() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_id_list_.empty() || !batch_waiters_.empty()) {
        return std::nullopt;
    }
    const size_t block_id = free_id_list_.back();
    free_id_list_.pop_back();
    return HostStagingBlockLease(this, block_id);
}

std::optional<HostStagingBlockPool::HostStagingBlockBatch>
HostStagingBlockPool::tryMallocBatch(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count == 0 || !batch_waiters_.empty() || free_id_list_.size() < count) {
        return std::nullopt;
    }
    return allocateBatchLocked(count);
}

HostStagingBlockPool::BatchWaiterId
HostStagingBlockPool::requestBatch(size_t count, BatchReadyCallback callback) {
    if (!callback) {
        return 0;
    }

    std::optional<HostStagingBlockBatch> immediate_result;
    BatchWaiterId                        waiter_id = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count == 0 || count > block_count_) {
            // Keep the empty result; dispatch the invalid request after unlocking.
        } else if (batch_waiters_.empty() && free_id_list_.size() >= count) {
            immediate_result.emplace(allocateBatchLocked(count));
        } else {
            waiter_id = next_waiter_id_++;
            batch_waiters_.push_back(BatchWaiter{waiter_id, count, std::move(callback)});
            return waiter_id;
        }
    }
    callback(std::move(immediate_result));
    return waiter_id;
}

bool HostStagingBlockPool::cancelBatchWaiter(BatchWaiterId waiter_id) {
    if (waiter_id == 0) {
        return false;
    }

    BatchReadyCallback     cancelled_callback;
    std::vector<ReadyBatch> ready_batches;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto waiter = std::find_if(batch_waiters_.begin(),
                                         batch_waiters_.end(),
                                         [waiter_id](const BatchWaiter& item) { return item.id == waiter_id; });
        if (waiter == batch_waiters_.end()) {
            return false;
        }
        cancelled_callback = std::move(waiter->callback);
        batch_waiters_.erase(waiter);
        ready_batches = collectReadyBatchesLocked();
    }
    cancelled_callback(std::nullopt);
    dispatchReadyBatches(std::move(ready_batches));
    return true;
}

void HostStagingBlockPool::cancelAllBatchWaiters() {
    std::vector<BatchReadyCallback> callbacks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks.reserve(batch_waiters_.size());
        while (!batch_waiters_.empty()) {
            callbacks.push_back(std::move(batch_waiters_.front().callback));
            batch_waiters_.pop_front();
        }
    }
    for (auto& callback : callbacks) {
        callback(std::nullopt);
    }
}

std::optional<HostStagingBlockPool::HostStagingBlockLease>
HostStagingBlockPool::mallocWithBackoff(std::chrono::milliseconds timeout) {
    if (auto lease = malloc(); lease.has_value()) {
        return lease;
    }

    const auto wait_start = std::chrono::steady_clock::now();
    const auto deadline   = wait_start + timeout;
    auto       delay      = kInitialBackoff;

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            reportAcquireTimeout(wait_start);
            return std::nullopt;
        }

        const auto remaining = deadline - now;
        const auto sleep_duration =
            std::min(std::chrono::duration_cast<std::chrono::steady_clock::duration>(delay), remaining);
        std::this_thread::sleep_for(sleep_duration);

        // Final non-blocking try at the deadline; never start a new sleep past it.
        if (auto lease = malloc(); lease.has_value()) {
            return lease;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            reportAcquireTimeout(wait_start);
            return std::nullopt;
        }

        if (delay >= kMaxBackoff / 2) {
            delay = kMaxBackoff;
        } else {
            delay = std::min(delay * 2, kMaxBackoff);
        }
    }
}

// Rate-limited to ~1 log/s.
void HostStagingBlockPool::reportAcquireTimeout(std::chrono::steady_clock::time_point wait_start) {
    const auto        now       = std::chrono::steady_clock::now();
    const int64_t     waited_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - wait_start).count();
    const size_t      total     = timeout_count_.fetch_add(1) + 1;
    const int64_t     now_ns    = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    int64_t           last_ns   = last_timeout_log_ns_.load();
    constexpr int64_t kLogIntervalNs = 1'000'000'000;
    if (now_ns - last_ns < kLogIntervalNs || !last_timeout_log_ns_.compare_exchange_strong(last_ns, now_ns)) {
        return;
    }
    RTP_LLM_LOG_WARNING("host staging acquire timed out after %ld ms: blocks=%zu total_timeouts=%zu; "
                        "consider raising staging block count or investigating disk latency",
                        waited_ms,
                        block_count_,
                        total);
}

void HostStagingBlockPool::free(size_t block_id) {
    std::vector<ReadyBatch> ready_batches;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        free_id_list_.push_back(block_id);
        ready_batches = collectReadyBatchesLocked();
    }
    dispatchReadyBatches(std::move(ready_batches));
}

HostStagingBlockPool::HostStagingBlockBatch HostStagingBlockPool::allocateBatchLocked(size_t count) {
    HostStagingBlockBatch leases;
    leases.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const size_t block_id = free_id_list_.back();
        free_id_list_.pop_back();
        leases.emplace_back(this, block_id);
    }
    return leases;
}

std::vector<HostStagingBlockPool::ReadyBatch> HostStagingBlockPool::collectReadyBatchesLocked() {
    std::vector<ReadyBatch> ready_batches;
    while (!batch_waiters_.empty() && free_id_list_.size() >= batch_waiters_.front().count) {
        BatchWaiter waiter = std::move(batch_waiters_.front());
        batch_waiters_.pop_front();
        ready_batches.push_back(ReadyBatch{std::move(waiter.callback), allocateBatchLocked(waiter.count)});
    }
    return ready_batches;
}

void HostStagingBlockPool::dispatchReadyBatches(std::vector<ReadyBatch> ready_batches) {
    for (auto& ready : ready_batches) {
        ready.callback(std::move(ready.leases));
    }
}

HostBufferView HostStagingBlockPool::blockBuffer(size_t block_id, size_t payload_bytes) const {
    void* base = backing_.data() + block_id * stride_bytes_;
    return HostBufferView{base, payload_bytes, stride_bytes_};
}

}  // namespace rtp_llm
