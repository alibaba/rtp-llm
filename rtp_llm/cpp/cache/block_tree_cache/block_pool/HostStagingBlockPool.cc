#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"

#include <mutex>
#include <optional>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

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

std::optional<HostStagingBlockPool::HostStagingBlockBatch> HostStagingBlockPool::tryMallocBatch(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count == 0 || !batch_waiters_.empty() || free_id_list_.size() < count) {
        return std::nullopt;
    }
    return allocateBatchLocked(count);
}

void HostStagingBlockPool::requestBatch(size_t count, BatchReadyCallback callback) {
    if (!callback) {
        return;
    }

    std::optional<HostStagingBlockBatch> immediate_result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count == 0 || count > block_count_) {
            // Keep the empty result; dispatch the invalid request after unlocking.
        } else if (batch_waiters_.empty() && free_id_list_.size() >= count) {
            immediate_result.emplace(allocateBatchLocked(count));
        } else {
            batch_waiters_.push_back(BatchWaiter{count, std::move(callback)});
            return;
        }
    }
    callback(std::move(immediate_result));
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
