#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostStagingBlockPool.h"

#include <exception>
#include <algorithm>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {
size_t checkedStagingBytes(size_t count, size_t stride) {
    // AlignedHostMemory adds alignment bytes and passes the result to an int64 tensor dimension.
    const size_t max_bytes =
        std::min(std::numeric_limits<size_t>::max() - HostStagingBlockPool::kAlignment,
                 static_cast<size_t>(std::numeric_limits<int64_t>::max()) - HostStagingBlockPool::kAlignment);
    if (stride != 0 && count > max_bytes / stride) {
        throw std::length_error("host staging capacity exceeds supported allocation size");
    }
    return count * stride;
}
}  // namespace

HostStagingBlockPool::HostStagingBlockPool(size_t block_count, size_t stride_bytes):
    block_count_(block_count),
    stride_bytes_(stride_bytes),
    backing_(checkedStagingBytes(block_count_, stride_bytes_), kAlignment, "host staging block pool") {
    const size_t total_bytes = block_count_ * stride_bytes_;
    free_id_list_.reserve(block_count_);
    for (size_t block_id = 0; block_id < block_count_; ++block_id) {
        free_id_list_.push_back(block_id);
    }
    RTP_LLM_LOG_INFO("host staging block pool ready: blocks=%zu stride=%zu total_bytes=%zu",
                     block_count_,
                     stride_bytes_,
                     total_bytes);
}

std::optional<HostStagingBlockPool::HostStagingBlockBatch> HostStagingBlockPool::tryMallocBatch(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count == 0 || !batch_waiters_.empty() || free_id_list_.size() < count) {
        return std::nullopt;
    }
    return allocateBatchLocked(count);
}

void HostStagingBlockPool::requestBatch(size_t count, Clock::time_point deadline, BatchReadyCallback callback) {
    if (!callback) {
        return;
    }

    std::optional<HostStagingBlockBatch> immediate_result;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count == 0 || count > block_count_ || Clock::now() >= deadline) {
            // Keep the empty result; dispatch the invalid request after unlocking.
        } else if (batch_waiters_.empty() && free_id_list_.size() >= count) {
            immediate_result.emplace(allocateBatchLocked(count));
        } else {
            batch_waiters_.push_back(BatchWaiter{count, deadline, std::move(callback)});
            return;
        }
    }
    callback(std::move(immediate_result));
}

void HostStagingBlockPool::cancelAllBatchWaiters() noexcept {
    std::deque<BatchWaiter> cancelled_waiters;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_waiters.swap(batch_waiters_);
    }
    for (auto& waiter : cancelled_waiters) {
        try {
            waiter.callback(std::nullopt);
        } catch (const std::exception& error) {
            RTP_LLM_LOG_ERROR("staging cancellation callback failed: %s", error.what());
        } catch (...) {
            RTP_LLM_LOG_ERROR("staging cancellation callback failed with unknown exception");
        }
    }
}

void HostStagingBlockPool::free(size_t block_id) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // Valid leases return into the capacity reserved by the constructor.
        free_id_list_.push_back(block_id);
    }
    while (true) {
        ReadyBatch ready;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (batch_waiters_.empty()) {
                return;
            }
            auto& waiter = batch_waiters_.front();
            if (Clock::now() < waiter.deadline) {
                if (free_id_list_.size() < waiter.count) {
                    return;
                }
                try {
                    // Reserve before consuming IDs or moving the waiter. After
                    // reserve, noexcept lease moves cannot allocate or throw.
                    ready.leases.emplace(allocateBatchLocked(waiter.count));
                } catch (const std::bad_alloc&) {
                    // Fail this admission with the existing null result contract.
                    // No blocks were consumed, and later waiters still get notified.
                }
            }
            ready.callback = std::move(waiter.callback);
            batch_waiters_.pop_front();
        }
        dispatchReadyBatch(std::move(ready));
    }
}

HostStagingBlockPool::HostStagingBlockBatch HostStagingBlockPool::allocateBatchLocked(size_t count) {
    HostStagingBlockBatch leases;
    if (before_batch_allocation_for_test_ != nullptr) {
        before_batch_allocation_for_test_();
    }
    leases.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const size_t block_id = free_id_list_.back();
        free_id_list_.pop_back();
        leases.emplace_back(this, block_id);
    }
    return leases;
}

void HostStagingBlockPool::dispatchReadyBatch(ReadyBatch ready) {
    // Dispatch runs outside mutex_, including lease destruction after callbacks.
    try {
        ready.callback(std::move(ready.leases));
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("staging ready callback failed: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("staging ready callback failed with unknown exception");
    }
}

HostBufferView HostStagingBlockPool::blockBuffer(size_t block_id, size_t payload_bytes) const {
    if (block_id >= block_count_) {
        throw std::out_of_range("host staging block id is outside the pool");
    }
    if (payload_bytes > stride_bytes_) {
        throw std::invalid_argument("host staging payload exceeds block capacity");
    }
    void* base = backing_.data() + block_id * stride_bytes_;
    return HostBufferView{base, payload_bytes, stride_bytes_};
}

}  // namespace rtp_llm
