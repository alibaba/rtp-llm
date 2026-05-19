#include "rtp_llm/cpp/disaggregate/cache_store/CacheTransferBufferPool.h"

#include <cuda_runtime.h>
#include <algorithm>

#include "autil/Log.h"

namespace rtp_llm {

AUTIL_DECLARE_AND_SETUP_LOGGER(CacheTransferBufferPool, CacheTransferBufferPool);

CacheTransferBufferPool::CacheTransferBufferPool(size_t pool_size, const std::shared_ptr<MemoryUtil>& memory_util):
    pool_size_(pool_size), memory_util_(memory_util) {
    if (pool_size_ == 0) {
        return;
    }
    auto err = cudaHostAlloc(&base_, pool_size_, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        AUTIL_LOG(WARN, "CacheTransferBufferPool: cudaHostAlloc(%zu) failed: %s", pool_size_, cudaGetErrorString(err));
        base_      = nullptr;
        pool_size_ = 0;
        return;
    }

    if (memory_util_ && memory_util_->isRdmaMode()) {
        if (memory_util_->regUserMr(base_, pool_size_, false)) {
            mr_registered_ = true;
        } else {
            AUTIL_LOG(WARN, "CacheTransferBufferPool: regUserMr failed, RDMA may use fallback path");
        }
    }

    free_list_.push_back(FreeBlock{0, pool_size_});
    free_bytes_ = pool_size_;
}

CacheTransferBufferPool::~CacheTransferBufferPool() {
    if (base_ != nullptr) {
        if (mr_registered_ && memory_util_) {
            memory_util_->deregUserMr(base_, false);
        }
        cudaFreeHost(base_);
    }
}

CacheTransferBufferPool::BufferHandle* CacheTransferBufferPool::tryAllocate(size_t size) {
    if (size == 0 || base_ == nullptr) {
        return nullptr;
    }
    size_t                      aligned_size = alignGatherSize(size);
    std::lock_guard<std::mutex> lock(mutex_);
    return allocateInternal(aligned_size);
}

void CacheTransferBufferPool::free(BufferHandle* handle) {
    if (handle == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);

    FreeBlock new_block{handle->offset, handle->size};

    auto it = free_list_.begin();
    while (it != free_list_.end() && it->offset < new_block.offset) {
        ++it;
    }
    it = free_list_.insert(it, new_block);

    // merge with next
    auto next = std::next(it);
    if (next != free_list_.end() && it->offset + it->size == next->offset) {
        it->size += next->size;
        free_list_.erase(next);
    }
    // merge with prev
    if (it != free_list_.begin()) {
        auto prev = std::prev(it);
        if (prev->offset + prev->size == it->offset) {
            prev->size += it->size;
            free_list_.erase(it);
        }
    }

    free_bytes_ += handle->size;
    delete handle;
}

size_t CacheTransferBufferPool::totalBytes() const {
    return pool_size_;
}

size_t CacheTransferBufferPool::freeBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_bytes_;
}

size_t CacheTransferBufferPool::largestFreeBlock() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t                      largest = 0;
    for (const auto& fb : free_list_) {
        largest = std::max(largest, fb.size);
    }
    return largest;
}

void* CacheTransferBufferPool::baseAddr() const {
    return base_;
}

CacheTransferBufferPool::BufferHandle* CacheTransferBufferPool::allocateInternal(size_t aligned_size) {
    auto best = free_list_.end();
    for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
        if (it->size >= aligned_size) {
            if (best == free_list_.end() || it->size < best->size) {
                best = it;
            }
        }
    }
    if (best == free_list_.end()) {
        return nullptr;
    }

    auto* handle   = new BufferHandle();
    handle->ptr    = static_cast<char*>(base_) + best->offset;
    handle->size   = aligned_size;
    handle->offset = best->offset;

    if (best->size > aligned_size) {
        best->offset += aligned_size;
        best->size -= aligned_size;
    } else {
        free_list_.erase(best);
    }

    free_bytes_ -= aligned_size;
    return handle;
}

}  // namespace rtp_llm
