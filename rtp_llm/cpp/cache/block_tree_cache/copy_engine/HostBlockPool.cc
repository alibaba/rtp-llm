#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/HostBlockPool.h"

#include <cstdlib>
#include <cstring>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

HostBlockPool::HostBlockPool(size_t block_size_bytes, size_t block_count, bool use_pinned_memory):
    block_size_bytes_(block_size_bytes), block_count_(block_count), use_pinned_memory_(use_pinned_memory) {}

HostBlockPool::~HostBlockPool() {
    if (base_ptr_ && use_pinned_memory_) {
        std::free(base_ptr_);
        base_ptr_ = nullptr;
    }
}

bool HostBlockPool::init() {
    if (initialized_) {
        return true;
    }
    if (block_size_bytes_ == 0 || block_count_ == 0) {
        RTP_LLM_LOG_WARNING(
            "HostBlockPool::init: invalid config, block_size=%zu, block_count=%zu", block_size_bytes_, block_count_);
        return false;
    }

    const size_t total_bytes = block_size_bytes_ * block_count_;

    if (use_pinned_memory_) {
        // Allocate page-aligned pinned memory for O_DIRECT disk I/O
        void* ptr = nullptr;
        if (::posix_memalign(&ptr, 4096, total_bytes) != 0) {
            RTP_LLM_LOG_ERROR("HostBlockPool::init: posix_memalign failed, bytes=%zu", total_bytes);
            return false;
        }
        std::memset(ptr, 0, total_bytes);
        base_ptr_ = ptr;
    } else {
        backing_buffer_.resize(total_bytes, 0);
        base_ptr_ = backing_buffer_.data();
    }

    // Block indices: 1-based (0 = NULL_BLOCK_IDX)
    block_cache_ref_counter_.init(static_cast<int>(block_count_) + 1);
    request_ref_counter_.init(static_cast<int>(block_count_) + 1);

    for (size_t i = 1; i <= block_count_; ++i) {
        free_blocks_.insert(static_cast<BlockIdxType>(i));
    }

    initialized_ = true;
    RTP_LLM_LOG_INFO("HostBlockPool::init: block_size=%zu, block_count=%zu, total_bytes=%zu, pinned=%d",
                     block_size_bytes_,
                     block_count_,
                     total_bytes,
                     use_pinned_memory_);
    return true;
}

BlockIdxType HostBlockPool::malloc() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_blocks_.empty()) {
        return NULL_BLOCK_IDX;
    }
    auto         it  = free_blocks_.begin();
    BlockIdxType idx = *it;
    free_blocks_.erase(it);
    return idx;
}

void HostBlockPool::free(BlockIdxType block_idx) {
    if (isNullBlockIdx(block_idx))
        return;
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.insert(block_idx);
}

void HostBlockPool::blockCacheReference(BlockIdxType block_idx) {
    block_cache_ref_counter_.incrementRefCounter({block_idx});
}

void HostBlockPool::blockCacheFree(BlockIdxType block_idx) {
    block_cache_ref_counter_.decrementRefCounter({block_idx});
}

void HostBlockPool::requestReference(BlockIdxType block_idx) {
    request_ref_counter_.incrementRefCounter({block_idx});
}

void HostBlockPool::requestFree(BlockIdxType block_idx) {
    request_ref_counter_.decrementRefCounter({block_idx});
}

void* HostBlockPool::blockAddr(BlockIdxType block_idx) const {
    if (!validBlock(block_idx) || base_ptr_ == nullptr) {
        return nullptr;
    }
    return static_cast<uint8_t*>(base_ptr_) + static_cast<size_t>(block_idx - 1) * block_size_bytes_;
}

size_t HostBlockPool::freeBlocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_.size();
}

bool HostBlockPool::validBlock(BlockIdxType block_idx) const {
    return !isNullBlockIdx(block_idx) && block_idx >= 1 && static_cast<size_t>(block_idx) <= block_count_;
}

BlockInfo HostBlockPool::blockInfo(BlockIdxType block_idx) const {
    BlockInfo info;
    info.is_cuda      = false;
    info.device_index = 0;
    info.addr         = blockAddr(block_idx);
    info.size_bytes   = block_size_bytes_;
    return info;
}

}  // namespace rtp_llm
