#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/core/MemoryTracker.h"

using namespace std;
using ReadLock  = std::shared_lock<std::shared_mutex>;
using WriteLock = std::unique_lock<std::shared_mutex>;

namespace rtp_llm {

MemoryTracker::MemoryTracker(void* ptr, size_t size, const size_t align_size) {
    total_size_        = size;
    align_size_        = align_size;
    MemoryChunk* chunk = new MemoryChunk();
    chunk->ptr         = ptr;
    chunk->size        = size;
    chunk_map_[ptr]    = chunk;
    free_chunk_.insert({size, chunk});
    base_ptr_         = ptr;
    freezed_from_ptr_ = (void*)((size_t)ptr + size);  // initially, no memory is freezed.
}

MemoryTracker::~MemoryTracker() {
    auto status = getStatus();
    if (status.allocated_chunk_count) {
        RTP_LLM_LOG_ERROR("Memory tracker is destroyed with %lu allocated chunks of size %lu!",
                          status.allocated_chunk_count,
                          status.allocated_size);
    }
    for (auto& pair : chunk_map_) {
        delete pair.second;
    }
    free_chunk_.clear();
}

void* MemoryTracker::allocate(const size_t alloc_size) {
    WriteLock    lock(mutex_);
    const auto   aligned_size = checkAndAlign(alloc_size);
    MemoryChunk* chunk_to_use = nullptr;

    // Find the smallest chunk that holds the requested size, and is not freezed for private alloc.
    auto it = free_chunk_.lower_bound({aligned_size, nullptr});
    while (it != free_chunk_.end()) {
        if (((size_t)it->second->ptr + aligned_size) <= ((size_t)freezed_from_ptr_)) {
            chunk_to_use = it->second;
            free_chunk_.erase(it);
            break;  // found a chunk that can be used
        }
        it = next(it);  // skip chunks that reaches freezed memory area
    }

    // Allocate memory
    if (chunk_to_use) {
        if (chunk_to_use->size == aligned_size) {
            chunk_to_use->used = true;
        } else {
            auto chunk_size            = chunk_to_use->size;
            chunk_to_use->used         = true;
            chunk_to_use->size         = aligned_size;
            auto new_chunk             = new MemoryChunk();
            new_chunk->ptr             = (void*)((size_t)chunk_to_use->ptr + aligned_size);
            new_chunk->size            = chunk_size - aligned_size;
            new_chunk->used            = false;
            chunk_map_[new_chunk->ptr] = new_chunk;
            free_chunk_.insert({new_chunk->size, new_chunk});
        }
        // Update peak single allocation (includes all allocations)
        size_t current_peak = peak_single_allocation_.load(std::memory_order_relaxed);
        if (aligned_size > current_peak) {
            peak_single_allocation_.store(aligned_size, std::memory_order_relaxed);
        }
        // Update peak allocated size
        size_t current_allocated = aligned_size;
        for (const auto& pair : chunk_map_) {
            if (pair.second->used && pair.second != chunk_to_use) {
                current_allocated += pair.second->size;
            }
        }
        size_t current_peak_allocated = peak_allocated_size_.load(std::memory_order_relaxed);
        if (current_allocated > current_peak_allocated) {
            peak_allocated_size_.store(current_allocated, std::memory_order_relaxed);
        }
        return chunk_to_use->ptr;
    }

    // Failed to allocate
    RTP_LLM_LOG_DEBUG("Memory tracker failed to allocate memory of size %lu", aligned_size);
    return nullptr;
}

void* MemoryTracker::allocatePrivate(const size_t size) {
    WriteLock    lock(mutex_);
    const auto   aligned_size = checkAndAlign(size);
    MemoryChunk* chunk_to_use = nullptr;

    // allocate private starts from the end of the whole memory chunk
    auto iter = chunk_map_.rbegin();
    while (iter != chunk_map_.rend()) {
        auto chunk = iter->second;
        if ((!chunk->used) && (chunk->size >= aligned_size)) {
            chunk_to_use = chunk;
            free_chunk_.erase({chunk_to_use->size, chunk_to_use});
            break;
        }
        iter = next(iter);
    }

    if (chunk_to_use) {
        // Update peak single allocation
        size_t current_peak = peak_single_allocation_.load(std::memory_order_relaxed);
        if (aligned_size > current_peak) {
            peak_single_allocation_.store(aligned_size, std::memory_order_relaxed);
        }

        if (chunk_to_use->size == aligned_size) {
            chunk_to_use->used = true;
            freezed_from_ptr_  = min(freezed_from_ptr_, chunk_to_use->ptr);
            return chunk_to_use->ptr;
        } else {
            // shrink the chunk to use
            auto remained_size = chunk_to_use->size - aligned_size;
            chunk_to_use->size = remained_size;
            free_chunk_.insert({remained_size, chunk_to_use});

            // newly allocated chunk
            auto allocated_chunk             = new MemoryChunk();
            allocated_chunk->ptr             = ((char*)chunk_to_use->ptr) + remained_size;
            allocated_chunk->size            = aligned_size;
            allocated_chunk->used            = true;
            chunk_map_[allocated_chunk->ptr] = allocated_chunk;
            freezed_from_ptr_                = min(freezed_from_ptr_, allocated_chunk->ptr);
            return allocated_chunk->ptr;
        }
    }

    // TODO: should throw exception here ?
    RTP_LLM_LOG_ERROR("Memory tracker failed to allocate private memory of size %lu", aligned_size);
    return nullptr;
}

bool MemoryTracker::isTracking(void* ptr) const {
    ReadLock lock(mutex_);
    auto     iter = chunk_map_.find(ptr);
    return (iter != chunk_map_.end()) && iter->second->used;
}

void MemoryTracker::deallocate(void* ptr) {
    WriteLock lock(mutex_);

    // Find the chunk and free it
    auto chunk_iter = chunk_map_.find(ptr);
    if (chunk_iter == chunk_map_.end()) {
        RTP_LLM_LOG_ERROR("Memory tracker failed to deallocate [%p]!", ptr);
        return;
    }
    chunk_iter->second->used = false;
    MemoryChunk* new_chunk   = chunk_iter->second;
    // Merge with the next chunk if possible
    auto next_chunk_iter = next(chunk_iter);
    if ((next_chunk_iter != chunk_map_.end()) && (!next_chunk_iter->second->used)) {
        chunk_iter->second->size += next_chunk_iter->second->size;
        if (!free_chunk_.erase({next_chunk_iter->second->size, next_chunk_iter->second})) {
            RTP_LLM_LOG_ERROR("free_chunk erase error, can not found.");
        }
        delete next_chunk_iter->second;
        chunk_map_.erase(next_chunk_iter);
    }

    // Merge with the previous chunk if possible
    if ((chunk_iter != chunk_map_.begin()) && (!prev(chunk_iter)->second->used)) {
        auto prev_chunk_iter = prev(chunk_iter);
        if (!free_chunk_.erase({prev_chunk_iter->second->size, prev_chunk_iter->second})) {
            RTP_LLM_LOG_ERROR("free_chunk erase error, can not found.");
        }
        prev_chunk_iter->second->size += chunk_iter->second->size;
        new_chunk = prev_chunk_iter->second;
        delete chunk_iter->second;
        chunk_map_.erase(chunk_iter);
    }
    free_chunk_.insert({new_chunk->size, new_chunk});
}

vector<MemoryChunk*> MemoryTracker::getAllChunks() const {
    ReadLock             lock(mutex_);
    vector<MemoryChunk*> chunks;
    for (auto& pair : chunk_map_) {
        chunks.push_back(pair.second);
    }
    return chunks;
}

TrackerStatus MemoryTracker::getStatus() const {
    ReadLock      lock(mutex_);
    TrackerStatus status;
    status.freezed_bytes          = (size_t)(total_size_ - ((char*)freezed_from_ptr_ - (char*)base_ptr_));
    status.peak_single_allocation = peak_single_allocation_.load(std::memory_order_relaxed);

    for (auto iter = chunk_map_.begin(); iter != chunk_map_.end(); iter++) {
        auto chunk = iter->second;
        status.chunks.push_back(*chunk);
        if (chunk->used) {
            status.allocated_size += chunk->size;
            status.allocated_chunk_count++;
            if (chunk->ptr >= freezed_from_ptr_) {
                status.allocated_private_size += chunk->size;
            }
        } else {
            status.available_size += chunk->size;
            if (next(iter) != chunk_map_.end()) {
                status.fragmented_size += chunk->size;
                status.fragment_chunk_count++;
            }
        }
    }

    // Return peak allocated size (excluding KV cache allocations)
    status.peak_allocated_size = peak_allocated_size_.load(std::memory_order_relaxed);

    return status;
}

void MemoryTracker::resetStatus() {
    WriteLock lock(mutex_);
    // Calculate current allocated_size (excluding KV cache allocations that will be reset)
    size_t current_allocated = 0;
    for (const auto& pair : chunk_map_) {
        if (pair.second->used) {
            current_allocated += pair.second->size;
        }
    }
    // Reset peak_allocated_size to current allocated_size
    peak_allocated_size_.store(current_allocated, std::memory_order_relaxed);
}

size_t MemoryTracker::checkAndAlign(const size_t size) const {
    if (size == 0) {
        throw std::runtime_error("Memory tracker can not allocate memory of size 0!");
    }
    return ((size + align_size_ - 1) / align_size_) * align_size_;
}

}  // namespace rtp_llm
