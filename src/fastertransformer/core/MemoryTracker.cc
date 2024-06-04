#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/core/MemoryTracker.h"

using namespace std;
using ReadLock = std::shared_lock<std::shared_mutex>;
using WriteLock = std::unique_lock<std::shared_mutex>;

namespace fastertransformer {

MemoryTracker::MemoryTracker(void* ptr, const size_t size, const size_t align_size) {
    total_size_ = size;
    align_size_ = align_size;
    MemoryChunk* chunk = new MemoryChunk();
    chunk->ptr = ptr;
    chunk->size = size;
    chunk_map_[ptr] = chunk;
}

MemoryTracker::~MemoryTracker() {
    auto status = getStatus();
    if (status.allocated_chunk_count) {
        FT_LOG_ERROR("Memory tracker is destroyed with %lu allocated chunks of size %lu!",
                     status.allocated_chunk_count, status.allocated_size);
    }
    for (auto& pair : chunk_map_) {
        delete pair.second;
    }
}

void* MemoryTracker::allocate(const size_t alloc_size) {
    if (alloc_size == 0) {
        FT_LOG_ERROR("Memory tracker can not allocate memory of size 0!");
        return nullptr;
    }
    WriteLock lock(mutex_);
    MemoryChunk* chunk_to_use = nullptr;
    const auto aligned_size = align(alloc_size);

    // 1. find the smallest chunk that holds the requested size
    for (auto& pair : chunk_map_) {
        MemoryChunk* chunk = pair.second;
        if ((!chunk->used) && (chunk->size >= aligned_size) && \
            ((chunk_to_use == nullptr) || (chunk->size < chunk_to_use->size)))
        {
            chunk_to_use = chunk;
        }
    }

    // 2. allocate memory
    if (chunk_to_use) {
        if (chunk_to_use->size == aligned_size) {
            chunk_to_use->used = true;
            return chunk_to_use->ptr;
        } else {
            auto chunk_size = chunk_to_use->size;
            chunk_to_use->used = true;
            chunk_to_use->size = aligned_size;
            auto new_chunk = new MemoryChunk();
            new_chunk->ptr = (void*)((size_t)chunk_to_use->ptr + aligned_size);
            new_chunk->size = chunk_size - aligned_size;
            new_chunk->used = false;
            chunk_map_[new_chunk->ptr] = new_chunk;
            return chunk_to_use->ptr;
        }
    }

    // 3. failed to allocate
    FT_LOG_DEBUG("Memory tracker failed to allocate memory of size %lu", aligned_size);
    return nullptr;
}

bool MemoryTracker::isTracking(void* ptr) const {
    ReadLock lock(mutex_);
    auto iter = chunk_map_.find(ptr);
    return (iter != chunk_map_.end()) && iter->second->used;
}

void MemoryTracker::deallocate(void* ptr) {
    WriteLock lock(mutex_);

    // 1. find the chunk and free
    auto chunk_iter = chunk_map_.find(ptr);
    if (chunk_iter == chunk_map_.end()) {
        FT_LOG_ERROR("Memory tracker failed to deallocate [%p]!", ptr);
        return;
    }
    chunk_iter->second->used = false;

    // 2. merge with the next chunk if possible
    auto next_chunk_iter = next(chunk_iter);
    if ((next_chunk_iter != chunk_map_.end()) && (!next_chunk_iter->second->used)) {
        chunk_iter->second->size += next_chunk_iter->second->size;
        delete next_chunk_iter->second;
        chunk_map_.erase(next_chunk_iter);
    }

    // 3. merge with the previous chunk if possible
    if ((chunk_iter != chunk_map_.begin()) && (!prev(chunk_iter)->second->used)) {
        auto prev_chunk_iter = prev(chunk_iter);
        prev_chunk_iter->second->size += chunk_iter->second->size;
        delete chunk_iter->second;
        chunk_map_.erase(chunk_iter);
    }
}

void* MemoryTracker::getBasePtr() const {
    return chunk_map_.begin()->second->ptr;
}

size_t MemoryTracker::getTotalSize() const {
    return total_size_;
}

TrackerStatus MemoryTracker::getStatus() const {
    ReadLock lock(mutex_);
    TrackerStatus status;

    for (auto iter = chunk_map_.begin(); iter != chunk_map_.end(); iter++) {
        auto chunk = iter->second;
        status.chunks.push_back(*chunk);
        if (chunk->used) {
            status.allocated_size += chunk->size;
            status.allocated_chunk_count++;
        } else {
            status.available_size += chunk->size;
            if (next(iter) != chunk_map_.end()) {
                status.fragmented_size += chunk->size;
                status.fragment_chunk_count++;
            } else {
                status.free_size = chunk->size;
            }
        }
    }

    return status;
}


size_t MemoryTracker::align(const size_t size) const {
    return ((size + align_size_ - 1) / align_size_) * align_size_;
}

} // namespace fastertransformer
