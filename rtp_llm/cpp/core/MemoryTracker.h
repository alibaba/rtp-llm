#pragma once

#include <map>
#include <memory>
#include <atomic>
#include <set>
#include <shared_mutex>
#include <mutex>
#include <queue>
#include <vector>

namespace rtp_llm {

struct MemoryChunk {
    void*  ptr;
    size_t size;
    bool   used = false;
};

struct TrackerStatus {
public:
    size_t available_size         = 0;  // size of unused chunk, including fragmented size.
    size_t fragmented_size        = 0;
    size_t allocated_size         = 0;
    size_t fragment_chunk_count   = 0;
    size_t allocated_chunk_count  = 0;
    size_t allocated_private_size = 0;  // size of private allocated memory, for cuda graph and freezed.
    size_t freezed_bytes          = 0;  // size of freezed memory, only can be allocated for private allocation.
    size_t peak_single_allocation = 0;  // max size of a single allocation (aligned size)
    size_t peak_allocated_size    = 0;  // max value of allocated_size
    std::vector<MemoryChunk> chunks;
};

// This class is designed to completely manage assignable memories.
// It takes a large piece of pre-allocated memory and assigns them on-need.
// Note: this class is not responsible for memory allocation and free.
class MemoryTracker {
public:
    MemoryTracker(void* ptr, size_t size, const size_t align_size);
    ~MemoryTracker();

    void* allocate(const size_t size);
    void  deallocate(void* ptr);

    // allocatePrivate is used for cuda graph,
    // it allocates memory from the end of the whole memory chunk,
    // and freezes the memory once it is allocated privately.
    void* allocatePrivate(const size_t size);

    bool                      isTracking(void* ptr) const;
    TrackerStatus             getStatus() const;
    std::vector<MemoryChunk*> getAllChunks() const;

private:
    size_t checkAndAlign(const size_t size) const;

private:
    mutable std::shared_mutex                 mutex_;
    size_t                                    total_size_;
    size_t                                    align_size_;
    std::map<void*, MemoryChunk*>             chunk_map_;
    std::set<std::pair<size_t, MemoryChunk*>> free_chunk_;
    void*                                     base_ptr_;
    void* freezed_from_ptr_ = nullptr;  // the pointer where freezed memory starts, used for private allocation.
    mutable std::atomic<size_t> peak_single_allocation_{0};
    mutable std::atomic<size_t> peak_allocated_size_{0};
};

}  // namespace rtp_llm
