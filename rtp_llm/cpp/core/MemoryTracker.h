#pragma once

#include <vector>
#include <memory>
#include <map>
#include <set>
#include <shared_mutex>
#include <mutex>
#include <queue>

namespace rtp_llm {

struct MemoryChunk {
    void* ptr;
    size_t size;
    bool used = false;
};

struct TrackerStatus {
public:
    size_t available_size         = 0; // size of unused chunk, including fragmented size.
    size_t free_size              = 0; // size of last unused chunk, not including fragmented size.
    size_t fragmented_size        = 0;
    size_t allocated_size         = 0;
    size_t fragment_chunk_count   = 0;
    size_t allocated_chunk_count  = 0;
    std::vector<MemoryChunk> chunks;
};

// This class is designed to completely manage assignable memories.
// It takes a large piece of pre-allocated memory and assigns them on-need.
// Note: this class is not responsible for memory allocation and free.
class MemoryTracker {
public:
    MemoryTracker(void* ptr, size_t size, const size_t align_size, bool use_small_chunk=false);
    ~MemoryTracker();

    void* allocate(const size_t size);
    void deallocate(void* ptr);

    bool isTracking(void* ptr) const;
    TrackerStatus getStatus() const;
    std::vector<MemoryChunk *> getAllChunks() const;

private:
    size_t align(const size_t size) const;

private:
    mutable std::shared_mutex mutex_;
    size_t total_size_;
    size_t align_size_;
    std::map<void*, MemoryChunk*> chunk_map_;
    std::set<std::pair<size_t, MemoryChunk*>> free_chunk_;
    void* base_ptr_;
    size_t small_chunk_size_ = 256 * 1024;
    size_t small_chunk_num_ = 1024;
    std::queue<size_t> small_chunk_queue_;
};

} // namespace rtp_llm
