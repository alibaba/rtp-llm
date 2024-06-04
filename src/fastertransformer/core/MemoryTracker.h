#pragma once

#include <vector>
#include <memory>
#include <map>
#include <shared_mutex>

namespace fastertransformer {

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
    MemoryTracker(void* ptr, const size_t size, const size_t align_size);
    ~MemoryTracker();

    void* allocate(const size_t size);
    void deallocate(void* ptr);

    bool isTracking(void* ptr) const;
    void* getBasePtr() const;
    size_t getTotalSize() const;
    TrackerStatus getStatus() const;

private:
    size_t align(const size_t size) const;

private:
    mutable std::shared_mutex mutex_;
    size_t total_size_;
    size_t align_size_;
    std::map<void*, MemoryChunk*> chunk_map_;
};

} // namespace fastertransformer
