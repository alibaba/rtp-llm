#pragma once
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/MemoryTracker.h"
#include <unordered_set>

namespace fastertransformer {

struct TrackerAllocatorParams {
    IAllocator* real_allocator;
    size_t target_track_bytes;
    size_t bytes_try_step;
    size_t align_size;
};

class TrakcerAllocator : public IAllocator {
public:
    TrakcerAllocator(const TrackerAllocatorParams& params);
    ~TrakcerAllocator();

    MemoryType memoryType() const;
    void* malloc(size_t size, const bool is_set_zero = false);
    void  free(void** ptr);
    void* reMalloc(void* ptr, size_t size, const bool is_set_zero = false);

private:
    IAllocator* real_allocator_;
    std::unique_ptr<MemoryTracker> memory_tracker_;
};


} // namespace fastertransformer
