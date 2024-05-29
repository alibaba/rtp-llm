#pragma once
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/core/MemoryTracker.h"
#include <unordered_set>

namespace fastertransformer {

struct TrackerAllocatorParams {
    IAllocator* real_allocator = nullptr;
    size_t target_track_bytes  = 0;
    size_t bytes_try_step      = 64UL * 1024 * 1024; // 64 MiB
    size_t align_size          = 1024;
};

class TrakcerAllocator : public IAllocator {
public:
    TrakcerAllocator(const TrackerAllocatorParams& params);
    ~TrakcerAllocator();

    AllocatorType type() const;
    MemoryType memoryType() const;

    void* malloc(size_t size, const bool is_set_zero = false);
    void  free(void** ptr);
    void* reMalloc(void* ptr, size_t size, const bool is_set_zero = false);

private:
    IAllocator* real_allocator_;
    std::unique_ptr<MemoryTracker> memory_tracker_;
};


} // namespace fastertransformer
