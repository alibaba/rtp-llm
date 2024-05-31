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

class TrackerAllocator : public IAllocator {
public:
    TrackerAllocator(const TrackerAllocatorParams& params);
    ~TrackerAllocator();

    AllocatorType type() const override;
    MemoryType memoryType() const override;

    void* malloc(size_t size) override;
    void  free(void** ptr) override;
    void* reMalloc(void* ptr, size_t size) override;

    TrackerStatus getTrackerStatus() const;

private:
    IAllocator* real_allocator_;
    std::unique_ptr<MemoryTracker> memory_tracker_;
};

} // namespace fastertransformer
