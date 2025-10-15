#pragma once
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/core/MemoryTracker.h"
#include <unordered_set>

namespace rtp_llm {

struct TrackerAllocatorParams {
    IAllocator* real_allocator     = nullptr;
    int64_t     target_track_bytes = 0;
    size_t      bytes_try_step     = 64UL * 1024 * 1024;  // 64 MiB
    size_t      align_size         = 1024;
};

class TrackerAllocator: public IVirtualMemAllocator {
public:
    TrackerAllocator(const TrackerAllocatorParams& params);
    ~TrackerAllocator();

    AllocatorType type() const override;
    MemoryType    memoryType() const override;

    void* malloc(size_t size) override;
    void* mallocSync(size_t size) override;
    void  free(void** ptr) override;
    void* reMalloc(void* ptr, size_t size) override;

    void* mallocPrivate(size_t size) override;

    TrackerStatus getTrackerStatus() const;

    void* mallocPhysical(size_t size) override;
    void map() override;
    void unmap() override;

private:
    std::vector<MemoryChunk*> getChunks() const;
    friend class DeviceBase;

private:
    IAllocator*                    real_allocator_;
    std::unique_ptr<MemoryTracker> memory_tracker_;
};

}  // namespace rtp_llm
