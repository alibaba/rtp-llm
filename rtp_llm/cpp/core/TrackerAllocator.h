#pragma once
#include "rtp_llm/cpp/core/allocator.h"
#include "rtp_llm/cpp/core/MemoryTracker.h"
#include <atomic>
#include <thread>
#include <memory>
#include <string>

namespace kmonitor {
class MetricsReporter;
using MetricsReporterPtr = std::shared_ptr<MetricsReporter>;
}  // namespace kmonitor

namespace rtp_llm {

struct TrackerAllocatorParams {
    IAllocator*                  real_allocator     = nullptr;
    int64_t                      target_track_bytes = 0;
    size_t                       bytes_try_step     = 64UL * 1024 * 1024;  // 64 MiB
    size_t                       align_size         = 1024;
    kmonitor::MetricsReporterPtr metrics_reporter   = nullptr;
    std::string                  allocator_type_tag = "";  // "device" or "host" for metrics tagging
};

class TrackerAllocator: public IAllocator {
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

    // Reset tracker status (call after KV cache allocation)
    void resetStatus();

private:
    void                      reportMetricsLoop();
    std::vector<MemoryChunk*> getChunks() const;
    friend class DeviceBase;

private:
    IAllocator*                    real_allocator_;
    std::unique_ptr<MemoryTracker> memory_tracker_;
    kmonitor::MetricsReporterPtr   metrics_reporter_;
    std::shared_ptr<std::thread>   metrics_reporter_thread_{nullptr};
    std::atomic<bool>              stop_{false};
    std::string                    allocator_type_tag_;  // "device" or "host" for metrics tagging
};

}  // namespace rtp_llm
