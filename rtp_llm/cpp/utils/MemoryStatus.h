#pragma once

#include <cstddef>

namespace rtp_llm {

// Device memory counters shared by the device sampler and
// dependency-free sizing/assembly code. Keep this POD independent of bindings and torch.
struct MemoryStatus {
    size_t used_bytes = 0;
    size_t free_bytes = 0;
    // Device total as reported by cudaMemGetInfo/hipMemGetInfo.
    size_t total_bytes     = 0;
    size_t available_bytes = 0;  // free GPU memory available for allocation
    size_t allocated_bytes = 0;  // current bytes held by live torch allocations
    // Absolute torch allocated high-water mark since resetPeakStats(). Unlike reserved_bytes,
    // allocated_bytes excludes unused caching-allocator blocks. The warmup assembler compares
    // this with allocated_bytes after teardown to recover only the transient torch headroom.
    size_t torch_allocated_peak_bytes = 0;
};

}  // namespace rtp_llm
