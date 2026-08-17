#pragma once

#include <cstddef>

namespace rtp_llm {

// Device memory counters and warmup growth diagnostics shared by the device sampler and
// dependency-free sizing/assembly code. Keep this POD independent of bindings and torch.
struct MemoryStatus {
    size_t used_bytes = 0;
    size_t free_bytes = 0;
    // Device total as reported by cudaMemGetInfo/hipMemGetInfo.
    size_t total_bytes     = 0;
    size_t available_bytes = 0;  // free GPU memory available for allocation
    size_t allocated_bytes = 0;  // memory allocated via current device
    // Torch allocator peak growth over the traced window.
    size_t max_consumed_bytes = 0;
    // Torch allocator reserved growth at the moment of sampling; diagnostics only.
    size_t torch_current_increase_bytes = 0;
    // Driver-side resident delta at the moment of sampling.
    size_t non_torch_increase_bytes = 0;
};

struct MemoryGrowthBreakdown {
    size_t torch_peak_increase_bytes    = 0;
    size_t torch_current_increase_bytes = 0;
    size_t non_torch_increase_bytes     = 0;
};

inline MemoryGrowthBreakdown calculateMemoryGrowth(size_t reserved_baseline_bytes,
                                                   size_t reserved_peak_bytes,
                                                   size_t reserved_current_bytes,
                                                   size_t cuda_used_baseline_bytes,
                                                   size_t cuda_used_current_bytes) {
    const size_t torch_peak_increase =
        reserved_peak_bytes > reserved_baseline_bytes ? reserved_peak_bytes - reserved_baseline_bytes : 0;
    const size_t torch_current_increase =
        reserved_current_bytes > reserved_baseline_bytes ? reserved_current_bytes - reserved_baseline_bytes : 0;
    const size_t non_torch_current =
        cuda_used_current_bytes > reserved_current_bytes ? cuda_used_current_bytes - reserved_current_bytes : 0;
    const size_t non_torch_baseline =
        cuda_used_baseline_bytes > reserved_baseline_bytes ? cuda_used_baseline_bytes - reserved_baseline_bytes : 0;
    const size_t non_torch_increase =
        non_torch_current > non_torch_baseline ? non_torch_current - non_torch_baseline : 0;
    MemoryGrowthBreakdown breakdown;
    breakdown.torch_peak_increase_bytes    = torch_peak_increase;
    breakdown.torch_current_increase_bytes = torch_current_increase;
    breakdown.non_torch_increase_bytes     = non_torch_increase;
    return breakdown;
}

}  // namespace rtp_llm
