#pragma once

namespace rtp_llm {

struct WarmUpResult {
    size_t device_reserved_bytes = 0;
    size_t max_used_memory       = 0;
    // Breakdown of max_used_memory measured during warmup. CUDA graph allocations made while
    // tracing are already included in these torch/non-torch deltas.
    size_t torch_peak_increase = 0;
    size_t non_torch_increase  = 0;
};

}  // namespace rtp_llm
