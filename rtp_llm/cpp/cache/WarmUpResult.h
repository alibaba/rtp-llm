#pragma once

namespace rtp_llm {

struct WarmUpResult {
    size_t device_reserved_bytes = 0;
    size_t max_used_memory       = 0;
    // Breakdown of max_used_memory measured during warmup (torch caching-allocator peak vs
    // non-torch NCCL/all-to-all growth), surfaced only for the KV-sizing log.
    size_t torch_peak_increase = 0;
    size_t non_torch_increase  = 0;
};

}  // namespace rtp_llm
