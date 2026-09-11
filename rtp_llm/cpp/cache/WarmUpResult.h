#pragma once

#include <cstddef>

namespace rtp_llm {

struct WarmUpResult {
    // Free memory before profiling starts, after weights and distributed runtime state are loaded.
    // KV sizing compares it with the latest free-memory sample to detect interference from other
    // processes; persistent profiling allocations are already reflected in the latest sample.
    size_t init_free_memory_bytes = 0;
    // Trust is tracked independently for the normal forward and CUDA Graph measurements. A zero
    // forward growth is still a valid measurement, and must not discard a separately measured
    // CUDA Graph requirement.
    bool forward_measurement_trusted    = false;
    bool cuda_graph_measurement_trusted = false;
    // Torch allocation present at the normal-forward peak but absent after executor teardown.
    // Persistent growth is not stored or deducted separately.
    size_t transient_peak_headroom_bytes = 0;
    // Device-memory decrease while constructing and capturing the temporary CUDA graphs. Kept
    // separate from normal forward growth so graph-pool reservation and driver allocations that
    // are not represented by torch allocated peak are still deducted from the KV budget.
    size_t cuda_graph_memory_bytes = 0;
};

}  // namespace rtp_llm
