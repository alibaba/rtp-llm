#pragma once

#include <cstddef>

namespace rtp_llm {

struct WarmUpResult {
    // Free memory at the trace baseline: the weights are loaded and the warmup has allocated
    // nothing yet. This is the pool the KV budget divides whenever the measurement is used, and it
    // is paired with measured_total_growth_bytes below -- base and growth term are read at the same
    // instant, so neither side has to reason about what the teardown handed back.
    size_t available_bytes_pre_warmup = 0;
    // Free memory after the traced executor was released and emptyCache() ran, still inside the
    // trace window. The base for every path that *discards* the measurement: those reserve a static
    // amount that does not account for what the warmup left resident, so they have to divide the
    // pool that already excludes it. Also the inherited (pre-feature) sizing base.
    size_t device_reserved_bytes = 0;
    // Trust is tracked independently for the normal forward and CUDA Graph measurements. A zero
    // forward growth is still a valid measurement, and must not discard a separately measured
    // CUDA Graph requirement.
    bool forward_measurement_trusted    = false;
    bool cuda_graph_measurement_trusted = false;
    // Normal-forward persistent device growth plus transient torch headroom. This is paired with
    // the pre-warmup free-memory baseline for KV sizing and deliberately excludes CUDA graphs.
    size_t measured_total_growth_bytes = 0;
    // Device-memory decrease while constructing and capturing the temporary CUDA graphs. Kept
    // separate from normal forward growth so graph-pool reservation and driver allocations that
    // are not represented by torch allocated peak are still deducted from the KV budget.
    size_t cuda_graph_memory_bytes = 0;
};

}  // namespace rtp_llm
