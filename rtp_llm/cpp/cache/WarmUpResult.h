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
    //
    // On the measured path it still carries information: available_bytes_pre_warmup minus this is
    // how much the warmup permanently cost the device, which must not exceed the growth term (see
    // the unaccounted WARNING in MemoryEvaluationHelper).
    size_t device_reserved_bytes = 0;
    // False means the forward ran only to preserve the pre-warmup-feature lazy-init timing and the
    // post-forward device_reserved_bytes sample (PDFUSION): sizing must ignore
    // measured_total_growth_bytes and stay on the no-warmup formula against device_reserved_bytes.
    // Distinct from a broken measurement, which is detected separately via
    // measured_total_growth_bytes == 0.
    bool measurement_trusted = false;
    // Total growth over the traced window: the torch allocator's peak growth (including CUDA graph
    // pools captured while tracing) plus the non-torch growth sampled at the end of the forward.
    // A growth delta over the baseline, not an absolute peak. Serving needs all of it on top of the
    // weights, whether or not the teardown returned it, which is why it pairs with the pre-warmup
    // pool rather than the post-teardown one.
    size_t measured_total_growth_bytes = 0;
};

}  // namespace rtp_llm
