#pragma once

#include <cstddef>

namespace rtp_llm {

struct WarmUpResult {
    size_t device_reserved_bytes = 0;
    // What the KV-cache budget reserves. A *growth* delta, not an absolute peak: the torch
    // allocator peak growth over the warmup (including CUDA graph pools captured while tracing)
    // plus non_torch_transient below.
    size_t measured_peak_growth_bytes = 0;
    // The two halves of the measured non-torch growth, split by whether they survived the warmup
    // teardown (see transientNonTorchBytes in DeviceData.h). Only the transient half is part of
    // measured_peak_growth_bytes; the resident half is carried for logging because
    // available_bytes already excludes it.
    size_t non_torch_resident  = 0;
    size_t non_torch_transient = 0;
};

}  // namespace rtp_llm
