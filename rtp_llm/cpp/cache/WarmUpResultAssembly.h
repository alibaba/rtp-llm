#pragma once

#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/utils/MemoryStatus.h"

namespace rtp_llm {

// How much the warmup permanently cost the device: the pre-warmup free pool minus what was free
// after teardown. Clamped at 0 (an unrelated release between the samples can invert them). This is
// also the persistent component of measured_total_growth_bytes; sharing one definition keeps the
// sizing calculations consistent.
inline size_t poolShrinkBytes(const WarmUpResult& result) {
    return result.available_bytes_pre_warmup > result.device_reserved_bytes ?
               result.available_bytes_pre_warmup - result.device_reserved_bytes :
               0;
}

// Turns the samples taken inside the warmup trace window into the KV-sizing inputs.
//
// pre_warmup_available_bytes is free memory at the trace baseline (weights loaded, the warmup has
// allocated nothing); post_teardown_status is sampled after the traced executor has been released
// and emptyCache() has run, but still inside the window. Its allocator peak still covers the full
// window because releasing memory does not lower the high-water mark.
//
// The normal-forward requirement follows vLLM's accounting model:
//   persistent device growth = pre-warmup free - post-teardown free
//   transient torch headroom = post-teardown torch allocated peak - post-teardown torch allocated
// This counts every resident allocation once through cudaMemGetInfo, then adds only torch memory
// that existed at the high-water mark and was released before the final sample. Reading the peak
// from the final snapshot covers the complete trace window, including executor teardown.
// CUDA graph capture is intentionally outside this trace and is stored separately in
// WarmUpResult::cuda_graph_memory_bytes.
// measurement_trusted applies to the forward measurement. CUDA Graph trust is assigned only after
// graph capture succeeds; both fields default to false for fail-closed manual construction.
//
// Pure computation, deliberately free of logging and device access: it throws
// std::overflow_error instead of asserting so the caller can route the failure through its own
// error handling (same split as calculateRuntimeMemorySizing).
WarmUpResult assembleWarmUpResult(size_t              pre_warmup_available_bytes,
                                  const MemoryStatus& post_teardown_status,
                                  bool                measurement_trusted);

}  // namespace rtp_llm
