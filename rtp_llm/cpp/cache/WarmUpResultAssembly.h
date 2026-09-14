#pragma once

#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/utils/MemoryStatus.h"

namespace rtp_llm {

// Turns the samples taken inside the warmup trace window into the KV-sizing inputs.
//
// init_free_memory_bytes is sampled after weights are loaded but before temporary profiling state
// is allocated. profile_status is sampled at normal-forward teardown while the trace still exposes
// its allocator high-water mark.
//
// Persistent consumption is reflected in the latest free-memory sample used by KV sizing. The
// initial sample is retained only for consistency validation. CUDA Graph capture remains a separate
// measurement. measurement_trusted applies to the forward measurement. CUDA Graph trust is assigned
// only after graph capture succeeds; both fields default to false for fail-closed manual construction.
WarmUpResult
assembleWarmUpResult(size_t init_free_memory_bytes, const MemoryStatus& profile_status, bool measurement_trusted);

}  // namespace rtp_llm
