#pragma once

#include "rtp_llm/cpp/cache/WarmUpResult.h"
// Pulls the python-bindings layer into cache core for MemoryStatus -- the reverse of the intended
// bindings->core direction, and it transitively links torch (device_data -> model_config ->
// type_convert), so targets using this header are "needs no GPU", not "torch-free". Mid-term fix:
// sink MemoryStatus into a neutral header (e.g. rtp_llm/cpp/utils) included by both sides.
#include "rtp_llm/models_py/bindings/core/DeviceData.h"

namespace rtp_llm {

// How much the warmup permanently cost the device: the pre-warmup free pool minus what was free
// after teardown. Clamped at 0 (an unrelated allocation between the samples can invert them). Both
// the sizing layer (MemoryEvaluationHelper) and the logging layer (NormalEngine::makeWarmUpResult)
// compare this against measured_total_growth_bytes; sharing one definition keeps their WARNING and
// [KV_ALLOC] numbers from drifting.
inline size_t poolShrinkBytes(const WarmUpResult& result) {
    return result.available_bytes_pre_warmup > result.device_reserved_bytes ?
               result.available_bytes_pre_warmup - result.device_reserved_bytes :
               0;
}

// Turns the samples taken inside the warmup trace window into the KV-sizing inputs.
//
// pre_warmup_available_bytes is free memory at the trace baseline (weights loaded, the warmup has
// allocated nothing); peak_status is sampled at the end of the traced forward; post_teardown_status
// after the traced executor has been released and emptyCache() has run, but still inside the window.
//
// Base and growth term are deliberately read at the same instant: the pre-warmup pool is paired
// with the *total* growth from peak_status, so no consumer has to reason about which share the
// teardown returned. post_teardown_status contributes only its free-memory reading, which is the
// base for the paths that discard the measurement.
// measurement_trusted is required so every producer declares whether sizing may consume the
// measured growth; WarmUpResult itself defaults to false for fail-closed manual construction.
//
// Pure computation, deliberately free of logging and device access: it throws
// std::overflow_error instead of asserting so the caller can route the failure through its own
// error handling (same split as calculateRuntimeMemorySizing).
WarmUpResult assembleWarmUpResult(size_t              pre_warmup_available_bytes,
                                  const MemoryStatus& peak_status,
                                  const MemoryStatus& post_teardown_status,
                                  bool                measurement_trusted);

}  // namespace rtp_llm
