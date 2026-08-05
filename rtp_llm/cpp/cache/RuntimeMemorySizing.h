#pragma once

#include <cstddef>
#include <cstdint>

namespace rtp_llm {

// Pure sizing math, intentionally dependency-free. safety_ratio and
// no_warmup_floor_bytes carry no product defaults on purpose: every caller
// passes them in from KVCacheConfig, whose fields own those defaults
// (kDefaultRuntimeMemorySafetyRatio / kDefaultRuntimeNoWarmupFloorMb in
// rtp_llm/cpp/config/ConfigModules.h). Restating them here would duplicate a
// value this layer does not own.
struct RuntimeMemorySizingInput {
    bool   has_warmup               = false;
    size_t configured_reserve_bytes = 0;
    size_t warmup_required_bytes    = 0;
    size_t sampler_required_bytes   = 0;
    size_t total_gpu_bytes          = 0;
    double safety_ratio             = 0.0;
    size_t no_warmup_floor_bytes    = 0;
};

struct RuntimeMemorySizingResult {
    // total_gpu_bytes * safety_ratio. Added on top of the base requirement when a
    // warmup measurement exists; otherwise it is one more floor inside the max()
    // (the pre-warmup-feature "at least 5% of total GPU" minimum).
    size_t safety_ratio_bytes     = 0;
    size_t runtime_required_bytes = 0;
};

RuntimeMemorySizingResult calculateRuntimeMemorySizing(const RuntimeMemorySizingInput& input);

}  // namespace rtp_llm
