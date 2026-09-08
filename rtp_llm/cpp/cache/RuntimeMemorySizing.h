#pragma once

#include <cstddef>

namespace rtp_llm {

inline constexpr double kNoWarmupSafetyRatio = 0.05;

// Pure sizing math. A trusted warmup uses normal forward growth, an independent CUDA graph
// measurement, and additive safety headroom; the fallback path preserves the previous behavior.
struct RuntimeMemorySizingInput {
    bool   has_warmup               = false;
    size_t configured_reserve_bytes = 0;
    size_t warmup_required_bytes    = 0;
    size_t cuda_graph_memory_bytes  = 0;
    size_t sampler_required_bytes   = 0;
    size_t total_gpu_bytes          = 0;
    double safety_ratio             = 0.0;
    size_t no_warmup_floor_bytes    = 0;
};

struct RuntimeMemorySizingResult {
    size_t runtime_required_bytes = 0;
};

RuntimeMemorySizingResult calculateRuntimeMemorySizing(const RuntimeMemorySizingInput& input);

}  // namespace rtp_llm
