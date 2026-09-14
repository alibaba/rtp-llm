#pragma once

#include <cstddef>

namespace rtp_llm {

inline constexpr double kNoWarmupSafetyRatio = 0.05;

// Pure sizing math. A trusted profile reserves transient runtime memory, an independent CUDA Graph
// measurement, and additive safety headroom from the latest free-memory snapshot. The configured
// reserve is a lower bound on that headroom. The no-profile path uses the configured reserve, a fixed
// floor, and a total-memory safety ratio as fallbacks.
struct RuntimeMemorySizingInput {
    bool   has_memory_profile            = false;
    size_t configured_reserve_bytes      = 0;
    size_t transient_peak_headroom_bytes = 0;
    size_t cuda_graph_memory_bytes       = 0;
    size_t total_gpu_bytes               = 0;
    double safety_ratio                  = 0.0;
    size_t no_warmup_floor_bytes         = 0;
};

size_t calculateRuntimeMemorySizing(const RuntimeMemorySizingInput& input);

}  // namespace rtp_llm
