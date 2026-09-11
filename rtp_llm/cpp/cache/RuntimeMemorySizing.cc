#include "rtp_llm/cpp/cache/RuntimeMemorySizing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace rtp_llm {

size_t calculateRuntimeMemorySizing(const RuntimeMemorySizingInput& input) {
    if (!std::isfinite(input.safety_ratio) || input.safety_ratio < 0.0 || input.safety_ratio >= 1.0) {
        throw std::invalid_argument("runtime_mem_safety_ratio must be finite and in [0, 1), got "
                                    + std::to_string(input.safety_ratio));
    }

    const double applied_safety_ratio = input.has_memory_profile ? input.safety_ratio : kNoWarmupSafetyRatio;
    const size_t safety_ratio_bytes   = static_cast<size_t>(input.total_gpu_bytes * applied_safety_ratio);

    if (!input.has_memory_profile) {
        return std::max({input.configured_reserve_bytes, input.no_warmup_floor_bytes, safety_ratio_bytes});
    }

    if (input.transient_peak_headroom_bytes > std::numeric_limits<size_t>::max() - input.cuda_graph_memory_bytes) {
        throw std::overflow_error("runtime memory sizing overflow: transient_peak_headroom_bytes="
                                  + std::to_string(input.transient_peak_headroom_bytes)
                                  + " plus cuda_graph_memory_bytes=" + std::to_string(input.cuda_graph_memory_bytes)
                                  + " exceeds size_t");
    }
    const size_t required_with_cuda_graph = input.transient_peak_headroom_bytes + input.cuda_graph_memory_bytes;
    if (required_with_cuda_graph > std::numeric_limits<size_t>::max() - safety_ratio_bytes) {
        throw std::overflow_error("runtime memory sizing overflow: transient_peak_headroom_bytes="
                                  + std::to_string(input.transient_peak_headroom_bytes) + ", cuda_graph_memory_bytes="
                                  + std::to_string(input.cuda_graph_memory_bytes) + " plus safety_ratio_bytes="
                                  + std::to_string(safety_ratio_bytes) + " exceeds size_t");
    }
    const size_t profiled_required_bytes = required_with_cuda_graph + safety_ratio_bytes;
    return std::max(input.configured_reserve_bytes, profiled_required_bytes);
}

}  // namespace rtp_llm
