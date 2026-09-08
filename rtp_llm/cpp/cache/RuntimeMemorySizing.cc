#include "rtp_llm/cpp/cache/RuntimeMemorySizing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace rtp_llm {

RuntimeMemorySizingResult calculateRuntimeMemorySizing(const RuntimeMemorySizingInput& input) {
    if (!std::isfinite(input.safety_ratio) || input.safety_ratio < 0.0 || input.safety_ratio >= 1.0) {
        throw std::invalid_argument("runtime_mem_safety_ratio must be finite and in [0, 1), got "
                                    + std::to_string(input.safety_ratio));
    }

    const double applied_safety_ratio = input.has_warmup ? input.safety_ratio : kNoWarmupSafetyRatio;
    const size_t safety_ratio_bytes   = static_cast<size_t>(input.total_gpu_bytes * applied_safety_ratio);

    if (!input.has_warmup) {
        RuntimeMemorySizingResult result;
        result.runtime_required_bytes = std::max({input.configured_reserve_bytes,
                                                  input.sampler_required_bytes,
                                                  input.no_warmup_floor_bytes,
                                                  safety_ratio_bytes});
        return result;
    }

    const size_t base_required_bytes =
        std::max({input.configured_reserve_bytes, input.warmup_required_bytes, input.sampler_required_bytes});

    if (base_required_bytes > std::numeric_limits<size_t>::max() - input.cuda_graph_memory_bytes) {
        throw std::overflow_error(
            "runtime memory sizing overflow: base_required_bytes=" + std::to_string(base_required_bytes)
            + " plus cuda_graph_memory_bytes=" + std::to_string(input.cuda_graph_memory_bytes) + " exceeds size_t");
    }
    const size_t required_with_cuda_graph = base_required_bytes + input.cuda_graph_memory_bytes;
    if (required_with_cuda_graph > std::numeric_limits<size_t>::max() - safety_ratio_bytes) {
        throw std::overflow_error(
            "runtime memory sizing overflow: base_required_bytes=" + std::to_string(base_required_bytes)
            + " (configured_reserve=" + std::to_string(input.configured_reserve_bytes)
            + ", warmup_required=" + std::to_string(input.warmup_required_bytes)
            + ", sampler_required=" + std::to_string(input.sampler_required_bytes)
            + "), cuda_graph_memory_bytes=" + std::to_string(input.cuda_graph_memory_bytes)
            + " plus safety_ratio_bytes=" + std::to_string(safety_ratio_bytes) + " exceeds size_t");
    }
    RuntimeMemorySizingResult result;
    result.runtime_required_bytes = required_with_cuda_graph + safety_ratio_bytes;
    return result;
}

}  // namespace rtp_llm
