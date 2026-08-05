#include "rtp_llm/cpp/cache/RuntimeMemorySizing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace rtp_llm {

RuntimeMemorySizingResult calculateRuntimeMemorySizing(const RuntimeMemorySizingInput& input) {
    if (!std::isfinite(input.safety_ratio) || input.safety_ratio < 0.0 || input.safety_ratio >= 1.0) {
        // Echo the offending value and name the knob: this layer stays dependency-free, so
        // std::to_string is as much formatting as it can do, but an operator still needs to
        // know which setting to change.
        throw std::invalid_argument("runtime_mem_safety_ratio must be finite and in [0, 1), got "
                                    + std::to_string(input.safety_ratio));
    }

    const size_t safety_ratio_bytes = static_cast<size_t>(input.total_gpu_bytes * input.safety_ratio);

    if (!input.has_warmup) {
        // Pre-warmup-feature semantics, kept bit-for-bit so deployments that never
        // run a traced warmup see no sizing change across the upgrade: the ratio
        // term is a floor inside the max() (formerly the hardcoded
        // max(2048 MiB, 5% * total) minimum), not additive headroom.
        const size_t required = std::max({input.configured_reserve_bytes,
                                          input.sampler_required_bytes,
                                          input.no_warmup_floor_bytes,
                                          safety_ratio_bytes});
        return {safety_ratio_bytes, required};
    }

    // A traced warmup replaces the no-warmup floors. Operators needing an absolute
    // floor on this path pass it as configured_reserve_bytes. Safety is *added* here:
    // the measured peak is a point sample, and the headroom covers runtime variation
    // the sample cannot represent.
    const size_t base_required_bytes =
        std::max({input.configured_reserve_bytes, input.warmup_required_bytes, input.sampler_required_bytes});

    if (base_required_bytes > std::numeric_limits<size_t>::max() - safety_ratio_bytes) {
        throw std::overflow_error("runtime memory sizing overflow: base_required_bytes="
                                  + std::to_string(base_required_bytes) + " (configured_reserve="
                                  + std::to_string(input.configured_reserve_bytes) + ", warmup_required="
                                  + std::to_string(input.warmup_required_bytes) + ", sampler_required="
                                  + std::to_string(input.sampler_required_bytes) + ") plus safety_ratio_bytes="
                                  + std::to_string(safety_ratio_bytes) + " exceeds size_t");
    }
    return {safety_ratio_bytes, base_required_bytes + safety_ratio_bytes};
}

}  // namespace rtp_llm
