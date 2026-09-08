#include "rtp_llm/cpp/cache/WarmUpResultAssembly.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace rtp_llm {

WarmUpResult assembleWarmUpResult(size_t              pre_warmup_available_bytes,
                                  const MemoryStatus& post_teardown_status,
                                  bool                measurement_trusted) {
    WarmUpResult result;
    result.available_bytes_pre_warmup  = pre_warmup_available_bytes;
    result.device_reserved_bytes       = post_teardown_status.available_bytes;
    result.forward_measurement_trusted = measurement_trusted;

    const size_t persistent_device_growth = poolShrinkBytes(result);
    const size_t transient_torch_headroom =
        post_teardown_status.torch_allocated_peak_bytes > post_teardown_status.allocated_bytes ?
            post_teardown_status.torch_allocated_peak_bytes - post_teardown_status.allocated_bytes :
            0;
    if (persistent_device_growth > std::numeric_limits<size_t>::max() - transient_torch_headroom) {
        throw std::overflow_error("warmup memory measurement overflow: persistent_device_growth="
                                  + std::to_string(persistent_device_growth)
                                  + " plus transient_torch_headroom=" + std::to_string(transient_torch_headroom)
                                  + " exceeds size_t");
    }

    result.measured_total_growth_bytes = persistent_device_growth + transient_torch_headroom;
    return result;
}

}  // namespace rtp_llm
