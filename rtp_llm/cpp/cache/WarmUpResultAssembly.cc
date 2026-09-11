#include "rtp_llm/cpp/cache/WarmUpResultAssembly.h"

namespace rtp_llm {

WarmUpResult assembleWarmUpResult(size_t              init_free_memory_bytes,
                                  const MemoryStatus& profile_status,
                                  bool                measurement_trusted) {
    WarmUpResult result;
    result.init_free_memory_bytes      = init_free_memory_bytes;
    result.forward_measurement_trusted = measurement_trusted;
    result.transient_peak_headroom_bytes =
        profile_status.torch_allocated_peak_bytes > profile_status.allocated_bytes ?
            profile_status.torch_allocated_peak_bytes - profile_status.allocated_bytes :
            0;
    return result;
}

}  // namespace rtp_llm
