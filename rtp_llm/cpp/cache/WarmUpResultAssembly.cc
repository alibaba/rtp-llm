#include "rtp_llm/cpp/cache/WarmUpResultAssembly.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace rtp_llm {

WarmUpResult assembleWarmUpResult(size_t              pre_warmup_available_bytes,
                                  const MemoryStatus& peak_status,
                                  const MemoryStatus& post_teardown_status,
                                  bool                measurement_trusted) {
    const size_t torch_peak_growth = peak_status.max_consumed_bytes;
    const size_t non_torch_growth  = peak_status.non_torch_increase_bytes;
    if (torch_peak_growth > std::numeric_limits<size_t>::max() - non_torch_growth) {
        throw std::overflow_error("warmup memory measurement overflow: torch_peak_growth="
                                  + std::to_string(torch_peak_growth)
                                  + " plus non_torch_growth=" + std::to_string(non_torch_growth) + " exceeds size_t");
    }

    WarmUpResult result;
    result.available_bytes_pre_warmup  = pre_warmup_available_bytes;
    result.device_reserved_bytes       = post_teardown_status.available_bytes;
    result.measurement_trusted         = measurement_trusted;
    result.measured_total_growth_bytes = torch_peak_growth + non_torch_growth;
    return result;
}

}  // namespace rtp_llm
