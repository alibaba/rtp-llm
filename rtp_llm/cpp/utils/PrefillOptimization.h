#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace rtp_llm {
// Same process-lifetime contract as config/prefill_optimization.py.
inline bool prefillOptimizationsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("RTP_LLM_CP_PREFILL_PIPELINE_OPT");
        if (value == nullptr || std::strcmp(value, "1") == 0)
            return true;
        if (std::strcmp(value, "0") == 0)
            return false;
        throw std::invalid_argument("RTP_LLM_CP_PREFILL_PIPELINE_OPT must be 0 or 1");
    }();
    return enabled;
}
}  // namespace rtp_llm
