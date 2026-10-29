#pragma once

#include <cstdlib>
#include <string>

namespace rtp_llm {
inline bool mtpFusionEnabled() {
#if USING_CUDA
    static const bool enabled = [] {
        const char* value = std::getenv("RTP_LLM_MTP_OPTIMIZE");
        return value != nullptr && std::string(value) == "1";
    }();
    return enabled;
#else
    return false;
#endif
}
}  // namespace rtp_llm
