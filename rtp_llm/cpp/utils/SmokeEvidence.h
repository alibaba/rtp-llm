#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace rtp_llm {

// Optional request/path evidence. Read once so the disabled hot path only checks
// a cached flag. Enabling evidence never changes execution or synchronization.
inline bool smokeEvidenceEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("RTP_LLM_SMOKE_EVIDENCE");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline int64_t smokeEvidenceTimeNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

}  // namespace rtp_llm
