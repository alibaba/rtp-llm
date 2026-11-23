#pragma once

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::linear_prefill_trace {

inline bool enabled() {
    static const bool value = [] {
        const char* env = std::getenv("KIMI_K3_SMOKE_EVIDENCE");
        return env != nullptr && std::string_view(env) == "1";
    }();
    return value;
}

inline std::string quote(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') {
            out << '\\' << static_cast<char>(ch);
        } else if (ch < 0x20) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(ch);
        } else {
            out << static_cast<char>(ch);
        }
    }
    out << '"';
    return out.str();
}

// Fields contain host metadata only. No device reads, events, waits, transfer
// submission or cache state changes are introduced by this evidence path.
inline void emit(const char* event, const std::string& fields) {
    if (!enabled()) {
        return;
    }
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    RTP_LLM_LOG_INFO("[K3_SMOKE_EVENT] {\"event\":\"%s\",%s,\"time_ns\":%ld}",
                     event, fields.c_str(), static_cast<long>(now));
}

}  // namespace rtp_llm::linear_prefill_trace
