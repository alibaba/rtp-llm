#pragma once

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace rtp_llm {
namespace speculative {

// Match the Python CMP parser without changing legacy MTP flag semantics.
inline bool parseHy4CmpBool(const char* name, const char* value, bool default_on) {
    if (value == nullptr) {
        return default_on;
    }
    std::string normalized(value);
    const auto first = normalized.find_first_not_of(" \t\n\r\f\v");
    if (first == std::string::npos) {
        normalized.clear();
    } else {
        normalized = normalized.substr(first, normalized.find_last_not_of(" \t\n\r\f\v") - first + 1);
    }
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") {
        return true;
    }
    if (normalized.empty() || normalized == "0" || normalized == "false" || normalized == "no"
        || normalized == "off") {
        return false;
    }
    throw std::invalid_argument(std::string("invalid ") + name + "=" + value);
}

inline bool resolveHy4EarlyD2H(bool hy4_cmp_enabled, const char* override_value) {
    return parseHy4CmpBool("RTP_LLM_MTP_EARLY_SPEC_LOGITS_D2H", override_value, hy4_cmp_enabled);
}

}  // namespace speculative
}  // namespace rtp_llm
