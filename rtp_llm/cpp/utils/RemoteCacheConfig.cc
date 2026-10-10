#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::optional<bool> parseRemoteCacheGdrEnabled(std::string_view value) {
    std::string normalized(value);
    normalized.erase(normalized.begin(),
                     std::find_if(normalized.begin(), normalized.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    normalized.erase(
        std::find_if(normalized.rbegin(), normalized.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
        normalized.end());
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
        || normalized == "enable" || normalized == "enabled") {
        return true;
    }
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off"
        || normalized == "disable" || normalized == "disabled") {
        return false;
    }
    return std::nullopt;
}

bool remoteCacheGdrEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("RTP_LLM_REMOTE_CACHE_ENABLE_GDR_ZERO_COPY");
        if (value == nullptr) {
            return false;
        }
        const auto parsed = parseRemoteCacheGdrEnabled(value);
        if (!parsed.has_value()) {
            RTP_LLM_LOG_WARNING("invalid RTP_LLM_REMOTE_CACHE_ENABLE_GDR_ZERO_COPY value [%s]; disabling GDR",
                                value);
            return false;
        }
        return *parsed;
    }();
    return enabled;
}

}  // namespace rtp_llm
