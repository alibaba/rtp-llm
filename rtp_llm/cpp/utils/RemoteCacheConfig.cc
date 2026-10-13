#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

#include <cstdlib>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::optional<bool> parseRemoteCacheGdrEnabled(std::string_view value) {
    if (value == "1") {
        return true;
    }
    if (value == "0") {
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
