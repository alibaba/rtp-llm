#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

#include <cstdlib>

namespace rtp_llm {

bool remoteCacheGdrEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("RTP_LLM_REMOTE_CACHE_ENABLE_GDR_ZERO_COPY");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

}  // namespace rtp_llm
