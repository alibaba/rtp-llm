#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <optional>

namespace rtp_llm {

std::optional<double> remoteCacheQuarantineAbortRatio() {
    const char* value = std::getenv("RTP_LLM_REMOTE_CACHE_QUARANTINE_ABORT_RATIO");
    if (value == nullptr || *value == '\0') {
        return 0.0;
    }

    errno            = 0;
    char*      end   = nullptr;
    const auto ratio = std::strtod(value, &end);
    if (errno == ERANGE || end == value || *end != '\0' || !std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0) {
        return std::nullopt;
    }
    return ratio;
}

}  // namespace rtp_llm
