#include "rtp_llm/cpp/utils/RemoteCacheConfig.h"

#include <cerrno>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string_view>

namespace rtp_llm {

bool remoteCacheGdrEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("RTP_LLM_REMOTE_CACHE_ENABLE_GDR_ZERO_COPY");
        return value != nullptr && std::string_view(value) == "1";
    }();
    return enabled;
}

bool remoteCacheDecodePublicationEnabled() {
    const char* value = std::getenv("RTP_LLM_REMOTE_CACHE_ENABLE_DECODE_PUBLISH");
    return value != nullptr && std::string_view(value) == "1";
}

std::optional<DecodePublicationLimits> remoteCacheDecodePublicationLimits(size_t native_pool_bytes) {
    if (native_pool_bytes == 0)
        return std::nullopt;
    auto positive = [](const char* name, size_t fallback, size_t maximum) -> std::optional<size_t> {
        const char* value = std::getenv(name);
        if (!value)
            return fallback;
        if (*value == '\0')
            return std::nullopt;
        for (const char* p = value; *p; ++p)
            if (*p < '0' || *p > '9')
                return std::nullopt;
        errno             = 0;
        char*      end    = nullptr;
        const auto parsed = std::strtoull(value, &end, 10);
        if (errno == ERANGE || *end || parsed == 0 || parsed > maximum)
            return std::nullopt;
        return static_cast<size_t>(parsed);
    };
    DecodePublicationLimits limits;
    limits.bytes = std::max<size_t>(1, std::min<size_t>(native_pool_bytes / 8, 4ULL * 1024 * 1024 * 1024));
    // Preserve sub-MiB budgets for small native test pools. Only explicit
    // overrides are expressed in MiB; deriving the default must not round up.
    auto mib  = positive("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB", 1, 16384);
    auto jobs = positive("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_JOBS", limits.jobs, 64);
    if (!mib || !jobs)
        return std::nullopt;
    if (std::getenv("RTP_LLM_REMOTE_CACHE_DECODE_PENDING_MIB"))
        limits.bytes = *mib * 1024 * 1024;
    limits.jobs = *jobs;
    return limits;
}

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
