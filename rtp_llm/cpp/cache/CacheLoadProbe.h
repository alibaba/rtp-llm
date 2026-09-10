#pragma once

#include <cstdint>
#include <optional>

namespace rtp_llm {

enum class CacheDependency {
    NONE,
    LOOKUP_ONLY,
    DATA,
    UNKNOWN
};

struct CacheDependencyEvidence {
    bool all_sources_resolved  = false;
    bool has_data_dependency   = false;
    bool async_lookup_started  = false;
    bool had_error_or_fallback = false;

    CacheDependency dependency() const {
        return has_data_dependency   ? CacheDependency::DATA :
               !all_sources_resolved ? CacheDependency::UNKNOWN :
               async_lookup_started  ? CacheDependency::LOOKUP_ONLY :
                                       CacheDependency::NONE;
    }
};

struct CacheLoadTerminalSnapshot {
    CacheDependencyEvidence evidence;
    bool                    terminal = false;
    bool                    success  = false;
    std::optional<int64_t>  terminal_time_us;
};

inline const char* cacheDependencyName(CacheDependency dependency) {
    switch (dependency) {
        case CacheDependency::NONE:
            return "none";
        case CacheDependency::LOOKUP_ONLY:
            return "lookup_only";
        case CacheDependency::DATA:
            return "data";
        default:
            return "unknown";
    }
}

}  // namespace rtp_llm
