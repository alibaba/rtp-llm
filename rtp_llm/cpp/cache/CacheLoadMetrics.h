#pragma once

#include <cstdint>
#include <optional>

namespace rtp_llm {

struct CacheLoadTerminalSnapshot {
    bool has_async_cache_dependency = false;
    bool success                    = false;
    // Present only after the context has published its terminal state.
    std::optional<int64_t> terminal_time_us;
};

}  // namespace rtp_llm
