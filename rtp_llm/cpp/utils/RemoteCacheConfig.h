#pragma once

#include <optional>
#include <string_view>

namespace rtp_llm {

// Environment configuration is immutable after process startup. Cache the
// value on first use so every RemoteCache component observes the same setting.
bool remoteCacheGdrEnabled();

// Exposed as a pure parser so configuration tests do not mutate the process
// environment or depend on remoteCacheGdrEnabled()'s process-wide cache.
std::optional<bool> parseRemoteCacheGdrEnabled(std::string_view value);

}  // namespace rtp_llm
