#pragma once

#include <optional>

namespace rtp_llm {

bool remoteCacheGdrEnabled();

// Returns 0 when the environment variable is unset. Invalid or out-of-range
// values return nullopt so backend initialization can fail explicitly.
std::optional<double> remoteCacheQuarantineAbortRatio();

}  // namespace rtp_llm
