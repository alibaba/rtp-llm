#pragma once

#include <optional>
#include <cstddef>

namespace rtp_llm {

bool remoteCacheGdrEnabled();

// Startup-only configuration: cache the result on the backend/cache owner.
// Request, allocation and model-step paths must not re-read the environment.
// This opt-in is independent of the existing Prefill remote-cache/GDR flags.
bool remoteCacheDecodePublicationEnabled();

struct DecodePublicationLimits {
    // Existing GPU pages retained after Decode, not a new GPU or CPU pool.
    size_t bytes = 512ULL * 1024 * 1024;
    size_t jobs  = 4;
};

// Called only when publication is enabled, at backend initialization. The
// backend passes the sum of unique physical native pools, without CP copies.
// Default: pool / 8, capped at 4 GiB/process, with four pending publications.
// Explicit positive MiB/jobs overrides are strict; malformed values or a byte
// budget larger than the physical pool reject initialization. Disabled callers
// must not parse these settings or create publication resources.
std::optional<DecodePublicationLimits> remoteCacheDecodePublicationLimits(size_t native_pool_bytes);

// Returns 0 when the environment variable is unset. Invalid or out-of-range
// values return nullopt so backend initialization can fail explicitly.
std::optional<double> remoteCacheQuarantineAbortRatio();

}  // namespace rtp_llm
