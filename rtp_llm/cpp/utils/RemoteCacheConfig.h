#pragma once

#include <optional>
#include <cstddef>

namespace rtp_llm {

bool remoteCacheGdrEnabled();

// One feature switch for every P CP / D DP topology. No legacy or CP-specific
// flags are read. Publication is disabled unless explicitly enabled.
bool remoteCacheDecodePublicationEnabled();

struct DecodePublicationLimits {
    // Configuration-only default for a 4 GiB pool. Backend initialization
    // derives bytes from its actual unique physical pools instead.
    // Budgets cap existing pinned pages, not another GPU allocation.
    size_t bytes = 512ULL * 1024 * 1024;
    size_t jobs  = 4;
};

// Strictly parsed at backend initialization. Overrides are deployment budgets,
// not separate feature switches. Invalid values reject initialization.
// Initial policy: at most 1/8 of the native KV pool, capped at 4 GiB per D
// process, and four pending publications. The default argument supports config
// inspection/tests; the backend always supplies its measured pool capacity.
std::optional<DecodePublicationLimits> remoteCacheDecodePublicationLimits(size_t native_pool_bytes = 4ULL * 1024 * 1024
                                                                                                     * 1024);

// Returns 0 when the environment variable is unset. Invalid or out-of-range
// values return nullopt so backend initialization can fail explicitly.
std::optional<double> remoteCacheQuarantineAbortRatio();

}  // namespace rtp_llm
