#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CacheTier.h"

namespace rtp_llm {

enum class VictimSelectionTrigger : uint8_t {
    WATERMARK,
    ON_DEMAND,
};

enum class VictimSkipReason : uint8_t {
    CACHED_REJECTION,
    REQUEST_REF,
    FROZEN,
    NOT_RECLAIMABLE,
    CHILD_LOADING,
    INVALID_CANDIDATE,
    COUNT,
};

inline constexpr std::array<const char*, static_cast<size_t>(VictimSkipReason::COUNT)> kVictimSkipReasonNames = {
    "cached_rejection", "request_ref", "frozen", "not_reclaimable", "child_loading", "invalid_candidate"};

struct VictimSelectionCounters {
    uint64_t                                                           calls{0};
    uint64_t                                                           selected{0};
    uint64_t                                                           latency_sum_ns{0};
    uint64_t                                                           scanned_candidates{0};
    std::array<uint64_t, static_cast<size_t>(VictimSkipReason::COUNT)> skipped{};
    bool                                                               observed{false};
};

// Fixed cardinality: trigger x local tier x group type x drop/demote mode.
// All access is protected by the existing BlockTreeCache mutex, including take().
struct VictimSelectionMetrics {
    static constexpr size_t                           kBucketCount = 2 * 3 * 3 * 2;
    std::array<VictimSelectionCounters, kBucketCount> buckets{};

    static constexpr size_t index(VictimSelectionTrigger trigger, Tier tier, CacheGroupType group_type, bool drop) {
        return ((static_cast<size_t>(trigger) * 3 + static_cast<size_t>(tier)) * 3 + static_cast<size_t>(group_type))
                   * 2
               + static_cast<size_t>(drop);
    }

    VictimSelectionCounters& at(VictimSelectionTrigger trigger, Tier tier, CacheGroupType group_type, bool drop) {
        return buckets[index(trigger, tier, group_type, drop)];
    }

    const VictimSelectionCounters&
    at(VictimSelectionTrigger trigger, Tier tier, CacheGroupType group_type, bool drop) const {
        return buckets[index(trigger, tier, group_type, drop)];
    }

    VictimSelectionMetrics take() noexcept {
        const auto snapshot = *this;
        for (auto& counters : buckets) {
            const bool observed = counters.observed;
            counters            = {};
            // Continue reporting zero deltas for previously active series during idle periods.
            counters.observed = observed;
        }
        return snapshot;
    }
};

}  // namespace rtp_llm
