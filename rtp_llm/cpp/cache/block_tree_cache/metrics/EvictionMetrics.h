#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/CacheTier.h"

namespace rtp_llm {

struct CandidateMeta;

struct EvictionTimingMetrics {
    EvictionTimingMetrics() = default;
    explicit EvictionTimingMetrics(const CandidateMeta& candidate_meta);

    int64_t tier_enter_time_us{0};
    int64_t last_access_time_us{0};
    int64_t selected_time_us{0};
};

enum class EvictionVictimTrigger : uint8_t {
    WATERMARK,
    FORCE_DROP,
};

enum class EvictionVictimSkipReason : uint8_t {
    CACHED_REJECTION,
    REQUEST_REF,
    FROZEN,
    NOT_RECLAIMABLE,
    CHILD_LOADING,
    INVALID_CANDIDATE,
    COUNT,
};

inline constexpr std::array<const char*, static_cast<size_t>(EvictionVictimSkipReason::COUNT)>
    kEvictionVictimSkipReasonNames = {
        "cached_rejection", "request_ref", "frozen", "not_reclaimable", "child_loading", "invalid_candidate"};

struct EvictionVictimCounters {
    uint64_t                                                                   candidates_sum{0};
    uint64_t                                                                   calls{0};
    uint64_t                                                                   selected{0};
    uint64_t                                                                   latency_sum_ns{0};
    uint64_t                                                                   scanned_candidates{0};
    std::array<uint64_t, static_cast<size_t>(EvictionVictimSkipReason::COUNT)> skipped{};
    bool                                                                       observed{false};
};

// Fixed cardinality: trigger x local tier x group type x drop/demote mode.
// All access is protected by the existing BlockTreeCache mutex, including getAndReset().
struct EvictionVictimMetrics {
    static constexpr size_t                          kBucketCount = 2 * 3 * 3 * 2;
    std::array<EvictionVictimCounters, kBucketCount> buckets{};

    static constexpr size_t index(EvictionVictimTrigger trigger, Tier tier, CacheGroupType group_type, bool drop) {
        return ((static_cast<size_t>(trigger) * 3 + static_cast<size_t>(tier)) * 3 + static_cast<size_t>(group_type))
                   * 2
               + static_cast<size_t>(drop);
    }

    EvictionVictimCounters& at(EvictionVictimTrigger trigger, Tier tier, CacheGroupType group_type, bool drop) {
        return buckets[index(trigger, tier, group_type, drop)];
    }

    const EvictionVictimCounters&
    at(EvictionVictimTrigger trigger, Tier tier, CacheGroupType group_type, bool drop) const {
        return buckets[index(trigger, tier, group_type, drop)];
    }

    EvictionVictimMetrics getAndReset() noexcept {
        const auto metrics_data = *this;
        for (auto& counters : buckets) {
            const bool observed = counters.observed;
            counters            = {};
            // Continue reporting zero deltas for previously active series during idle periods.
            counters.observed = observed;
        }
        return metrics_data;
    }
};

}  // namespace rtp_llm
