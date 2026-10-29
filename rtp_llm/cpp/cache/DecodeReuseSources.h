#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/utils/DecodeCacheOrigin.h"

namespace rtp_llm {
struct DecodeReuseSource {
    std::shared_ptr<const DecodeCacheOrigin> origin;
    Tier                                     tier{Tier::NONE};
};
using DecodeReuseSources    = std::vector<DecodeReuseSource>;
using DecodeReuseSourcesPtr = std::shared_ptr<const DecodeReuseSources>;
struct DecodeReuseCounts {
    std::array<int64_t, 4> by_tier{};
    int64_t                total() const {
        return by_tier[0] + by_tier[1] + by_tier[2] + by_tier[3];
    }
};

// Union D-computed intervals clipped at P's adopted prefix. Overlapping groups,
// CP replicas and repeated descriptors count once. Attribute a mixed-tier
// overlap to its coldest source so per-tier totals also sum to the union.
inline DecodeReuseCounts countDecodeReuse(const DecodeReuseSources& sources, int64_t adopted_end) {
    struct Event {
        int64_t position;
        int     tier;
        int     delta;
    };
    std::vector<Event> events;
    for (const auto& value : sources) {
        if (!value.origin || value.tier < Tier::DEVICE || value.tier >= Tier::NONE)
            continue;
        const auto& origin = *value.origin;
        const auto  begin  = std::max(origin.begin, origin.handoff_end);
        const auto  end    = std::min(origin.end, adopted_end);
        if (origin.begin < 0 || origin.handoff_end < 0 || origin.begin >= origin.end || origin.end > origin.computed_end
            || origin.handoff_end > origin.computed_end || origin.computed_end == std::numeric_limits<int64_t>::max()
            || origin.sampled_end != origin.computed_end + 1 || end <= begin)
            continue;
        const auto tier = static_cast<int>(value.tier);
        events.push_back({begin, tier, 1});
        events.push_back({end, tier, -1});
    }
    std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) { return a.position < b.position; });
    DecodeReuseCounts      counts;
    std::array<int64_t, 4> active{};
    int64_t                previous = 0;
    for (size_t i = 0; i < events.size();) {
        const auto position = events[i].position;
        for (int tier = 3; tier >= 0; --tier) {
            if (active[static_cast<size_t>(tier)] != 0) {
                counts.by_tier[static_cast<size_t>(tier)] += position - previous;
                break;
            }
        }
        do {
            active[static_cast<size_t>(events[i].tier)] += events[i].delta;
            ++i;
        } while (i < events.size() && events[i].position == position);
        previous = position;
    }
    return counts;
}

// Request-owned candidates are independent of remote-restore timing. Only a
// successful P context with existing all-rank GPU proof consumes them once.
class DecodeReuseUseState {
public:
    void reset() {
        sources_.reset();
        adopted_end_ = 0;
        consumed_    = false;
    }
    void adopt(DecodeReuseSourcesPtr sources, int64_t adopted_end) {
        if (consumed_) {
            sources_.reset();
            return;
        }
        sources_     = std::move(sources);
        adopted_end_ = adopted_end;
    }
    bool pending() const {
        return !consumed_ && sources_ && !sources_->empty();
    }
    std::optional<DecodeReuseCounts> consume(bool all_rank_gpu_ready) {
        if (!all_rank_gpu_ready || !pending())
            return std::nullopt;
        auto result = countDecodeReuse(*sources_, adopted_end_);
        consumed_   = true;
        sources_.reset();
        return result;
    }

private:
    DecodeReuseSourcesPtr sources_;
    int64_t               adopted_end_{0};
    bool                  consumed_{false};
};
}  // namespace rtp_llm
