#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <string>

namespace navi_sched {

// Fixed stage names keep abort telemetry free of request-level cardinality.
// Counters record stage events, so delivery retries can repeat request IDs.
// vllm_applied includes every FINISHED_ABORTED lifecycle event, including local ones.
class AbortMetrics {
public:
    static constexpr size_t STAGE_COUNT = 7;
    AbortMetrics()
    {
        for (auto &count : _counts) {
            count.store(0, std::memory_order_relaxed);
        }
    }
    static const std::array<const char *, STAGE_COUNT> &stages()
    {
        static const std::array<const char *, STAGE_COUNT> names = {
            "received", "mapped", "deferred", "enqueued", "enqueue_failed",
            "engine_unbound", "vllm_applied"};
        return names;
    }
    bool record(const std::string &stage, uint64_t count)
    {
        for (size_t i = 0; i < STAGE_COUNT; ++i) {
            if (stage == stages()[i]) {
                _counts[i].fetch_add(count, std::memory_order_relaxed);
                return true;
            }
        }
        return false;
    }
    uint64_t count(size_t stage) const
    {
        return _counts.at(stage).load(std::memory_order_relaxed);
    }
private:
    std::array<std::atomic<uint64_t>, STAGE_COUNT> _counts{};
};

}
