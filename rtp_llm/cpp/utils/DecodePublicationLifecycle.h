#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace rtp_llm {

// Scheduler/stream-lock-owned metadata. No payload, heap allocation or timer.
// Preserve the actual GenerateDone event across delayed planning/fallbacks.
class DecodePublicationLifecycle {
public:
    using Clock = std::chrono::steady_clock;
    void recordEos(Clock::time_point at = Clock::now()) const {
        if (!eos_at_)
            eos_at_ = at;
    }
    const std::optional<Clock::time_point>& eosAt() const {
        return eos_at_;
    }
    // Set by the validated P-to-D handoff, never inferred from D's input length.
    bool recordPrefillComputedEnd(int64_t end) const {
        if (end < 0 || (prefill_computed_end_ && *prefill_computed_end_ != end)) {
            return false;
        }
        prefill_computed_end_ = end;
        return true;
    }
    const std::optional<int64_t>& prefillComputedEnd() const {
        return prefill_computed_end_;
    }
    bool recordFirstBudgetSkip() const {
        if (budget_skip_reported_)
            return false;
        budget_skip_reported_ = true;
        return true;
    }
    void reset() {
        eos_at_.reset();
        prefill_computed_end_.reset();
        budget_skip_reported_ = false;
    }

private:
    mutable std::optional<Clock::time_point> eos_at_;
    mutable std::optional<int64_t>           prefill_computed_end_;
    mutable bool                             budget_skip_reported_{false};
};

}  // namespace rtp_llm
