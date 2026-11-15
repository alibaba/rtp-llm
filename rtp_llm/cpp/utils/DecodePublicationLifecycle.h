#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

namespace rtp_llm {

// Scheduler/stream-lock-owned metadata. No payload, heap allocation or timer.
// Preserve the actual GenerateDone event across delayed planning/fallbacks.
class DecodePublicationLifecycle {
public:
    using Clock = std::chrono::steady_clock;
    enum class Admission { ADMITTED, DECLINED, ALREADY_ADMITTED, IN_PROGRESS, EXHAUSTED };

    // Called synchronously by the scheduler/stream-lock owner. Preparing closes
    // reentrant capture, not a cross-thread lock. No I/O callback owns this state.
    // Admission is permanent even if that task later fails or becomes UNKNOWN.
    template<typename Submit>
    Admission tryAdmit(bool& submitted, Submit&& submit, Clock::time_point now = Clock::now()) const {
        if (submitted)
            return Admission::ALREADY_ADMITTED;
        if (preparing_)
            return Admission::IN_PROGRESS;
        if (admission_attempts_ >= max_admission_attempts_ || (admission_deadline_ && now >= *admission_deadline_))
            return Admission::EXHAUSTED;
        ++admission_attempts_;
        preparing_ = true;
        try {
            const bool admitted = std::forward<Submit>(submit)();
            submitted = admitted;
            preparing_ = false;
            return admitted ? Admission::ADMITTED : Admission::DECLINED;
        } catch (...) {
            preparing_ = false;
            throw;
        }
    }
    // Stream-owned query only. tryAdmit remains authoritative and consumes
    // attempts; this never reserves an attempt or changes the EOS deadline.
    bool mayAttempt(bool submitted, Clock::time_point now = Clock::now()) const {
        return !submitted && !preparing_ && admission_attempts_ < max_admission_attempts_
               && (!admission_deadline_ || now < *admission_deadline_);
    }
    size_t admissionAttempts() const {
        return admission_attempts_;
    }
    // One budget for update and release fallback. Never restart it on retry.
    Clock::time_point admissionDeadline(std::chrono::milliseconds timeout,
                                        Clock::time_point now = Clock::now()) const {
        if (!admission_deadline_)
            admission_deadline_ = eos_at_.value_or(now) + timeout;
        return *admission_deadline_;
    }
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
        preparing_ = false;
        admission_attempts_ = 0;
        admission_deadline_.reset();
    }

private:
    mutable std::optional<Clock::time_point> eos_at_;
    mutable std::optional<int64_t>           prefill_computed_end_;
    mutable bool                             budget_skip_reported_{false};
    // At most the primary update attempt and one release-time fallback.
    static constexpr size_t                  max_admission_attempts_{2};
    mutable bool                             preparing_{false};
    mutable size_t                           admission_attempts_{0};
    mutable std::optional<Clock::time_point>  admission_deadline_;
};

}  // namespace rtp_llm
