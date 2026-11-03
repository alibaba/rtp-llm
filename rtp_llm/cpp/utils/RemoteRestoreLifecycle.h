#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace rtp_llm {

struct RemoteRestoreTiming {
    using Clock = std::chrono::steady_clock;
    Clock::time_point read_started_at;
    Clock::time_point gpu_ready_at;
};

// Stream-lock-owned metadata only. No payload, allocation, timer or device sync.
// A backend read is not actual use. Preserve its clock through adoption until
// the existing root context update is observed, once per logical request. This
// CPU boundary does not itself prove every CP rank's GPU tail has completed.
class RemoteRestoreLifecycle {
public:
    using Clock = RemoteRestoreTiming::Clock;

    bool adopt(const RemoteRestoreTiming& timing, std::optional<Clock::time_point> attempt_started_at = std::nullopt) {
        reset();
        if (timing.gpu_ready_at < timing.read_started_at
            || (attempt_started_at && *attempt_started_at > timing.read_started_at)) {
            return false;
        }
        timing_             = timing;
        attempt_started_at_ = attempt_started_at.value_or(timing.read_started_at);
        return true;
    }

    bool pendingForward() const {
        return timing_.has_value() && !forward_reported_;
    }

    std::optional<int64_t> checkpointReadyUs() const {
        if (!timing_) {
            return std::nullopt;
        }
        return elapsed(timing_->gpu_ready_at);
    }

    std::optional<int64_t> readToCheckpointUs() const {
        if (!timing_) {
            return std::nullopt;
        }
        return std::chrono::duration_cast<std::chrono::microseconds>(timing_->gpu_ready_at - timing_->read_started_at)
            .count();
    }

    std::optional<int64_t> completeContextForward(Clock::time_point at = Clock::now()) {
        if (!pendingForward() || at < timing_->gpu_ready_at) {
            return std::nullopt;
        }
        forward_reported_ = true;
        return elapsed(at);
    }

    void reset() {
        timing_.reset();
        attempt_started_at_.reset();
        forward_reported_ = false;
    }

private:
    int64_t elapsed(Clock::time_point at) const {
        return std::chrono::duration_cast<std::chrono::microseconds>(at - *attempt_started_at_).count();
    }
    std::optional<RemoteRestoreTiming> timing_;
    std::optional<Clock::time_point>   attempt_started_at_;
    bool                               forward_reported_{false};
};

}  // namespace rtp_llm
