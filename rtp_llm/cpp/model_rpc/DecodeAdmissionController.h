#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>

namespace rtp_llm {

class DecodeAdmissionController {
public:
    enum class AcquireResult {
        ACQUIRED,
        CANCELLED,
        TIMED_OUT,
        OVERSIZED,
    };

    explicit DecodeAdmissionController(size_t limit = 1): limit_(std::max<size_t>(limit, 1)) {}

    void setLimit(size_t limit) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            limit_ = std::max<size_t>(limit, 1);
        }
        condition_.notify_all();
    }

    AcquireResult acquire(size_t slots, const std::function<bool()>& cancelled, int64_t timeout_ms) {
        slots = std::max<size_t>(slots, 1);
        std::unique_lock<std::mutex> lock(mutex_);
        // Precondition check, not a servable outcome: a request needing more slots
        // than the whole limit can never satisfy the wait below, so it would spin
        // to the deadline instead of failing. DecodeRpcServer always asks for one
        // slot, so this only fires on a future caller that miscounts.
        if (slots > limit_) {
            return AcquireResult::OVERSIZED;
        }

        const bool has_deadline = timeout_ms >= 0;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max<int64_t>(0, timeout_ms));
        while (true) {
            if (cancelled && cancelled()) {
                return AcquireResult::CANCELLED;
            }
            if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
                return AcquireResult::TIMED_OUT;
            }
            if (active_slots_ + slots <= limit_) {
                active_slots_ += slots;
                return AcquireResult::ACQUIRED;
            }

            if (has_deadline) {
                condition_.wait_until(lock, std::min(deadline, std::chrono::steady_clock::now() + kCancelPollInterval));
            } else {
                condition_.wait_for(lock, kCancelPollInterval);
            }
        }
    }

    size_t activeSlots() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_slots_;
    }

    size_t limit() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return limit_;
    }

private:
    friend class DecodeAdmissionGuard;

    void release(size_t slots) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_slots_ -= std::min(active_slots_, std::max<size_t>(slots, 1));
        }
        condition_.notify_all();
    }

private:
    static constexpr std::chrono::milliseconds kCancelPollInterval{50};

    mutable std::mutex      mutex_;
    std::condition_variable condition_;
    size_t                  limit_        = 1;
    size_t                  active_slots_ = 0;
};

class DecodeAdmissionGuard {
public:
    DecodeAdmissionGuard(DecodeAdmissionController& controller, size_t slots):
        controller_(&controller), slots_(std::max<size_t>(slots, 1)) {}

    ~DecodeAdmissionGuard() {
        if (controller_ != nullptr) {
            controller_->release(slots_);
        }
    }

    DecodeAdmissionGuard(const DecodeAdmissionGuard&)            = delete;
    DecodeAdmissionGuard& operator=(const DecodeAdmissionGuard&) = delete;

private:
    DecodeAdmissionController* controller_;
    size_t                     slots_;
};

}  // namespace rtp_llm
