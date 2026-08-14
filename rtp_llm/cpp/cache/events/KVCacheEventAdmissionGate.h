#pragma once

#include <atomic>
#include <cstddef>
#include <limits>
#include <thread>
#include <utility>

namespace rtp_llm::detail {

// Admission gate for non-blocking cache-event hot paths. close() and
// tryEnter() share one atomic modification order, so close()+quiesce() cannot
// miss a caller on weakly ordered CPUs. A gate may be a one-way lifetime fence
// or a reusable pause epoch; reusable owners must reopen only after quiesce()
// and use a separate one-way gate when object destruction can race recovery.
// The owner must still keep the gate alive until callers that observed a
// closed gate have returned.
class KVCacheEventAdmissionGate {
public:
    class Guard {
    public:
        Guard() noexcept               = default;
        Guard(const Guard&)            = delete;
        Guard& operator=(const Guard&) = delete;
        Guard(Guard&& other) noexcept: gate_(std::exchange(other.gate_, nullptr)) {}
        Guard& operator=(Guard&&) = delete;

        ~Guard() {
            if (gate_) {
                gate_->leave();
            }
        }

        explicit operator bool() const noexcept {
            return gate_ != nullptr;
        }

    private:
        friend class KVCacheEventAdmissionGate;
        explicit Guard(KVCacheEventAdmissionGate* gate) noexcept: gate_(gate) {}

        KVCacheEventAdmissionGate* gate_{nullptr};
    };

    KVCacheEventAdmissionGate()                                            = default;
    KVCacheEventAdmissionGate(const KVCacheEventAdmissionGate&)            = delete;
    KVCacheEventAdmissionGate& operator=(const KVCacheEventAdmissionGate&) = delete;

    [[nodiscard]] Guard tryEnter() noexcept {
        size_t state = state_.load(std::memory_order_acquire);
        for (;;) {
            if ((state & kClosedBit) != 0) {
                return Guard{};
            }
            // Reaching kClosedBit-1 would make the next admission look closed.
            // This requires roughly half the address space of simultaneous
            // callers and cannot occur in practice, but fail closed rather
            // than wrapping the counter if the invariant is ever violated.
            if (state == kClosedBit - 1) {
                return Guard{};
            }
            if (state_.compare_exchange_weak(state, state + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return Guard{this};
            }
        }
    }

    void close() noexcept {
        state_.fetch_or(kClosedBit, std::memory_order_acq_rel);
    }

    void quiesce() noexcept {
        while ((state_.load(std::memory_order_acquire) & ~kClosedBit) != 0) {
            std::this_thread::yield();
        }
    }

    // Reopen a temporarily paused gate after quiesce(). The lifetime owner
    // must provide a separate one-way gate when destruction can race this
    // operation. False means the gate was not in the exact closed/quiescent
    // state required for a safe new admission epoch.
    bool reopenAfterQuiesce() noexcept {
        size_t expected = kClosedBit;
        return state_.compare_exchange_strong(expected, 0, std::memory_order_release, std::memory_order_relaxed);
    }

    bool closed() const noexcept {
        return (state_.load(std::memory_order_acquire) & kClosedBit) != 0;
    }

private:
    void leave() noexcept {
        state_.fetch_sub(1, std::memory_order_release);
    }

private:
    static constexpr size_t kClosedBit = size_t{1} << (std::numeric_limits<size_t>::digits - 1);
    std::atomic<size_t>     state_{0};
};

}  // namespace rtp_llm::detail
