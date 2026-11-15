#pragma once

#include <atomic>

namespace rtp_llm::kvcm {

// Source BUFFER_SAFE is deliberately absent: it cannot prove destination untouched.
class PayloadDispatchProof {
public:
    enum class State { NOT_HANDED_OFF, HANDOFF_PENDING, PROVEN_NOT_STARTED, MAY_HAVE_STARTED };
    void pending() noexcept {
        auto expected = State::NOT_HANDED_OFF;
        state_.compare_exchange_strong(expected, State::HANDOFF_PENDING);
    }
    void started() noexcept { state_.store(State::MAY_HAVE_STARTED); }
    // Single-carrier expire/reject, or a local peer latch after ALL carriers
    // are terminal. Never infer this from one peer while another can start.
    void notStarted() noexcept {
        auto expected = State::HANDOFF_PENDING;
        state_.compare_exchange_strong(expected, State::PROVEN_NOT_STARTED);
    }
    bool neverStarted() const noexcept {
        const auto state = state_.load();
        return state == State::NOT_HANDED_OFF || state == State::PROVEN_NOT_STARTED;
    }
    State state() const noexcept { return state_.load(); }
private:
    std::atomic<State> state_{State::NOT_HANDED_OFF};
};

}  // namespace rtp_llm::kvcm
