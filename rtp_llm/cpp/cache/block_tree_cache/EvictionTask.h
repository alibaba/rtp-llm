#pragma once

#include <cstdint>
#include <string>

namespace rtp_llm {

// EvictionTask state machine for tracking async eviction lifecycle.
// Phase 3: PENDING -> RUNNING -> COMPLETED/FAILED
// On FAILED: rollback restores source tier heap and frees target block.
enum class EvictionTaskState : int8_t {
    PENDING   = 0,  // Task created, not yet started
    RUNNING   = 1,  // Copy in progress (no lock held)
    COMPLETED = 2,  // Copy succeeded, onEvictionComplete done
    FAILED    = 3,  // Copy failed, rollback executed
};

inline const char* evictionTaskStateName(EvictionTaskState state) {
    switch (state) {
        case EvictionTaskState::PENDING:
            return "PENDING";
        case EvictionTaskState::RUNNING:
            return "RUNNING";
        case EvictionTaskState::COMPLETED:
            return "COMPLETED";
        case EvictionTaskState::FAILED:
            return "FAILED";
    }
    return "UNKNOWN";
}

struct EvictionTask {
    EvictionTaskState state{EvictionTaskState::PENDING};
    std::string       error_message;

    // State transition validation
    bool canTransition(EvictionTaskState from, EvictionTaskState to) const {
        switch (from) {
            case EvictionTaskState::PENDING:
                return to == EvictionTaskState::RUNNING;
            case EvictionTaskState::RUNNING:
                return to == EvictionTaskState::COMPLETED || to == EvictionTaskState::FAILED;
            case EvictionTaskState::FAILED:
                return to == EvictionTaskState::PENDING;  // retry
            case EvictionTaskState::COMPLETED:
                return false;  // terminal state
        }
        return false;
    }

    // Transition with validation
    bool transition(EvictionTaskState new_state) {
        if (!canTransition(state, new_state))
            return false;
        state = new_state;
        return true;
    }

    bool isTerminal() const {
        return state == EvictionTaskState::COMPLETED || state == EvictionTaskState::FAILED;
    }
};

}  // namespace rtp_llm
