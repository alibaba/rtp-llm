#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace rtp_llm {

// Scheduler-owned admission ledger. Callers receive an execution identity and
// a completion notification, never ownership of a lease. Notifications are
// idempotent and remain safe after the scheduler has been destroyed.
class SchedulerAdmission {
public:
    struct Result {
        bool                  accepted     = false;
        uint64_t              execution_id = 0;
        std::function<void()> complete;
    };

    SchedulerAdmission();
    Result admit(bool continuation = false);
    void   closeRoots();
    void   sealContinuations();
    void   beginTermination();
    bool   reopen();
    bool   rootsOpen() const;
    bool   continuationsSealed() const;
    bool   terminating() const;
    size_t activeCount() const;

private:
    // The lease never leaves this ledger. Its presence, rather than a second
    // independent counter, is the authoritative active-admission accounting.
    struct AdmissionLease {};
    struct State {
        mutable std::mutex                           mutex;
        bool                                         roots_open         = true;
        bool                                         continuations_open = true;
        bool                                         terminating        = false;
        uint64_t                                     next_id            = 1;
        std::unordered_map<uint64_t, AdmissionLease> active;
    };
    std::shared_ptr<State> state_;
};

}  // namespace rtp_llm
