#pragma once

#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"

namespace rtp_llm {

// Controller-only tests supply the same ledger that a real SchedulerBase owns.
// Production never creates a controller-private fallback ledger.
class BoundSleepLifecycleController: public SleepLifecycleController {
public:
    explicit BoundSleepLifecycleController(bool enabled = false): SleepLifecycleController(enabled) {
        bindAdmission(std::make_shared<SchedulerAdmission>());
    }
};

inline bool admitAndComplete(SchedulerAdmission::Result result) {
    const bool accepted = result.accepted;
    if (result.complete) {
        result.complete();
    }
    return accepted;
}

}  // namespace rtp_llm
