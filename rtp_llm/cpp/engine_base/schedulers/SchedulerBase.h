#pragma once

#include <list>
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerAdmission.h"
#include "rtp_llm/cpp/engine_base/sleep/DrainManager.h"

namespace rtp_llm {

class SchedulerBase {
public:
    SchedulerBase(): admission_(std::make_shared<SchedulerAdmission>()) {
        drain_manager_.registerCounter("admission_leases",
                                       [admission = admission_]() { return admission->activeCount(); });
        // Queries only run after construction and must be joined before the
        // scheduler is destroyed. Never evaluate providers under a queue lock.
        drain_manager_.registerCounter(
            "scheduler_onflight", [this]() { return static_cast<size_t>(std::max<int64_t>(0, onflightStreams())); });
    }
    virtual ~SchedulerBase() {}
    std::shared_ptr<SchedulerAdmission> admission() const {
        return admission_;
    }
    DrainManager& drainManager() {
        return drain_manager_;
    }
    virtual absl::Status enqueue(const GenerateStreamPtr& stream) = 0;
    virtual std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueGroup(const std::vector<GenerateStreamPtr>& streams)     = 0;
    virtual absl::StatusOr<std::list<GenerateStreamPtr>> schedule() = 0;

    // Conservative-KV scheduling variant for async execution. The async path
    // schedules step N+1 before step N's specUpdate has run, so seq_len is not
    // yet authoritative. Conservative variants reserve the maximum possible
    // accept_len (propose_step + 1), then release surplus blocks once the real
    // accept_len is known.
    virtual absl::StatusOr<std::list<GenerateStreamPtr>> scheduleConservative(int /*propose_step*/) {
        return schedule();
    }
    virtual absl::Status stop() = 0;
    virtual void         wake() {}
    // When enabled, schedule() must NOT block indefinitely on an empty queue: it polls with a
    // short timeout so empty peers can match busy peers during all-rank sleep drain
    // and catch up to the common stopping round. The engine parks at the round fence
    // before scheduling again; wake/cancel clears forced polling. No-op by default.
    virtual void    setForcePoll(bool /*enable*/) {}
    virtual bool    empty()            = 0;
    virtual int64_t lastScheduleTime() = 0;
    virtual int64_t onflightStreams()  = 0;

    virtual std::vector<EngineScheduleInfo::TaskInfo> waitingTaskList() {
        return {};
    }
    virtual std::vector<EngineScheduleInfo::TaskInfo> runningTaskList() {
        return {};
    }
    virtual void updateSchedulerInfo(const std::string& scheduler_info) {}

private:
    std::shared_ptr<SchedulerAdmission> admission_;
    DrainManager                        drain_manager_;
};

}  // namespace rtp_llm
