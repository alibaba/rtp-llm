#pragma once

#include <chrono>
#include <list>
#include <memory>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "autil/EnvUtil.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"

namespace rtp_llm {

class SchedulerBase {
public:
    virtual ~SchedulerBase() {}
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

protected:
    static std::chrono::milliseconds readPollInterval() {
        constexpr int default_interval_ms = 10;
        const int     interval_ms = autil::EnvUtil::getEnv("RTP_LLM_SCHEDULER_POLL_INTERVAL_MS", default_interval_ms);
        // A nonpositive timeout would turn idle collective polling into a busy loop.
        return std::chrono::milliseconds(interval_ms > 0 ? interval_ms : default_interval_ms);
    }

    // Read once at construction; notifications still wake a timed wait immediately.
    const std::chrono::milliseconds poll_interval_ = readPollInterval();
};

}  // namespace rtp_llm
