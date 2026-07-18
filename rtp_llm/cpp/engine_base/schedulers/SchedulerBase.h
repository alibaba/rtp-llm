#pragma once

#include <list>
#include <vector>
#include <memory>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"

namespace rtp_llm {

class SchedulerBase {
public:
    virtual ~SchedulerBase() {}
    virtual absl::Status                   enqueue(const GenerateStreamPtr& stream)                    = 0;
    virtual std::vector<GenerateStreamPtr> enqueueGroup(const std::vector<GenerateStreamPtr>& streams) = 0;
    virtual absl::StatusOr<std::list<GenerateStreamPtr>> schedule()                                    = 0;

    // Conservative-KV scheduling variant for async execution. The async path
    // schedules step N+1 before step N's specUpdate has run, so seq_len is not
    // yet authoritative. Conservative variants reserve the maximum possible
    // accept_len (propose_step + 1), then release surplus blocks once the real
    // accept_len is known.
    virtual absl::StatusOr<std::list<GenerateStreamPtr>> scheduleConservative(int /*propose_step*/) {
        return schedule();
    }
    virtual absl::Status stop()             = 0;
    virtual bool         empty()            = 0;
    virtual int64_t      lastScheduleTime() = 0;
    virtual int64_t      onflightStreams()  = 0;

    virtual std::vector<EngineScheduleInfo::TaskInfo> waitingTaskList() {
        return {};
    }
    virtual std::vector<EngineScheduleInfo::TaskInfo> runningTaskList() {
        return {};
    }
    virtual void updateSchedulerInfo(const std::string& scheduler_info) {}
};

}  // namespace rtp_llm
