#pragma once

#include <list>
#include <memory>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/engine_base/schedulers/EngineScheduleInfo.h"

namespace rtp_llm {

// Snapshot of the actual admission path, never a speculative readiness probe.
struct ScheduleObservation {
    bool    valid          = false;
    bool    intent_prefill = false;
    int64_t waiting = 0, loading = 0, running = 0, pending = 0;
    int64_t kv_available = 0, kv_reserved = 0;
    int64_t rejected_batch = 0, rejected_tokens = 0, rejected_kv = 0;
    int64_t committed_prefill = 0, committed_decode = 0, committed_input_tokens = 0;
    int64_t waiting_after = 0, loading_after = 0;
    int64_t oldest_waiting_us = 0;
};

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
    virtual ScheduleObservation lastScheduleObservation() {
        return {};
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
