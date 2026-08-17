#pragma once

#include <vector>
#include <string>

namespace rtp_llm {

enum class TaskPhase {
    PENDING      = 0,
    RECEIVED     = 1,
    KV_ALLOCATED = 2,
    RUNNING      = 3,
};

// Priority-preemption progress is orthogonal to TaskPhase. CANCELING means
// the Prefill control owner accepted the cancel intent; resources are still
// accounted. CANCELED is published only after the request execution and
// resource cleanup paths have completed.
enum class PriorityPreemptionProgress {
    NONE      = 0,
    CANCELING = 1,
    CANCELED  = 2,
};

struct EngineScheduleInfo {
    struct TaskInfo {
        int64_t                    request_id;
        int64_t                    prefix_length;
        int64_t                    input_length;
        int64_t                    waiting_time_ms;
        int64_t                    iterate_count = 0;
        int64_t                    end_time_ms   = -1;
        TaskPhase                  phase         = TaskPhase::PENDING;
        PriorityPreemptionProgress priority_preemption_progress = PriorityPreemptionProgress::NONE;
        int64_t                    error_code    = 0;
        std::string                error_message;
        int64_t                    batch_id          = -1;
        int64_t                    execution_time_ms = -1;
    };
    std::vector<TaskInfo> running_task_info_list;
    std::vector<TaskInfo> finished_task_info_list;
    int64_t               last_schedule_delta;
    int64_t               latest_finished_version = 0;
};

}  // namespace rtp_llm
