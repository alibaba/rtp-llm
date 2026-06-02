#pragma once

#include <vector>
#include <string>

namespace rtp_llm {

struct EngineScheduleInfo {
    struct TaskInfo {
        int64_t request_id;
        int64_t prefix_length;
        int64_t input_length;
        int64_t waiting_time_ms;
        int64_t iterate_count = 0;
        int64_t end_time_ms   = -1;
        bool    is_waiting    = true;
    };
    std::vector<TaskInfo> running_task_info_list;
    std::vector<TaskInfo> finished_task_info_list;
    int64_t               last_schedule_delta;
    int64_t               latest_finished_version = 0;
};

}  // namespace rtp_llm
