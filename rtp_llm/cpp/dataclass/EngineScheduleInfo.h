#pragma once

#include <vector>
#include <string>
#include "rtp_llm/cpp/utils/PyUtils.h"

namespace rtp_llm {

struct EngineScheduleInfo {
    struct TaskInfo {
        int64_t request_id;
        int64_t inter_request_id;
        int     prefix_length;
        int     input_length;
        int64_t waiting_time_ms;
        size_t  iterate_count = 0;
        int64_t end_time_ms   = -1;
    };
    std::vector<TaskInfo> running_task_info_list;
    std::vector<TaskInfo> finished_task_info_list;
    int64_t               last_schedule_delta;
};

void registerEngineScheduleInfo(const pybind11::module& m);

}  // namespace rtp_llm
