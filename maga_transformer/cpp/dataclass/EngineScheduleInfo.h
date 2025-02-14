#pragma once

#include <vector>
#include <string>
#include "maga_transformer/cpp/utils/PyUtils.h"

namespace rtp_llm {

struct EngineScheduleInfo {
    struct TaskInfo {
        int64_t request_id;
        int     prefix_length;
        int     input_length;
    };
    std::vector<TaskInfo> running_task_info_list;
    std::vector<TaskInfo> finished_task_info_list;
    int64_t               last_schedule_delta;
};

void registerEngineScheduleInfo(const pybind11::module& m);

}  // namespace rtp_llm
