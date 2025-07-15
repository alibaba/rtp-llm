#include "rtp_llm/cpp/dataclass/EngineScheduleInfo.h"

namespace rtp_llm {
void registerEngineScheduleInfo(const pybind11::module& m) {
    pybind11::class_<EngineScheduleInfo::TaskInfo>(m, "EngineTaskInfo")
        .def(pybind11::init<>())
        .def_readwrite("request_id", &EngineScheduleInfo::TaskInfo::request_id)
        .def_readwrite("inter_request_id", &EngineScheduleInfo::TaskInfo::inter_request_id)
        .def_readwrite("prefix_length", &EngineScheduleInfo::TaskInfo::prefix_length)
        .def_readwrite("input_length", &EngineScheduleInfo::TaskInfo::input_length)
        .def_readwrite("waiting_time_ms", &EngineScheduleInfo::TaskInfo::waiting_time_ms)
        .def_readwrite("iterate_count", &EngineScheduleInfo::TaskInfo::iterate_count)
        .def_readwrite("end_time_ms", &EngineScheduleInfo::TaskInfo::end_time_ms);

    pybind11::class_<EngineScheduleInfo>(m, "EngineScheduleInfo")
        .def(pybind11::init<>())
        .def_readwrite("last_schedule_delta", &EngineScheduleInfo::last_schedule_delta)
        .def_readwrite("finished_task_info_list", &EngineScheduleInfo::finished_task_info_list)
        .def_readwrite("running_task_info_list", &EngineScheduleInfo::running_task_info_list);
}

}  // namespace rtp_llm