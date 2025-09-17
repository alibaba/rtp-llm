#include "rtp_llm/cpp/dataclass/WorkerStatusInfo.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

void registerWorkerStatusInfo(const py::module& m) {
    pybind11::class_<WorkerStatusInfo>(m, "WorkerStatusInfo")
        .def(pybind11::init<>())
        .def_readwrite("role", &WorkerStatusInfo::role)
        .def_readwrite("engine_schedule_info", &WorkerStatusInfo::engine_schedule_info)
        .def_readwrite("status_version", &WorkerStatusInfo::status_version)
        .def_readwrite("alive", &WorkerStatusInfo::alive)
        .def_readwrite("dp_size", &WorkerStatusInfo::dp_size)
        .def_readwrite("tp_size", &WorkerStatusInfo::tp_size)
        .def_readwrite("dp_rank", &WorkerStatusInfo::dp_rank)
        .def_readwrite("precision", &WorkerStatusInfo::precision);
}

}  // namespace rtp_llm
