// Test-only access to the same pure timing library used by C++ callers.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "aios/navi_sched/common/TPUPrefillPP4Timing.h"

namespace py = pybind11;

PYBIND11_MODULE(_navi_sched_pp4_timing_test, module)
{
    using namespace navi_sched;
    py::class_<TPUPrefillPipelineSlot>(module, "Slot")
        .def(py::init<>())
        .def_readwrite("ordinal", &TPUPrefillPipelineSlot::ordinal)
        .def_readwrite("dispatch_time_ms", &TPUPrefillPipelineSlot::dispatchTimeMs)
        .def_readwrite("features", &TPUPrefillPipelineSlot::features);
    py::class_<TPUPrefillPP4Timing>(module, "Timing")
        .def(
            py::init<TPUPrefillPP4Timing::Parameters, std::vector<TPUPrefillPipelineSlot>,
                     TPUPrefillPP4Timing::ReplyTimes, TPUPrefillPP4Timing::ReplyTimes>())
        .def("append_slot", &TPUPrefillPP4Timing::appendSlot,
             py::call_guard<py::gil_scoped_release>())
        .def("step_cost_ms", &TPUPrefillPP4Timing::stepCostMs,
             py::call_guard<py::gil_scoped_release>())
        .def("last_stage_reply_time_ms", &TPUPrefillPP4Timing::lastStageReplyTimeMs,
             py::call_guard<py::gil_scoped_release>())
        .def("result_ready_time_ms", &TPUPrefillPP4Timing::resultReadyTimeMs,
             py::call_guard<py::gil_scoped_release>());
}
