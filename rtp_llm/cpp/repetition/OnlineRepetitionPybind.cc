#include "rtp_llm/cpp/repetition/OnlineRepetitionTracker.h"
#include "rtp_llm/cpp/repetition/TokenToolCallLoopGuard.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(libonline_repetition_tracker, m) {
    using namespace rtp_llm;

    py::class_<OnlineRepetitionConfig>(m, "OnlineRepetitionConfig")
        .def(py::init<>())
        .def_readwrite("min_repeats", &OnlineRepetitionConfig::min_repeats)
        .def_readwrite("min_duplicate_tokens", &OnlineRepetitionConfig::min_duplicate_tokens)
        .def_readwrite("max_period", &OnlineRepetitionConfig::max_period)
        .def_readwrite("non_contiguous_min_span", &OnlineRepetitionConfig::non_contiguous_min_span)
        .def_readwrite("non_contiguous_min_occurrences", &OnlineRepetitionConfig::non_contiguous_min_occurrences)
        .def_readwrite("non_contiguous_max_span", &OnlineRepetitionConfig::non_contiguous_max_span);

    py::class_<OnlineRepetitionResult>(m, "OnlineRepetitionResult")
        .def_readonly("hit", &OnlineRepetitionResult::hit)
        .def_readonly("repeat_unit_size", &OnlineRepetitionResult::repeat_unit_size)
        .def_readonly("repeat_count", &OnlineRepetitionResult::repeat_count)
        .def_readonly("partial_tail_tokens", &OnlineRepetitionResult::partial_tail_tokens)
        .def_readonly("covered_token_count", &OnlineRepetitionResult::covered_token_count)
        .def_readonly("duplicate_token_count", &OnlineRepetitionResult::duplicate_token_count)
        .def_readonly("start_index", &OnlineRepetitionResult::start_index)
        .def_readonly("end_index", &OnlineRepetitionResult::end_index)
        .def_readonly("first_detect_index", &OnlineRepetitionResult::first_detect_index)
        .def_readonly("non_contiguous", &OnlineRepetitionResult::non_contiguous)
        .def_readonly("occurrence_count", &OnlineRepetitionResult::occurrence_count);

    py::class_<OnlineRepetitionTracker>(m, "OnlineRepetitionTracker")
        .def(py::init<OnlineRepetitionConfig>())
        .def("reset", &OnlineRepetitionTracker::reset)
        .def("update_many",
             [](OnlineRepetitionTracker& tracker, const std::vector<int>& token_ids) {
                 py::gil_scoped_release release;
                 return tracker.updateMany(token_ids);
             })
        .def("finalize", &OnlineRepetitionTracker::considerFinalTail)
        .def_property_readonly("result", [](const OnlineRepetitionTracker& tracker) { return tracker.result(); })
        .def_property_readonly("token_count", &OnlineRepetitionTracker::tokenCount);

    m.def(
        "check_tool_call_loop",
        [](const std::vector<int>&              input_ids,
           const std::vector<int>&              output_ids,
           const std::vector<std::vector<int>>& marker_begin_ids,
           const std::vector<std::vector<int>>& marker_end_ids,
           int                                  repeat_threshold,
           int                                  max_span_tokens) {
            ToolCallLoopCheckResult result;
            {
                py::gil_scoped_release release;
                result = checkToolCallLoop(
                    input_ids, output_ids, marker_begin_ids, marker_end_ids, repeat_threshold, max_span_tokens);
            }
            return py::make_tuple(result.hit, result.repeat_count, result.current_span_tokens, result.marker_index);
        },
        py::arg("input_ids"),
        py::arg("output_ids"),
        py::arg("marker_begin_ids"),
        py::arg("marker_end_ids"),
        py::arg("repeat_threshold") = 5,
        py::arg("max_span_tokens")  = 16384);
}
