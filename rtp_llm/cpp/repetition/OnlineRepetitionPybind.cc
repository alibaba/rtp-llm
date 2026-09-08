#include "rtp_llm/cpp/repetition/TokenToolCallLoopGuard.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(libonline_repetition_tracker, m) {
    using namespace rtp_llm;

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
