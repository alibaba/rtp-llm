#include "rtp_llm/cpp/repetition/OnlineRepetitionTracker.h"
#include "rtp_llm/cpp/repetition/TokenToolCallLoopGuard.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace rtp_llm {
namespace {

OnlineRepetitionConfig makeConfig(int min_repeats, int min_duplicate_tokens, int max_period) {
    OnlineRepetitionConfig config;
    config.min_repeats = min_repeats;
    config.min_duplicate_tokens = min_duplicate_tokens;
    config.max_period = max_period;
    return config;
}

}  // namespace
}  // namespace rtp_llm

PYBIND11_MODULE(libonline_repetition_tracker, m) {
    using namespace rtp_llm;

    py::class_<OnlineRepetitionConfig>(m, "OnlineRepetitionConfig")
        .def(py::init<>())
        .def_readwrite("min_repeats", &OnlineRepetitionConfig::min_repeats)
        .def_readwrite("min_duplicate_tokens", &OnlineRepetitionConfig::min_duplicate_tokens)
        .def_readwrite("max_period", &OnlineRepetitionConfig::max_period);

    py::class_<OnlineRepetitionResult>(m, "OnlineRepetitionResult")
        .def(py::init<>())
        .def_readonly("hit", &OnlineRepetitionResult::hit)
        .def_readonly("repeat_unit_size", &OnlineRepetitionResult::repeat_unit_size)
        .def_readonly("repeat_count", &OnlineRepetitionResult::repeat_count)
        .def_readonly("partial_tail_tokens", &OnlineRepetitionResult::partial_tail_tokens)
        .def_readonly("covered_token_count", &OnlineRepetitionResult::covered_token_count)
        .def_readonly("duplicate_token_count", &OnlineRepetitionResult::duplicate_token_count)
        .def_readonly("start_index", &OnlineRepetitionResult::start_index)
        .def_readonly("end_index", &OnlineRepetitionResult::end_index)
        .def_readonly("first_detect_index", &OnlineRepetitionResult::first_detect_index);

    py::class_<OnlineRepetitionTracker>(m, "OnlineRepetitionTracker")
        .def(py::init<OnlineRepetitionConfig>(), py::arg("config") = OnlineRepetitionConfig())
        .def("reset", &OnlineRepetitionTracker::reset)
        .def("update", &OnlineRepetitionTracker::update, py::arg("token_id"))
        .def("update_many",
             [](OnlineRepetitionTracker& tracker, const std::vector<int>& token_ids) {
                 py::gil_scoped_release release;
                 return tracker.updateMany(token_ids);
             },
             py::arg("token_ids"))
        .def_property_readonly("result",
                               [](const OnlineRepetitionTracker& tracker) {
                                   return tracker.result();
                               })
        .def_property_readonly("token_count", &OnlineRepetitionTracker::tokenCount);

    m.def("detect_repetition_hit_only",
          [](const std::vector<int>& token_ids,
             int min_repeats,
             int min_duplicate_tokens,
             int max_period) {
              py::gil_scoped_release release;
              return detectOnlineRepetitionHitOnly(
                  token_ids, makeConfig(min_repeats, min_duplicate_tokens, max_period));
          },
          py::arg("token_ids"),
          py::arg("min_repeats") = 3,
          py::arg("min_duplicate_tokens") = 32,
          py::arg("max_period") = 512);

    m.def("detect_repetition_max",
          [](const std::vector<int>& token_ids,
             int min_repeats,
             int min_duplicate_tokens,
             int max_period) {
              py::gil_scoped_release release;
              return detectOnlineRepetitionMax(
                  token_ids, makeConfig(min_repeats, min_duplicate_tokens, max_period));
          },
          py::arg("token_ids"),
          py::arg("min_repeats") = 3,
          py::arg("min_duplicate_tokens") = 32,
          py::arg("max_period") = 512);

    py::class_<ToolCallMarkerIds>(m, "ToolCallMarkerIds")
        .def(py::init<>())
        .def_readwrite("begin_ids", &ToolCallMarkerIds::begin_ids)
        .def_readwrite("end_ids", &ToolCallMarkerIds::end_ids)
        .def_readwrite("name", &ToolCallMarkerIds::name);

    py::class_<CompletedToolCallSpan>(m, "CompletedToolCallSpan")
        .def(py::init<>())
        .def_readwrite("marker_index", &CompletedToolCallSpan::marker_index)
        .def_readwrite("token_ids", &CompletedToolCallSpan::token_ids)
        .def_readwrite("overflow", &CompletedToolCallSpan::overflow);

    py::class_<ToolCallLoopResult>(m, "ToolCallLoopResult")
        .def(py::init<>())
        .def_readonly("hit", &ToolCallLoopResult::hit)
        .def_readonly("repeat_count", &ToolCallLoopResult::repeat_count)
        .def_readonly("threshold", &ToolCallLoopResult::threshold)
        .def_readonly("current_span_tokens", &ToolCallLoopResult::current_span_tokens)
        .def_readonly("marker_index", &ToolCallLoopResult::marker_index)
        .def_readonly("history_suffix_count", &ToolCallLoopResult::history_suffix_count)
        .def_readonly("current_suffix_count", &ToolCallLoopResult::current_suffix_count)
        .def_readonly("span_overflow", &ToolCallLoopResult::span_overflow);

    py::class_<ToolCallSpanRecorder>(m, "ToolCallSpanRecorder")
        .def(py::init<std::vector<ToolCallMarkerIds>, int>(),
             py::arg("markers"),
             py::arg("max_span_tokens") = 16384)
        .def("reset", &ToolCallSpanRecorder::reset)
        .def("update_many",
             [](ToolCallSpanRecorder& recorder, const std::vector<int>& token_ids) {
                 py::gil_scoped_release release;
                 return recorder.updateMany(token_ids);
             },
             py::arg("token_ids"))
        .def_property_readonly("inside_span", &ToolCallSpanRecorder::insideSpan);

    py::class_<TokenToolCallLoopGuard>(m, "TokenToolCallLoopGuard")
        .def(py::init<std::vector<ToolCallMarkerIds>, int, int>(),
             py::arg("markers"),
             py::arg("repeat_threshold") = 5,
             py::arg("max_span_tokens") = 16384)
        .def("reset", &TokenToolCallLoopGuard::reset)
        .def("check_completed_span",
             [](TokenToolCallLoopGuard& guard,
                const std::vector<int>& input_ids,
                const std::vector<int>& current_span_ids,
                int marker_index,
                bool span_overflow) {
                 py::gil_scoped_release release;
                 return guard.checkCompletedSpan(
                     input_ids, current_span_ids, marker_index, span_overflow);
             },
             py::arg("input_ids"),
             py::arg("current_span_ids"),
             py::arg("marker_index"),
             py::arg("span_overflow") = false);

    m.def("scan_tool_call_spans_once",
          [](const std::vector<int>& token_ids,
             const std::vector<ToolCallMarkerIds>& markers,
             int max_span_tokens) {
              py::gil_scoped_release release;
              return scanToolCallSpansOnce(token_ids, markers, max_span_tokens);
          },
          py::arg("token_ids"),
          py::arg("markers"),
          py::arg("max_span_tokens") = 16384);

    m.def("check_tool_call_loop_once",
          [](const std::vector<int>& input_ids,
             const std::vector<int>& current_span_ids,
             int marker_index,
             const std::vector<ToolCallMarkerIds>& markers,
             int repeat_threshold,
             int max_span_tokens) {
              py::gil_scoped_release release;
              return checkToolCallLoopOnce(input_ids,
                                           current_span_ids,
                                           marker_index,
                                           markers,
                                           repeat_threshold,
                                           max_span_tokens);
          },
          py::arg("input_ids"),
          py::arg("current_span_ids"),
          py::arg("marker_index"),
          py::arg("markers"),
          py::arg("repeat_threshold") = 5,
          py::arg("max_span_tokens") = 16384);
}
