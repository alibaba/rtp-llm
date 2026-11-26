// Test-only access to the production C++ forecast, with conversion outside GIL release.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "aios/navi_sched/common/TPUPrefillPP4Forecast.h"
#include "aios/navi_sched/test/TPUPrefillPP4Baseline.h"

namespace py = pybind11;

PYBIND11_MODULE(_navi_sched_pp4_forecast_test, module)
{
    using namespace navi_sched;
    py::class_<TPUPrefillPipelineSlot>(module, "Slot")
        .def(py::init<>())
        .def_readwrite("ordinal", &TPUPrefillPipelineSlot::ordinal)
        .def_readwrite("dispatch_time_ms", &TPUPrefillPipelineSlot::dispatchTimeMs)
        .def_readwrite("features", &TPUPrefillPipelineSlot::features);
    py::class_<TPUPrefillPP4HostResult>(module, "HostResult")
        .def(py::init<>())
        .def_readwrite("slot_ordinal", &TPUPrefillPP4HostResult::slotOrdinal)
        .def_readwrite("ready", &TPUPrefillPP4HostResult::ready)
        .def_readwrite("control_ready_time_ms",
                       &TPUPrefillPP4HostResult::controlReadyTimeMs);
    py::class_<TPUPrefillPP4State>(module, "State")
        .def(py::init<>())
        .def_readwrite("now_ms", &TPUPrefillPP4State::nowMs)
        .def_readwrite("measured_forward_ms", &TPUPrefillPP4State::measuredForwardMs)
        .def_readwrite("last_real_ordinal", &TPUPrefillPP4State::lastRealOrdinal)
        .def_readwrite("slots", &TPUPrefillPP4State::slots)
        .def_readwrite("expected_forward_ms", &TPUPrefillPP4State::expectedForwardMs)
        .def_readwrite("observed_replies", &TPUPrefillPP4State::observedReplies)
        .def_readwrite("reply_lower_bounds", &TPUPrefillPP4State::replyLowerBounds)
        .def_readwrite("results", &TPUPrefillPP4State::results)
        .def_readwrite("max_running_request_count",
                       &TPUPrefillPP4State::maxRunningRequestCount)
        .def_readwrite("other_request_release_times_ms",
                       &TPUPrefillPP4State::otherRequestReleaseTimesMs)
        .def_readwrite("request_completion_slots",
                       &TPUPrefillPP4State::requestCompletionSlots);
    py::enum_<TPUPrefillForecastQueue>(module, "Queue")
        .value("RUNNING", TPUPrefillForecastQueue::RUNNING)
        .value("SKIPPED_WAITING", TPUPrefillForecastQueue::SKIPPED_WAITING)
        .value("WAITING", TPUPrefillForecastQueue::WAITING);
    py::class_<TPUPrefillKVReceiveDependency>(module, "ReceiveDependency")
        .def(py::init<>())
        .def_readwrite("carrier_ordinal", &TPUPrefillKVReceiveDependency::carrierOrdinal)
        .def_readwrite("reporter_ordinal",
                       &TPUPrefillKVReceiveDependency::reporterOrdinal)
        .def_readwrite("decided_proposal_fence",
                       &TPUPrefillKVReceiveDependency::decidedProposalFence);
    py::class_<TPUPrefillPP4Request>(module, "Request")
        .def(py::init<>())
        .def_readwrite("request_id", &TPUPrefillPP4Request::requestId)
        .def_readwrite("context_tokens", &TPUPrefillPP4Request::contextTokens)
        .def_readwrite("remaining_tokens", &TPUPrefillPP4Request::remainingTokens)
        .def_readwrite("reserved_tokens", &TPUPrefillPP4Request::reservedTokens)
        .def_readwrite("shared_prefix_boundary",
                       &TPUPrefillPP4Request::sharedPrefixBoundary)
        .def_readwrite("queue", &TPUPrefillPP4Request::queue)
        .def_readwrite("owned", &TPUPrefillPP4Request::owned)
        .def_readwrite("release_slot_after_prefill",
                       &TPUPrefillPP4Request::releaseSlotAfterPrefill)
        .def_readwrite("needs_first_query", &TPUPrefillPP4Request::needsFirstQuery)
        .def_readwrite("ready_time_ms", &TPUPrefillPP4Request::readyTimeMs)
        .def_readwrite("receive_dependency", &TPUPrefillPP4Request::receiveDependency)
        .def_readwrite("receive_host_result_index",
                       &TPUPrefillPP4Request::receiveHostResultIndex)
        .def_readwrite("metadata_only_load", &TPUPrefillPP4Request::metadataOnlyLoad);
    py::class_<TPUPrefillMetadataKVLoad>(module, "MetadataKVLoad")
        .def(py::init<>())
        .def_readwrite("local_load_ms", &TPUPrefillMetadataKVLoad::localLoadMs)
        .def_readwrite("control_reply_ms", &TPUPrefillMetadataKVLoad::controlReplyMs);
    py::class_<TPUPrefillExecutionPolicyConfig>(module, "Policy")
        .def(py::init<>())
        .def_readwrite("pcp_size", &TPUPrefillExecutionPolicyConfig::pcpSize)
        .def_readwrite("interleave_size",
                       &TPUPrefillExecutionPolicyConfig::interleaveSize)
        .def_readwrite("compile_sizes", &TPUPrefillExecutionPolicyConfig::compileSizes)
        .def_readwrite("max_scheduled_tokens",
                       &TPUPrefillExecutionPolicyConfig::maxScheduledTokens)
        .def_readwrite("long_prefill_token_threshold",
                       &TPUPrefillExecutionPolicyConfig::longPrefillTokenThreshold)
        .def_readwrite("hash_block_size", &TPUPrefillExecutionPolicyConfig::hashBlockSize)
        .def_readwrite("scheduler_block_size",
                       &TPUPrefillExecutionPolicyConfig::schedulerBlockSize)
        .def_readwrite("use_eagle", &TPUPrefillExecutionPolicyConfig::useEagle)
        .def_readwrite("mamba_has_prefill_checkpoint_blocks",
                       &TPUPrefillExecutionPolicyConfig::mambaHasPrefillCheckpointBlocks)
        .def_readwrite("mamba_partial_cache_hit",
                       &TPUPrefillExecutionPolicyConfig::mambaPartialCacheHit);
    py::class_<TPUPrefillPP4ForecastSpan>(module, "Span")
        .def_readonly("request_id", &TPUPrefillPP4ForecastSpan::requestId)
        .def_readonly("computed_tokens", &TPUPrefillPP4ForecastSpan::computedTokens)
        .def_readonly("scheduled_tokens", &TPUPrefillPP4ForecastSpan::scheduledTokens);
    py::class_<TPUPrefillPP4ForecastDispatch>(module, "Dispatch")
        .def_readonly("ordinal", &TPUPrefillPP4ForecastDispatch::ordinal)
        .def_readonly("time_ms", &TPUPrefillPP4ForecastDispatch::timeMs)
        .def_readonly("real", &TPUPrefillPP4ForecastDispatch::real)
        .def_readonly("spans", &TPUPrefillPP4ForecastDispatch::spans)
        .def_readonly("finished_load_request_ids",
                      &TPUPrefillPP4ForecastDispatch::finishedLoadRequestIds);
    py::class_<TPUPrefillPP4ForecastOutput::ControlDispatch>(module, "ControlDispatch")
        .def_readonly("time_ms", &TPUPrefillPP4ForecastOutput::ControlDispatch::timeMs)
        .def_readonly("ready_time_ms",
                      &TPUPrefillPP4ForecastOutput::ControlDispatch::readyTimeMs)
        .def_readonly("pp_ordinal",
                      &TPUPrefillPP4ForecastOutput::ControlDispatch::ppOrdinal)
        .def_readonly("load_request_ids",
                      &TPUPrefillPP4ForecastOutput::ControlDispatch::loadRequestIds)
        .def_readonly(
            "finished_load_request_ids",
            &TPUPrefillPP4ForecastOutput::ControlDispatch::finishedLoadRequestIds);
    py::class_<TPUPrefillPP4ForecastOutput>(module, "Output")
        .def_readonly("control_dispatches",
                      &TPUPrefillPP4ForecastOutput::controlDispatches)
        .def_readonly("queue_ms", &TPUPrefillPP4ForecastOutput::queueMs)
        .def_readonly("execution_ms", &TPUPrefillPP4ForecastOutput::executionMs)
        .def_readonly("total_ms", &TPUPrefillPP4ForecastOutput::totalMs)
        .def_readonly("real_steps", &TPUPrefillPP4ForecastOutput::realSteps)
        .def_readonly("owned_steps", &TPUPrefillPP4ForecastOutput::ownedSteps)
        .def_readonly("real_prefix_steps", &TPUPrefillPP4ForecastOutput::realPrefixSteps)
        .def_readonly("first_scheduled_time_ms",
                      &TPUPrefillPP4ForecastOutput::firstScheduledTimeMs)
        .def_readonly("scheduled_tokens", &TPUPrefillPP4ForecastOutput::scheduledTokens)
        .def_readonly("dispatches", &TPUPrefillPP4ForecastOutput::dispatches);
    module.def("predict", &TPUPrefillPP4Forecast::predict,
               py::call_guard<py::gil_scoped_release>());
    module.def("predict_baseline", &TPUPrefillPP4ForecastBaseline::predict,
               py::call_guard<py::gil_scoped_release>());
}
