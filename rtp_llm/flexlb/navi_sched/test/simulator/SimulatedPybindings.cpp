#include "aios/navi_sched/bindings/NaviSchedBindings.h"
#include "aios/navi_sched/service/NaviSchedAgent.h"
#include "aios/navi_sched/service/NaviSchedClient.h"
#include "aios/navi_sched/service/NaviSchedMaster.h"
#include "aios/navi_sched/test/simulator/SimulatedAgentAccess.h"
#include "aios/navi_sched/test/simulator/SimulatedExecutionR.h"
#include "aios/navi_sched/test/simulator/TrafficReplayAccess.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include "navi/pybind/NaviPybind.h"

namespace py = pybind11;

namespace navi_sched {
namespace {

template <typename T>
T &unwrapRoleFacade(const py::object &role)
{
    try {
        return role.cast<T &>();
    } catch (const py::cast_error &) {
    }
    if (!py::hasattr(role, "_service")) {
        throw py::type_error("expected a NaviSched role object");
    }
    return role.attr("_service").cast<T &>();
}

}

void bindSimulatedNaviSched(py::module_ &m)
{
    py::class_<SimulatedAgentAccess, std::shared_ptr<SimulatedAgentAccess>>(
        m, "NaviSchedSimulatedAgentHandle")
        .def("stop", &SimulatedAgentAccess::stop,
             py::call_guard<py::gil_scoped_release>());
    py::class_<test::SimulatedExecutionConfig>(m, "NaviSchedSimulatedExecutionConfig")
        .def(py::init<>())
        .def_readwrite("cache_capacity_bytes",
                       &test::SimulatedExecutionConfig::cacheCapacityBytes)
        .def_readwrite("token_cache_bytes",
                       &test::SimulatedExecutionConfig::tokenCacheBytes)
        .def_readwrite("block_size", &test::SimulatedExecutionConfig::blockSize)
        .def_readwrite("formula_parameters",
                       &test::SimulatedExecutionConfig::formulaParameters)
        .def_readwrite("noise_stddev_ratio",
                       &test::SimulatedExecutionConfig::noiseStddevRatio)
        .def_readwrite("random_seed", &test::SimulatedExecutionConfig::randomSeed)
        .def_readwrite("time_compression",
                       &test::SimulatedExecutionConfig::timeCompression)
        .def_readwrite("queue_trace_output_path",
                       &test::SimulatedExecutionConfig::queueTraceOutputPath);
    m.def(
        "init_simulated_agent",
        [](const py::object &agent,
           const test::SimulatedExecutionConfig &simulateConfig) {
            auto &nativeAgent = unwrapRoleFacade<NaviSchedAgent>(agent);
            return SimulatedAgentAccess::create(nativeAgent, simulateConfig);
        },
        py::arg("agent"), py::arg("simulate_config") = test::SimulatedExecutionConfig());
    m.def(
        "report_simulated_batch",
        [](NaviSchedAgent &agent, uint64_t requestId, int64_t totalTokenCount,
           int64_t cacheHitTokenCount) {
            return SimulatedAgentAccess::reportBatch(agent, requestId, totalTokenCount,
                                                     cacheHitTokenCount);
        },
        py::arg("agent"), py::arg("request_id"), py::arg("total_token_count") = 1,
        py::arg("cache_hit_token_count") = 0, py::call_guard<py::gil_scoped_release>());
    py::class_<test::TrafficReplayConfig>(m, "NaviSchedTrafficReplayConfig")
        .def(py::init<>())
        .def_readwrite("client_biz_name", &test::TrafficReplayConfig::clientBizName)
        .def_readwrite("traffic_log", &test::TrafficReplayConfig::trafficLog)
        .def_readwrite("target_biz_name", &test::TrafficReplayConfig::targetBizName)
        .def_readwrite("master_biz_name", &test::TrafficReplayConfig::masterBizName)
        .def_readwrite("service_name", &test::TrafficReplayConfig::serviceName)
        .def_readwrite("client_count", &test::TrafficReplayConfig::clientCount)
        .def_readwrite("client_index", &test::TrafficReplayConfig::clientIndex)
        .def_readwrite("time_compression", &test::TrafficReplayConfig::timeCompression)
        .def_readwrite("request_timeout_ms", &test::TrafficReplayConfig::requestTimeoutMs)
        .def_readwrite("start_delay_ms", &test::TrafficReplayConfig::startDelayMs)
        .def_readwrite("block_size", &test::TrafficReplayConfig::blockSize)
        .def_readwrite("collect_replay_sequences",
                       &test::TrafficReplayConfig::collectReplaySequences)
        .def_readwrite("replay_output_path", &test::TrafficReplayConfig::replayOutputPath)
        .def_readwrite("latency_output_path",
                       &test::TrafficReplayConfig::latencyOutputPath);
    py::class_<TrafficReplayAccess, std::shared_ptr<TrafficReplayAccess>>(
        m, "NaviSchedTrafficReplayHandle")
        .def("stop", &TrafficReplayAccess::stop,
             py::call_guard<py::gil_scoped_release>());
    m.def(
        "init_simulated_traffic_replay",
        [](const py::object &client, const test::TrafficReplayConfig &trafficConfig) {
            auto &nativeClient = unwrapRoleFacade<NaviSchedClient>(client);
            return TrafficReplayAccess::create(nativeClient, trafficConfig);
        },
        py::arg("client"), py::arg("traffic_config"));
    m.def(
        "get_simulated_execution_stats",
        [](const py::object &master) {
            const auto &nativeMaster = unwrapRoleFacade<NaviSchedMaster>(master);
            auto stats = nativeMaster.getStats();
            py::dict result;
            result["received_request_count"] = stats.agentRequestReceivedCount;
            result["finished_request_count"] = stats.agentRequestFinishedCount;
            result["average_latency_ms"] =
                stats.agentRequestFinishedCount == 0
                    ? 0.0
                    : static_cast<double>(stats.agentRequestTotalLatencyUs) /
                          stats.agentRequestFinishedCount / 1000.0;
            result["cache_hit_tokens"] = stats.agentRequestCacheHitTokens;
            result["cache_miss_tokens"] = stats.agentRequestCacheMissTokens;
            py::list latencyModels;
            py::list reportTransportLatencyMs;
            const auto &expected = test::SimulatedExecutionR::getFormulaParameters();
            for (const auto &pair : nativeMaster.getNodes()) {
                const auto &node = pair.second;
                py::dict value;
                std::vector<double> deltas;
                deltas.reserve(std::min(node.latencyParameters.size(), expected.size()));
                double squaredDistance = 0.0;
                double absoluteDeltaSum = 0.0;
                double maxAbsoluteDelta = 0.0;
                for (size_t index = 0;
                     index < node.latencyParameters.size() && index < expected.size();
                     ++index) {
                    auto delta = node.latencyParameters[index] - expected[index];
                    auto absoluteDelta = std::abs(delta);
                    deltas.emplace_back(delta);
                    squaredDistance += delta * delta;
                    absoluteDeltaSum += absoluteDelta;
                    maxAbsoluteDelta = std::max(maxAbsoluteDelta, absoluteDelta);
                }
                value["agent_id"] = node.agentId;
                value["revision"] = node.latencyModelRevision;
                value["parameters"] = node.latencyParameters;
                value["parameter_deltas"] = deltas;
                value["parameter_l2_distance"] = std::sqrt(squaredDistance);
                value["mean_absolute_parameter_delta"] =
                    deltas.empty() ? 0.0 : absoluteDeltaSum / deltas.size();
                value["max_absolute_parameter_delta"] = maxAbsoluteDelta;
                const auto &validation = node.latencyValidation;
                value["prediction_sample_count"] = validation.sampleCount;
                value["prediction_unmatched_batch_count"] =
                    validation.unmatchedBatchCount;
                value["batch_request_count_mean"] =
                    validation.batchRequestCountMean;
                value["predicted_cache_hit_tokens_mean"] =
                    validation.predictedCacheHitTokensMean;
                value["actual_cache_hit_tokens_mean"] =
                    validation.actualCacheHitTokensMean;
                value["cache_hit_tokens_bias_mean"] =
                    validation.cacheHitTokensBiasMean;
                value["cache_hit_tokens_mae"] = validation.cacheHitTokensMae;
                value["master_predicted_queue_mean_ms"] =
                    validation.masterPredictedQueueMeanMs;
                value["actual_queue_mean_ms"] = validation.actualQueueMeanMs;
                value["queue_actual_minus_predicted_bias_ms"] =
                    validation.queueBiasMeanMs;
                value["queue_mae_ms"] = validation.queueMaeMs;
                value["queue_raw_predicted_mean_ms"] =
                    validation.queueRawPredictedMeanMs;
                value["queue_raw_bias_mean_ms"] = validation.queueRawBiasMeanMs;
                value["queue_raw_mae_ms"] = validation.queueRawMaeMs;
                value["agent_predicted_queue_mean_ms"] =
                    validation.agentPredictedQueueMeanMs;
                value["agent_queue_bias_mean_ms"] =
                    validation.agentQueueBiasMeanMs;
                value["agent_queue_mae_ms"] = validation.agentQueueMaeMs;
                value["queue_feature_batch_count_mean"] =
                    validation.queueFeatureBatchCountMean;
                value["queue_model_weights"] = validation.queueModelWeights;
                value["queue_model_fit_step"] = validation.queueModelFitStep;
                value["master_predicted_execution_mean_ms"] =
                    validation.masterPredictedExecutionMeanMs;
                value["agent_pre_update_predicted_execution_mean_ms"] =
                    validation.agentPredictedExecutionMeanMs;
                value["actual_execution_mean_ms"] =
                    validation.actualExecutionMeanMs;
                value["execution_actual_minus_master_predicted_bias_ms"] =
                    validation.masterExecutionBiasMeanMs;
                value["execution_master_prediction_mae_ms"] =
                    validation.masterExecutionMaeMs;
                value["execution_master_prediction_mape_pct"] =
                    validation.masterExecutionMapePct;
                value["execution_actual_minus_agent_pre_update_predicted_bias_ms"] =
                    validation.agentExecutionBiasMeanMs;
                value["execution_agent_pre_update_prediction_mae_ms"] =
                    validation.agentExecutionMaeMs;
                value["execution_agent_pre_update_prediction_mape_pct"] =
                    validation.agentExecutionMapePct;
                value["master_predicted_total_mean_ms"] =
                    validation.masterPredictedTotalMeanMs;
                value["actual_total_mean_ms"] = validation.actualTotalMeanMs;
                value["total_actual_minus_predicted_bias_ms"] =
                    validation.totalBiasMeanMs;
                value["total_mae_ms"] = validation.totalMaeMs;
                value["total_mape_pct"] = validation.totalMapePct;
                latencyModels.append(std::move(value));
                if (node.lastEventTimeUs > 0 &&
                    node.lastReceiveTimeUs >= node.lastEventTimeUs) {
                    reportTransportLatencyMs.append(
                        static_cast<double>(node.lastReceiveTimeUs -
                                            node.lastEventTimeUs) /
                        1000.0);
                }
            }
            result["latency_models"] = std::move(latencyModels);
            result["report_transport_latency_ms"] = std::move(reportTransportLatencyMs);
            return result;
        },
        py::arg("master"));
}

}

PYBIND11_MODULE(_navi_sched_core, module)
{
    navi_sched::bindNaviSchedModule(module, "navi_sched simulate");
    navi_sched::bindSimulatedNaviSched(module);
}
