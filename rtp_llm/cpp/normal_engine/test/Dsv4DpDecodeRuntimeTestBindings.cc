#include "rtp_llm/cpp/normal_engine/Dsv4DpDecodeRuntime.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

namespace rtp_llm {
namespace py = pybind11;

namespace {

py::tuple prepareModel(Dsv4DpDecodeRuntime&    runtime,
                       const AgreedDecodePlan& plan,
                       bool                    graph_available,
                       bool                    fail_graph,
                       bool                    fail_eager,
                       bool                    fail_discard) {
    const auto&                    planned = plan.execution;
    std::optional<DecodeExecution> pending = planned;
    AgreedDecodeGraphPreparation   graph;
    std::vector<std::string>       stages;
    const auto                     consume = [&] {
        return graph.consume(planned.phase, planned, planned.token_width, planned.bucket, planned.bucket);
    };
    try {
        if (!plan.valid_local_input) {
            throw std::logic_error("invalid input geometry");
        }
        const auto execution = prepareAgreedDecodeExecution(
            runtime,
            pending,
            [&](const DecodeExecution& decision) {
                stages.emplace_back("prepare_graph");
                return graph.prepare([&] {
                    if (fail_graph) {
                        throw std::runtime_error("injected graph preparation failure");
                    }
                    return graph_available ? decision.bucket : 0;
                });
            },
            [&] {
                stages.emplace_back("discard_graph");
                if (fail_discard) {
                    throw std::runtime_error("injected graph discard failure");
                }
                graph.discard();
            },
            [&](const DecodeExecution&) {
                stages.emplace_back("prepare_eager");
                if (consume()) {
                    throw std::logic_error("eager preparation saw a stale graph key");
                }
                if (fail_eager) {
                    throw std::runtime_error("injected eager preparation failure");
                }
            });
        return py::make_tuple(execution, stages, consume());
    } catch (...) {
        if (pending) {
            (void)finishPendingDecodePreparation(runtime, pending, DecodePreparationStatus::INVALID);
        }
        throw;
    }
}

}  // namespace

PYBIND11_MODULE(libdsv4_dp_decode_runtime_test_bindings, m) {
    py::enum_<DecodeExecutionMode>(m, "Mode")
        .value("EAGER", DecodeExecutionMode::EAGER)
        .value("GRAPH", DecodeExecutionMode::GRAPH);
    py::enum_<DecodeExecutionPhase>(m, "Phase")
        .value("TARGET", DecodeExecutionPhase::TARGET)
        .value("TARGET_VERIFY", DecodeExecutionPhase::TARGET_VERIFY)
        .value("MTP_DRAFT", DecodeExecutionPhase::MTP_DRAFT)
        .value("MTP_COMMIT", DecodeExecutionPhase::MTP_COMMIT)
        .value("DSPARK_PROPOSAL", DecodeExecutionPhase::DSPARK_PROPOSAL)
        .value("DSPARK_COMMIT", DecodeExecutionPhase::DSPARK_COMMIT);
    py::enum_<DecodePreparationStatus>(m, "Status")
        .value("READY", DecodePreparationStatus::READY)
        .value("GRAPH_UNAVAILABLE", DecodePreparationStatus::GRAPH_UNAVAILABLE)
        .value("INVALID", DecodePreparationStatus::INVALID);
    py::enum_<DecodeReplicaRound>(m, "Round")
        .value("IDLE", DecodeReplicaRound::IDLE)
        .value("EXECUTE", DecodeReplicaRound::EXECUTE)
        .value("PAUSED", DecodeReplicaRound::PAUSED)
        .value("STOP", DecodeReplicaRound::STOP)
        .value("INVALID", DecodeReplicaRound::INVALID);
    py::enum_<DecodeReplicaStartupStage>(m, "StartupStage")
        .value("WARMUP_INPUTS", DecodeReplicaStartupStage::WARMUP_INPUTS)
        .value("CACHE_MANAGER", DecodeReplicaStartupStage::CACHE_MANAGER)
        .value("ENGINE_READY", DecodeReplicaStartupStage::ENGINE_READY);
    py::class_<DecodeExecution>(m, "Execution")
        .def_readonly("mode", &DecodeExecution::mode)
        .def_readonly("phase", &DecodeExecution::phase)
        .def_readonly("bucket", &DecodeExecution::bucket)
        .def_readonly("token_width", &DecodeExecution::token_width);
    py::class_<AgreedDecodePlan>(m, "Plan")
        .def_readonly("execution", &AgreedDecodePlan::execution)
        .def_readonly("valid_local_input", &AgreedDecodePlan::valid_local_input);
    py::class_<Dsv4DpDecodeRuntime>(m, "Runtime")
        .def(py::init([](int world_size, DecodeExecutionPhase first_phase) {
                 return std::make_unique<Dsv4DpDecodeRuntime>(world_size, nullptr, first_phase);
             }),
             py::arg("world_size"),
             py::arg("first_phase") = DecodeExecutionPhase::TARGET)
        .def("begin_round", &Dsv4DpDecodeRuntime::beginRound)
        .def("agree_scheduled_requests", &Dsv4DpDecodeRuntime::agreeScheduledRequests)
        .def("reject_unprepared_target", &Dsv4DpDecodeRuntime::rejectUnpreparedTarget)
        .def("agree_speculative_status", &Dsv4DpDecodeRuntime::agreeSpeculativeStatus)
        .def("reject_unprepared_round", &Dsv4DpDecodeRuntime::rejectUnpreparedRound)
        .def("verify_model_initialization", &Dsv4DpDecodeRuntime::verifyModelInitialization)
        .def("verify_capture", &Dsv4DpDecodeRuntime::verifyCapture)
        .def("verify_capture_readiness", &Dsv4DpDecodeRuntime::verifyCaptureReadiness)
        .def("finish_capture", &Dsv4DpDecodeRuntime::finishCapture)
        .def("reject_speculative_input", &Dsv4DpDecodeRuntime::rejectSpeculativeInput)
        .def("plan", &Dsv4DpDecodeRuntime::plan)
        .def("finish_preparation", &Dsv4DpDecodeRuntime::finishPreparation);
    m.def("prepare_startup", [](Dsv4DpDecodeRuntime& runtime, DecodeReplicaStartupStage stage, bool fail) {
        prepareDecodeReplicaStartup(&runtime, stage, [&] {
            if (fail) {
                throw std::runtime_error("injected local startup failure");
            }
        });
    });
    m.def("prepare_cache_capacity", [](Dsv4DpDecodeRuntime& runtime, uint32_t local_blocks, bool fail) {
        return prepareDecodeReplicaCacheCapacity(&runtime, [&] {
            if (fail) {
                throw std::runtime_error("injected local cache capacity failure");
            }
            return local_blocks;
        });
    });
    m.def("prepare_model",
          &prepareModel,
          py::arg("runtime"),
          py::arg("plan"),
          py::arg("graph_available"),
          py::arg("fail_graph"),
          py::arg("fail_eager"),
          py::arg("fail_discard") = false);
    m.def("prepare_speculative_input",
          [](Dsv4DpDecodeRuntime& runtime, DecodeExecutionPhase phase, int64_t requests, bool fail) {
              return prepareSpeculativeInput(&runtime, phase, [&] {
                  if (fail) {
                      throw std::runtime_error("injected rank-local preparation failure");
                  }
                  return requests;
              });
          });
    m.def("report_invalid_speculative_input",
          [](Dsv4DpDecodeRuntime& runtime, DecodeExecutionPhase phase, int original_status) {
              reportInvalidSpeculativeInput(&runtime, phase);
              return original_status;
          });
}

}  // namespace rtp_llm
