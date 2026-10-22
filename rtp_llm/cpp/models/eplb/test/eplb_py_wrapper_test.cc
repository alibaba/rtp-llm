#include "rtp_llm/cpp/models/eplb/ExpertBalancerPythonWrapper.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/models/eplb/ExpertLoadMonitor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

#include <stdexcept>

namespace rtp_llm {

// Keep the production Python collective callback in the test module, without
// loading the service DSO alongside the test's static dependencies.
void execAllGather(const AllGatherParams& params) {
    py::gil_scoped_acquire gil;
    py::list               recv_list, send_list;
    for (const auto& tensor : params.recv_buffers) {
        recv_list.append(tensor);
    }
    for (const auto& tensor : params.send_buffers) {
        send_list.append(tensor);
    }
    py::module_::import("rtp_llm.cpp.models.eplb.test.libth_eplb_py_wrapper_test")
        .attr("_allgather_callback")(recv_list, static_cast<int>(params.mode), send_list, params.inplace);
}

AllReduceOutput execAllReduce(const AllReduceParams&) {
    throw std::logic_error("single-rank balancer tests must not call execAllReduce");
}

void execBroadcast(const BroadcastParams&) {
    throw std::logic_error("single-rank balancer tests must not call execBroadcast");
}

}  // namespace rtp_llm

using namespace rtp_llm;

namespace unittest {

class ExpertBalancerStepTest {
public:
    ExpertBalancerStepTest(const py::object&  model,
                           int                snapshot_max_files,
                           int                interval,
                           size_t             model_id,
                           bool               moe,
                           int                tp_rank,
                           int                tp_size,
                           std::optional<int> mode) {
        config_.eplb_metrics_report_interval    = interval;
        config_.eplb_metrics_snapshot_max_files = snapshot_max_files;
        config_.eplb_update_time                = 1;
        config_.eplb_control_step               = 1;
        config_.eplb_mode = mode ? static_cast<EplbMode>(*mode) : (model.is_none() ? EplbMode::NONE : EplbMode::EPLB);
        Weights weights;
        weights.layers.resize(1);
        GptModelDescription description{};
        description.ffn_conf.activation_type = ActivationType::Swiglu;
        if (moe) {
            description.ffn_conf.moe_configs = MoeConfigs{2, 0, 1};
        }
        GptModelInitParams params{weights, description, std::nullopt};
        params.hidden_size                   = 1;
        params.model_id                      = model_id;
        params.eplb_config                   = config_;
        params.parallelism_config.ep_size    = 1;
        params.parallelism_config.ep_rank    = 0;
        params.parallelism_config.tp_rank    = tp_rank;
        params.parallelism_config.tp_size    = tp_size;
        params.parallelism_config.world_rank = tp_rank;
        params.parallelism_config.world_size = tp_size;
        balancer_ = std::make_unique<ExpertBalancer>(description.ffn_conf.moe_configs, params);
        balancer_->initialize(model);
    }

    void step(bool warmup) {
        balancer_->step(warmup);
    }

    bool updateConfig(int mode, int64_t update_time) {
        config_.eplb_mode        = static_cast<EplbMode>(mode);
        config_.eplb_update_time = update_time;
        return balancer_->updateEplbConfig(config_);
    }

    void destroy() {
        balancer_.reset();
    }

    torch::Tensor counts() const {
        return balancer_->stable_expert_token_counts();
    }

private:
    EPLBConfig                      config_;
    std::unique_ptr<ExpertBalancer> balancer_;
};

class EplbPyWrapperOP {
public:
    void initialize(const py::object& model, size_t ep_rank, size_t ep_size, size_t model_id) {
        EPLBConfig config;
        config.eplb_mode                   = EplbMode::ALL;
        config.eplb_update_time            = 7;
        config.eplb_stats_window_size      = 3;
        config.eplb_balance_layer_per_step = 2;
        config.balance_method              = "round";
        wrapper_                           = std::make_unique<ExpertBalancerPythonWrapper>();
        wrapper_->initialize(model, config, ep_rank, ep_size, model_id);
    }

    void recordStep(const torch::Tensor& counts) {
        wrapper_->recordStep(counts);
    }

    bool balance() {
        return wrapper_->balance();
    }

    void updateConfig(int mode, int64_t update_time) {
        EPLBConfig config;
        config.eplb_mode        = static_cast<EplbMode>(mode);
        config.eplb_update_time = update_time;
        wrapper_->updateConfig(config);
    }

    void destroy() {
        wrapper_.reset();
    }

private:
    std::unique_ptr<ExpertBalancerPythonWrapper> wrapper_;
};

PYBIND11_MODULE(libth_eplb_py_wrapper_test, m) {
    py::class_<ExpertBalancerStepTest>(m, "ExpertBalancerStepTest")
        .def(py::init<const py::object&, int, int, size_t, bool, int, int, std::optional<int>>(),
             py::arg("model"),
             py::arg("snapshot_max_files"),
             py::arg("interval") = 2,
             py::arg("model_id") = 0,
             py::arg("moe")      = true,
             py::arg("tp_rank")  = 0,
             py::arg("tp_size")  = 1,
             py::arg("mode")     = py::none())
        .def("step", &ExpertBalancerStepTest::step, py::arg("warmup") = false, py::call_guard<py::gil_scoped_release>())
        .def("update_config", &ExpertBalancerStepTest::updateConfig, py::arg("mode"), py::arg("update_time") = 1)
        .def("destroy", &ExpertBalancerStepTest::destroy, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("counts", &ExpertBalancerStepTest::counts);
    py::class_<ExpertLoadMonitor>(m, "ExpertLoadMonitor")
        .def(py::init([](size_t layers,
                         size_t experts,
                         int    world_size,
                         int    rank,
                         int    interval,
                         size_t model_id,
                         int    snapshot_max_files) {
            ParallelismConfig parallelism;
            parallelism.world_size = world_size;
            parallelism.world_rank = rank;
            parallelism.ep_size    = world_size;
            parallelism.ep_rank    = rank;
            EPLBConfig config;
            config.eplb_metrics_report_interval    = interval;
            config.eplb_metrics_snapshot_max_files = snapshot_max_files;
            return std::make_unique<ExpertLoadMonitor>(
                layers, experts, parallelism, config, model_id, nullptr, model_id == 0 ? "target" : "draft");
        }))
        .def("clear_record_buffer", &ExpertLoadMonitor::clearRecordBuffer)
        .def("record_step", &ExpertLoadMonitor::recordStep)
        .def_property_readonly("counts", &ExpertLoadMonitor::stepCountsBuffer);
    py::class_<EplbPyWrapperOP>(m, "EplbPyWrapperOP")
        .def(py::init<>())
        .def("initialize",
             &EplbPyWrapperOP::initialize,
             py::arg("model"),
             py::arg("ep_rank")  = 0,
             py::arg("ep_size")  = 1,
             py::arg("model_id") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("record_step", &EplbPyWrapperOP::recordStep, py::call_guard<py::gil_scoped_release>())
        .def("balance", &EplbPyWrapperOP::balance, py::call_guard<py::gil_scoped_release>())
        .def("update_config",
             &EplbPyWrapperOP::updateConfig,
             py::arg("mode"),
             py::arg("update_time"),
             py::call_guard<py::gil_scoped_release>())
        .def("destroy", &EplbPyWrapperOP::destroy, py::call_guard<py::gil_scoped_release>());
}
}  // namespace unittest
