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

class BalancerTestModel: public ModelBase {
public:
    GptModelOutputs forward(const GptModelInputs&) override {
        return {};
    }
};

class ExpertBalancerStepTest {
public:
    ExpertBalancerStepTest(py::object planner,
                           int        snapshot_max_files,
                           int        interval,
                           size_t     model_id,
                           bool       moe,
                           int        tp_rank,
                           int        tp_size,
                           bool       afd) {
        config_.eplb_metrics_report_interval    = interval;
        config_.eplb_metrics_snapshot_max_files = snapshot_max_files;
        config_.eplb_update_time                = 1;
        config_.eplb_control_step               = 1;
        config_.eplb_test_mode                  = true;
        Weights            weights;
        OverallExpertStats expert_stats;
        weights.layers.resize(1);
        if (!planner.is_none()) {
            config_.eplb_mode = EplbMode::EPLB;
            const auto i32    = torch::TensorOptions(torch::kInt32).device(torch::kCUDA);
            const auto bf16   = torch::TensorOptions(torch::kBFloat16).device(torch::kCUDA);
            expert_stats      = {1, 1, 2, 2, {torch::tensor({2, 3}, i32).reshape({1, 2}), torch::full({1, 1}, 5, i32)}};
            auto& ffn_weights = weights.layers[0].ffn_weights;
            ffn_weights.moe_gate_weight         = std::make_shared<DenseWeights>();
            ffn_weights.moe_down_weight         = std::make_shared<DenseWeights>();
            ffn_weights.moe_gate_weight->kernel = torch::zeros({2, 2, 1}, bf16);
            ffn_weights.moe_down_weight->kernel = torch::zeros({2, 1, 1}, bf16);
            ffn_weights.logic_expert_cnt        = torch::ones({2}, i32);
            ffn_weights.log2phy                 = torch::arange(2, i32).reshape({2, 1});
        }
        GptModelDescription description{};
        description.ffn_conf.activation_type = ActivationType::Swiglu;
        if (moe) {
            description.ffn_conf.moe_configs = MoeConfigs{2, 0, 1};
        }
        GptModelInitParams params{weights, description, std::nullopt};
        params.parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate = afd;

        params.hidden_size                   = 1;
        params.model_id                      = model_id;
        params.eplb_config                   = config_;
        params.py_eplb                       = planner;
        params.parallelism_config.tp_rank    = tp_rank;
        params.parallelism_config.tp_size    = tp_size;
        params.parallelism_config.world_rank = tp_rank;
        params.parallelism_config.world_size = tp_size;
        balancer_                            = std::make_unique<ExpertBalancer>(
            description.ffn_conf.moe_configs, params, model_.weights_, model_.overall_expert_stats_);
        model_.weights_              = params.weights;
        model_.overall_expert_stats_ = std::move(expert_stats);
    }

    void step(bool warmup) {
        balancer_->step(warmup);
    }

    bool disablePlanner() {
        config_.eplb_mode = EplbMode::NONE;
        return balancer_->updateEplbConfig(config_);
    }

    void destroy() {
        balancer_.reset();
    }

    torch::Tensor counts() const {
        return balancer_->stable_expert_token_counts();
    }

    torch::Tensor gateWeights() const {
        return model_.weights_.layers[0].ffn_weights.moe_gate_weight->kernel;
    }

    torch::Tensor log2phy() const {
        return model_.weights_.layers[0].ffn_weights.log2phy;
    }

private:
    BalancerTestModel               model_;
    EPLBConfig                      config_;
    std::unique_ptr<ExpertBalancer> balancer_;
};

class EplbPyWrapperOP: public torch::jit::CustomClassHolder {
public:
    void init(py::object py_eplb) {
        eplb_py_wrapper_ = ExpertBalancerPythonWrapper(py_eplb);
    }

    void createBalancePlan(torch::Tensor log_stats, torch::Tensor gpu_loads) {
        eplb_py_wrapper_.createBalancePlan(log_stats, gpu_loads, eplb_plan_);
    }

    void loadMoeWeight(int64_t ep_rank, int64_t ep_size) {
        eplb_py_wrapper_.loadBalanceWeight(ep_rank, ep_size, eplb_plan_);
    }

    EplbPlanTensors::tuple_type getResult() {
        return eplb_plan_.to_tuple();
    }

private:
    ExpertBalancerPythonWrapper eplb_py_wrapper_;
    EplbPlanTensors             eplb_plan_;
};

PYBIND11_MODULE(libth_eplb_py_wrapper_test, m) {
    py::class_<ExpertBalancerStepTest>(m, "ExpertBalancerStepTest")
        .def(py::init<py::object, int, int, size_t, bool, int, int, bool>(),
             py::arg("planner"),
             py::arg("snapshot_max_files"),
             py::arg("interval") = 2,
             py::arg("model_id") = 0,
             py::arg("moe")      = true,
             py::arg("tp_rank")  = 0,
             py::arg("tp_size")  = 1,
             py::arg("afd")      = false)
        .def("step", &ExpertBalancerStepTest::step, py::arg("warmup") = false, py::call_guard<py::gil_scoped_release>())
        .def("disable_planner", &ExpertBalancerStepTest::disablePlanner)
        .def("destroy", &ExpertBalancerStepTest::destroy, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("counts", &ExpertBalancerStepTest::counts)
        .def_property_readonly("gate_weights", &ExpertBalancerStepTest::gateWeights)
        .def_property_readonly("log2phy", &ExpertBalancerStepTest::log2phy);
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
            return ExpertLoadMonitor::create(
                layers, experts, parallelism, config, model_id, nullptr, model_id == 0 ? "target" : "draft");
        }))
        .def("clear_record_buffer", &ExpertLoadMonitor::clearRecordBuffer)
        .def("record_step", &ExpertLoadMonitor::recordStep)
        .def_property_readonly("counts", &ExpertLoadMonitor::stepCountsBuffer);
    py::class_<EplbPyWrapperOP>(m, "EplbPyWrapperOP")
        .def(py::init<>())
        .def("init", &EplbPyWrapperOP::init)
        .def("create_balance_plan", &EplbPyWrapperOP::createBalancePlan, py::arg("log_stats"), py::arg("gpu_loads"))
        .def("load_moe_weight", &EplbPyWrapperOP::loadMoeWeight, py::arg("ep_rank"), py::arg("ep_size"))
        .def("get_result", &EplbPyWrapperOP::getResult);
}
}  // namespace unittest
