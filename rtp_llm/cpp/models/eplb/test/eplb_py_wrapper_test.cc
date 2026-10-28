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
                           std::optional<int> mode,
                           bool               afd,
                           int                pp_size,
                           bool               context_parallel,
                           const std::string& role_type,
                           int                dp_size,
                           const std::string& sp_type,
                           int64_t            update_time) {
        EPLBConfig config;
        config.eplb_metrics_report_interval    = interval;
        config.eplb_metrics_snapshot_max_files = snapshot_max_files;
        config.eplb_update_time                = update_time;
        config.eplb_stats_window_size          = 1;
        config.eplb_mode = mode ? static_cast<EplbMode>(*mode) : (model.is_none() ? EplbMode::NONE : EplbMode::EPLB);
        Weights weights;
        weights.layers.resize(1);
        GptModelDescription description{};
        description.ffn_conf.activation_type = ActivationType::Swiglu;
        if (moe) {
            description.ffn_conf.moe_configs = MoeConfigs{2, 0, 1};
        }
        GptModelInitParams params{weights, description, std::nullopt};
        params.parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate = afd;

        params.hidden_size                = 1;
        params.model_id                   = model_id;
        params.eplb_config                = config;
        params.parallelism_config.pp_size = pp_size;
        if (context_parallel) {
            params.parallelism_config.prefill_cp_config.method = CPRotateMethod::ALL_GATHER;
        }
        params.sp_config.type                = SpeculativeExecutionConfig::from_string(sp_type);
        params.parallelism_config.role_type  = role_type == "PDFUSION" ?
                                                   RoleType::PDFUSION :
                                                   (role_type == "DECODE" ? RoleType::DECODE : RoleType::PREFILL);
        params.parallelism_config.dp_size    = dp_size;
        params.parallelism_config.ep_size    = dp_size > 1 ? tp_size * dp_size : 1;
        params.parallelism_config.ep_rank    = 0;
        params.parallelism_config.tp_rank    = tp_rank;
        params.parallelism_config.tp_size    = tp_size;
        params.parallelism_config.world_rank = tp_rank;
        params.parallelism_config.world_size = tp_size * dp_size;
        balancer_ = std::make_unique<ExpertBalancer>(description.ffn_conf.moe_configs, params);
        balancer_->initialize(model);
    }

    void step(bool warmup, bool is_prompt) {
        balancer_->step(warmup, is_prompt);
    }

    void destroy() {
        balancer_.reset();
    }

    torch::Tensor counts() const {
        return balancer_->stable_expert_token_counts();
    }

private:
    std::unique_ptr<ExpertBalancer> balancer_;
};

class EplbPyWrapperOP {
public:
    void initialize(const py::object& model,
                    size_t            ep_rank,
                    size_t            ep_size,
                    size_t            model_id,
                    bool              is_normal_engine,
                    bool              sync_prompt_phase) {
        EPLBConfig config;
        config.eplb_mode                    = EplbMode::ALL;
        config.eplb_update_time             = 7;
        config.eplb_stats_window_size       = 3;
        config.eplb_rearrange_after_prefill = true;
        config.eplb_rearrange_p2p_groups    = 2;
        wrapper_                            = std::make_unique<ExpertBalancerPythonWrapper>();
        wrapper_->initialize(model, config, ep_rank, ep_size, model_id, is_normal_engine, sync_prompt_phase);
    }

    bool step(const torch::Tensor& counts, bool is_prompt) {
        return wrapper_->step(counts, is_prompt);
    }

    void destroy() {
        wrapper_.reset();
    }

private:
    std::unique_ptr<ExpertBalancerPythonWrapper> wrapper_;
};

PYBIND11_MODULE(libth_eplb_py_wrapper_test, m) {
    py::class_<ExpertBalancerStepTest>(m, "ExpertBalancerStepTest")
        .def(py::init<const py::object&,
                      int,
                      int,
                      size_t,
                      bool,
                      int,
                      int,
                      std::optional<int>,
                      bool,
                      int,
                      bool,
                      const std::string&,
                      int,
                      const std::string&,
                      int64_t>(),
             py::arg("model"),
             py::arg("snapshot_max_files"),
             py::arg("interval")         = 2,
             py::arg("model_id")         = 0,
             py::arg("moe")              = true,
             py::arg("tp_rank")          = 0,
             py::arg("tp_size")          = 1,
             py::arg("mode")             = py::none(),
             py::arg("afd")              = false,
             py::arg("pp_size")          = 1,
             py::arg("context_parallel") = false,
             py::arg("role_type")        = "PREFILL",
             py::arg("dp_size")          = 1,
             py::arg("sp_type")          = "none",
             py::arg("update_time")      = 1)
        .def("step",
             &ExpertBalancerStepTest::step,
             py::arg("warmup")    = false,
             py::arg("is_prompt") = false,
             py::call_guard<py::gil_scoped_release>())
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
            return ExpertLoadMonitor::create(
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
             py::arg("ep_rank")           = 0,
             py::arg("ep_size")           = 1,
             py::arg("model_id")          = 0,
             py::arg("is_normal_engine")  = false,
             py::arg("sync_prompt_phase") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("step",
             &EplbPyWrapperOP::step,
             py::arg("counts"),
             py::arg("is_prompt") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("destroy", &EplbPyWrapperOP::destroy, py::call_guard<py::gil_scoped_release>());
}
}  // namespace unittest
