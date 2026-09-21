#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/config/RankLayout.h"

namespace rtp_llm {
void registerExecCtxOps(pybind11::module& m);

namespace {

class StatsOnlyModel: public ModelBase {
public:
    GptModelOutputs forward(const GptModelInputs&) override {
        return {};
    }
};

// Keep all access to balancer internals in this test extension. Communication,
// controller synchronization and plan creation execute the production methods.
py::dict exerciseStage(const ParallelismConfig& config,
                       py::object               python_balancer,
                       torch::Tensor            log_stats,
                       torch::Tensor            gpu_loads) {
    const auto layout = RankLayout::fromParallelismConfig(config);
    const auto range  = layout.myLayerRange(4);
    EPLBConfig initial;
    initial.eplb_control_step = 1;
    ExpertBalancer balancer(2, 4, 4, 8, 8, config, range, python_balancer,
                            DataType::TYPE_FP16, QuantAlgo(), nullptr, initial);
    StatsOnlyModel model;
    model.overall_expert_stats_.stats_buf.log_stats_buf = log_stats;
    model.overall_expert_stats_.stats_buf.gpu_loads_buf = gpu_loads;

    if (balancer.is_eplb_group_root_) {
        EPLBConfig updated;
        updated.eplb_mode = config.pp_rank == 0 ? EplbMode::EPLB : EplbMode::ALL;
        updated.eplb_update_time = 100 + config.pp_rank;
        balancer.updateEplbConfig(updated);
    }

    py::dict result;
    RtpLLMExecutorMetricsCollector collector;
    balancer.stepForward(model, collector, false);
    result["mode"] = static_cast<int>(balancer.eplb_control_data_.eplb_mode);
    result["update_time"] = balancer.eplb_control_data_.eplb_update_time;
    result["fake_step_count"] = balancer.update_cnt_;
    result["fake_step_stats"] = balancer.stats_.log_stats_gpu.cpu();
    result["fake_step_loads"] = balancer.stats_.gpu_loads_gpu.cpu();
    balancer.stepForward(model, collector, true);
    result["real_step_count"] = balancer.update_cnt_;

    balancer.createPlan();
    result["log_stats"] = balancer.stats_.log_stats.clone();
    result["gpu_loads"] = balancer.stats_.gpu_loads.clone();
    result["layer_id"] = balancer.eplb_plan_tensors_.layer_id_buf.item<int>();
    result["logic_expert_cnt"] = balancer.eplb_plan_tensors_.logic_expert_cnt.clone();
    result["log2phy"] = balancer.eplb_plan_tensors_.log2phy.clone();
    result["phy2log"] = balancer.eplb_plan_tensors_.phy2log.clone();

    // Stage 0 is ready; the other stage has one rank still loading.
    balancer.load_flags_.setReady(config.pp_rank == 0 || config.ep_rank == 0);
    result["partially_ready"] = balancer.syncPlanWeightsLoadStatus();
    balancer.load_flags_.setReady(true);
    result["all_ready"] = balancer.syncPlanWeightsLoadStatus();

    balancer.metrics_reporter_ = std::make_shared<kmonitor::MetricsReporter>("", "", kmonitor::MetricsTags());
    balancer.eplb_control_data_.eplb_mode = EplbMode::STATS;
    balancer.executor_collector_.update_weights_qps = false;
    balancer.reportStats(model.overall_expert_stats_);
    result["metrics_layer_begin"] = balancer.executor_collector_.layer_begin;
    result["metrics_gpu_loads"] = balancer.executor_collector_.gpu_loads;
    return result;
}

}  // namespace

PYBIND11_MODULE(libth_eplb_stage_test, m) {
    registerExecCtxOps(m);
    m.def("exercise_stage", &exerciseStage);
}

}  // namespace rtp_llm
