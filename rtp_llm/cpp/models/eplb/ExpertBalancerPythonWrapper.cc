#include "rtp_llm/cpp/models/eplb/ExpertBalancerPythonWrapper.h"

namespace rtp_llm {
ExpertBalancerPythonWrapper::~ExpertBalancerPythonWrapper() {
    if (runtime_) {
        py::gil_scoped_acquire gil;
        runtime_ = py::object();
    }
}

void ExpertBalancerPythonWrapper::initialize(const py::object& model,
                                             const EPLBConfig& config,
                                             size_t            ep_rank,
                                             size_t            ep_size,
                                             size_t            model_id,
                                             bool              is_normal_engine,
                                             bool              sync_prompt_phase) {
    py::gil_scoped_acquire gil;
    // Avoid importing the service config DSO into the independent Python policy.
    auto options                                 = py::module_::import("types").attr("SimpleNamespace")();
    options.attr("eplb_mode")                    = static_cast<int>(config.eplb_mode);
    options.attr("eplb_update_time")             = config.eplb_update_time;
    options.attr("eplb_stats_window_size")       = config.eplb_stats_window_size;
    options.attr("eplb_rearrange_after_prefill") = config.eplb_rearrange_after_prefill;
    options.attr("eplb_rearrange_p2p_groups")    = config.eplb_rearrange_p2p_groups;
    runtime_                                     = py::module_::import("rtp_llm.eplb.runtime")
                   .attr("RuntimeExpertBalancer")(model.attr("get_eplb_layers")(),
                                                  options,
                                                  ep_rank,
                                                  ep_size,
                                                  model_id,
                                                  is_normal_engine,
                                                  sync_prompt_phase);
}

bool ExpertBalancerPythonWrapper::step(const torch::Tensor& counts, bool is_prompt) {
    py::gil_scoped_acquire gil;
    return runtime_.attr("step")(counts, is_prompt).cast<bool>();
}

void ExpertBalancerPythonWrapper::updateConfig(const EPLBConfig& config) {
    py::gil_scoped_acquire gil;
    runtime_.attr("update_config")(static_cast<int>(config.eplb_mode), config.eplb_update_time);
}
}  // namespace rtp_llm
