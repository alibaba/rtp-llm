#pragma once
#include <memory>
#include <mutex>
#include <optional>
#include <torch/extension.h>
#include "rtp_llm/cpp/models/eplb/ExpertBalancerPythonWrapper.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {
class ExpertLoadMonitor;
struct GptModelInitParams;

class EplbController {
public:
    void init(const EPLBConfig& config, size_t world_size);
    void setData(const EPLBConfig& config);
    // All ranks call at the same boundary; only root's request is published.
    std::optional<EPLBConfig> step();

private:
    std::mutex    mutex_;
    EPLBConfig    requested_;
    torch::Tensor host_;
    torch::Tensor device_;
    size_t        world_size_ = 1;
    int           interval_   = 100;
    int           steps_      = 0;
};

class ExpertBalancer {
public:
    __attribute__((visibility("default"))) ExpertBalancer(const std::optional<MoeConfigs>& moe,
                                                          const GptModelInitParams&        params);
    ~ExpertBalancer();
    void initialize(const py::object& model);
    // Forward completed on the current GPU stream. Warmup discards counts;
    // real steps sample before clearing, then may migrate.
    void step(bool warmup = false);
    bool updateEplbConfig(const EPLBConfig& config);
    // Fixed int32 [model_layers, local_physical] storage, shared with graph capture.
    const torch::Tensor& stable_expert_token_counts() const {
        return expert_token_counts_;
    }

private:
    EPLBConfig                         config_;
    bool                               runtime_requested_ = false;
    size_t                             ep_rank_           = 0;
    size_t                             ep_size_           = 1;
    size_t                             model_id_          = 0;
    std::unique_ptr<ExpertLoadMonitor> load_monitor_;
    torch::Tensor                      expert_token_counts_;
    EplbController                     controller_;
    kmonitor::MetricsReporterPtr       metrics_reporter_;
    ExpertBalancerPythonWrapper        python_;
};
}  // namespace rtp_llm
