#pragma once
#include <memory>
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

class ExpertBalancer {
public:
    __attribute__((visibility("default"))) ExpertBalancer(const std::optional<MoeConfigs>& moe,
                                                          const GptModelInitParams&        params);
    ~ExpertBalancer();
    void initialize(const py::object& model);
    // Forward completed on the current GPU stream. Warmup discards counts;
    // Real steps sample/migrate before the monitor records and clears counts.
    void step(bool warmup = false, bool is_prompt = false);
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
    bool                               is_normal_engine_  = false;
    bool                               sync_prompt_phase_ = false;
    std::unique_ptr<ExpertLoadMonitor> load_monitor_;
    torch::Tensor                      expert_token_counts_;
    kmonitor::MetricsReporterPtr       metrics_reporter_;
    ExpertBalancerPythonWrapper        python_;
};
}  // namespace rtp_llm
