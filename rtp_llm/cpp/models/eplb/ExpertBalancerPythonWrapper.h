#pragma once
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/config/EplbConfig.h"

namespace rtp_llm {
// All calls and destruction acquire the GIL. Initialize after model.initialize()
// and before graph capture; runtime weights and mappings keep their storage.
class ExpertBalancerPythonWrapper {
public:
    ExpertBalancerPythonWrapper() = default;
    ~ExpertBalancerPythonWrapper();
    ExpertBalancerPythonWrapper(const ExpertBalancerPythonWrapper&)            = delete;
    ExpertBalancerPythonWrapper& operator=(const ExpertBalancerPythonWrapper&) = delete;
    void                         initialize(const py::object& model,
                                            const EPLBConfig& config,
                                            size_t            ep_rank,
                                            size_t            ep_size,
                                            size_t            model_id,
                                            bool              is_normal_engine,
                                            bool              sync_prompt_phase);
    // Borrows counts without clearing them; the C++ monitor owns the step reset.
    bool step(const torch::Tensor& counts, bool is_prompt);
    void updateConfig(const EPLBConfig& config);

private:
    py::object runtime_;
};

}  // namespace rtp_llm
