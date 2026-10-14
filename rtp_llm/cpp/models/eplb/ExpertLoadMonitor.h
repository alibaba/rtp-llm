#pragma once

#include <cstddef>
#include <memory>
#include <torch/torch.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

// One main-model instance owns one monitor. All methods are called from the
// serialized model-forward path on its current GPU producing stream.
class ExpertLoadMonitor {
public:
    ExpertLoadMonitor(size_t                       layer_count,
                      size_t                       physical_expert_count,
                      const ParallelismConfig&     parallelism,
                      const EPLBConfig&            config,
                      size_t                       model_id,
                      kmonitor::MetricsReporterPtr metrics_reporter);
    ~ExpertLoadMonitor();

    const torch::Tensor& stepCounts() const;
    void                 beginStep(bool warmup);
    void                 finishStep();
    void                 abortStep() noexcept;
    // Only call at an all-rank coordinated map transition, before remapping.
    void beforeMappingChange();

    ExpertLoadMonitor(const ExpertLoadMonitor&)            = delete;
    ExpertLoadMonitor& operator=(const ExpertLoadMonitor&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
