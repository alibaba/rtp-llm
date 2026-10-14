#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <torch/torch.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

// Each model's balancer owns its monitor. All forward methods are called from the
// serialized model-forward path on its current GPU producing stream. Full
// windows use the existing GPU collective, then rank 0 copies to the CPU and
// reports synchronously.
class ExpertLoadMonitor {
public:
    ExpertLoadMonitor(size_t                       layer_count,
                      size_t                       physical_expert_count,
                      const ParallelismConfig&     parallelism,
                      const EPLBConfig&            config,
                      size_t                       model_id,
                      kmonitor::MetricsReporterPtr metrics_reporter,
                      std::string                  model_type = "target");
    ~ExpertLoadMonitor();

    // Int32 accumulators at a fixed address for eager kernels and CUDA graph capture.
    const torch::Tensor& stepCountsBuffer() const;
    // Clear graph capture/warmup counts without recording a forward.
    void clearRecordBuffer();
    // Commit the forward's counts to the int64 window and clear the record buffer.
    void recordStep();
    // Discard an unfinished window at a coordinated map transition.
    void beforeMappingChange();

    ExpertLoadMonitor(const ExpertLoadMonitor&)            = delete;
    ExpertLoadMonitor& operator=(const ExpertLoadMonitor&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rtp_llm
