#pragma once
#include <cstdint>
#include <memory>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
namespace rtp_llm {
enum class ModelInputsModelRole {
    NORMAL,
    TARGET,
    DRAFT,
    DRAFT_PREFILL,
};
class ModelInputsLogger {
public:
    ModelInputsLogger(int64_t rank_id, int backup_count, kmonitor::MetricsReporterPtr metrics_reporter);
    ~ModelInputsLogger();
    void log(const GptModelInputs& inputs, ModelInputsModelRole role, int64_t model_id);

private:
    class Worker;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::unique_ptr<Worker>      worker_;
};
}  // namespace rtp_llm
