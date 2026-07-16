#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

class ModelInputsLogger {
public:
    ModelInputsLogger(int64_t rank_id, int backup_count, kmonitor::MetricsReporterPtr metrics_reporter);
    ~ModelInputsLogger();

    void log(const GptModelInputs& inputs);

private:
    class Writer;

    int64_t                                    rank_id_{0};
    int                                        backup_count_{0};
    kmonitor::MetricsReporterPtr               metrics_reporter_;
    std::once_flag                             init_once_;
    std::unique_ptr<ModelInputsLogger::Writer> writer_;
};

}  // namespace rtp_llm
