#pragma once

#include <cstdint>
#include <memory>
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

    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::unique_ptr<Writer>      writer_;
};

}  // namespace rtp_llm
