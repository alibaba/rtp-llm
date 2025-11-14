#pragma once
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include <cstdint>

namespace rtp_llm {

class NormalGenerateStream: public GenerateStream {
public:
    NormalGenerateStream(const GenerateStream& stream): GenerateStream(stream) {
        CopyOnWrite(stream);
        generate_outputs_queue_.setCapacity(1000);
    }

    NormalGenerateStream(const std::shared_ptr<GenerateInput>& query,
                         const ModelConfig&                    model_config,
                         const RuntimeConfig&                  runtime_config,
                         const ResourceContext&                resource_context,
                         kmonitor::MetricsReporterPtr          metrics_reporter,
                         size_t                                extra_reserve_token_num = 0,
                         bool                                  perf_test               = false):
        GenerateStream(query, model_config, runtime_config, resource_context, metrics_reporter, extra_reserve_token_num, perf_test),
        request_id_(query->request_id) {
        generate_outputs_queue_.setCapacity(1000);
    }

    ~NormalGenerateStream() {
        generate_outputs_queue_.wakeup();
    }

    bool                         hasOutput() override;
    ErrorResult<GenerateOutputs> nextOutput() override;
    void                         updateOutput(const StreamUpdateInfo& update_info) override;

private:
    GenerateOutputs prepareGenerateOutput(const StreamUpdateInfo& update_info);
    void            enqueueGenerateOutput(GenerateOutputs&& generate_results);

    int64_t                                   request_id_{0};
    bool                                      finished_{false};
    autil::SynchronizedQueue<GenerateOutputs> generate_outputs_queue_;
};
}  // namespace rtp_llm
