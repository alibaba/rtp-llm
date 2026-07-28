#pragma once
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include <cstdint>
#include <deque>

namespace rtp_llm {

class NormalGenerateStream: public GenerateStream {
public:
    NormalGenerateStream(const GenerateStream& stream): GenerateStream(stream) {
        CopyOnWrite(stream);
    }

    NormalGenerateStream(const std::shared_ptr<GenerateInput>& query,
                         const ModelConfig&                    model_config,
                         const RuntimeConfig&                  runtime_config,
                         const ResourceContext&                resource_context,
                         kmonitor::MetricsReporterPtr          metrics_reporter,
                         size_t                                extra_reserve_token_num = 0,
                         bool                                  perf_test               = false):
        GenerateStream(query,
                       model_config,
                       runtime_config,
                       resource_context,
                       metrics_reporter,
                       extra_reserve_token_num,
                       perf_test),
        request_id_(query->request_id) {}

    bool                         hasOutput() override;
    ErrorResult<GenerateOutputs> nextOutput(int64_t wait_timeout_ms = 0) override;
    void                         updateOutput(const StreamUpdateInfo& update_info) override;

private:
    GenerateOutputs prepareGenerateOutput(const StreamUpdateInfo& update_info);
    void            enqueueGenerateOutput(GenerateOutputs&& generate_results);
    bool            consumerReadyWithoutLock() const override;

    static constexpr size_t kOutputCapacity = 1000;

    int64_t                     request_id_{0};
    bool                        finished_{false};
    std::deque<GenerateOutputs> generate_outputs_;
};
}  // namespace rtp_llm
