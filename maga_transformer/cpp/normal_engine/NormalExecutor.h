#pragma once

#include <memory>
#include "kmonitor/client/MetricsReporter.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "src/fastertransformer/core/Types.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/lora/LoraManager.h"

namespace rtp_llm {

class NormalExecutor: public Executor {
public:
    explicit NormalExecutor(const EngineInitParams& params,
                            const std::shared_ptr<CacheManager>& cache_manager,
                            ft::DeviceBase* device,
                            const std::shared_ptr<lora::LoraManager>& lora_manager = nullptr,
                            bool warm_up = false);
    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    void reportMetrics(const StreamGroups& stream_groups,
                       RtpLLMExecutorMetricsCollector& executor_collector,
                       RtpLLMTokenPSMetricsCollector& tps_collector);

    void setBatchProcessor(std::unique_ptr<NormalBatchStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
    }

    void setGptModel(std::unique_ptr<GptModel> model) {
        model_ = std::move(model);
    }

private:
    std::unique_ptr<GptModel>                   model_;
    std::unique_ptr<Sampler>                    sampler_;
    std::unique_ptr<NormalBatchStreamProcessor> batch_stream_processor_;
    std::shared_ptr<CacheManager>               cache_manager_;
    std::shared_ptr<lora::LoraManager>          lora_manager_;
    bool                                        warm_up_;
    kmonitor::MetricsReporterPtr                metrics_reporter_ = nullptr;
    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter_;
};

}  // namespace rtp_llm
