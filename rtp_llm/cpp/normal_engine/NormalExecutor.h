#pragma once

#include <memory>
#include <optional>
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/models/lora/LoraManager.h"
#include "rtp_llm/cpp/models/eplb/ExpertBalancer.h"

namespace rtp_llm {

class NormalExecutor: public Executor {
public:
    explicit NormalExecutor(const EngineInitParams&                   params,
                            const std::shared_ptr<KVCacheManager>&    cache_manager,
                            rtp_llm::DeviceBase*                      device,
                            const std::shared_ptr<lora::LoraManager>& lora_manager        = nullptr,
                            bool                                      warm_up             = false,
                            bool                                      is_propose          = false,
                            int                                       propose_model_index = 0);
    ~NormalExecutor() {
        device_->profileStop();
    }
    absl::Status process(const std::list<GenerateStreamPtr>& streams) override;
    void         reportMetrics(const StreamGroups&             stream_groups,
                               RtpLLMExecutorMetricsCollector& executor_collector,
                               RtpLLMTokenPSMetricsCollector&  tps_collector);

    void setBatchProcessor(std::unique_ptr<NormalBatchStreamProcessor> processor) {
        batch_stream_processor_ = std::move(processor);
    }

    void setGptModel(std::unique_ptr<GptModel> model) {
        model_ = std::move(model);
    }

    bool updateEplbConfig(const EPLBConfig& config) override;

private:
    std::unique_ptr<GptModel>                                                model_;
    std::unique_ptr<Sampler>                                                 sampler_;
    std::unique_ptr<NormalBatchStreamProcessor>                              batch_stream_processor_;
    std::shared_ptr<KVCacheManager>                                          cache_manager_;
    std::shared_ptr<lora::LoraManager>                                       lora_manager_;
    std::shared_ptr<ExpertBalancer>                                          expert_balancer_;
    bool                                                                     warm_up_;
    bool                                                                     use_all_gather_;
    kmonitor::MetricsReporterPtr                                             metrics_reporter_ = nullptr;
    MetricsLoopReporter<RtpLLMTokenPSMetrics, RtpLLMTokenPSMetricsCollector> tps_reporter_;
    bool                                                                     is_ffn_service_ = false;
    bool                                                                     enable_detail_log_       = false;

    bool is_propose_          = false;
    int  propose_model_index_ = 0;
};

}  // namespace rtp_llm
