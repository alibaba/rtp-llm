#pragma once

#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeOnlineAdaptor.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeExecutor.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreExecutor.h"
#include "maga_transformer/cpp/speculative_engine/speculative_sampler/SpeculativeSampler.h"
#include "maga_transformer/cpp/speculative_engine/speculative_updater/SpeculativeUpdater.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class SpeculativeEngine: public EngineBase {
public:
    explicit SpeculativeEngine(const EngineInitParams&                       engine_init_params,
                               std::unique_ptr<ProposeModelEngineInitParams> propose_model_engine_init_params);
    ~SpeculativeEngine();
    absl::Status                    init();
    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) override;
    absl::Status                    stop() override;
    KVCacheInfo                     getKVCacheInfo() const override;

private:
    absl::Status step();
    absl::Status startLoop();
    void         loop();
    absl::Status trySaveStepError() const;
    absl::Status initCacheManager();
    void         initSystemPrompt();
    void         reportMetrics(const SpeculativeSamplerOutput& sampler_output,
                               int64_t                         propose_begin_time_us,
                               int64_t                         score_begin_time_us,
                               int64_t                         sampler_begin_time_us,
                               int64_t                         update_begin_time_us);

private:
    kmonitor::MetricsReporterPtr metrics_reporter_ = nullptr;
    MetricsLoopReporter<RtpLLMSpeculativeEngineMetrics, RtpLLMSpeculativeEngineMetricsCollector>
                                                  speculative_engine_reporter_;
    std::unique_ptr<ProposeModelEngineInitParams> propose_model_params_;
    const EngineInitParams                        score_model_params_;

    std::unique_ptr<ProposeExecutor>          propose_executor_    = nullptr;
    std::unique_ptr<ScoreExecutor>            score_executor_      = nullptr;
    std::unique_ptr<SpeculativeOnlineAdaptor> online_adaptor_      = nullptr;
    std::unique_ptr<SpeculativeSampler>       speculative_sampler_ = nullptr;
    std::unique_ptr<SchedulerBase>            scheduler_           = nullptr;
    std::unique_ptr<SpeculativeUpdater>       speculative_updater_ = nullptr;
    std::shared_ptr<SystemPrompt>             system_prompt_       = nullptr;

    std::thread       loop_thread_;
    std::atomic<bool> running_{false};
    ResourceContext   resource_context_;
};

}  // namespace rtp_llm