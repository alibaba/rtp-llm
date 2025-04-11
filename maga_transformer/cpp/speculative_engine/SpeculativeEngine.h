#pragma once

#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
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
    absl::Status                      init();
    std::shared_ptr<GenerateStream> makeStream(const std::shared_ptr<GenerateInput>& input) override;
    std::shared_ptr<GenerateStream>   enqueue(const std::shared_ptr<GenerateInput>& input) override;
    void enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                             preRunMode                            mode) override;
    absl::Status                      stop() override;
    LoadBalanceInfo                   getLoadBalanceInfo() override;

    const ResourceContext& resourceContext() const {
        return resource_context_;
    }

private:
    WarmUpResult       warmUp();
    void         initLoadBalance();
    absl::Status step();

    // do not walk through speculative process.
    absl::Status normStep(std::list<GenerateStreamPtr>& streams);

    // walk through sp process, but prefill do not propose.
    absl::Status noPrefillProposeStep(std::list<GenerateStreamPtr>& streams);

    // walk through sp process, prefill can propose.
    absl::Status prefillProposeStep(std::list<GenerateStreamPtr>& streams);

    std::list<GenerateStreamPtr> extractPrefillStreams(std::list<GenerateStreamPtr>& streams) {
        std::list<GenerateStreamPtr> need_prefill_streams;
        streams.erase(std::remove_if(streams.begin(),
                                     streams.end(),
                                    [&](GenerateStreamPtr stream) {
                                        if (stream->getLastHiddenStates() == nullptr) {
                                            need_prefill_streams.push_back(stream);
                                            return true;
                                        } else {
                                            return false;
                                        }
                                    }), streams.end());
        return need_prefill_streams;
    };


    absl::Status startLoop();
    void         loop();
    absl::Status trySaveStepError() const;
    absl::Status initCacheManager(std::optional<WarmUpResult> warm_up_result);
    absl::Status initSystemPrompt();
    void         tpSyncDisableSPRun(bool& all_streams_disable_sp_run);
    void         reportMetrics(int64_t                         propose_begin_time_us,
                               int64_t                         score_begin_time_us,
                               int64_t                         sampler_begin_time_us,
                               int64_t                         update_begin_time_us,
                               int64_t                         total_propose_token_num,
                               int64_t                         total_accepted_token_num);


    bool checkAllHasHiddenStates(std::list<GenerateStreamPtr>& streams);

    std::list<GenerateStreamPtr> extractFirstPrefillStreams(std::list<GenerateStreamPtr>& streams);

private:
    kmonitor::MetricsReporterPtr                  metrics_reporter_ = nullptr;
    std::unique_ptr<ProposeModelEngineInitParams> propose_model_params_;
    const EngineInitParams                        score_model_params_;

    std::unique_ptr<ProposeExecutor>          propose_executor_    = nullptr;
    std::unique_ptr<ScoreExecutor>            score_executor_      = nullptr;
    std::unique_ptr<SpeculativeOnlineAdaptor> online_adaptor_      = nullptr;
    std::unique_ptr<SpeculativeSampler>       speculative_sampler_ = nullptr;
    std::unique_ptr<SpeculativeUpdater>       speculative_updater_ = nullptr;
    std::shared_ptr<SystemPrompt>             system_prompt_       = nullptr;

    const std::string sp_type_;
    std::thread       loop_thread_;
    std::atomic<bool> running_{false};
    ResourceContext   resource_context_;
    StepRecorder      step_recorder_;
};

}  // namespace rtp_llm
