#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include "absl/status/status.h"
#include "kmonitor/client/MetricsReporter.h"
#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"

namespace rtp_llm {

class NormalEngine: public EngineBase {
public:
    NormalEngine(const EngineInitParams& params);
    ~NormalEngine();

    std::shared_ptr<GenerateStream> makeStream(const std::shared_ptr<GenerateInput>& input) override;
    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) override;
    void enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input, preRunMode mode) override;
    absl::Status                    stop() override;
    LoadBalanceInfo                 getLoadBalanceInfo() override;
    absl::Status step();
    absl::Status startLoop();
    int64_t getLastScheduleTime() override;
    const rtp_llm::GptInitParameter gptInitParameter() const;
    void reportMetrics(RtpLLMEngineMetricsCollector collector) {
        if (metrics_reporter_) {
            metrics_reporter_->report<RtpLLMEngineMetrics, RtpLLMEngineMetricsCollector>(nullptr, &collector);
        }
    }

private:
    void         initScheduler();
    std::shared_ptr<GenerateStream> enqueueMinFakeQuery(int32_t max_new_tokens);
    WarmUpResult warmUp(const EngineInitParams& params);
    WarmUpResult prefillWarmUp(const EngineInitParams& params);
    WarmUpResult decodeWarmUp(const EngineInitParams& params);
    void         initLoadBalance();
    absl::Status trySaveStepError() const;
    void         loop();
    void         initCacheManager(std::optional<WarmUpResult> warm_up_result);
    absl::Status initSystemPrompt();
    std::shared_ptr<GenerateInput> makeFakeInput(size_t seq_len);

private:
    autil::ThreadPtr               loop_thread_;
    std::atomic<bool>              running_{false};
    std::unique_ptr<Executor>      executor_;
    const rtp_llm::GptInitParameter     params_;
    StepRecorder                   step_recorder_;
    kmonitor::MetricsReporterPtr   metrics_reporter_;
};

}  // namespace rtp_llm
