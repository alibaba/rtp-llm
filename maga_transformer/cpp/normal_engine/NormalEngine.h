#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include "absl/status/status.h"
#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/cache/CacheManager.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/engine_base/Executor.h"
#include "maga_transformer/cpp/models/GptModel.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "maga_transformer/cpp/system_prompt/SystemPrompt.h"
#include "kmonitor/client/MetricsReporter.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"

namespace rtp_llm {

class NormalEngine: public EngineBase {
public:
    NormalEngine(const EngineInitParams& params);
    ~NormalEngine();

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) override;
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input, preRunMode mode) override;
    absl::Status                    stop() override;
    LoadBalanceInfo                 getLoadBalanceInfo() override;
    absl::Status step();
    absl::Status startLoop();
    void reportMetrics(RtpLLMEngineMetricsCollector collector) {
        if (metrics_reporter_) {
            metrics_reporter_->report<RtpLLMEngineMetrics, RtpLLMEngineMetricsCollector>(nullptr, &collector);
        }
    }

public:
    // just for test
    const ResourceContext& resourceContext() const {
        return resource_context_;
    }

private:
    size_t warmUp(const EngineInitParams& params);
    absl::Status trySaveStepError() const;
    void         loop();
    void         initCacheManager(size_t kv_cache_mem_size);
    absl::Status initSystemPrompt();

private:
    std::thread                    loop_thread_;
    std::atomic<bool>              running_{false};
    std::unique_ptr<Executor>      executor_;
    std::unique_ptr<SchedulerBase> scheduler_;
    const ft::GptInitParameter     params_;
    ResourceContext                resource_context_;
    StepRecorder                   step_recorder_;
    kmonitor::MetricsReporterPtr   metrics_reporter_ = nullptr;
};

}  // namespace rtp_llm
