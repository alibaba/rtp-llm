#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include "absl/status/status.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include "rtp_llm/cpp/cache_new/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/cache_new/WarmUpResult.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/models/GptModel.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

class NormalEngine: public EngineBase {
public:
    NormalEngine(const EngineInitParams& params);
    ~NormalEngine();

    std::shared_ptr<GenerateStream>   makeStream(const std::shared_ptr<GenerateInput>& input) override;
    std::shared_ptr<GenerateStream>   enqueue(const std::shared_ptr<GenerateInput>& input) override;
    std::vector<GenerateStreamPtr>    batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override;
    void                              enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                             preRunMode                            mode) override;
    absl::Status                      stop() override;

    KVCacheInfo                     getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) override;
    absl::Status                    step();
    absl::Status                    startLoop();
    int64_t                         getLastScheduleTime() override;
    const rtp_llm::GptInitParameter gptInitParameter() const;
    void                            reportMetrics(RtpLLMEngineMetricsCollector collector) {
        if (metrics_reporter_) {
            metrics_reporter_->report<RtpLLMEngineMetrics, RtpLLMEngineMetricsCollector>(nullptr, &collector);
        }
    }
    bool updateEplbConfig(const EplbConfig& config) override;

private:
    void                            initScheduler();
    std::shared_ptr<GenerateStream> createMinFakeStream(int32_t max_new_tokens);
    WarmUpResult                    warmUp(const EngineInitParams& params);
    WarmUpResult                    prefillWarmUp(const EngineInitParams& params);
    WarmUpResult                    decodeWarmUp(const EngineInitParams& params);
    void                            initLoadBalance();
    absl::Status                    trySaveStepError() const;
    void                            loop();
    void                            initCacheManager(std::optional<WarmUpResult> warm_up_result);
    absl::Status                    initSystemPrompt();
    std::shared_ptr<GenerateInput>  makeFakeInput(size_t seq_len);

private:
    autil::ThreadPtr                loop_thread_;
    std::atomic<bool>               running_{false};
    std::unique_ptr<Executor>       executor_;
    const rtp_llm::GptInitParameter params_;
    kmonitor::MetricsReporterPtr    metrics_reporter_;
    std::shared_ptr<CudaProfiler>   profiler_;
    int                             profiler_step_     = 0;
    bool                            gen_timeline_sync_ = false;
};

}  // namespace rtp_llm
