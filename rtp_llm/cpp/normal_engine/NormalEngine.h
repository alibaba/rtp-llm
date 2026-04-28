#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include "absl/status/status.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/cache/WarmUpResult.h"
#include "rtp_llm/cpp/engine_base/Executor.h"
#include "rtp_llm/cpp/models/ModelTypes.h"
#include "rtp_llm/cpp/normal_engine/BatchFuture.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

class NormalEngine: public EngineBase {
public:
    NormalEngine(const EngineInitParams& params, std::unique_ptr<ProposeModelEngineInitParams> propose_params);
    ~NormalEngine();

    std::shared_ptr<GenerateStream>   makeStream(const std::shared_ptr<GenerateInput>& input) override;
    std::shared_ptr<GenerateStream>   enqueue(const std::shared_ptr<GenerateInput>& input) override;
    std::vector<GenerateStreamPtr>    batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override;
    void                              enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                             preRunMode                            mode) override;
    absl::Status                      stop() override;

    KVCacheInfo  getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) override;
    absl::Status step();
    // Async-scheduling entry point. When use_async_scheduling_ is
    // false (the default), this delegates to step() to preserve the current
    // synchronous behaviour bit-for-bit. When enabled, the main thread will
    // schedule + launch a batch and hand a BatchFuture to the result thread
    // for bookkeeping (specUpdate / KV release / output dispatch), letting
    // the next schedule + launch overlap with the previous batch's GPU work.
    absl::Status asyncStep();
    absl::Status startLoop();
    int64_t      getLastScheduleTime() override;
    void         reportMetrics(RtpLLMEngineMetricsCollector collector) {
        if (metrics_reporter_) {
            metrics_reporter_->report<RtpLLMEngineMetrics, RtpLLMEngineMetricsCollector>(nullptr, &collector);
        }
    }
    bool updateEplbConfig(const EPLBConfig& config) override;
    void startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) override;

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
    void                            mayAddFakeStream(std::list<GenerateStreamPtr>& streams);

    void initExecutor(const EngineInitParams& params, std::unique_ptr<ProposeModelEngineInitParams>& propose_params);

    bool isMTPEagle() override;
    bool isEagle() override;

private:
    // Drives the result-thread bookkeeping loop. Started lazily
    // when async scheduling is enabled; otherwise the synchronous path
    // never touches it.
    void resultLoop();
    // Wait for any outstanding BatchFuture to finish bookkeeping so the
    // next scheduler call sees the post-update stream state. Returns the
    // bookkeeping status so the caller can short-circuit on error.
    absl::Status awaitLastBookkeeping();

private:
    autil::ThreadPtr          loop_thread_;
    std::atomic<bool>         running_{false};
    std::unique_ptr<Executor> executor_;
    // Async scheduling state. The synchronous path leaves these untouched;
    // RTP_LLM_ASYNC_SCHEDULING controls whether the result thread is used.
    bool                                          use_async_scheduling_ = false;
    std::thread                                   result_thread_;
    std::mutex                                    result_mutex_;
    std::condition_variable                       result_cv_;
    std::queue<BatchFuturePtr>                    result_queue_;
    BatchFuturePtr                                last_future_;
    std::atomic<bool>                             result_thread_stop_{false};
    ModelConfig                                   model_config_;
    ParallelismConfig                             parallelism_config;
    RuntimeConfig                                 runtime_config;
    EPLBConfig                                    eplb_config;
    PDSepConfig                                   pd_sep_config;
    ProfilingDebugLoggingConfig                   profiling_debug_logging_config;
    KVCacheConfig                                 kv_cache_config;
    FfnDisAggregateConfig                         ffn_disaggregate_config;
    ModelSpecificConfig                           model_specific_config;
    SpeculativeExecutionConfig                    sp_config;
    kmonitor::MetricsReporterPtr                  metrics_reporter_;
    std::unique_ptr<ProposeModelEngineInitParams> propose_params_;
    StepWindowProfiler                            step_profiler_;
    int                                           reserve_step_ = 0;
};

}  // namespace rtp_llm
