#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
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
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/system_prompt/SystemPrompt.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

class NormalEngine: public EngineBase {
public:
    NormalEngine(const EngineInitParams& params, std::unique_ptr<ProposeModelEngineInitParams> propose_params);
    ~NormalEngine();

    std::shared_ptr<GenerateStream> makeStream(const std::shared_ptr<GenerateInput>& input) override;
    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) override;
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
         enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override;
    void enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                             preRunMode                            mode) override;
    absl::Status                      stop() override;

    KVCacheInfo  getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) override;
    absl::Status step();
    absl::Status pp_step();
    absl::Status startLoop();
    absl::Status waitStartupResult(std::chrono::milliseconds timeout) override;
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
    void                            initCacheConfigForSystemPrompt();
    absl::Status                    buildAndInstallSystemPrompt();

    // Single per-round execution step shared by PP serving (pp_step) and the system-prompt
    // bootstrap (driveSystemPromptBuild), so both drive the executor identically. Future PP
    // DP/CP lane coordination (e.g. fake participation for empty lanes) belongs here, added once.
    absl::Status executeOneRound(const ScheduleOutput& schedule_output, int64_t schedule_time_us = 0);

    // Completion contract for running one system-prompt build stream through the engine.
    // Wrapping the inflight pump behind this handle keeps the raw loop out of the build driver
    // and gives MTP support a single seam to add the rejected-before-submit path.
    enum class BuildRunOutcome {
        kDispatched,
        kRejectedBeforeSubmit,
        kInterrupted
    };
    struct BuildRunResult {
        BuildRunOutcome outcome;
        absl::Status    status;
    };
    absl::StatusOr<BuildRunResult> driveSystemPromptBuild(const GenerateStreamPtr& stream);
    absl::Status                   buildSystemPromptsDirect();
    bool                           isFirstStageRoot() const;
    bool                           buildsSystemPromptsOnLoopThread() const;
    void                           publishStartupReady();
    void                           publishStartupFailed(absl::Status error);
    std::shared_ptr<GenerateInput> makeFakeInput(size_t seq_len);
    size_t                         getWarmUpInputLength() const;
    void                           mayAddFakeStream(std::list<GenerateStreamPtr>& streams);

    void initExecutor(const EngineInitParams& params);

    bool isMTPEagle() override;
    bool isEagle() override;
    bool isDSpark() override;

private:
    autil::ThreadPtr                              loop_thread_;
    std::atomic<bool>                             running_{false};
    std::function<bool()>                         should_loop_;
    std::unique_ptr<Executor>                     executor_;
    ModelConfig                                   model_config_;
    ParallelismConfig                             parallelism_config;
    RuntimeConfig                                 runtime_config;
    EPLBConfig                                    eplb_config;
    PDSepConfig                                   pd_sep_config;
    ProfilingDebugLoggingConfig                   profiling_debug_logging_config;
    KVCacheConfig                                 kv_cache_config;
    CacheStoreConfig                              cache_store_config;
    FfnDisAggregateConfig                         ffn_disaggregate_config;
    ModelSpecificConfig                           model_specific_config;
    SpeculativeExecutionConfig                    sp_config;
    kmonitor::MetricsReporterPtr                  metrics_reporter_;
    std::unique_ptr<ProposeModelEngineInitParams> propose_params_;
    StepWindowProfiler                            step_profiler_;
    int                                           reserve_step_ = 0;

    enum class StartupState {
        kPending,
        kReady,
        kFailed
    };
    std::mutex              startup_mu_;
    std::condition_variable startup_cv_;
    StartupState            startup_state_{StartupState::kPending};
    absl::Status            startup_error_;
};

}  // namespace rtp_llm
