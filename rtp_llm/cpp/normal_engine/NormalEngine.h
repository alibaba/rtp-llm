#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include "absl/status/status.h"
#include "kmonitor/client/MetricsReporter.h"
#include "rtp_llm/cpp/engine_base/TorchProfiler.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/engine_base/sleep/SleepRoundFence.h"
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
    NormalEngine(const EngineInitParams&                       params,
                 std::unique_ptr<ProposeModelEngineInitParams> propose_params,
                 bool                                          defer_loop_start = false);
    ~NormalEngine();

    std::shared_ptr<GenerateStream> makeStream(const std::shared_ptr<GenerateInput>& input) override;
    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) override;
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
         enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override;
    void enqueue(std::shared_ptr<GenerateStream>& stream) override;
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                             preRunMode                            mode) override;
    absl::Status                      stop() override;
    absl::Status                      start() override;
    absl::Status quiesce(int64_t timeout_ms, std::optional<uint64_t> target_round = std::nullopt) override;
    absl::Status resume() override;
    absl::Status terminate() override;
    bool         executionQuiesceSupported() const override;
    void         requestTermination() override;
    void         pause() override;
    void         restart() override;
    void         armCollectiveSleepQuiesce() override;
    bool         requiresCoordinatedSleepQuiesce() const override;
    uint64_t     freezeSleepRounds() override;

    KVCacheInfo  getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) override;
    absl::Status step();
    absl::Status startLoop();
    int64_t      getLastScheduleTime() override;
    void         reportMetrics(RtpLLMEngineMetricsCollector collector) {
        if (metrics_reporter_) {
            metrics_reporter_->report<RtpLLMEngineMetrics, RtpLLMEngineMetricsCollector>(nullptr, &collector);
        }
    }
    bool updateEplbConfig(const EPLBConfig& config) override;
    void startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) override;
    bool isTimelineProfilingEnabled() const override;

private:
    void                            initScheduler();
    std::shared_ptr<GenerateStream> createMinFakeStream(int32_t max_new_tokens);
    WarmUpResult                    warmUp(const EngineInitParams& params);
    WarmUpResult                    prefillWarmUp(const EngineInitParams& params);
    WarmUpResult                    decodeWarmUp(const EngineInitParams& params);
    void                            initLoadBalance();
    absl::Status                    trySaveStepError() const;
    void                            loop();
    void                            normalizeSystemPromptCacheConfig();
    void                            initCacheManager(std::optional<WarmUpResult> warm_up_result);
    static void                     initializeAndPublishCacheManager(ResourceContext&                            resource_context,
                                                                     int&                                        kv_cache_group_num,
                                                                     RoleType                                    role_type,
                                                                     std::shared_ptr<KVCacheManager>             cache_manager,
                                                                     const std::function<bool(KVCacheManager&)>& initializer);
    absl::Status                    initSystemPrompt();
    std::shared_ptr<GenerateInput>  makeFakeInput(size_t seq_len);
    size_t                          getWarmUpInputLength() const;
    static size_t warmUpReservedBlockCount(size_t seq_len, size_t reserve_tokens, size_t tokens_per_block);
    void          mayAddFakeStream(std::list<GenerateStreamPtr>& streams);
    bool          collectiveSleepQuiesceEnabled() const;
    bool          acquireSleepRound();
    void          enterPausedState();
    void          requestPause(bool require_drain);
    absl::Status  drainExecution();
    absl::Status  resumeExecution(bool require_quiesced);
    absl::Status  terminateExecution(bool require_quiesced);

    void initExecutor(const EngineInitParams& params, std::unique_ptr<ProposeModelEngineInitParams>& propose_params);

    bool isMTPEagle() override;
    bool isEagle() override;
    bool isDSpark() override;

private:
    autil::ThreadPtr  loop_thread_;
    std::atomic<bool> running_{false};
    // The control-plane owner serializes start/resume/join; the loop only takes pause_mutex_.
    std::mutex                                    execution_mutex_;
    bool                                          terminated_{false};
    bool                                          loop_ready_{false};
    absl::Status                                  loop_start_status_;
    absl::Status                                  loop_exit_status_;
    std::atomic<bool>                             execution_quiesced_{false};
    SleepRoundFence                               sleep_round_fence_;
    std::mutex                                    pause_mutex_;
    std::condition_variable                       pause_cv_;
    uint64_t                                      quiesced_pause_epoch_{0};
    bool                                          pause_quiescing_{false};
    bool                                          safe_pause_requested_{false};
    absl::Status                                  pause_failure_;
    std::atomic<uint64_t>                         pause_epoch_{0};
    std::unique_ptr<Executor>                     executor_;
    ModelConfig                                   model_config_;
    py::object                                    custom_output_selector_;
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
};

}  // namespace rtp_llm
