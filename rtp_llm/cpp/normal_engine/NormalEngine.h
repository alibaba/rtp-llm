#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
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
    void                              pause() override;
    void                              restart() override;
    absl::Status                      pauseAndWaitQuiesced(int64_t timeout_ms) override;

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
    absl::Status                    runExecutorProcess(const std::list<GenerateStreamPtr>& streams);
    absl::Status                    releasePendingTpCollectiveForPause(uint64_t pause_epoch);
    bool                            collectiveSleepQuiesceEnabled() const;
    absl::Status                    maybeReachCollectiveSleepQuiesce();
    void                            enterPausedState();
    void                            markPauseQuiesced(uint64_t pause_epoch);

    void initExecutor(const EngineInitParams& params, std::unique_ptr<ProposeModelEngineInitParams>& propose_params);

    bool isMTPEagle() override;
    bool isEagle() override;

private:
    autil::ThreadPtr  loop_thread_;
    std::atomic<bool> running_{false};
    std::mutex        process_mutex_;
    std::mutex        collective_quiesce_state_mutex_;
    torch::Tensor     collective_quiesce_state_;
    // Async arm-on-demand sleep-quiesce consensus state (see maybeReachCollectiveSleepQuiesce).
    // engaged_: latched true once this rank observes pause_, cleared on a terminal verdict
    //           (consensus reached, or globally cancelled). Steady serving keeps it false so
    //           the step loop issues ZERO consensus collectives.
    // handle_:  opaque non-zero id of the in-flight async all-reduce (0 = none in flight). A
    //           rank issues round k+1 only after round k completes, so per-rank round counts
    //           stay matched across the group.
    bool                    collective_quiesce_engaged_ = false;
    uint64_t                collective_quiesce_handle_  = 0;
    std::mutex              pause_mutex_;
    std::condition_variable pause_cv_;
    // Monotonic quiesce acknowledgement: the highest pause epoch a quiesce has
    // completed for. pauseAndWaitQuiesced() waits for this to reach the epoch it
    // captured. Monotonic-and-epoch-stamped so a fresh pause() (which only bumps
    // pause_epoch_) can never race-erase a quiesce already recorded for that epoch.
    uint64_t                                      quiesced_pause_epoch_{0};
    std::atomic<uint64_t>                         pause_epoch_{0};
    std::atomic<uint64_t>                         processed_pause_epoch_{0};
    std::unique_ptr<Executor>                     executor_;
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
