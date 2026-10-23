#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <memory>
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
    absl::Status startLoop();
    int64_t      getLastScheduleTime() override;
    void         reportMetrics(RtpLLMEngineMetricsCollector collector) {
        if (metrics_reporter_) {
            metrics_reporter_->report<RtpLLMEngineMetrics, RtpLLMEngineMetricsCollector>(nullptr, &collector);
        }
    }
    bool updateEplbConfig(const EPLBConfig& config) override;
    void startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) override;
    bool rlEnabled() const override {
        return rl_enabled_;
    }
    bool         rlPaused() const override;
    absl::Status rlPause(int64_t timeout_ms) override;
    absl::Status rlResume(int64_t timeout_ms) override;
    absl::Status rlSetModelVersion(const std::string& version, int64_t global_step) override;
    std::string  rlModelVersion() const override;
    absl::Status rlReleaseKVCache() override;
    absl::Status rlRestoreKVCache() override;
    bool         rlKVCacheReady() const override;
    size_t       rlKVCacheBytes() const override;

private:
    void                            initScheduler();
    std::shared_ptr<GenerateStream> createMinFakeStream(int32_t max_new_tokens, bool prefill = false);
    WarmUpResult                    warmUp(const EngineInitParams& params);
    WarmUpResult                    prefillWarmUp(const EngineInitParams& params);
    WarmUpResult                    decodeWarmUp(const EngineInitParams& params);
    void                            initLoadBalance();
    absl::Status                    trySaveStepError() const;
    void                            loop();
    void                            initCacheManager(std::optional<WarmUpResult> warm_up_result);
    absl::Status                    initSystemPrompt();
    std::shared_ptr<GenerateInput>  makeFakeInput(size_t seq_len);
    size_t                          getWarmUpInputLength() const;
    static size_t warmUpReservedBlockCount(size_t seq_len, size_t reserve_tokens, size_t tokens_per_block);
    void mayAddFakeStream(std::list<GenerateStreamPtr>& streams, bool global_prefill = false, bool coordinated = false);
    absl::Status rlCheckpoint();
    // Caller holds rl_mutex_ when RL mode is enabled.
    bool rlAdmitStream(const GenerateStreamPtr& stream);

    void initExecutor(const EngineInitParams& params, std::unique_ptr<ProposeModelEngineInitParams>& propose_params);

    bool isMTPEagle() override;
    bool isEagle() override;
    bool isDSpark() override;

private:
    autil::ThreadPtr                              loop_thread_;
    std::atomic<bool>                             running_{false};
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
    // Engine-loop local IDs, starting after warmup/capture. Cross-rank equality must
    // be validated; these counters do not claim a synchronized global control plan.
    int64_t                 trace_control_epoch_ = 0;
    int64_t                 trace_model_step_id_ = 0;
    bool                    rl_enabled_          = false;
    bool                    rl_uses_cuda_graph_  = false;
    mutable std::mutex      rl_mutex_;
    std::condition_variable rl_cv_;
    bool                    rl_pause_requested_ = false;
    bool                    rl_paused_          = false;
    bool                    rl_failed_          = false;
    bool                    rl_kv_ready_        = true;
    std::string             rl_model_version_;
    int64_t                 rl_global_step_ = 0;
};

}  // namespace rtp_llm
