#pragma once

#include <atomic>
#include <cstdint>
#include <utility>
#include <optional>

#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/EplbConfig.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/DeviceData.h"
#include "rtp_llm/cpp/disaggregate/cache_store/NormalCacheStore.h"

namespace rtp_llm {

enum preRunMode {
    prefill_warm_up     = 0,
    decode_warm_up      = 1,
    build_system_prompt = 2
};

inline std::string preRunModeToString(preRunMode mode) {
    switch (mode) {
        case prefill_warm_up:
            return "prefill_warm_up";
        case decode_warm_up:
            return "decode_warm_up";
        case build_system_prompt:
            return "build_system_prompt";
        default:
            return "unknown pre run mode";
    }
}

class EngineBase {
public:
    EngineBase(const EngineInitParams& params);
    virtual ~EngineBase();

    void initRuntime(const EngineInitParams& params);

    // These are rank-local execution guarantees, not a distributed lifecycle
    // coordinator. The caller owns admission/drain and all-rank acknowledgement.
    // Unsupported engines must never report a safe resource-release boundary.
    virtual absl::Status start() {
        return absl::UnimplementedError("first execution start is not supported");
    }
    virtual absl::Status quiesce(int64_t timeout_ms, std::optional<uint64_t> target_round = std::nullopt) {
        (void)timeout_ms;
        (void)target_round;
        return absl::UnimplementedError("safe execution quiesce is not supported");
    }
    virtual absl::Status resume() {
        return absl::UnimplementedError("safe execution resume is not supported");
    }
    virtual absl::Status terminate() {
        return absl::UnimplementedError("safe execution termination is not supported");
    }
    virtual bool executionQuiesceSupported() const {
        return false;
    }
    // Irreversible intent: drain/quiesce may continue, but resume may not reopen
    // execution. This is separate from terminating/joining the engine thread.
    virtual void requestTermination();
    bool         terminationRequested() const {
        return termination_requested_.load(std::memory_order_acquire);
    }

    virtual void pause() {
        pause_.store(true, std::memory_order_release);
    }

    virtual void restart() {
        pause_.store(false, std::memory_order_release);
    }

    // Keep empty DP/EP peers polling during drain, before the control plane
    // freezes executor rounds. This does not pause the engine or run a collective.
    virtual void armCollectiveSleepQuiesce() {}

    virtual bool requiresCoordinatedSleepQuiesce() const {
        return false;
    }
    virtual uint64_t freezeSleepRounds() {
        return 0;
    }
    virtual std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) = 0;

    virtual void enqueue(std::shared_ptr<GenerateStream>& stream) = 0;

    virtual std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs);

    virtual std::shared_ptr<GenerateStream> makeStream(const std::shared_ptr<GenerateInput>& input);

    virtual absl::Status stop() = 0;

    virtual absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input,
                                                     preRunMode                            mode) = 0;

    virtual KVCacheInfo getCacheStatusInfo(int64_t latest_version, bool need_cache_keys) = 0;

    virtual const ResourceContext& resourceContext() const {
        return resource_context_;
    }

    virtual SchedulerBase& getScheduler() {
        return *scheduler_;
    }

    virtual int64_t getLastScheduleTime() {
        return autil::TimeUtility::currentTimeInMilliSeconds();
    }

    virtual bool isMTPEagle() {
        return false;
    }

    virtual bool isEagle() {
        return false;
    }

    virtual bool isDSpark() {
        return false;
    }

    virtual bool updateEplbConfig(const EPLBConfig& config) {
        return false;
    }
    virtual void startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) {}
    virtual bool isTimelineProfilingEnabled() const {
        return false;
    }

    std::shared_ptr<KVCacheManager> getCacheManager() const;

    // Sleep/wake_up lifecycle. Distinct from pause()/restart(),
    // which only stall the scheduling loop and are kept for RL training flows.
    SleepLifecycleController& sleepController() {
        return sleep_controller_;
    }

protected:
    std::atomic<bool> termination_requested_{false};
    SleepLifecycleController sleep_controller_;

    ResourceContext                resource_context_;
    MlaOpsType                     mla_ops_type_       = MlaOpsType::AUTO;
    int32_t                        kv_cache_group_num_ = 1;
    std::vector<int32_t>           kv_cache_layer_to_group_;
    std::unique_ptr<SchedulerBase> scheduler_ = nullptr;
    std::atomic<bool>              pause_{false};
};

}  // namespace rtp_llm
