#pragma once

#include <atomic>
#include <cstdint>

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

    virtual void pause() {
        pause_.store(true, std::memory_order_release);
    }

    virtual void restart() {
        pause_.store(false, std::memory_order_release);
    }

    virtual absl::Status pauseAndWaitQuiesced(int64_t timeout_ms) {
        (void)timeout_ms;
        pause();
        return absl::OkStatus();
    }

    // Arm the collective sleep-quiesce consensus at the DRAINING transition (before the
    // rank-asymmetric drain), symmetrically on every rank. Overridden by NormalEngine; no-op in
    // the base so engines without a collective quiesce path (and single-rank) do nothing.
    virtual void armCollectiveSleepQuiesce() {}

    virtual std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) = 0;

    virtual void enqueue(std::shared_ptr<GenerateStream>& stream) = 0;

    virtual std::vector<GenerateStreamPtr> batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs);

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

    virtual bool updateEplbConfig(const EPLBConfig& config) {
        return false;
    }
    virtual void startTimelineProfiling(const std::string& trace_name, int start_step, int num_steps) {}

    std::shared_ptr<KVCacheManager> getCacheManager() const;

    // Sleep/wake_up lifecycle. Distinct from pause()/restart(),
    // which only stall the scheduling loop and are kept for RL training flows.
    SleepLifecycleController& sleepController() {
        return sleep_controller_;
    }

protected:
    SleepLifecycleController sleep_controller_;

    ResourceContext                resource_context_;
    MlaOpsType                     mla_ops_type_       = MlaOpsType::AUTO;
    int32_t                        kv_cache_group_num_ = 1;
    std::vector<int32_t>           kv_cache_layer_to_group_;
    std::unique_ptr<SchedulerBase> scheduler_ = nullptr;
    std::atomic<bool>              pause_{false};
};

}  // namespace rtp_llm
