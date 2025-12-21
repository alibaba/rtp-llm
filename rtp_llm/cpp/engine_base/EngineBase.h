#pragma once

#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/SchedulerBase.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/models/eplb/EplbConfig.h"
#include "rtp_llm/cpp/models/lora/LoraManager.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
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

    void                 initDevices(const EngineInitParams& params);
    rtp_llm::DeviceBase* getDevice() {
        return device_;
    }

    void addLora(const std::string&                 adapter_name,
                 rtp_llm::lora::loraLayerWeightsMap lora_a,
                 rtp_llm::lora::loraLayerWeightsMap lora_b);

    void removeLora(const std::string& adapter_name);

    void pause() {
        // This very simple function sets the pause_ flag to true.
        // At the beginning of the Engine's Step method, it checks if pause_ is true.
        // If it is, the Engine will sleep for a moment, thus pausing its execution.
        // Pausing the Engine is necessary for tasks like updating model weights, swapping LoRA adapters,
        // or clearing GPU memory, as it prevents model forwarding during these updates.
        // The pause_ parameter doesn't need to guarantee thread-safe access; only this interface modifies it.
        pause_ = true;
    }

    void restart() {
        // This very simple function sets the pause_ flag to false, resuming the model's execution.
        // At the beginning of the Engine's Step method, it checks if pause_ is true.
        // If it is, the Engine will sleep for a moment, thus pausing its execution.
        // Pausing the Engine is necessary for tasks like updating model weights, swapping LoRA adapters,
        // or clearing GPU memory, as it prevents model forwarding during these updates.
        // The pause_ parameter doesn't need to guarantee thread-safe access; only this interface modifies it.
        pause_ = false;
    }

    std::shared_ptr<lora::LoraManager> getLoraManager();

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

    virtual bool updateEplbConfig(const EPLBConfig& config) {
        return false;
    }

    std::shared_ptr<KVCacheManager> getCacheManager() const;

protected:
    rtp_llm::DeviceBase*               device_;
    ResourceContext                    resource_context_;
    std::shared_ptr<lora::LoraManager> lora_manager_;
    std::unique_ptr<SchedulerBase>     scheduler_ = nullptr;
    bool                               pause_     = false;
};

}  // namespace rtp_llm
