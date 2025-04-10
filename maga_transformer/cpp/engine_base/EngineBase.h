#pragma once

#include "absl/status/status.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "src/fastertransformer/devices/DeviceBase.h"
#include "maga_transformer/cpp/lora/LoraManager.h"
#include "maga_transformer/cpp/schedulers/SchedulerBase.h"
#include "maga_transformer/cpp/disaggregate/cache_store/NormalCacheStore.h"

namespace ft = fastertransformer;

namespace rtp_llm {

enum preRunMode {
    warm_up = 0,
    build_system_prompt = 1
};

class EngineBase {
public:
    EngineBase(const EngineInitParams& params);
    virtual ~EngineBase();

    void initDevices(const EngineInitParams& params);
    ft::DeviceBase* getDevice() {
        return device_;
    }

    void addLora(const std::string& adapter_name, ft::lora::loraLayerWeightsMap lora_a, ft::lora::loraLayerWeightsMap lora_b);

    void removeLora(const std::string& adapter_name);

    std::shared_ptr<lora::LoraManager> getLoraManager();

    virtual std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>& input) = 0;

    virtual void enqueue(std::shared_ptr<GenerateStream>& stream) = 0;

    virtual std::shared_ptr<GenerateStream> makeStream(const std::shared_ptr<GenerateInput>& input);

    virtual absl::Status stop() = 0;

    virtual absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>& generate_input, preRunMode mode) = 0;

    virtual LoadBalanceInfo getLoadBalanceInfo() {
        return LoadBalanceInfo();
    }

    virtual const ResourceContext& resourceContext() const {
        return resource_context_;
    }

    virtual SchedulerBase& getScheduler() {
        return *scheduler_;
    }

    virtual int64_t getLastScheduleTime() { return autil::TimeUtility::currentTimeInMilliSeconds(); }

protected:
    ft::DeviceBase*                      device_;
    ResourceContext                      resource_context_;
    std::shared_ptr<lora::LoraManager>   lora_manager_;
    std::unique_ptr<SchedulerBase>       scheduler_ = nullptr;
};

}  // namespace rtp_llm
