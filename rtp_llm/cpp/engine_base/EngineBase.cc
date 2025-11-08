#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "autil/EnvUtil.h"
#include <stdexcept>

using namespace autil;

namespace rtp_llm {

EngineBase::EngineBase(const EngineInitParams& params) {
    initDevices(params);
    lora_manager_ =
        std::make_shared<lora::LoraManager>(params.gpt_init_parameter.model_specific_config.max_lora_model_size);
}

EngineBase::~EngineBase() {}

std::vector<GenerateStreamPtr> EngineBase::batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    throw std::runtime_error("not implemeted");
}

std::shared_ptr<GenerateStream> EngineBase::makeStream(const std::shared_ptr<GenerateInput>& input) {
    throw std::runtime_error("not implemeted");
}

void EngineBase::initDevices(const EngineInitParams& params) {
    const auto rank =
        params.gpt_init_parameter.dp_rank_ * params.gpt_init_parameter.tp_size_ + params.gpt_init_parameter.tp_rank_;
    Logger::getEngineLogger().setRank(rank);
    Logger::getEngineLogger().flush();
    rtp_llm::DeviceFactory::initDevices(params.gpt_init_parameter);
    device_ = rtp_llm::DeviceFactory::getDefaultDevice();
}

void EngineBase::addLora(const std::string&                 adapter_name,
                         rtp_llm::lora::loraLayerWeightsMap lora_a,
                         rtp_llm::lora::loraLayerWeightsMap lora_b) {
    lora_manager_->addLora(adapter_name, lora_a, lora_b);
}

void EngineBase::removeLora(const std::string& adapter_name) {
    lora_manager_->removeLora(adapter_name);
}

std::shared_ptr<lora::LoraManager> EngineBase::getLoraManager() {
    return lora_manager_;
}

std::shared_ptr<KVCacheManager> EngineBase::getCacheManager() const {
    return resource_context_.cache_manager;
}

}  // namespace rtp_llm
