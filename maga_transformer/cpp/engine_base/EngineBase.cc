#include "maga_transformer/cpp/engine_base/EngineBase.h"
#include "maga_transformer/cpp/utils/SignalUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "autil/EnvUtil.h"
#include <stdexcept>

using namespace autil;

namespace rtp_llm {

EngineBase::EngineBase(const EngineInitParams& params) {
    EngineBase::initDevices(params);
    device_       = ft::DeviceFactory::getDefaultDevice();
    lora_manager_ = std::make_shared<lora::LoraManager>();
}

EngineBase::~EngineBase() {}

std::shared_ptr<GenerateStream> EngineBase::makeStream(const std::shared_ptr<GenerateInput>& input) {
    throw std::runtime_error("not implemeted");
}

void EngineBase::initDevices(const EngineInitParams& params) {
    Logger::getEngineLogger().setRank(params.gpt_init_parameter.tp_rank_);
    Logger::getEngineLogger().flush();
    ft::DeviceFactory::initDevices(params.gpt_init_parameter);
}

void EngineBase::addLora(const std::string&            adapter_name,
                         ft::lora::loraLayerWeightsMap lora_a,
                         ft::lora::loraLayerWeightsMap lora_b) {
    lora_manager_->addLora(adapter_name, lora_a, lora_b);
}

void EngineBase::removeLora(const std::string& adapter_name) {
    lora_manager_->removeLora(adapter_name);
}

std::shared_ptr<lora::LoraManager> EngineBase::getLoraManager() {
    return lora_manager_;
}

}  // namespace rtp_llm
