#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "autil/EnvUtil.h"
#include <stdexcept>

using namespace autil;

namespace rtp_llm {

EngineBase::EngineBase(const EngineInitParams& params) {
    initDevices(params);
    lora_manager_ = std::make_shared<lora::LoraManager>(params.model_specific_config.max_lora_model_size);
}

EngineBase::~EngineBase() {
    DeviceFactory::releaseDevices();
    RTP_LLM_LOG_INFO("engine base destrutor done");
}

std::vector<GenerateStreamPtr> EngineBase::batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) {
    throw std::runtime_error("not implemeted");
}

std::shared_ptr<GenerateStream> EngineBase::makeStream(const std::shared_ptr<GenerateInput>& input) {
    throw std::runtime_error("not implemeted");
}

void EngineBase::initDevices(const EngineInitParams& params) {
    const auto rank =
        params.parallelism_config.dp_rank * params.parallelism_config.tp_size + params.parallelism_config.tp_rank;
    Logger::getEngineLogger().setRank(rank);
    Logger::getEngineLogger().flush();
    rtp_llm::DeviceFactory::initDevices(params.parallelism_config,
                                        params.model_config_,
                                        params.eplb_config,
                                        params.fmha_config,
                                        params.device_resource_config,
                                        params.moe_config,
                                        params.sp_config,
                                        params.misc_config,
                                        params.profiling_debug_logging_config,
                                        params.hw_kernel_config,
                                        params.concurrency_config,
                                        params.ffn_disaggregate_config,
                                        params.runtime_config,
                                        params.model_specific_config,
                                        params.nccl_comm_config);
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
