#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeService.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/LocalSubscribeService.h"

namespace rtp_llm {

std::shared_ptr<SubscribeService> createInstanceFromCm2Config(const CM2SubscribeServiceConfig& config) {
    throw std::runtime_error("not support to create service from CM2SubscribeServiceConfig");
}

std::shared_ptr<SubscribeService> createInstanceFromLocalConfig(const LocalSubscribeServiceConfig& config) {
    auto service = std::make_shared<LocalSubscribeService>();
    if (service->init(config)) {
        return service;
    }
    return nullptr;
}

}  // namespace rtp_llm