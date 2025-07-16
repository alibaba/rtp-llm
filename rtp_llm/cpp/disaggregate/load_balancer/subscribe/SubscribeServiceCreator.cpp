#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeService.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/LocalSubscribeService.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/NacosSubscribeService.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/DomainSubscribeService.h"

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

std::shared_ptr<SubscribeService> createInstanceFromNacosConfig(const NacosSubscribeServiceConfig& config) {
    auto service = std::make_shared<NacosSubscribeService>();
    if (service->init(config)) {
        return service;
    }
    return nullptr;
}

std::shared_ptr<SubscribeService> createInstanceFromVIPConfig(const VIPSubscribeServiceConfig& config) {
    throw std::runtime_error("not support to create service from VIPSubscribeServiceConfig");
}

std::shared_ptr<SubscribeService> createInstanceFromDomainConfig(const DomainSubscribeServiceConfig& config) {
    auto service = std::make_shared<DomainSubscribeService>();
    if (service->init(config)) {
        return service;
    }
    return nullptr;
}

}  // namespace rtp_llm