#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeService.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/CM2SubscribeService.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/LocalSubscribeService.h"

namespace rtp_llm {

std::shared_ptr<SubscribeService> SubscribeService::createInstance(const CM2SubscribeServiceConfig& config) {
    auto service = std::make_shared<CM2SubscribeService>();
    if (service->init(config)) {
        return service;
    }
    return nullptr;
}

std::shared_ptr<SubscribeService> SubscribeService::createInstance(const LocalSubscribeServiceConfig& config) {
    auto service = std::make_shared<LocalSubscribeService>();
    if (service->init(config)) {
        return service;
    }
    return nullptr;
}

}  // namespace rtp_llm