#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/SubscribeServiceFactory.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/VIPSubscribeService.h"
#include <mutex>
#include <utility>

namespace multi_call {

namespace {
std::mutex clientFactoryMutex;
SubscribeServiceFactory::VipClientFactory clientFactory;
}

void SubscribeServiceFactory::setVipClientFactory(VipClientFactory factory) {
    std::lock_guard<std::mutex> lock(clientFactoryMutex);
    clientFactory = std::move(factory);
}

SubscribeServicePtr SubscribeServiceFactory::createVipSub(const VipConfig &vipConfig) {
    VipClientFactory factory;
    {
        std::lock_guard<std::mutex> lock(clientFactoryMutex);
        factory = clientFactory;
    }
    return SubscribeServicePtr(new VIPSubscribeService(vipConfig, factory ? factory() : nullptr));
}

} // namespace multi_call
