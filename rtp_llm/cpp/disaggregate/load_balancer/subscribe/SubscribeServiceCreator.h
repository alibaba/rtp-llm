#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeService.h"

namespace rtp_llm {

std::shared_ptr<SubscribeService> createInstanceFromCm2Config(const CM2SubscribeServiceConfig& config);

std::shared_ptr<SubscribeService> createInstanceFromLocalConfig(const LocalSubscribeServiceConfig& config);

std::shared_ptr<SubscribeService> createInstanceFromNacosConfig(const NacosSubscribeServiceConfig& config);

std::shared_ptr<SubscribeService> createInstanceFromVIPConfig(const VIPSubscribeServiceConfig& config);

std::shared_ptr<SubscribeService> createInstanceFromDomainConfig(const DomainSubscribeServiceConfig& config);

}  // namespace rtp_llm
