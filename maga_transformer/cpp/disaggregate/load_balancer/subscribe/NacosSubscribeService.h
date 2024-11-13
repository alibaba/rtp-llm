#pragma once

#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeService.h"
#include "Nacos.h"

namespace rtp_llm {

class NacosSubscribeService : public SubscribeService {
public:
    NacosSubscribeService() = default;

public:
    bool init(const NacosSubscribeServiceConfig& config);
    bool isReady() override;
    bool getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) override;

private:
    bool inited_{false};
    NacosSubscribeServiceConfig config_;

    std::unique_ptr<nacos::INacosServiceFactory> factory_;
    std::unique_ptr<nacos::NamingService> naming_service_;
};

}