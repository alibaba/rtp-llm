#pragma once

#include "autil/Log.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeService.h"

namespace rtp_llm {

class SubscribeServiceManager {
public:
    SubscribeServiceManager() = default;
    ~SubscribeServiceManager();

public:
    bool init(const SubscribeServiceConfig& config);

    bool isAllReady();
    bool getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes);

private:
    std::vector<std::shared_ptr<SubscribeService>> subscribe_service_vec_;
};

}  // namespace rtp_llm