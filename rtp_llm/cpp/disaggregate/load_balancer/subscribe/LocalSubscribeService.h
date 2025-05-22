#pragma once

#include "autil/Log.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeService.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"

namespace rtp_llm {

class LocalSubscribeService: public SubscribeService {
public:
    LocalSubscribeService()  = default;
    ~LocalSubscribeService() = default;

public:
    bool init(const LocalSubscribeServiceConfig& config);
    bool isReady() override;
    bool getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) override;

private:
    bool                                         inited_{false};
    std::vector<std::shared_ptr<const TopoNode>> nodes_;
};

}  // namespace rtp_llm