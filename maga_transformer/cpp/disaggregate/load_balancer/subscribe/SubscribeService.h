#pragma once

#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/TopoNode.h"

namespace rtp_llm {

class SubscribeService {
public:
    virtual bool isReady()                                                              = 0;
    virtual bool getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) = 0;
};

}  // namespace rtp_llm