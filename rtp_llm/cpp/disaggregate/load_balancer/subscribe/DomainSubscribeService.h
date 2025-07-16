#pragma once

#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeService.h"

namespace rtp_llm {

class DomainSubscribeService: public SubscribeService {
public:
    DomainSubscribeService()  = default;
    ~DomainSubscribeService() = default;

public:
    bool init(const DomainSubscribeServiceConfig& config);
    bool isReady() override;
    bool getTopoNodes(std::vector<std::shared_ptr<const TopoNode>>& topo_nodes) override;

private:
    bool        inited_{false};
    std::string domain_;
    int         http_port_;
    int         rpc_port_;
};

}  // namespace rtp_llm