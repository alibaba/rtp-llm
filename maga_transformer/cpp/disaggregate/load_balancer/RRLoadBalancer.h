#pragma once

#include "maga_transformer/cpp/disaggregate/load_balancer/BaseLoadBalancer.h"

namespace rtp_llm {

class RRLoadBalancer: public BaseLoadBalancer {
public:
    RRLoadBalancer()          = default;
    virtual ~RRLoadBalancer();

public:
    bool                        init(const LoadBalancerInitParams& params) override;
    std::shared_ptr<const Host> chooseHost(const std::string& biz) override;
};

}  // namespace rtp_llm