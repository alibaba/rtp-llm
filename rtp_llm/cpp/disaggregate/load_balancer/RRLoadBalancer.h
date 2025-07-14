#pragma once

#include "rtp_llm/cpp/disaggregate/load_balancer/BaseLoadBalancer.h"

namespace rtp_llm {

class RRLoadBalancer: public BaseLoadBalancer {
public:
    RRLoadBalancer() = default;
    virtual ~RRLoadBalancer();

public:
    bool                        init(const LoadBalancerInitParams& params) override;
    std::shared_ptr<const Host> chooseHost(const std::string& biz, int32_t global_counter = -1) override;
};

}  // namespace rtp_llm