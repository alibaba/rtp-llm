#pragma once

#include "maga_transformer/cpp/disaggregate/load_balancer/WorkerAwaredLoadBalancer.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"

namespace rtp_llm {

class WRRLoadBalancer: public WorkerAwaredLoadBalancer {
public:
    WRRLoadBalancer() = default;
    virtual ~WRRLoadBalancer();

public:
    std::shared_ptr<const Host> chooseHost(const std::string& biz) override;    

private:
    std::shared_ptr<const Host> chooseHostByWeight(std::vector<std::shared_ptr<const Host>> biz_hosts) const;
    double                      calculateThreshold(std::vector<std::shared_ptr<const Host>> biz_hosts) const;
    virtual void updateWorkerStatusImpl(std::unordered_map<std::string, WorkerStatusResponse>& result) override;    

    mutable std::shared_mutex                                     host_load_balance_info_map_mutex_;
    mutable std::unordered_map<std::string, WorkerStatusResponse> host_load_balance_info_map_;
};

}  // namespace rtp_llm
