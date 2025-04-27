#pragma once

#include "maga_transformer/cpp/utils/ErrorCode.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/WorkerAwaredLoadBalancer.h"

namespace rtp_llm {

class WRRLoadBalancer: public WorkerAwaredLoadBalancer {
public:
    WRRLoadBalancer();
    virtual ~WRRLoadBalancer();

public:
    std::shared_ptr<const Host> chooseHost(const std::string& biz, int32_t global_counter = -1) override;    

private:
    std::shared_ptr<const Host> chooseHostByWeight(const std::shared_ptr<BizHosts>& biz_hosts) const;
    double                      calculateThreshold(std::vector<std::shared_ptr<const Host>> biz_hosts) const;
    virtual void updateWorkerStatusImpl(ErrorResult<HeartbeatSynchronizer::NodeStatus>& result) override;    

    mutable std::shared_mutex                   host_load_balance_info_map_mutex_;
    mutable HeartbeatSynchronizer::NodeStatus   host_load_balance_info_map_;
    int available_ratio_ = 0;
    int rank_factor_     = 0;
};

}  // namespace rtp_llm
