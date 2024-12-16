#pragma once

#include <shared_mutex>

#include "maga_transformer/cpp/disaggregate/load_balancer/BaseLoadBalancer.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/HeartbeatSynchronizer.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"

namespace rtp_llm {

class WRRLoadBalancer: public BaseLoadBalancer {
public:
    WRRLoadBalancer() = default;
    virtual ~WRRLoadBalancer();

public:
    std::shared_ptr<const Host> chooseHost(const std::string& biz) const override;
    bool                        init(const LoadBalancerInitParams& params) override;

private:
    std::shared_ptr<const Host> chooseHostByWeight(std::vector<std::shared_ptr<const Host>> biz_hosts) const;
    double                      calculateThreshold(std::vector<std::shared_ptr<const Host>> biz_hosts) const;

    void syncWorkerThread();
    void syncWorkerStatus();

private:
    bool                    sync_worker_status_stop_{false};
    int                     sync_worker_status_interval_ms_{10};
    autil::ThreadPtr        sync_worker_status_thread_;
    std::shared_ptr<HeartbeatSynchronizer> heartbeat_synchronizer_;

    mutable std::shared_mutex                    host_load_balance_info_map_mutex_;
    mutable std::unordered_map<std::string, WorkerStatusResponse> host_load_balance_info_map_;
};

}  // namespace rtp_llm
