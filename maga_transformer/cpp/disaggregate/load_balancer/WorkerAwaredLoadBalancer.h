#pragma once

#include <shared_mutex>

#include "maga_transformer/cpp/utils/ErrorCode.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/BaseLoadBalancer.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/HeartbeatSynchronizer.h"

namespace rtp_llm {

class WorkerAwaredLoadBalancer: public BaseLoadBalancer {
public:
    WorkerAwaredLoadBalancer() = default;
    virtual ~WorkerAwaredLoadBalancer();

public:
    bool init(const LoadBalancerInitParams& params) override;

protected:
    void stop();

private:
    void syncWorkerThread();
    void syncWorkerStatus();
    virtual void updateWorkerStatusImpl(ErrorResult<HeartbeatSynchronizer::NodeStatus>& result) = 0;

private:    
    bool                                   sync_worker_status_stop_{true};
    int                                    sync_worker_status_interval_ms_{200};
    autil::ThreadPtr                       sync_worker_status_thread_;
    std::shared_ptr<HeartbeatSynchronizer> heartbeat_synchronizer_;
};

}  // namespace rtp_llm
