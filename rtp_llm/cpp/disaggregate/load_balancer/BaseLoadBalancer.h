#pragma once

#include <shared_mutex>
#include <mutex>
#include "autil/Log.h"
#include "autil/LoopThread.h"
#include "autil/legacy/jsonizable.h"

#include "rtp_llm/cpp/dataclass/LoadBalance.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceManager.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/subscribe/TopoNode.h"

namespace rtp_llm {

struct LoadBalancerInitParams {
public:
    SubscribeServiceConfig subscribe_config;
    uint32_t               update_interval_ms{1000};
    uint32_t               sync_status_interval_ms{1000};
};

class BaseLoadBalancer {
public:
    BaseLoadBalancer()          = default;
    virtual ~BaseLoadBalancer() = default;

public:
    virtual bool                        init(const LoadBalancerInitParams& params)                      = 0;
    virtual std::shared_ptr<const Host> chooseHost(const std::string& biz, int32_t global_counter = -1) = 0;
    bool                                isReady(const std::string& biz);

protected:
    void discovery();

protected:
    mutable std::shared_mutex                        biz_hosts_mutex_;
    std::map<std::string, std::shared_ptr<BizHosts>> biz_hosts_;

    std::unique_ptr<SubscribeServiceManager> subscribe_service_manager_;
    autil::LoopThreadPtr                     service_discovery_thread_;
};

}  // namespace rtp_llm