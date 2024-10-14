#pragma once

#include <shared_mutex>
#include "autil/Log.h"
#include "autil/LoopThread.h"
#include "autil/legacy/jsonizable.h"

#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceManager.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/TopoNode.h"

namespace rtp_llm {

struct Host {
    std::string ip;
    uint32_t    port;

    Host(const std::string& ip_, uint32_t port_): ip(ip_), port(port_) {}
};

struct LoadBalancerInitParams {
public:
    SubscribeServiceConfig subscribe_config;
    uint32_t               update_interval_ms{1000};
};

class RRLoadBalancer {
public:
    bool                        init(const LoadBalancerInitParams& params);
    std::shared_ptr<const Host> chooseHost(const std::string& biz) const;

private:
    void discovery();

private:
    std::unique_ptr<SubscribeServiceManager> subscribe_service_manager_;
    autil::LoopThreadPtr                     service_discovery_thread_;

    struct BizHosts {
        std::string                              biz;
        std::shared_ptr<std::atomic_uint32_t>    index{0};
        std::vector<std::shared_ptr<const Host>> hosts;
    };

    mutable std::shared_mutex                        biz_hosts_mutex_;
    std::map<std::string, std::shared_ptr<BizHosts>> biz_hosts_;

private:
    AUTIL_LOG_DECLARE();
};

}  // namespace rtp_llm