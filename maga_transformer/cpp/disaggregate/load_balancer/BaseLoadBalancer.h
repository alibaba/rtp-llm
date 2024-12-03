#pragma once

#include <shared_mutex>
#include <mutex>
#include "autil/Log.h"
#include "autil/LoopThread.h"
#include "autil/legacy/jsonizable.h"

#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceConfig.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/SubscribeServiceManager.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/subscribe/TopoNode.h"

namespace rtp_llm {

struct Host {
    std::string ip;
    uint32_t    rpc_port;
    uint32_t    http_port = 0;

    Host(const std::string& ip_, uint32_t rpc_port_, uint32_t http_port_):
        ip(ip_), rpc_port(rpc_port_), http_port(http_port_) {}
    Host(const std::string& ip_, uint32_t rpc_port_): ip(ip_), rpc_port(rpc_port_) {}
};

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
    virtual bool                        init(const LoadBalancerInitParams& params) = 0;
    virtual std::shared_ptr<const Host> chooseHost(const std::string& biz) const   = 0;
    bool                                isReady(const std::string& biz);

protected:
    void discovery();

protected:
    struct BizHosts {
        std::string                              biz;
        std::shared_ptr<std::atomic_uint32_t>    index{0};
        std::vector<std::shared_ptr<const Host>> hosts;
        BizHosts() {}
        BizHosts(const std::string&                       biz_,
                 std::shared_ptr<std::atomic_uint32_t>    index_,
                 std::vector<std::shared_ptr<const Host>> hosts_):
            biz(biz_), index(index_), hosts(hosts_) {}
    };

    mutable std::shared_mutex                        biz_hosts_mutex_;
    std::map<std::string, std::shared_ptr<BizHosts>> biz_hosts_;

    std::unique_ptr<SubscribeServiceManager> subscribe_service_manager_;
    autil::LoopThreadPtr                     service_discovery_thread_;
};

}  // namespace rtp_llm