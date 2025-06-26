#include "rtp_llm/cpp/disaggregate/load_balancer/BaseLoadBalancer.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool BaseLoadBalancer::isReady(const std::string& biz) {
    std::shared_lock<std::shared_mutex> lock(biz_hosts_mutex_);
    auto                                iter = biz_hosts_.find(biz);
    if (iter == biz_hosts_.end() || iter->second == nullptr) {
        return false;
    }
    return iter->second->hosts.size() > 0;
}

void BaseLoadBalancer::discovery() {
    if (!subscribe_service_manager_->isAllReady()) {
        return;
    }

    std::vector<std::shared_ptr<const TopoNode>> topo_nodes;
    if (!subscribe_service_manager_->getTopoNodes(topo_nodes)) {
        RTP_LLM_LOG_WARNING("random load balancer discovery failed, get cluster info map failed");
        return;
    }

    if (topo_nodes.empty()) {
        return;
    }

    std::map<std::string, std::shared_ptr<BizHosts>> new_biz_hosts;
    for (auto& toponode : topo_nodes) {
        auto iter = new_biz_hosts.find(toponode->biz);
        if (iter == new_biz_hosts.end()) {
            iter              = new_biz_hosts.insert(std::make_pair(toponode->biz, std::make_shared<BizHosts>())).first;
            iter->second->biz = toponode->biz;
            iter->second->index = std::make_shared<std::atomic_uint32_t>(0);
        }

        auto host = std::make_shared<const Host>(toponode->ip, toponode->rpc_port, toponode->http_port);
        iter->second->hosts.push_back(host);
    }
    {
        std::shared_lock<std::shared_mutex> lock(biz_hosts_mutex_);
        for (auto& [biz, biz_hosts] : new_biz_hosts) {
            auto iter = biz_hosts_.find(biz);
            if (iter != biz_hosts_.end()) {
                biz_hosts->index = iter->second->index;
            }
        }
    }
    {
        for (auto& [biz, biz_hosts] : new_biz_hosts) {
            biz_hosts->shuffleIndex();
            biz_hosts->sortHosts();
        }
        std::unique_lock<std::shared_mutex> lock(biz_hosts_mutex_);
        biz_hosts_.swap(new_biz_hosts);
    }
}


bool BaseLoadBalancer::makeLocalSubscribeConfig(SubscribeServiceConfig& config,
                                                const std::string&      cluster_name,
                                                const std::string&      str_ips,
                                                int64_t                 default_port) {
    if (str_ips.empty()) {
        return false;
    }
    std::vector<std::string>    remote_addrs = split(str_ips, ',');
    LocalSubscribeServiceConfig local_config;
    if (remote_addrs.size() == 1) {
        const auto& addr    = remote_addrs.front();
        auto [ip, port_str] = split_ip_port(addr);
        uint32_t port       = default_port;

        if (ip.empty() || port_str.empty()) {
            RTP_LLM_LOG_WARNING("Using Deprecated method to get remote rpc server addr");
            ip = remote_addrs.front();
        } else {
            port = parse_port(port_str);
        }
        local_config.nodes.emplace_back(cluster_name, ip, port);
    } else {
        for (const auto& addr : remote_addrs) {
            auto [ip, port_str] = split_ip_port(addr);
            if (ip.empty() || port_str.empty()) {
                RTP_LLM_LOG_WARNING("Using Deprecated method to get remote rpc server addr %s", addr.c_str());
                continue;
            }
            uint32_t port = parse_port(port_str);
            local_config.nodes.emplace_back(cluster_name, ip, port);
            RTP_LLM_LOG_INFO("Adding remote rpc server addr: %s:%u", ip.c_str(), port);
        }
    }
    config.local_configs.push_back(local_config);
    return true;
}

bool BaseLoadBalancer::makeCm2SubscribeConfig(SubscribeServiceConfig& config,
                                              std::string&            cluster_name,
                                              const std::string&      str_cm2_cluster_desc) {
    CM2SubscribeClusterConfig cm2_config;
    try {
        FromJsonString(cm2_config, str_cm2_cluster_desc);
    } catch (autil::legacy::ExceptionBase& e) {
        RTP_LLM_LOG_ERROR("create json from str[%s] failed", str_cm2_cluster_desc.c_str());
        return false;
    }
    cluster_name = cm2_config.cluster_name;

    CM2SubscribeServiceConfig cm2_service_config;
    cm2_service_config.zk_host       = cm2_config.zk_host;
    cm2_service_config.zk_path       = cm2_config.zk_path;
    cm2_service_config.zk_timeout_ms = 10 * 1000;
    cm2_service_config.clusters      = {cm2_config.cluster_name};
    config.cm2_configs.push_back(cm2_service_config);
    return true;
}

}  // namespace rtp_llm