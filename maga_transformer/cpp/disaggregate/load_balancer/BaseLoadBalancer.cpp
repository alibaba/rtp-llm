#include "maga_transformer/cpp/disaggregate/load_balancer/BaseLoadBalancer.h"

#include "maga_transformer/cpp/utils/Logger.h"

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
        FT_LOG_WARNING("random load balancer discovery failed, get cluster info map failed");
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
        std::unique_lock<std::shared_mutex> lock(biz_hosts_mutex_);
        biz_hosts_.swap(new_biz_hosts);
    }
}

}  // namespace rtp_llm