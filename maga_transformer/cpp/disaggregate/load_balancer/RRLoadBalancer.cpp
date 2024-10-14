#include "maga_transformer/cpp/disaggregate/load_balancer/RRLoadBalancer.h"

namespace rtp_llm {

AUTIL_LOG_SETUP(rtp_llm, RRLoadBalancer);

bool RRLoadBalancer::init(const LoadBalancerInitParams& params) {

    subscribe_service_manager_.reset(new SubscribeServiceManager);
    if (!subscribe_service_manager_->init(params.subscribe_config)) {
        AUTIL_LOG(WARN, "random load balancer init failed, subscribe service manager init failed");
        return false;
    }

    service_discovery_thread_ = autil::LoopThread::createLoopThread(
        std::bind(&RRLoadBalancer::discovery, this), params.update_interval_ms * 1000, "discovery");
    return true;
}

void RRLoadBalancer::discovery() {
    if (!subscribe_service_manager_->isAllReady()) {
        return;
    }

    std::vector<std::shared_ptr<const TopoNode>> topo_nodes;
    if (!subscribe_service_manager_->getTopoNodes(topo_nodes)) {
        AUTIL_LOG(WARN, "random load balancer discovery failed, get cluster info map failed");
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

        auto host = std::make_shared<const Host>(toponode->ip, toponode->arpc_port);
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

std::shared_ptr<const Host> RRLoadBalancer::chooseHost(const std::string& biz) const {
    std::shared_lock<std::shared_mutex> lock(biz_hosts_mutex_);
    auto                                iter = biz_hosts_.find(biz);
    if (iter == biz_hosts_.end() || iter->second == nullptr) {
        return nullptr;
    }

    auto& biz_hosts = iter->second;
    if (biz_hosts->hosts.empty()) {
        return nullptr;
    }

    return biz_hosts->hosts[(*(biz_hosts->index))++ % biz_hosts->hosts.size()];
}

}  // namespace rtp_llm