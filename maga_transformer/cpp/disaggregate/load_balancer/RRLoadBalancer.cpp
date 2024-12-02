#include "maga_transformer/cpp/disaggregate/load_balancer/RRLoadBalancer.h"

#include "maga_transformer/cpp/utils/Logger.h"

namespace rtp_llm {

bool RRLoadBalancer::init(const LoadBalancerInitParams& params) {

    subscribe_service_manager_.reset(new SubscribeServiceManager);
    if (!subscribe_service_manager_->init(params.subscribe_config)) {
        FT_LOG_WARNING("random load balancer init failed, subscribe service manager init failed");
        return false;
    }

    service_discovery_thread_ = autil::LoopThread::createLoopThread(
        std::bind(&RRLoadBalancer::discovery, this), params.update_interval_ms * 1000, "discovery");
    return true;
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