
#include <random>

#include "maga_transformer/cpp/utils/Logger.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/WRRLoadBalancer.h"

namespace rtp_llm {

WRRLoadBalancer::~WRRLoadBalancer() {
    stop();
}

double generateRandomDouble() {
    static int               seed = (int)std::chrono::system_clock::now().time_since_epoch().count();
    static std::minstd_rand0 generator(seed);
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);

    auto rand = distribution(generator);
    return rand;
}

std::shared_ptr<const Host> WRRLoadBalancer::chooseHost(const std::string& biz) {
    std::shared_lock<std::shared_mutex> lock(biz_hosts_mutex_);
    auto                                iter = biz_hosts_.find(biz);
    if (iter == biz_hosts_.end() || iter->second == nullptr) {
        return nullptr;
    }

    auto& biz_hosts = iter->second;
    if (biz_hosts->hosts.empty()) {
        return nullptr;
    }
    auto current_host = chooseHostByWeight(biz_hosts->hosts);
    if (current_host == nullptr) {
        FT_LOG_WARNING("choose host by concurrency failed");
        return biz_hosts->hosts[(*(biz_hosts->index))++ % biz_hosts->hosts.size()];  // choose host by RR
    }
    return current_host;
}

std::shared_ptr<const Host>
WRRLoadBalancer::chooseHostByWeight(std::vector<std::shared_ptr<const Host>> biz_hosts) const {
    std::shared_lock<std::shared_mutex> lock(host_load_balance_info_map_mutex_);
    double                              threshold  = calculateThreshold(biz_hosts);
    double                              weight_acc = 0;
    for (auto& host : biz_hosts) {
        // calculate weight sum
        const std::string spec = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
        auto              iter = host_load_balance_info_map_.find(spec);
        if (iter == host_load_balance_info_map_.end()) {
            continue;
        }
        weight_acc += iter->second.load_balance_info.available_kv_cache;
        if (weight_acc >= threshold) {
            return host;
        }
    }
    return nullptr;
}

double WRRLoadBalancer::calculateThreshold(std::vector<std::shared_ptr<const Host>> biz_hosts) const {
    double weight_sum = 0;
    for (auto& host : biz_hosts) {
        // calculate weight sum
        const std::string spec = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
        auto              iter = host_load_balance_info_map_.find(spec);
        if (iter == host_load_balance_info_map_.end()) {
            continue;
        }
        weight_sum += iter->second.load_balance_info.available_kv_cache;
    }
    return weight_sum * generateRandomDouble();
}

void WRRLoadBalancer::updateWorkerStatusImpl(ErrorResult<HeartbeatSynchronizer::NodeStatus>& result) {
    const int wait_success_times = 100;
    static int part_success_times = 0;
    if (result.ok()) {
        part_success_times = 0;
        HeartbeatSynchronizer::NodeStatus temp = std::move(result.value());
        std::swap(host_load_balance_info_map_, temp);
    } else {
        if (result.status().code() == ErrorCode::GET_PART_NODE_STATUS_FAILED) {
            if (part_success_times == wait_success_times) {
                part_success_times = 0;
                HeartbeatSynchronizer::NodeStatus temp = std::move(result.value());
                std::swap(host_load_balance_info_map_, temp);
            } else {
                part_success_times++;
            }
        }
    }
    
}

}  // namespace rtp_llm