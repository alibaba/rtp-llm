
#include <random>

#include "maga_transformer/cpp/utils/Logger.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/WRRLoadBalancer.h"

namespace rtp_llm {

WRRLoadBalancer::WRRLoadBalancer() {
    available_ratio_ = autil::EnvUtil::getEnv("WRR_AVAILABLE_RATIO", 80);
    // rank factor: 0: KV_CACHE, 1: ONFLIGHT_REQUESTS
    rank_factor_ = autil::EnvUtil::getEnv("RANK_FACTOR", 0);
    FT_CHECK_WITH_INFO(rank_factor_ == 0 || rank_factor_ == 1, "rank factor should be 0 or 1");
    FT_LOG_INFO("wrr load balance avaiable ratio %lu, rank factor = %ld", available_ratio_, rank_factor_);
}

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
    auto current_host = chooseHostByWeight(biz_hosts);
    if (current_host == nullptr) {
        FT_LOG_WARNING("choose host by concurrency failed");
        return biz_hosts->hosts[(*(biz_hosts->index))++ % biz_hosts->hosts.size()];  // choose host by RR
    }
    return current_host;
}

std::shared_ptr<const Host>
WRRLoadBalancer::chooseHostByWeight(const std::shared_ptr<BizHosts>& biz_hosts) const {
    std::shared_lock<std::shared_mutex> lock(host_load_balance_info_map_mutex_);
    auto& hosts = biz_hosts->hosts;
    if (host_load_balance_info_map_.size() < hosts.size() * available_ratio_ / 100) {
        // use round robin load balance
        return hosts[(*(biz_hosts->index))++ % hosts.size()];
    }

    int max_onflight_requests = 0;
    for (const auto& pair : host_load_balance_info_map_) {
        const WorkerStatusResponse& response = pair.second;
        if (response.load_balance_info.onflight_requests > max_onflight_requests) {
            max_onflight_requests = response.load_balance_info.onflight_requests;
        }
    }
    for (auto& pair : host_load_balance_info_map_) {
        WorkerStatusResponse& response = pair.second;
        response.load_balance_info.onflight_requests = max_onflight_requests - response.load_balance_info.onflight_requests;
    }

    double                              threshold  = calculateThreshold(hosts);
    double                              weight_acc = 0;
    
    for (auto& host : hosts) {
        // calculate weight sum
        const std::string spec = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
        auto              iter = host_load_balance_info_map_.find(spec);
        if (iter == host_load_balance_info_map_.end()) {
            continue;
        }
        weight_acc += rank_factor_ == 0 ?
            iter->second.load_balance_info.available_kv_cache : iter->second.load_balance_info.onflight_requests;
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
        weight_sum += rank_factor_ == 0 ?
            iter->second.load_balance_info.available_kv_cache : iter->second.load_balance_info.onflight_requests;
    }
    return weight_sum * generateRandomDouble();
}

void WRRLoadBalancer::updateWorkerStatusImpl(ErrorResult<HeartbeatSynchronizer::NodeStatus>& result) {
    const int wait_success_times = 100;
    static int part_success_times = 0;
    if (result.ok()) {
        part_success_times = 0;
        HeartbeatSynchronizer::NodeStatus temp = std::move(result.value());
        std::unique_lock<std::shared_mutex> lock(host_load_balance_info_map_mutex_);
        std::swap(host_load_balance_info_map_, temp);
        return;
    }

    if (result.status().code() == ErrorCode::GET_PART_NODE_STATUS_FAILED) {
        if (part_success_times == wait_success_times) {
            FT_LOG_INFO("part success times reached [%d], so update load balance info map", wait_success_times);
            part_success_times = 0;
            HeartbeatSynchronizer::NodeStatus temp = std::move(result.value());
            std::unique_lock<std::shared_mutex> lock(host_load_balance_info_map_mutex_);
            std::swap(host_load_balance_info_map_, temp);
        } else {
            part_success_times++;
        }
    } else {
        FT_LOG_ERROR("worker status is failed, error msg is [%s]", ErrorCodeToString(result.status().code()).c_str());
    }
}

}  // namespace rtp_llm