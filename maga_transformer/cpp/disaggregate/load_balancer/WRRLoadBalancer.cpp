
#include <random>

#include "maga_transformer/cpp/utils/Logger.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/WRRLoadBalancer.h"

namespace rtp_llm {

double generateRandomDouble() {
    static int               seed = (int)std::chrono::system_clock::now().time_since_epoch().count();
    static std::minstd_rand0 generator(seed);
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);

    auto rand = distribution(generator);
    return rand;
}

WRRLoadBalancer::~WRRLoadBalancer() {
    sync_worker_status_stop_ = true;
    sync_worker_status_thread_->join();
    sync_worker_status_thread_.reset();
    service_discovery_thread_->stop();
    service_discovery_thread_.reset();

    FT_LOG_INFO("destroy WRRLoadBalancer done");
}

bool WRRLoadBalancer::init(const LoadBalancerInitParams& params) {
    subscribe_service_manager_.reset(new SubscribeServiceManager);
    heartbeat_synchronizer_.reset(new HeartbeatSynchronizer);
    if (!subscribe_service_manager_->init(params.subscribe_config)) {
        FT_LOG_WARNING("subscribe service manager init failed, WRRLoadBalancer init failed");
        return false;
    }
    if (!heartbeat_synchronizer_->init()) {
        FT_LOG_WARNING("heartbeat synchronizer init failed, WRRLoadBalancer init failed");
        return false;
    }

    service_discovery_thread_ = autil::LoopThread::createLoopThread(
        std::bind(&WRRLoadBalancer::discovery, this), params.update_interval_ms * 1000, "discovery");

    sync_worker_status_interval_ms_ = params.sync_status_interval_ms;
    sync_worker_status_thread_ =
        autil::Thread::createThread(std::bind(&WRRLoadBalancer::syncWorkerThread, this), "sync_woker_status");

    FT_LOG_INFO("WRRLoadBalancer init done");
    return true;
}

void WRRLoadBalancer::syncWorkerThread() {
    while (!sync_worker_status_stop_) {
        int64_t                            start_time_us = autil::TimeUtility::currentTime();
        syncWorkerStatus();
        int64_t end_time_us  = autil::TimeUtility::currentTime();
        int     wait_time_us = sync_worker_status_interval_ms_ * 1000 - (end_time_us - start_time_us);
        if (wait_time_us > 0) {
            usleep(wait_time_us);
        }
    }
}

void WRRLoadBalancer::syncWorkerStatus() {
    std::unique_lock<std::shared_mutex> lock(biz_hosts_mutex_);
    std::unordered_map<std::string, WorkerStatusResponse> result =
        heartbeat_synchronizer_->getHeartbeatFromHost(biz_hosts_, sync_worker_status_interval_ms_);
    {
        std::unique_lock<std::shared_mutex> lock(host_load_balance_info_map_mutex_);
        host_load_balance_info_map_ = std::move(result);
    }
}

std::shared_ptr<const Host> WRRLoadBalancer::chooseHost(const std::string& biz) const {
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
}  // namespace rtp_llm