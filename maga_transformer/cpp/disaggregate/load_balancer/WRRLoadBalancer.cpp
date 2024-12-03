
#include <random>

#include "aios/network/anet/connection.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/WRRLoadBalancer.h"
#include "maga_transformer/cpp/http_server/http_client/HandleHttpPacket.h"

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
    if (!subscribe_service_manager_->init(params.subscribe_config)) {
        FT_LOG_WARNING("random load balancer init failed, subscribe service manager init failed");
        return false;
    }

    http_client_ = std::make_shared<http_server::SimpleHttpClient>();
    if (!http_client_) {
        FT_LOG_WARNING("sync concurrency failed, http client is null");
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
        std::shared_ptr<std::atomic_int>   sync_cnt(new std::atomic_int(0));
        std::shared_ptr<std::shared_mutex> sync_result_map_mutex(new std::shared_mutex);
        std::shared_ptr<std::unordered_map<std::string, int>> sync_result_map(
            new std::unordered_map<std::string, int>());
        syncWorkerStatus(sync_cnt, sync_result_map_mutex, sync_result_map);
        int64_t end_time_us  = autil::TimeUtility::currentTime();
        int     wait_time_us = sync_worker_status_interval_ms_ * 1000 - (end_time_us - start_time_us);
        if (wait_time_us > 0) {
            usleep(wait_time_us);
        }
    }
}

void WRRLoadBalancer::syncWorkerStatus(
    const std::shared_ptr<std::atomic_int>&                      sync_cnt,
    const std::shared_ptr<std::shared_mutex>&                    sync_result_map_mutex,
    const std::shared_ptr<std::unordered_map<std::string, int>>& sync_result_map) {
    {
        std::shared_lock<std::shared_mutex> lock(biz_hosts_mutex_);
        total_host_cnt_ = getHostCnt();
        for (auto& hosts_in_one_biz : biz_hosts_) {
            for (auto& host : hosts_in_one_biz.second->hosts) {
                const std::string spec = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
                getConcurrencyFromHost(
                    spec, sync_cnt, sync_result_map_mutex, sync_result_map);
            }
        }
    }
    if (!waitDone(sync_cnt)) {
        FT_LOG_WARNING("sync work status timeout, sync_worker_status_interval_ms:%d, sync_cnt:%d",
                       sync_worker_status_interval_ms_,
                       sync_cnt->load());
    }
    {
        std::unique_lock<std::shared_mutex> lock(host_load_balance_info_map_mutex_);
        host_load_balance_info_map_ = std::move(*sync_result_map);
    }
}

void WRRLoadBalancer::getConcurrencyFromHost(
    const std::string&                                           spec,
    const std::shared_ptr<std::atomic_int>&                      sync_cnt,
    const std::shared_ptr<std::shared_mutex>&                    sync_result_map_mutex,
    const std::shared_ptr<std::unordered_map<std::string, int>>& sync_result_map) {
    http_server::HandleHttpPacket::HttpCallBack http_call_back =
        [this, spec, sync_cnt, sync_result_map_mutex, sync_result_map](
            bool ok, const std::string& response_body) {
            if (!ok) {
                FT_LOG_WARNING("http get request failed in callback, address:%s", spec.c_str());
                return;
            }
            processWorkerStatusResponse(
                spec, response_body, sync_result_map_mutex, sync_result_map);
            if (++(*sync_cnt) == total_host_cnt_) {
                sync_worker_status_cv_.notify_all();
            }
        };
    if (!http_client_->get(spec, "/worker_status", "", std::move(http_call_back))) {
        FT_LOG_WARNING("http get request failed, host address:%s", spec.c_str());
    }
}

void WRRLoadBalancer::processWorkerStatusResponse(
    const std::string&                                           spec,
    const std::string&                                           response_body,
    const std::shared_ptr<std::shared_mutex>&                    sync_result_map_mutex,
    const std::shared_ptr<std::unordered_map<std::string, int>>& sync_result_map) {
    try {
        WorkerStatusResponse worker_status_response;
        autil::legacy::FromJsonString(worker_status_response, response_body);
        {
            std::unique_lock<std::shared_mutex> lock(*sync_result_map_mutex);
            (*sync_result_map)[spec] = worker_status_response.load_balance_info.available_kv_cache;
        }
    } catch (...) {
        FT_LOG_WARNING("response deserialize failed, address:%s, response: %s", spec.c_str(), response_body.c_str());
    }
}

bool WRRLoadBalancer::waitDone(const std::shared_ptr<std::atomic_int>& sync_cnt) {
    std::unique_lock<std::mutex> lock(cv_mutex_);
    auto                         now = std::chrono::system_clock::now();
    return sync_worker_status_cv_.wait_until(lock,
                                             now + std::chrono::milliseconds(sync_worker_status_interval_ms_),
                                             [this, sync_cnt]() { return sync_cnt->load() == total_host_cnt_; });
}

int WRRLoadBalancer::getHostCnt() {
    int total_host_cnt = 0;
    for (auto& hosts_in_one_biz : biz_hosts_) {
        total_host_cnt += hosts_in_one_biz.second->hosts.size();
    }
    return total_host_cnt;
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
        weight_acc += iter->second;
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
        weight_sum += iter->second;
    }
    return weight_sum * generateRandomDouble();
}
}  // namespace rtp_llm