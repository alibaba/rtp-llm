#include "maga_transformer/cpp/disaggregate/load_balancer/HeartbeatSynchronizer.h"
#include "maga_transformer/cpp/utils/Logger.h"

namespace rtp_llm {
bool HeartbeatSynchronizer::init() {
    http_client_ = std::make_shared<http_server::SimpleHttpClient>();
    if (!http_client_) {
        FT_LOG_WARNING("sync concurrency failed, http client is null");
        return false;
    }
    return true;
}

int HeartbeatSynchronizer::getHostCnt(const std::map<std::string, std::shared_ptr<BizHosts>>& biz_hosts) {
    int total_host_cnt = 0;
    for (auto& hosts_in_one_biz : biz_hosts) {
        total_host_cnt += hosts_in_one_biz.second->hosts.size();
    }
    return total_host_cnt;
}

void HeartbeatSynchronizer::getStatusFromHost(
    const std::string&                                                      spec,
    std::shared_ptr<std::atomic_int>&                                       sync_cnt,
    int                                                                     total_count,
    std::shared_ptr<std::shared_mutex>&                                     mutex,
    std::shared_ptr<std::unordered_map<std::string, WorkerStatusResponse>>& result) {
    http_server::HttpCallBack http_call_back =
        [this, spec, sync_cnt, total_count, mutex, result](bool ok, const std::string& response_body) {
            if (!ok) {
                FT_LOG_WARNING("http get request failed in callback, address:%s", spec.c_str());
                return;
            }
            processWorkerStatusResponse(spec, response_body, mutex, result);
            if (++(*sync_cnt) == total_count) {
                sync_worker_status_cv_.notify_all();
            }
        };
    if (!http_client_->get(spec, "/worker_status", "", std::move(http_call_back))) {
        FT_LOG_WARNING("http get request failed, host address:%s", spec.c_str());
    }
}

bool HeartbeatSynchronizer::waitDone(const std::shared_ptr<std::atomic_int>& sync_cnt,
                                     int                                     total_count,
                                     int                                     timeout_ms) {
    std::unique_lock<std::mutex> lock(cv_mutex_);
    auto                         now = std::chrono::system_clock::now();
    return sync_worker_status_cv_.wait_until(
        lock, now + std::chrono::milliseconds(timeout_ms), [this, sync_cnt, total_count]() {
            return sync_cnt->load() == total_count;
        });
}

std::unordered_map<std::string, WorkerStatusResponse>
HeartbeatSynchronizer::getHeartbeatFromHost(std::map<std::string, std::shared_ptr<BizHosts>>& biz_hosts,
                                            int                                               timeout_ms) {
    // we need shared ptr because http call is async
    std::shared_ptr<std::unordered_map<std::string, WorkerStatusResponse>> worker_stat_map(
        new std::unordered_map<std::string, WorkerStatusResponse>);
    std::shared_ptr<std::atomic_int>   sync_cnt(new std::atomic_int(0));
    std::shared_ptr<std::shared_mutex> mutex(new std::shared_mutex);
    auto                               total_host_cnt = getHostCnt(biz_hosts);
    for (auto& hosts_in_one_biz : biz_hosts) {
        for (auto& host : hosts_in_one_biz.second->hosts) {
            const std::string spec = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
            getStatusFromHost(spec, sync_cnt, total_host_cnt, mutex, worker_stat_map);
        }
    }
    if (!waitDone(sync_cnt, total_host_cnt, timeout_ms)) {
        auto sync_cnt_value = sync_cnt->load();
        if (total_host_cnt > 0 && sync_cnt_value * 1.0 / total_host_cnt < 0.9) {
            FT_LOG_WARNING("sync work status timeout, sync_worker_status_interval_ms:%d, sync_cnt:%d, total_cnt:%d",
                           timeout_ms,
                           sync_cnt_value,
                           total_host_cnt);
        }
    }
    {
        std::unique_lock<std::shared_mutex> lock(*mutex);
        std::unordered_map<std::string, WorkerStatusResponse> result;
        std::swap(result, *worker_stat_map);
        return result;
    }
}

void HeartbeatSynchronizer::processWorkerStatusResponse(
    const std::string&                                                            spec,
    const std::string&                                                            response_body,
    const std::shared_ptr<std::shared_mutex>&                                     sync_result_map_mutex,
    const std::shared_ptr<std::unordered_map<std::string, WorkerStatusResponse>>& sync_result_map) {
    try {
        WorkerStatusResponse worker_status_response;
        autil::legacy::FromJsonString(worker_status_response, response_body);
        {
            std::unique_lock<std::shared_mutex> lock(*sync_result_map_mutex);
            (*sync_result_map)[spec] = worker_status_response;
        }
    } catch (const std::exception& e) {
        FT_LOG_WARNING("response deserialize failed, address:%s, response: %s, error: %s",
                       spec.c_str(),
                       response_body.c_str(),
                       e.what());
    } catch (...) {
        FT_LOG_WARNING("response deserialize failed, address:%s, response: %s", spec.c_str(), response_body.c_str());
    }
}

}  // namespace rtp_llm