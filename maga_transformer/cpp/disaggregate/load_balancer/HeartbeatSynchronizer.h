#pragma once

#include <shared_mutex>

#include "autil/legacy/jsonizable.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"
#include "maga_transformer/cpp/http_server/http_client/SimpleHttpClient.h"

namespace rtp_llm {

class WorkerStatusResponse: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("available_concurrency", available_concurrency);
        json.Jsonize("available_kv_cache", load_balance_info.available_kv_cache);
        json.Jsonize("total_kv_cache", load_balance_info.total_kv_cache);
        json.Jsonize("step_latency_ms", load_balance_info.step_latency_us / 1000.0);
        json.Jsonize("step_per_minute", load_balance_info.step_per_minute);
        json.Jsonize("iterate_count", load_balance_info.iterate_count);
        json.Jsonize("alive", alive);
    }

public:
    int             available_concurrency;
    LoadBalanceInfo load_balance_info;
    bool            alive;
};

class HeartbeatSynchronizer {
public:
    HeartbeatSynchronizer() = default;

    bool init();

    std::unordered_map<std::string, WorkerStatusResponse>
    getHeartbeatFromHost(std::map<std::string, std::shared_ptr<BizHosts>>& biz_hosts, int timeout_ms = 10);

protected:
    int getHostCnt(const std::map<std::string, std::shared_ptr<BizHosts>>& biz_hosts);

    void getStatusFromHost(const std::string&                                                      spec,
                           std::shared_ptr<std::atomic_int>&                                       sync_cnt,
                           int                                                                     total_count,
                           std::shared_ptr<std::shared_mutex>&                                     mutex,
                           std::shared_ptr<std::unordered_map<std::string, WorkerStatusResponse>>& result);

    void processWorkerStatusResponse(
        const std::string&                                                            spec,
        const std::string&                                                            response_body,
        const std::shared_ptr<std::shared_mutex>&                                     sync_result_map_mutex,
        const std::shared_ptr<std::unordered_map<std::string, WorkerStatusResponse>>& sync_result_map);

    bool waitDone(const std::shared_ptr<std::atomic_int>& sync_cnt, int total_count, int timeout_ms);

private:
    std::mutex                                     cv_mutex_;
    std::condition_variable                        sync_worker_status_cv_;
    std::shared_ptr<http_server::SimpleHttpClient> http_client_;
};
}  // namespace rtp_llm