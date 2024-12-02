
#pragma once

#include <shared_mutex>

#include "maga_transformer/cpp/http_server/http_client/SimpleHttpClient.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/BaseLoadBalancer.h"
#include "maga_transformer/cpp/dataclass/LoadBalance.h"

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

class WRRLoadBalancer: public BaseLoadBalancer {
public:
    WRRLoadBalancer() = default;
    virtual ~WRRLoadBalancer();

public:
    std::shared_ptr<const Host> chooseHost(const std::string& biz) const override;
    bool                        init(const LoadBalancerInitParams& params) override;

private:
    bool                                         sync_worker_status_stop_{false};
    int                                          sync_worker_status_interval_ms_{10};
    autil::ThreadPtr                             sync_worker_status_thread_;
    mutable std::shared_mutex                    host_load_balance_info_map_mutex_;
    mutable std::unordered_map<std::string, int> host_load_balance_info_map_;

    mutable std::shared_mutex                      new_host_load_balance_info_map_mutex_;
    mutable std::unordered_map<std::string, int>   new_host_load_balance_info_map_;
    std::shared_ptr<http_server::SimpleHttpClient> http_client_;

private:
    std::shared_ptr<const Host> chooseHostByWeight(std::vector<std::shared_ptr<const Host>> biz_hosts) const;
    double                      calculateThreshold(std::vector<std::shared_ptr<const Host>> biz_hosts) const;
    void                        syncWorkerStatus();
    void                        syncWorkerThread();
    void                        getConcurrencyFromHost(const std::string& spec);
    void processWorkerStatusResponse(const std::string& host_ip, const std::string& response_body);
};

}  // namespace rtp_llm
