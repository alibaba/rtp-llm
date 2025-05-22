#pragma once

#include <shared_mutex>

#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/dataclass/LoadBalance.h"
#include "rtp_llm/cpp/http_server/http_client/SimpleHttpClient.h"

namespace rtp_llm {

class WorkerTaskStatus: public autil::legacy::Jsonizable {
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("prefix_length", prefix_length);
        json.Jsonize("input_length", input_length);
        int64_t request_id;
        json.Jsonize("request_id", request_id);
        task_id = std::to_string(request_id);
    }

public:
    WorkerTaskStatus() = default;
    WorkerTaskStatus(int prefix_length, int input_length, std::string task_id):
        prefix_length(prefix_length), input_length(input_length), task_id(task_id) {}
    int         prefix_length;
    int         input_length;
    std::string task_id;
};

class WorkerStatusResponse: public autil::legacy::Jsonizable {
public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("available_concurrency", available_concurrency);
        json.Jsonize("available_kv_cache", load_balance_info.available_kv_cache);
        json.Jsonize("total_kv_cache", load_balance_info.total_kv_cache);
        json.Jsonize("step_latency_ms", load_balance_info.step_latency_us / 1000.0);
        json.Jsonize("step_per_minute", load_balance_info.step_per_minute);
        json.Jsonize("iterate_count", load_balance_info.iterate_count);
        json.Jsonize("onflight_requests", load_balance_info.onflight_requests);
        json.Jsonize("alive", alive);
        json.Jsonize("running_task_list", running_task_list, {});
        json.Jsonize("finished_task_list", finished_task_list, {});
        json.Jsonize("last_schedule_delta", last_schedule_delta, 0);
        json.Jsonize("machine_info", machine_info, "");
    }

public:
    int                           available_concurrency;
    LoadBalanceInfo               load_balance_info;
    bool                          alive;
    int64_t                       last_schedule_delta;
    std::vector<WorkerTaskStatus> running_task_list;
    std::vector<WorkerTaskStatus> finished_task_list;
    std::string                   machine_info;
};

class HeartbeatSynchronizer {
public:
    typedef std::unordered_map<std::string, WorkerStatusResponse> NodeStatus;
    HeartbeatSynchronizer() = default;

    bool init();

    ErrorResult<NodeStatus> getHeartbeatFromHost(std::map<std::string,
                                                 std::shared_ptr<BizHosts>>& biz_hosts, int timeout_ms = 10);

protected:
    int getHostCnt(const std::map<std::string, std::shared_ptr<BizHosts>>& biz_hosts);

    void getStatusFromHost(const std::string&                                                      spec,
                           std::shared_ptr<std::atomic_int>&                                       sync_cnt,
                           int                                                                     total_count,
                           std::shared_ptr<std::shared_mutex>&                                     mutex,
                           std::shared_ptr<NodeStatus>& result);

    void processWorkerStatusResponse(
        const std::string&                                                            spec,
        const std::string&                                                            response_body,
        const std::shared_ptr<std::shared_mutex>&                                     sync_result_map_mutex,
        const std::shared_ptr<NodeStatus>&                                            sync_result_map);

    bool waitDone(const std::shared_ptr<std::atomic_int>& sync_cnt, int total_count, int timeout_ms);

private:
    std::mutex                                     cv_mutex_;
    std::condition_variable                        sync_worker_status_cv_;
    std::shared_ptr<http_server::SimpleHttpClient> http_client_;
};
}  // namespace rtp_llm