#pragma once
#include "rtp_llm/cpp/disaggregate/rtpllm_master/cluster/PrefillWorkerInfo.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/PrefillTimeEstimator.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/estimator/EstimatorConfig.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/WorkerAwaredLoadBalancer.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/common/TaskDescription.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/HeartbeatSynchronizer.h"

namespace rtp_llm {
namespace rtp_llm_master {

struct EstimateInfo {
    std::shared_ptr<const Host> host                     = nullptr;
    int64_t                     expect_execute_time_ms = -1;
    int64_t                     expect_wait_time_ms    = -1;
    std::string                 machine_info;
};

class PrefillLoadBalancer: public WorkerAwaredLoadBalancer {
public:
    PrefillLoadBalancer() = default;
    PrefillLoadBalancer(int64_t pending_timeout_ms): pending_task_timeout_ms_(pending_timeout_ms) {}
    virtual ~PrefillLoadBalancer();
    bool                         init(const LoadBalancerInitParams& params) override;
    std::shared_ptr<const Host>  chooseHost(const std::string& biz, int32_t global_counter = -1) override;
    bool                         initWithEstimator(const LoadBalancerInitParams& params, const EstimatorConfig& config);
    absl::StatusOr<EstimateInfo> chooseHostWithTask(const std::string& biz, const TaskDescription& task);
    // for tokenize service
    std::shared_ptr<const Host> getRandomHost(const std::string& biz) const;

private:
    void updateWorkerStatusImpl(ErrorResult<HeartbeatSynchronizer::NodeStatus>& result) override;
    bool updateWorkerExpectFinishTime(PrefillWorkerInfo& worker);
    std::unordered_map<std::string, PrefillWorkerInfo> worker_map_; 
    std::shared_ptr<TimeEstimatorBase>                 estimator_;

    int                       pending_task_timeout_ms_{1000};
    int                       max_update_failed_times_{3};
    mutable std::shared_mutex host_load_balance_info_map_mutex_;
};

}  // namespace rtp_llm_master

}  // namespace rtp_llm
