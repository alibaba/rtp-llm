#include <random>
#include <chrono>
#include "maga_transformer/cpp/disaggregate/rtpllm_master/cluster/PrefillLoadBalancer.h"
#include "maga_transformer/cpp/utils/Logger.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"

namespace rtp_llm {
namespace rtp_llm_master {

int getRandomNumber(int n) {
    // Use the current time as the seed
    unsigned                           seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937                       generator(seed);  // Mersenne Twister generator
    std::uniform_int_distribution<int> distribution(0, n - 1);
    return distribution(generator);
}

bool PrefillLoadBalancer::init(const LoadBalancerInitParams& params) {
    EstimatorConfig config;
    config.estimator_type = "local";
    return PrefillLoadBalancer::initWithEstimator(params, config);
}

PrefillLoadBalancer::~PrefillLoadBalancer() {
    stop();
}

bool PrefillLoadBalancer::initWithEstimator(const LoadBalancerInitParams& params, const EstimatorConfig& config) {
    RETURN_IF_NOT_SUCCESS(WorkerAwaredLoadBalancer::init(params));
    estimator_ = createPrefillTimeEstimator(config);
    return estimator_ != nullptr;
}

std::shared_ptr<const Host> PrefillLoadBalancer::chooseHost(const std::string& biz, int32_t global_counter) {
    auto res = chooseHostWithTask(biz, createDummyTask());
    if (!res.ok()) {
        FT_LOG_WARNING("choose host failed with error: %s", res.status().message().data());
        return nullptr;
    }
    return res.value().host;
}

std::shared_ptr<const Host> PrefillLoadBalancer::getRandomHost(const std::string& biz) const {
    std::shared_lock<std::shared_mutex> lock(biz_hosts_mutex_);
    auto                                iter = biz_hosts_.find(biz);
    if (iter == biz_hosts_.end() || iter->second == nullptr) {
        return nullptr;
    }
    auto& biz_hosts = iter->second;
    if (biz_hosts->hosts.empty()) {
        return nullptr;
    }
    auto index = getRandomNumber(biz_hosts->hosts.size());
    return biz_hosts->hosts[index];
}

absl::StatusOr<EstimateInfo> PrefillLoadBalancer::chooseHostWithTask(const std::string&     biz,
                                                                     const TaskDescription& task) {
    if (!estimator_) {
        return absl::InternalError("estimator is nullptr!");
    }
    std::shared_lock<std::shared_mutex> lock(biz_hosts_mutex_);
    auto                                iter = biz_hosts_.find(biz);
    if (iter == biz_hosts_.end() || iter->second == nullptr) {
        return absl::InternalError(autil::StringUtil::formatString("failed to find biz: %s in biz_hosts", biz.c_str()));
    }
    auto& biz_hosts = iter->second;
    if (biz_hosts->hosts.empty()) {
        return absl::InternalError(autil::StringUtil::formatString("biz host is empty in biz: %s", biz.c_str()));
    }
    std::shared_ptr<const Host>         choosed_host = nullptr;
    std::string                         choosed_host_key;
    std::string                         choosed_machine_info;
    int64_t                             min_finish_time    = -1;
    int64_t                             estimate_cost_time = -1;
    int64_t                             choosed_wait_time = -1;
    std::unique_lock<std::shared_mutex> worker_lock(host_load_balance_info_map_mutex_);
    std::string last_error_msg;
    for (auto& host : biz_hosts->hosts) {
        const std::string spec = "tcp:" + host->ip + ":" + std::to_string(host->http_port);
        auto              iter = worker_map_.find(spec);
        if (iter == worker_map_.end()) {
            continue;
        }
        auto& worker    = iter->second;
        auto  cost_time = estimator_->estimate(worker.machine_info(), {task.prefix_length, task.input_length});
        if (!cost_time.ok()) {
            last_error_msg = autil::StringUtil::formatString("failed to get task cost time with error %s", cost_time.status().message().data());
            FT_LOG_WARNING("%s", last_error_msg.c_str());
            continue;
        }
        auto expect_wait_time = worker.expect_wait_time();
        if (!choosed_host || min_finish_time > cost_time.value() + expect_wait_time) {
            choosed_host       = host;
            choosed_host_key   = spec;
            estimate_cost_time = cost_time.value();
            min_finish_time    = cost_time.value() + expect_wait_time;
            choosed_machine_info = worker.machine_info();
            choosed_wait_time = expect_wait_time;
        }
    }
    if (!choosed_host) {
        return absl::InternalError(autil::StringUtil::formatString("here7 failed to choose host in biz: %s, last error: %s", biz.c_str(), last_error_msg.c_str()));
    }
    auto& choosed_worker = worker_map_[choosed_host_key];
    choosed_worker.insertPendingTaskUpdateTime(task, estimate_cost_time);
    return EstimateInfo({choosed_host, estimate_cost_time, choosed_wait_time, choosed_machine_info});
}

bool PrefillLoadBalancer::updateWorkerExpectFinishTime(PrefillWorkerInfo& worker) {
    //TODO: what happens if worker get error time? maybe use timedelta is better
    int64_t running_time = 0;
    for (auto& task : worker.running_task_list()) {
        auto stat = estimator_->estimate(worker.machine_info(), {task.prefix_length, task.input_length});
        if (!stat.ok()) {
            FT_LOG_WARNING("failed to update worker: %s with task: (%d, %d), err: %s",
                           worker.machine_info().c_str(),
                           task.prefix_length,
                           task.input_length,
                           stat.status().message().data());
            return false;
        }
        running_time += stat.value();
    }
    worker.set_running_task_cost_time(running_time);
    int64_t pending_time = 0;
    for (auto& task : worker.pending_task_list()) {
        auto stat = estimator_->estimate(worker.machine_info(), {task.prefix_length, task.input_length});
        if (!stat.ok()) {
            FT_LOG_WARNING("failed to update worker: %s with task: (%d, %d), err: %s",
                           worker.machine_info().c_str(),
                           task.prefix_length,
                           task.input_length,
                           stat.status().message().data());
            return false;
        }
        pending_time += stat.value();
    }
    worker.set_pending_task_cost_time(pending_time);
    return true;
}

void PrefillLoadBalancer::updateWorkerStatusImpl(ErrorResult<HeartbeatSynchronizer::NodeStatus>& result) {
    HeartbeatSynchronizer::NodeStatus response = std::move(result.value());
    std::unique_lock<std::shared_mutex>                lock(host_load_balance_info_map_mutex_);
    // update exist worker info, remove unhealthy worker
    for (auto it = worker_map_.begin(); it != worker_map_.end();) {
        if (response.find(it->first) == response.end()) {
            it->second.addUpdateFailedTimes();
            if (it->second.getUpdateFailedTimes() >= max_update_failed_times_) {
                FT_LOG_WARNING("worker [%s] update failed times: %d, do remove", it->first.c_str(), max_update_failed_times_);
                it = worker_map_.erase(it);
            }
        } else {
            it->second.updateWithResponse(response[it->first], pending_task_timeout_ms_);            
            it++;
        }

    }
    // add new worker info
    for (auto& it : response) {        
        if (worker_map_.find(it.first) == worker_map_.end()) {
            worker_map_[it.first] = PrefillWorkerInfo(it.second);
        }
    }
    // update worker expect finish time
    for (auto it = worker_map_.begin(); it != worker_map_.end();) {
        if (!updateWorkerExpectFinishTime(it->second)) {
            it = worker_map_.erase(it);
        } else {
            it++;
        }
    }
}

}  // namespace rtp_llm_master

}  // namespace rtp_llm
