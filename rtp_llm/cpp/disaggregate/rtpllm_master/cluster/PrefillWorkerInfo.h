#pragma once
#include <vector>
#include <cstdint>
#include "rtp_llm/cpp/disaggregate/load_balancer/HeartbeatSynchronizer.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/common/TaskDescription.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace rtp_llm_master {

class KVCacheRadixTree {};

struct PendingTask {
public:
    int         prefix_length;
    int         input_length;
    std::string task_id;
    int64_t     create_time;

public:
    PendingTask(int prefix_length, int input_length, std::string task_id):
        prefix_length(prefix_length),
        input_length(input_length),
        task_id(task_id),
        create_time(autil::TimeUtility::currentTimeInMilliSeconds()) {}

    PendingTask(const PendingTask& other) {
        prefix_length = other.prefix_length;
        input_length  = other.input_length;
        task_id       = other.task_id;
        create_time   = other.create_time;
    }

    bool timeout(int64_t timeout_ms) const {
        return autil::TimeUtility::currentTimeInMilliSeconds() - create_time > timeout_ms;
    }
};

class PrefillWorkerInfo {
public:
    PrefillWorkerInfo() = default;

    PrefillWorkerInfo(WorkerStatusResponse& response) {
        updateInfoFromResponse(response);
    }

    void updateWithResponse(WorkerStatusResponse& response, int64_t timeout_ms) {
        std::vector<PendingTask> new_pending_task_list;
        for (auto& task: pending_task_list_) {
            if (!timeoutOrTaskEnqueued(response, task, timeout_ms)) {
                new_pending_task_list.push_back(task);
            }
        }
        std::swap(pending_task_list_, new_pending_task_list);
        updateInfoFromResponse(response);
    }

    int64_t expect_wait_time() const {
        RTP_LLM_LOG_DEBUG("running_task_cost_time_ %ld, pending_task_cost_time_ %ld, update_time_ %ld, last_running_delta_ %ld", running_task_cost_time_, pending_task_cost_time_, update_time_, last_running_delta_);
        auto running_task_cost_time = std::max((int64_t)0, running_task_cost_time_ - (autil::TimeUtility::currentTimeInMilliSeconds() - update_time_ + last_running_delta_));
        auto pending_task_penalty = pending_task_cost_time_;
        return running_task_cost_time + pending_task_penalty;
    }


    const std::vector<WorkerTaskStatus>& running_task_list() const {
        return running_task_list_;
    }

    const std::vector<PendingTask>& pending_task_list() const {
        return pending_task_list_;
    }

    const std::string& machine_info() const {
        return machine_info_;
    }

    void set_running_task_cost_time(int64_t v) {
        running_task_cost_time_ = v;
    }

    void set_pending_task_cost_time(int64_t v) {
        pending_task_cost_time_ = v;
    }

    void insertPendingTaskUpdateTime(const TaskDescription& task, int64_t task_cost) {
        pending_task_list_.emplace_back(PendingTask(task.prefix_length, task.input_length, task.task_id));
        pending_task_cost_time_ += task_cost;
    }

    void addUpdateFailedTimes() {
        update_failed_times_++;
    }

    void resetUpdateFailedTimes() {
        update_failed_times_ = 0;
    }

    int getUpdateFailedTimes() const {
        return update_failed_times_;
    }

protected:
    bool
    timeoutOrTaskEnqueued(const WorkerStatusResponse& remote, const PendingTask& pending_task, int64_t timeout_ms) {
        if (pending_task.timeout(timeout_ms)) {
            RTP_LLM_LOG_DEBUG("pending task %s set timeout", pending_task.task_id.c_str());
            return true;
        }
        for (auto& finish_task : remote.finished_task_list) {
            if (finish_task.task_id == pending_task.task_id) {
                RTP_LLM_LOG_DEBUG("pending task %s set finished", pending_task.task_id.c_str());
                return true;
            }
        }
        for (auto& running_task : remote.running_task_list) {
            if (running_task.task_id == pending_task.task_id) {
                RTP_LLM_LOG_DEBUG("pending task %s set running", pending_task.task_id.c_str());
                return true;
            }
        }
        return false;
    }

    void updateInfoFromResponse(WorkerStatusResponse& response) {
        std::swap(running_task_list_, response.running_task_list);
        std::swap(machine_info_, response.machine_info);
        last_running_delta_ = response.last_schedule_delta;
        update_time_ = autil::TimeUtility::currentTimeInMilliSeconds();
    }

protected:
    std::vector<WorkerTaskStatus> running_task_list_;
    std::vector<PendingTask>      pending_task_list_;
    int64_t                       last_running_delta_;
    int64_t                       update_time_;
    int64_t                       running_task_cost_time_;
    int64_t                       pending_task_cost_time_;
    std::string                   machine_info_;
    int64_t                       update_failed_times_;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm
