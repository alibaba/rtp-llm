#include "rtp_llm/cpp/cache/block_tree_cache/transfer/FrozenTaskReclaimer.h"

namespace rtp_llm {

void FrozenTaskReclaimer::freezeTargetBlocksWithTraceIDAndStillIOWorkerID(
    const Trace& trace_id, std::vector<FrozenTask::TargetBlock> target_blocks, std::vector<size_t> still_io_worker_ids) {
    if (still_io_worker_ids.empty())
        return;
    FrozenTask::ExecutionInfo execution;
    execution.key = trace_id.key;
    execution.backend = trace_id.backend;
    execution.uses_rpc = trace_id.participants.remote;
    execution.rank_count = trace_id.participants.rank_count;
    execution.target_blocks = std::move(target_blocks);
    auto task = std::make_shared<FrozenTask>(FrozenTask::prepare(std::move(execution)), trace_id.result,
        cleanup_->prepare(trace_id.participants, trace_id.key), std::move(still_io_worker_ids));
    std::lock_guard<std::mutex> lock(mutex_);
    if (tasks_.try_emplace(trace_id.key.trace_id, task).second)
        task->freeze();
}

bool FrozenTaskReclaimer::unfreezeTargetBlocksInOneWorkerWithTraceID(const TransferCompletionKey& trace_id,
                                                                   size_t worker_id) {
    std::shared_ptr<FrozenTask> task;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = tasks_.find(trace_id.trace_id);
        if (found == tasks_.end())
            return false;
        task = found->second;
    }
    if (!task->confirmWorker(worker_id) || !task->completeRecovery(*cleanup_))
        return false;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = tasks_.find(trace_id.trace_id);
    if (found != tasks_.end() && found->second == task)
        tasks_.erase(found);
    return true;
}

std::vector<std::shared_ptr<FrozenTask>> FrozenTaskReclaimer::frozenTasks() const {
    std::vector<std::shared_ptr<FrozenTask>> active, retired;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = tasks_.begin(); it != tasks_.end();) {
        if (it->second->snapshot().state == FrozenTask::State::RELEASED) {
            retired.push_back(std::move(it->second));
            it = tasks_.erase(it);
        } else {
            active.push_back(it->second);
            ++it;
        }
    }
    return active;
}

}
