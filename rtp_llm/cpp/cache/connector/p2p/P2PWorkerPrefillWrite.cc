#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerPrefillWrite.h"

#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <exception>

namespace rtp_llm {

P2PWorkerPrefillWrite::P2PWorkerPrefillWrite(P2PConnectorWorkerConfig             config,
                                             std::shared_ptr<LayerBlockConverter> converter,
                                             transfer::IKVCacheReceiverPtr        receiver):
    config_(std::move(config)), converter_(std::move(converter)), receiver_(std::move(receiver)) {}

P2PWorkerPrefillWrite::~P2PWorkerPrefillWrite() {
    if (cleanup_thread_) {
        cleanup_thread_->stop();
    }
    for (const auto& [key, group] : groups_) {
        group->cancel();
        removeRecvTasks(group);
    }
}

bool P2PWorkerPrefillWrite::init() {
    if (!receiver_ || !converter_ || !config_.topology || config_.p2p_resource_store_timeout_check_interval_ms <= 0) {
        return false;
    }
    cleanup_thread_ =
        autil::LoopThread::createLoopThread([this]() { cleanup(); },
                                            int64_t(config_.p2p_resource_store_timeout_check_interval_ms) * 1000,
                                            "PrefillWriteCleanup");
    return cleanup_thread_ != nullptr;
}

ErrorInfo P2PWorkerPrefillWrite::handleWrite(int64_t,
                                             const std::string&         unique_key,
                                             int64_t                    deadline_ms,
                                             const P2PWorkerRoutePlan&  worker_plan,
                                             std::function<ErrorInfo()> on_registered) {
    using namespace p2p_internal;
    std::vector<WriteTransferUnit> units;
    const auto error = buildWriteUnits(unique_key, deadline_ms, worker_plan, config_, converter_, false, units);
    if (error.hasError()) {
        return error;
    }
    auto group = std::make_shared<WriteTaskGroup>(deadline_ms);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!groups_.emplace(unique_key, group).second) {
            return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "duplicate write task");
        }
    }
    {
        std::lock_guard<std::mutex> lock(group->mutex);
        for (auto& unit : units) {
            if (group->cancelled || currentTimeMs() >= group->deadline_ms) {
                group->error = ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "write registration expired or cancelled");
                break;
            }
            transfer::RecvRequest recv_request;
            recv_request.unique_key  = unit.key;
            recv_request.block_info  = std::move(unit.blocks);
            recv_request.deadline_ms = deadline_ms;
            try {
                auto task = receiver_->recv(recv_request);
                if (!task) {
                    group->error = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                                             "write recv registration failed");
                    break;
                }
                group->tasks.emplace(unit.key, std::move(task));
                group->lease->onTransferStarted();
            } catch (const std::exception& e) {
                group->error = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, e.what());
                break;
            }
        }
        group->lease->seal();
    }
    std::vector<std::pair<std::string, transfer::IKVCacheRecvTaskPtr>> registered_tasks;
    {
        std::lock_guard<std::mutex> lock(group->mutex);
        registered_tasks.assign(group->tasks.begin(), group->tasks.end());
    }
    for (const auto& [key, task] : registered_tasks) {
        group->watchTask(key, task);
    }
    bool can_register;
    {
        std::lock_guard<std::mutex> lock(group->mutex);
        can_register = group->error.ok() && !group->cancelled;
    }
    if (can_register && on_registered) {
        ErrorInfo registered;
        try {
            registered = on_registered();
        } catch (const std::exception& e) {
            registered = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, e.what());
        } catch (...) {
            registered = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                                   "write registration notification failed");
        }
        std::lock_guard<std::mutex> lock(group->mutex);
        if (group->error.ok()) {
            group->error = registered;
        }
    }
    const auto result = group->waitDone();
    if (result.hasError()) {
        group->cancel();
        removeRecvTasks(group);
    }
    return result;
}

bool P2PWorkerPrefillWrite::cancelWrite(const std::string& unique_key, int64_t deadline_ms) {
    if (unique_key.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = groups_.find(unique_key);
    if (it == groups_.end()) {
        auto group = std::make_shared<p2p_internal::WriteTaskGroup>(deadline_ms);
        group->lease->seal();
        it = groups_.emplace(unique_key, std::move(group)).first;
    }
    it->second->cancel();
    removeRecvTasks(it->second);
    return true;
}

bool P2PWorkerPrefillWrite::queryLeaseStatus(
    const std::string& unique_key, bool& sealed, int& started_ops, int& finished_ops, bool& stopped) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  it = groups_.find(unique_key);
    if (it == groups_.end()) {
        sealed      = false;
        started_ops = finished_ops = 0;
        stopped                    = false;
        return false;
    }
    const auto& lease = it->second->lease;
    sealed            = lease->isSealed();
    started_ops       = lease->startedOps();
    finished_ops      = lease->finishedOps();
    stopped           = lease->isStopped();
    return true;
}

void P2PWorkerPrefillWrite::removeRecvTasks(const std::shared_ptr<p2p_internal::WriteTaskGroup>& group) {
    std::lock_guard<std::mutex> lock(group->mutex);
    for (const auto& [key, task] : group->tasks) {
        receiver_->stealTask(key);
    }
}

void P2PWorkerPrefillWrite::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  now_ms = currentTimeMs();
    for (auto it = groups_.begin(); it != groups_.end();) {
        auto& group   = it->second;
        bool  stopped = group->lease->isStopped();
        if (!stopped && now_ms >= group->deadline_ms) {
            group->cancel();
            removeRecvTasks(group);
            stopped = group->lease->isStopped();
        }
        if (stopped) {
            removeRecvTasks(group);
            group->releaseStoppedTasks();
        }
        if (stopped && now_ms >= group->deadline_ms
            && now_ms - group->deadline_ms >= std::max<int64_t>(1, config_.p2p_cancelled_keys_ttl_ms)) {
            it = groups_.erase(it);
        } else {
            ++it;
        }
    }
}

}  // namespace rtp_llm
