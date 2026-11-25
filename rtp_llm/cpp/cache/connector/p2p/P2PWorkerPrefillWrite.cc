#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerPrefillWrite.h"

#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <exception>

namespace rtp_llm {

P2PWorkerPrefillWrite::P2PWorkerPrefillWrite(P2PConnectorWorkerConfig             config,
                                             std::shared_ptr<LayerBlockConverter> converter,
                                             transfer::IKVCacheReceiverPtr        receiver):
    config_(std::move(config)), converter_(std::move(converter)), receiver_(std::move(receiver)) {}

P2PWorkerPrefillWrite::~P2PWorkerPrefillWrite() {
    stopCleanupThread();
    for (const auto& [key, group] : groups_) {
        group->cancel();
        removeRecvTasks(group);
    }
}

void P2PWorkerPrefillWrite::stopCleanupThread() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cleanup_cv_.notify_all();
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }
}

bool P2PWorkerPrefillWrite::init() {
    if (!receiver_ || !converter_ || !config_.topology || cleanup_thread_.joinable()) {
        return false;
    }
    stopping_       = false;
    cleanup_thread_ = std::thread(&P2PWorkerPrefillWrite::runCleanupLoop, this);
    return true;
}

ErrorInfo P2PWorkerPrefillWrite::handleWrite(const std::string&         unique_key,
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
        if (groups_.size() >= config_.writeback_max_request_entries && !groups_.count(unique_key)) {
            return ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback worker table full");
        }
        if (!groups_.emplace(unique_key, group).second) {
            return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "duplicate write task");
        }
        active_groups_.insert(unique_key);
    }
    cleanup_cv_.notify_one();
    {
        std::lock_guard<std::mutex> lock(group->mutex);
        for (auto& unit : units) {
            if (group->cancelled || P2PNotification::nowMs() >= group->stop_at_ms) {
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
        removeRecvTasks(group);
    }
    cleanup_cv_.notify_one();
    return result;
}

bool P2PWorkerPrefillWrite::cancelWrite(const std::string& unique_key, int64_t deadline_ms) {
    if (unique_key.empty()) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = groups_.find(unique_key);
        if (it == groups_.end()) {
            if (groups_.size() >= config_.writeback_max_request_entries) {
                return false;
            }
            auto group = std::make_shared<p2p_internal::WriteTaskGroup>(deadline_ms);
            group->lease->seal();
            it = groups_.emplace(unique_key, std::move(group)).first;
            active_groups_.insert(unique_key);
        }
        it->second->cancel();
        removeRecvTasks(it->second);
    }
    cleanup_cv_.notify_one();
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
    it->second->queryLeaseStatus(sealed, started_ops, finished_ops, stopped);
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
    cleanupLocked(P2PNotification::nowMs());
}

int64_t P2PWorkerPrefillWrite::cleanupLocked(int64_t now_ms) {
    constexpr int64_t kPostDeadlinePollMs = 100;
    const int64_t     ttl_ms              = std::max<int64_t>(1, config_.p2p_cancelled_keys_ttl_ms);
    int64_t           next_wakeup_ms      = std::numeric_limits<int64_t>::max();
    for (auto it = active_groups_.begin(); it != active_groups_.end();) {
        auto git = groups_.find(*it);
        if (git == groups_.end()) {
            it = active_groups_.erase(it);
            continue;
        }
        auto&   group = git->second;
        int64_t deadline_ms;
        bool    was_cancelled;
        {
            std::lock_guard<std::mutex> group_lock(group->mutex);
            deadline_ms   = group->stop_at_ms;
            was_cancelled = group->cancelled;
        }
        bool stopped = group->lease->isStopped();
        if (!stopped && now_ms >= deadline_ms && !was_cancelled) {
            group->cancel();
            removeRecvTasks(group);
            stopped = group->lease->isStopped();
        }
        if (stopped) {
            removeRecvTasks(group);
            group->releaseStoppedTasks();
            const int64_t expiry_ms = deadline_ms <= std::numeric_limits<int64_t>::max() - ttl_ms ?
                                          deadline_ms + ttl_ms :
                                          std::numeric_limits<int64_t>::max();
            if (now_ms >= expiry_ms) {
                groups_.erase(git);
            } else {
                expiring_groups_.emplace(expiry_ms, ExpiringEntry{git->first, group});
            }
            it = active_groups_.erase(it);
        } else {
            const int64_t target_ms = now_ms >= deadline_ms ? now_ms + kPostDeadlinePollMs : deadline_ms;
            next_wakeup_ms          = std::min(next_wakeup_ms, target_ms);
            ++it;
        }
    }
    while (!expiring_groups_.empty() && expiring_groups_.begin()->first <= now_ms) {
        auto        node  = expiring_groups_.extract(expiring_groups_.begin());
        const auto& key   = node.mapped().first;
        const auto  group = node.mapped().second.lock();
        if (!group) {
            continue;
        }
        auto git = groups_.find(key);
        if (git == groups_.end() || git->second != group) {
            continue;
        }
        int64_t deadline_ms;
        {
            std::lock_guard<std::mutex> group_lock(group->mutex);
            deadline_ms = group->stop_at_ms;
        }
        if (group->lease->isStopped() && now_ms >= deadline_ms && now_ms - deadline_ms >= ttl_ms) {
            groups_.erase(git);
        } else {
            active_groups_.insert(key);
            next_wakeup_ms = std::min(next_wakeup_ms, now_ms >= deadline_ms ? now_ms + kPostDeadlinePollMs : deadline_ms);
        }
    }
    if (!expiring_groups_.empty()) {
        next_wakeup_ms = std::min(next_wakeup_ms, expiring_groups_.begin()->first);
    }
    return next_wakeup_ms;
}

void P2PWorkerPrefillWrite::runCleanupLoop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stopping_) {
        const int64_t now_ms         = P2PNotification::nowMs();
        const int64_t next_wakeup_ms = cleanupLocked(now_ms);
        if (next_wakeup_ms == std::numeric_limits<int64_t>::max()) {
            cleanup_cv_.wait(lock);
        } else {
            const int64_t wait_ms = std::max<int64_t>(1, next_wakeup_ms - P2PNotification::nowMs());
            cleanup_cv_.wait_for(lock, std::chrono::milliseconds(wait_ms));
        }
    }
}

}  // namespace rtp_llm
