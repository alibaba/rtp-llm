#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>

namespace rtp_llm {

P2PWorkerDecodeWrite::P2PWorkerDecodeWrite(P2PConnectorWorkerConfig                            config,
                                           std::shared_ptr<LayerBlockConverter>                converter,
                                           transfer::IKVCacheSenderPtr                         sender,
                                           std::shared_ptr<WritebackStepTracker> step_tracker):
    config_(std::move(config)),
    converter_(std::move(converter)),
    sender_(std::move(sender)),
    step_tracker_(std::move(step_tracker)) {}

P2PWorkerDecodeWrite::~P2PWorkerDecodeWrite() {
    stopCleanupThread();
    for (const auto& [key, group] : groups_) {
        group->cancel();
    }
    if (sender_pool_) {
        sender_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    }
}

void P2PWorkerDecodeWrite::stopCleanupThread() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cleanup_cv_.notify_all();
    if (cleanup_thread_.joinable()) {
        cleanup_thread_.join();
    }
}

bool P2PWorkerDecodeWrite::init(size_t sender_thread_count, size_t sender_queue_size) {
    if (!sender_ || !converter_ || !config_.topology
        || sender_thread_count == 0 || sender_queue_size == 0 || sender_pool_) {
        return false;
    }
    auto pool =
        std::make_shared<autil::ThreadPool>(sender_thread_count, sender_queue_size, nullptr, "DecodeWriteSender");
    if (!pool->start()) {
        return false;
    }
    sender_pool_    = std::move(pool);
    stopping_       = false;
    cleanup_thread_ = std::thread(&P2PWorkerDecodeWrite::runCleanupLoop, this);
    return true;
}

ErrorInfo P2PWorkerDecodeWrite::write(const std::string&        unique_key,
                                      int64_t                   deadline_ms,
                                      const P2PWorkerRoutePlan& worker_plan,
                                      uint64_t                  kv_step) {
    using namespace p2p_internal;
    if (!sender_pool_) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "write sender not initialized");
    }
    std::vector<WriteTransferUnit> units;
    const auto error = buildWriteUnits(unique_key, deadline_ms, worker_plan, config_, converter_, true, units);
    if (error.hasError()) {
        return error;
    }
    using PendingSend = std::pair<transfer::SendRequest, std::shared_ptr<transfer::TransferTask>>;
    std::vector<PendingSend> pending;
    pending.reserve(units.size());
    std::shared_ptr<WriteTaskGroup> group;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto [it, inserted] = groups_.try_emplace(unique_key, nullptr);
        if (inserted) {
            it->second = std::make_shared<WriteTaskGroup>(deadline_ms);
        }
        group = it->second;
        std::lock_guard<std::mutex> group_lock(group->mutex);
        if (group->started || group->cancelled || group->lease->isSealed()) {
            return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                             "duplicate or cancelled write task");
        }
        group->stop_at_ms  = std::min(group->stop_at_ms, writebackSteadyDeadline(deadline_ms));
        group->started     = true;
        active_groups_.insert(unique_key);
        for (auto& unit : units) {
            auto task = std::make_shared<transfer::TransferTask>(unit.blocks, deadline_ms);
            group->tasks.emplace(unit.key, task);
            group->lease->onTransferStarted();
            transfer::SendRequest send_request;
            send_request.ip          = std::move(unit.ip);
            send_request.port        = unit.port;
            send_request.unique_key  = std::move(unit.key);
            send_request.block_info  = std::move(unit.blocks);
            send_request.deadline_ms = deadline_ms;
            pending.emplace_back(std::move(send_request), std::move(task));
        }
        if (pending.empty()) {
            group->lease->seal();
            cleanup_cv_.notify_one();
            return ErrorInfo::OkStatus();
        }
    }
    cleanup_cv_.notify_one();
    for (const auto& [request, task] : pending) {
        group->watchTask(request.unique_key, task);
    }
    if (step_tracker_) {
        // Keep this wait on the RPC thread. A slow compute rank must not occupy a sender slot.
        auto ready = [&]() {
            try {
                return step_tracker_->wait(kv_step, *group);
            } catch (const std::exception& e) {
                return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, e.what());
            }
        }();
        if (ready.hasError()) {
            group->cancel();
            cleanup_cv_.notify_one();
            return ready;
        }
    }
    // Publish every task before enqueueing so CANCEL also covers work still in the queue.
    auto submit = [sender = sender_, group, pending = std::move(pending)]() mutable {
        // Keep the source alive through send(), including its synchronous copy.
        auto source = group->sourceResource();
        for (auto& [send_request, task] : pending) {
            int64_t remaining_ms;
            {
                std::lock_guard<std::mutex> lock(group->mutex);
                remaining_ms = group->stop_at_ms - P2PNotification::nowMs();
            }
            if (remaining_ms <= 0) {
                group->cancel();
            }
            // A cancelled queued task must not pass its borrowed buffers to the backend.
            if (!task->startTransfer()) {
                continue;
            }
            send_request.timeout_ms  = std::min<int64_t>(remaining_ms, std::numeric_limits<int32_t>::max());
            send_request.deadline_ms = currentTimeMs() + std::max<int64_t>(0, remaining_ms);
            try {
                // All copies of a retained backend callback share one releasable source reference.
                auto source_hold = std::make_shared<KVCacheResourcePtr>(source);
                sender->send(send_request,
                             [task, source_hold](transfer::TransferErrorCode code, const std::string& message) {
                                 source_hold->reset();
                                 task->notifyDone(code == transfer::TransferErrorCode::OK, code, message);
                             });
            } catch (const std::exception& e) {
                task->notifyDone(false, transfer::TransferErrorCode::UNKNOWN, e.what());
                group->cancel();
            } catch (...) {
                task->notifyDone(false, transfer::TransferErrorCode::UNKNOWN, "write sender threw an exception");
                group->cancel();
            }
        }
    };
    const bool accepted = sender_pool_->pushTask(std::move(submit), /*isBlocked=*/false, /*executeWhenFail=*/false)
                          == autil::ThreadPoolBase::ERROR_NONE;
    if (!accepted) {
        group->cancel();
    }
    {
        std::lock_guard<std::mutex> lock(group->mutex);
        if (!accepted) {
            group->error =
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "write sender queue full or stopped");
        }
        group->lease->seal();
    }
    group->cv.notify_all();
    const auto result = group->waitDone();
    cleanup_cv_.notify_one();
    return result;
}

std::shared_ptr<p2p_internal::WriteTaskGroup> P2PWorkerDecodeWrite::holdSourceResource(
    const std::string& unique_key, int64_t deadline_ms, KVCacheResourcePtr source_resource) {
    if (unique_key.empty() || !source_resource) {
        return nullptr;
    }
    auto group = std::make_shared<p2p_internal::WriteTaskGroup>(deadline_ms, std::move(source_resource));
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!groups_.emplace(unique_key, group).second) {
            return nullptr;
        }
        active_groups_.insert(unique_key);
    }
    cleanup_cv_.notify_one();
    return group;
}

bool P2PWorkerDecodeWrite::cancelWrite(const std::string& unique_key, int64_t deadline_ms) {
    if (unique_key.empty()) {
        return false;
    }
    std::shared_ptr<p2p_internal::WriteTaskGroup> group;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = groups_.find(unique_key);
        if (it == groups_.end()) {
            group = std::make_shared<p2p_internal::WriteTaskGroup>(deadline_ms);
            group->lease->seal();
            groups_.emplace(unique_key, group);
            active_groups_.insert(unique_key);
        } else {
            group = it->second;
        }
    }
    group->cancel();
    cleanup_cv_.notify_one();
    return true;
}

bool P2PWorkerDecodeWrite::queryLeaseStatus(
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

void P2PWorkerDecodeWrite::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    cleanupLocked(P2PNotification::nowMs());
}

int64_t P2PWorkerDecodeWrite::cleanupLocked(int64_t now_ms) {
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
            stopped = group->lease->isStopped();
        }
        if (stopped) {
            group->releaseStoppedTasks();
        }
        if (stopped && !group->sourceResource()) {
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
            const int64_t target_ms =
                (stopped || now_ms >= deadline_ms) ? now_ms + kPostDeadlinePollMs : deadline_ms;
            next_wakeup_ms = std::min(next_wakeup_ms, target_ms);
            ++it;
        }
    }
    while (!expiring_groups_.empty() && expiring_groups_.begin()->first <= now_ms) {
        auto node = expiring_groups_.extract(expiring_groups_.begin());
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
        if (group->lease->isStopped() && !group->sourceResource() && now_ms >= deadline_ms
            && now_ms - deadline_ms >= ttl_ms) {
            groups_.erase(git);
        } else {
            active_groups_.insert(key);
            next_wakeup_ms = std::min(next_wakeup_ms, now_ms + kPostDeadlinePollMs);
        }
    }
    if (!expiring_groups_.empty()) {
        next_wakeup_ms = std::min(next_wakeup_ms, expiring_groups_.begin()->first);
    }
    return next_wakeup_ms;
}

void P2PWorkerDecodeWrite::runCleanupLoop() {
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
