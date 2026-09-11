#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <exception>

namespace rtp_llm {

P2PWorkerDecodeWrite::P2PWorkerDecodeWrite(P2PConnectorWorkerConfig             config,
                                           std::shared_ptr<LayerBlockConverter> converter,
                                           transfer::IKVCacheSenderPtr          sender):
    config_(std::move(config)), converter_(std::move(converter)), sender_(std::move(sender)) {}

P2PWorkerDecodeWrite::~P2PWorkerDecodeWrite() {
    if (cleanup_thread_) {
        cleanup_thread_->stop();
    }
    for (const auto& [key, group] : groups_) {
        group->cancel();
    }
    if (sender_pool_) {
        sender_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    }
}

bool P2PWorkerDecodeWrite::init(size_t sender_thread_count, size_t sender_queue_size) {
    if (!sender_ || !converter_ || !config_.topology || config_.p2p_resource_store_timeout_check_interval_ms <= 0
        || sender_thread_count == 0 || sender_queue_size == 0 || sender_pool_) {
        return false;
    }
    auto pool =
        std::make_shared<autil::ThreadPool>(sender_thread_count, sender_queue_size, nullptr, "DecodeWriteSender");
    if (!pool->start()) {
        return false;
    }
    sender_pool_ = std::move(pool);
    cleanup_thread_ =
        autil::LoopThread::createLoopThread([this]() { cleanup(); },
                                            int64_t(config_.p2p_resource_store_timeout_check_interval_ms) * 1000,
                                            "DecodeWriteCleanup");
    return cleanup_thread_ != nullptr;
}

ErrorInfo P2PWorkerDecodeWrite::write(int64_t,
                                      const std::string&        unique_key,
                                      int64_t                   deadline_ms,
                                      const P2PWorkerRoutePlan& worker_plan) {
    using namespace p2p_internal;
    if (!sender_pool_) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "write sender not initialized");
    }
    std::vector<WriteTransferUnit> units;
    const auto error = buildWriteUnits(unique_key, deadline_ms, worker_plan, config_, converter_, true, units);
    if (error.hasError()) {
        return error;
    }
    auto group        = std::make_shared<WriteTaskGroup>(deadline_ms);
    using PendingSend = std::pair<transfer::SendRequest, std::shared_ptr<transfer::TransferTask>>;
    std::vector<PendingSend> pending;
    pending.reserve(units.size());
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
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!groups_.emplace(unique_key, group).second) {
            return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "duplicate write task");
        }
        if (pending.empty()) {
            group->lease->seal();
            return ErrorInfo::OkStatus();
        }
    }
    // Publish every task before enqueueing so CANCEL also covers work still in the queue.
    auto submit = [sender = sender_, group, pending = std::move(pending)]() {
        for (const auto& [send_request, task] : pending) {
            if (currentTimeMs() >= send_request.deadline_ms) {
                group->cancel();
            }
            // A cancelled queued task must not pass its borrowed buffers to the backend.
            if (!task->startTransfer()) {
                continue;
            }
            try {
                sender->send(send_request, [task](transfer::TransferErrorCode code, const std::string& message) {
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
    return group->error;
}

bool P2PWorkerDecodeWrite::cancelWrite(const std::string& unique_key, int64_t deadline_ms) {
    if (unique_key.empty()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = groups_.find(unique_key);
    if (it == groups_.end()) {
        auto group = std::make_shared<p2p_internal::WriteTaskGroup>(std::max(deadline_ms, currentTimeMs()));
        group->lease->seal();
        it = groups_.emplace(unique_key, std::move(group)).first;
    }
    it->second->cancel();
    return true;
}

bool P2PWorkerDecodeWrite::queryWriteStatus(const std::string& unique_key, WriteTaskStatus& status) {
    status = {};
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = groups_.find(unique_key);
    if (it == groups_.end()) {
        return false;
    }
    it->second->fillStatus(status);
    return true;
}

void P2PWorkerDecodeWrite::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  now_ms = currentTimeMs();
    for (auto it = groups_.begin(); it != groups_.end();) {
        auto&           group = it->second;
        WriteTaskStatus status;
        bool            stopped = group->fillStatus(status);
        if (!stopped && now_ms >= group->deadline_ms) {
            group->cancel();
            stopped = group->fillStatus(status);
        }
        if (stopped) {
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
