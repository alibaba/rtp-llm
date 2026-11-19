#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"

#include <exception>
#include <limits>
#include <mutex>

#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {

namespace {
void invokeDoneCallback(const std::function<void()>& callback) noexcept {
    if (!callback) {
        return;
    }
    try {
        callback();
    } catch (const std::exception& error) {
        RTP_LLM_LOG_ERROR("TransferTask completion callback failed: %s", error.what());
    } catch (...) {
        RTP_LLM_LOG_ERROR("TransferTask completion callback failed with an unknown exception");
    }
}
}  // namespace

// ==================== TransferTask ====================

bool TransferTask::done() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return done_;
}

bool TransferTask::success() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return done_ && error_code_ == TransferErrorCode::OK;
}

void TransferTask::cancel() {
    std::function<void()> done_callback;
    decltype(cancel_callbacks_) cancel_callbacks;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (done_) {
            return;
        }
        cancel_requested_ = true;
        cancel_callbacks.swap(cancel_callbacks_);
        if (in_flight_ == 0) {
            // PENDING: fast fail，立即终止
            done_               = true;
            if (error_code_ == TransferErrorCode::OK) {
                error_code_ = TransferErrorCode::CANCELLED;
                error_msg_  = "TransferTask cancelled";
            }
            total_cost_time_us_ = currentTimeUs() - start_time_us_;
            done_callback       = std::move(done_callback_);
        } else {
            // TRANSFERRING: 仅记录取消意图，等待 notifyDone() 真正结束
            if (error_code_ == TransferErrorCode::OK) {
                error_code_ = TransferErrorCode::CANCELLED;
                error_msg_  = "TransferTask cancelled during transfer";
            }
        }
    }
    for (const auto& entry : cancel_callbacks) {
        invokeDoneCallback(entry.second);
    }
    invokeDoneCallback(done_callback);
}

size_t TransferTask::addCancelCallback(std::function<void()> callback) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (done_ || cancel_requested_) {
        return 0;
    }
    const auto id = next_cancel_callback_id_++;
    cancel_callbacks_.emplace(id, std::move(callback));
    return id;
}

void TransferTask::removeCancelCallback(size_t id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    cancel_callbacks_.erase(id);
}

bool TransferTask::startTransfer() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (done_ || cancel_requested_ || error_code_ != TransferErrorCode::OK) {
        return false;
    }
    ++in_flight_;
    return true;
}

bool TransferTask::isCancelled() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return cancel_requested_;
}

void TransferTask::forceCancel() {
    std::function<void()> done_callback;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (done_) {
            return;
        }
        cancel_requested_   = true;
        done_               = true;
        if (error_code_ == TransferErrorCode::OK) {
            error_code_ = TransferErrorCode::CANCELLED;
            error_msg_  = "TransferTask force cancelled";
        }
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
        done_callback       = std::move(done_callback_);
    }
    invokeDoneCallback(done_callback);
}

TransferErrorCode TransferTask::errorCode() const {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (error_code_ == TransferErrorCode::OK && !done_ && currentTimeMs() >= deadline_ms_) {
        // Capture an observed timeout without claiming physical completion.
        error_code_ = TransferErrorCode::TIMEOUT;
        error_msg_  = "TransferTask timed out";
    }
    return error_code_;
}

std::string TransferTask::errorMessage() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return error_msg_;
}

void TransferTask::recordError(TransferErrorCode error_code, const std::string& error_message) {
    if (error_code == TransferErrorCode::OK) {
        return;
    }
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (!done_ && error_code_ == TransferErrorCode::OK) {
        error_code_ = error_code;
        error_msg_  = error_message;
    }
}

void TransferTask::notifyDone(bool success, TransferErrorCode error_code, const std::string& error_msg, bool started) {
    completeTransfer(success ? TransferErrorCode::OK : error_code, error_msg, nullptr, started);
}

void TransferTask::notifyBatchDone(const std::vector<int64_t>& keys, bool started) {
    completeTransfer(TransferErrorCode::OK, "", &keys, started);
}

void TransferTask::completeTransfer(TransferErrorCode           error_code,
                                    const std::string&          error_msg,
                                    const std::vector<int64_t>* batch_keys,
                                    bool                        started) {
    std::function<void()> done_callback;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (done_) {
            return;
        }
        if (started && in_flight_ > 0) {
            --in_flight_;
        }
        if (error_code_ == TransferErrorCode::OK && error_code != TransferErrorCode::OK) {
            error_code_ = error_code;
            error_msg_  = error_msg;
        }
        if (error_code_ == TransferErrorCode::OK && currentTimeMs() >= deadline_ms_) {
            // deadline 已过，无论传输是否物理完成，对调用方均视为超时
            error_code_ = TransferErrorCode::TIMEOUT;
            error_msg_  = "TransferTask timed out";
        }
        if (!completion_requested_) {
            completion_requested_ = true;
            if (error_code_ == TransferErrorCode::OK && batch_keys && on_batch_copied_) {
                try {
                    completion_requested_ = on_batch_copied_(*batch_keys);
                } catch (...) {
                    error_code_ = TransferErrorCode::COPY_FAILED;
                    error_msg_  = "receive batch callback failed";
                }
            }
        }
        // A duplicate receive can still own the destination after the logical
        // result is known. Publish completion only after all holders finish.
        if (!completion_requested_ || in_flight_ > 0) {
            return;
        }
        if (error_code_ == TransferErrorCode::OK) {
            error_msg_ = error_msg;
        }
        done_               = true;
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
        done_callback       = std::move(done_callback_);
    }
    invokeDoneCallback(done_callback);
}

void TransferTask::setDoneCallback(std::function<void()> callback) {
    bool invoke_now = false;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (done_) {
            invoke_now = true;
        } else {
            done_callback_ = std::move(callback);
        }
    }
    if (invoke_now) {
        invokeDoneCallback(callback);
    }
}

// ==================== TransferTaskStore ====================

std::shared_ptr<TransferTask>
TransferTaskStore::addTask(const std::string& unique_key,
                           KeyBlockInfoMap block_infos,
                           int64_t deadline_ms,
                           BatchCopiedCallback on_batch_copied) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (task_map_.find(unique_key) != task_map_.end()) {
        RTP_LLM_LOG_WARNING("P2P recv task registration rejected, stage=duplicate_recv_task "
                            "partition_layer_key=%s deadline_ms=%ld",
                            unique_key.c_str(), deadline_ms);
        return nullptr;
    }
    auto task = std::make_shared<TransferTask>(std::move(block_infos), deadline_ms, std::move(on_batch_copied));
    completed_batches_.erase(unique_key);
    task_map_[unique_key] = task;
    auto callback = task_added_callback_;
    lock.unlock();
    if (callback) {
        callback();
    }
    return task;
}

void TransferTaskStore::setTaskAddedCallback(std::function<void()> callback) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    task_added_callback_ = std::move(callback);
}

std::shared_ptr<TransferTask> TransferTaskStore::getTask(const std::string& unique_key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto                                it = task_map_.find(unique_key);
    if (it != task_map_.end()) {
        return it->second;
    }
    const auto completed = completed_batches_.find(unique_key);
    if (completed == completed_batches_.end() || currentTimeMs() >= completed->second) {
        return nullptr;
    }
    static const auto completed_receipt = [] {
        auto receipt = std::make_shared<TransferTask>(KeyBlockInfoMap{}, std::numeric_limits<int64_t>::max());
        receipt->notifyDone(true);
        return receipt;
    }();
    return completed_receipt;
}

std::shared_ptr<TransferTask> TransferTaskStore::stealTask(const std::string& unique_key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = task_map_.find(unique_key);
    if (it == task_map_.end()) {
        return nullptr;
    }
    auto task = it->second;
    task_map_.erase(it);
    const auto now_ms = currentTimeMs();
    while (!completed_expiry_.empty() && completed_expiry_.begin()->first <= now_ms) {
        const auto expired = completed_expiry_.begin();
        const auto completed = completed_batches_.find(expired->second);
        if (completed != completed_batches_.end() && completed->second == expired->first) {
            completed_batches_.erase(completed);
        }
        completed_expiry_.erase(expired);
    }
    if (task->acceptsBatches() && task->success() && task->deadlineMs() > now_ms) {
        completed_batches_.emplace(unique_key, task->deadlineMs());
        completed_expiry_.emplace(task->deadlineMs(), unique_key);
    }
    return task;
}

int64_t TransferTaskStore::getTaskCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return static_cast<int64_t>(task_map_.size());
}

}  // namespace transfer
}  // namespace rtp_llm
