#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"

#include <mutex>

#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace transfer {

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
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (done_) {
            return;
        }
        if (!transferring_) {
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
            cancel_requested_ = true;
            if (error_code_ == TransferErrorCode::OK) {
                error_code_ = TransferErrorCode::CANCELLED;
                error_msg_  = "TransferTask cancelled during transfer";
            }
        }
    }
    if (done_callback) {
        done_callback();
    }
}

bool TransferTask::startTransfer() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (done_) {
        return false;
    }
    transferring_ = true;
    return true;
}

void TransferTask::forceCancel() {
    std::function<void()> done_callback;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (done_) {
            return;
        }
        done_               = true;
        if (error_code_ == TransferErrorCode::OK) {
            error_code_ = TransferErrorCode::CANCELLED;
            error_msg_  = "TransferTask force cancelled";
        }
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
        done_callback       = std::move(done_callback_);
    }
    if (done_callback) {
        done_callback();
    }
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

void TransferTask::notifyDone(bool success, TransferErrorCode error_code, const std::string& error_msg) {
    std::function<void()> done_callback;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (done_) {
            return;
        }
        done_ = true;
        if (error_code_ != TransferErrorCode::OK) {
            // An earlier cancellation/failure already owns the result.
        } else if (!success) {
            error_code_ = error_code;
            error_msg_  = error_msg;
        } else if (currentTimeMs() >= deadline_ms_) {
            // deadline 已过，无论传输是否物理完成，对调用方均视为超时
            error_code_ = TransferErrorCode::TIMEOUT;
            error_msg_  = "TransferTask timed out";
        } else if (cancel_requested_) {
            error_code_ = TransferErrorCode::CANCELLED;
            error_msg_  = "TransferTask cancelled during transfer";
        } else {
            error_code_ = success ? TransferErrorCode::OK : error_code;
            error_msg_  = error_msg;
        }
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
        done_callback       = std::move(done_callback_);
    }
    if (done_callback) {
        done_callback();
    }
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
    if (invoke_now && callback) {
        callback();
    }
}

// ==================== TransferTaskStore ====================

std::shared_ptr<TransferTask>
TransferTaskStore::addTask(const std::string& unique_key, transfer::KeyBlockInfoMap block_infos, int64_t deadline_ms) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (task_map_.find(unique_key) != task_map_.end()) {
        return nullptr;
    }
    auto task             = std::make_shared<TransferTask>(std::move(block_infos), deadline_ms);
    task_map_[unique_key] = task;
    return task;
}

std::shared_ptr<TransferTask> TransferTaskStore::getTask(const std::string& unique_key) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto                                it = task_map_.find(unique_key);
    return it != task_map_.end() ? it->second : nullptr;
}

std::shared_ptr<TransferTask> TransferTaskStore::stealTask(const std::string& unique_key) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto                                it = task_map_.find(unique_key);
    if (it == task_map_.end()) {
        return nullptr;
    }
    auto task = it->second;
    task_map_.erase(it);
    return task;
}

int64_t TransferTaskStore::getTaskCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return static_cast<int64_t>(task_map_.size());
}

}  // namespace transfer
}  // namespace rtp_llm
