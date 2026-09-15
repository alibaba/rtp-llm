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
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (done_) {
        return;
    }
    if (!transferring_) {
        // PENDING: fast fail，立即终止
        done_               = true;
        error_code_         = TransferErrorCode::CANCELLED;
        error_msg_          = "TransferTask cancelled";
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
    } else {
        // TRANSFERRING: 仅记录取消意图，等待 notifyDone() 真正结束
        cancel_requested_ = true;
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
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (done_) {
        return;
    }
    done_               = true;
    error_code_         = TransferErrorCode::CANCELLED;
    error_msg_          = "TransferTask force cancelled";
    total_cost_time_us_ = currentTimeUs() - start_time_us_;
}

TransferErrorCode TransferTask::errorCode() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    if (!done_ && currentTimeMs() >= deadline_ms_) {
        return TransferErrorCode::TIMEOUT;
    }
    return error_code_;
}

std::string TransferTask::errorMessage() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return error_msg_;
}

void TransferTask::notifyDone(bool success, TransferErrorCode error_code, const std::string& error_msg) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    if (done_) {
        return;
    }
    done_ = true;
    if (currentTimeMs() >= deadline_ms_) {
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
