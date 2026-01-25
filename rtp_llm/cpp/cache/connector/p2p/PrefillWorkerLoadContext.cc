#include "rtp_llm/cpp/cache/connector/p2p/PrefillWorkerLoadContext.h"

#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

PrefillWorkerLoadContext::PrefillWorkerLoadContext(int64_t            request_id,
                                                   const std::string& unique_key,
                                                   int64_t            deadline_ms,
                                                   int                transfer_count):
    request_id_(request_id), unique_key_(unique_key), deadline_ms_(deadline_ms), transfer_count_(transfer_count) {
    for (int i = 0; i < transfer_count_; i++) {
        need_transfer_ids_.insert(i);
    }
}

bool PrefillWorkerLoadContext::done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return transferred_ids_.size() == transfer_count_;
}

bool PrefillWorkerLoadContext::canceled() const {
    return canceled_;
}

bool PrefillWorkerLoadContext::timeout() const {
    return currentTimeMs() >= deadline_ms_;
}

bool PrefillWorkerLoadContext::isAllTransfersDone() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return transferring_ids_.empty();
}

bool PrefillWorkerLoadContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return all_success_ && transferred_ids_.size() == transfer_count_;
}

bool PrefillWorkerLoadContext::startTransfer(int id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        iter = need_transfer_ids_.find(id);
    if (iter == need_transfer_ids_.end()) {
        RTP_LLM_LOG_DEBUG("PrefillWorkerLoadContext startTransfer failed, id: %d not need transfer, unique_key: %s",
                          id,
                          unique_key_.c_str());
        return false;
    }
    need_transfer_ids_.erase(iter);
    transferring_ids_.insert(id);
    RTP_LLM_LOG_DEBUG(
        "PrefillWorkerLoadContext startTransfer success, id: %d, unique_key: %s", id, unique_key_.c_str());
    return true;
}

void PrefillWorkerLoadContext::notifyDone(int id, ErrorCode error_code, const std::string& error_msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (transferring_ids_.find(id) == transferring_ids_.end()) {
        RTP_LLM_LOG_DEBUG("PrefillWorkerLoadContext notifyDone failed, id: %d not transferring, unique_key: %s",
                          id,
                          unique_key_.c_str());
        return;
    }
    if (error_code != ErrorCode::NONE_ERROR) {
        all_success_ = false;
        // 如果这是第一个失败，记录错误信息
        if (error_code_ == ErrorCode::NONE_ERROR) {
            error_code_ = error_code;
            error_msg_  = error_msg;
        }
    }
    RTP_LLM_LOG_DEBUG("PrefillWorkerLoadContext notifyDone, id: %d, unique_key: %s, error_code: %s",
                      id,
                      unique_key_.c_str(),
                      ErrorCodeToString(error_code).c_str());
    transferring_ids_.erase(id);
    transferred_ids_.insert(id);
}

ErrorCode PrefillWorkerLoadContext::errorCode() const {
    // 先检查取消和超时，这些状态优先级最高
    if (canceled()) {
        return ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED;
    }
    if (timeout()) {
        return ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT;
    }
    // 如果所有传输都成功完成，返回 NONE_ERROR
    if (success()) {
        return ErrorCode::NONE_ERROR;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    // 如果 error_code_ 已设置，使用它；否则使用默认错误码
    if (error_code_ != ErrorCode::NONE_ERROR) {
        return error_code_;
    }
    return ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED;
}

std::string PrefillWorkerLoadContext::errorMessage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_msg_.empty()) {
        return error_msg_;
    }
    // 如果没有设置错误信息，根据状态生成默认错误信息
    if (canceled_) {
        return "P2PConnectorWorker handleRead cancelled, request_id: " + std::to_string(request_id_)
               + ", unique_key: " + unique_key_;
    }
    if (timeout()) {
        return "P2PConnectorWorker handleRead timeout, request_id: " + std::to_string(request_id_)
               + ", unique_key: " + unique_key_;
    }
    if (!all_success_) {
        return "P2PConnectorWorker handleRead transfer failed, request_id: " + std::to_string(request_id_)
               + ", unique_key: " + unique_key_;
    }
    return "";
}

void PrefillWorkerLoadContext::setCanceled() {
    std::lock_guard<std::mutex> lock(mutex_);
    canceled_ = true;
}

bool PrefillWorkerLoadContext::isAllTransferStarted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return need_transfer_ids_.empty();
}

PrefillWorkerLoadContextStore::PrefillWorkerLoadContextStore() {}

PrefillWorkerLoadContextStore::~PrefillWorkerLoadContextStore() {}

std::shared_ptr<PrefillWorkerLoadContext>
PrefillWorkerLoadContextStore::addContext(int64_t                                 request_id,
                                          const std::string&                      unique_key,
                                          int64_t                                 deadline_ms,
                                          const std::vector<AsymmetricTPContext>& asymmetric_tp_contexts,
                                          int                                     num_layers) {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto                        prefill_worker_load_context = std::make_shared<PrefillWorkerLoadContext>(
        request_id, unique_key, deadline_ms, static_cast<int>(asymmetric_tp_contexts.size()) * num_layers);
    contexts_[request_id] = prefill_worker_load_context;
    return prefill_worker_load_context;
}

std::shared_ptr<PrefillWorkerLoadContext> PrefillWorkerLoadContextStore::getContext(int request_id) const {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto                        iter = contexts_.find(request_id);
    if (iter == contexts_.end()) {
        return nullptr;
    }
    return iter->second;
}

void PrefillWorkerLoadContextStore::removeContext(int64_t request_id) {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    contexts_.erase(request_id);
}

void PrefillWorkerLoadContextStore::checkTimeout() {
    std::unique_lock<std::mutex> lock(contexts_mutex_);
    for (auto iter = contexts_.begin(); iter != contexts_.end();) {
        if (iter->second->timeout()) {
            iter = contexts_.erase(iter);
        } else {
            ++iter;
        }
    }
}

int PrefillWorkerLoadContextStore::getContextsCount() const {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    return contexts_.size();
}

bool PrefillWorkerLoadContextStore::cancelByUniqueKey(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    for (auto& [request_id, context] : contexts_) {
        if (context->uniqueKey() == unique_key) {
            context->setCanceled();
            RTP_LLM_LOG_INFO("PrefillWorkerLoadContextStore cancelByUniqueKey success, unique_key: %s, request_id: %ld",
                             unique_key.c_str(),
                             request_id);
            return true;
        }
    }
    RTP_LLM_LOG_INFO("PrefillWorkerLoadContextStore cancelByUniqueKey failed, unique_key not found: %s",
                     unique_key.c_str());
    return true;  // return true even if not found, cancel is best-effort
}
}  // namespace rtp_llm