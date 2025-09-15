#include "rtp_llm/cpp/disaggregate/cache_store/RemoteStoreTaskImpl.h"

namespace rtp_llm {

RemoteStoreTaskImpl::RemoteStoreTaskImpl(const std::shared_ptr<RemoteStoreRequest>& request,
                                         const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>& collector,
                                         CheckCancelFunc                            check_cancel_func):
    RemoteStoreTask(request, check_cancel_func) {
    to_load_buffers_          = request_->buffer_pairs;
    expect_done_buffer_count_ = to_load_buffers_.size();
    collector_ = collector;
    collector_->markStart();
}

RemoteStoreTaskImpl::~RemoteStoreTaskImpl() {}

// SYNC结束判断, SYNC 结束后不会再触发新的更新请求, 正在处理的请求会继续处理到结束.
// 1. 超时: 外部超时/Cancel, 直接退出并判断为失败;
// 2. 失败: 目前实现中, 一个请求失败后, 不会再触发新的请求, 因为没有意义;
// 3. 成功: 正常退出, 所有block都成功传输完成, 并且没有取消
void RemoteStoreTaskImpl::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto                         once_time_ms = 5;
    while (true) {
        if (check_cancel_func_ != nullptr && check_cancel_func_()) {
            auto error_code = ErrorCode::CANCELLED;
            error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
            all_success_    = false;
            done_           = true;
            RTP_LLM_LOG_WARNING("remote store task %s wait done on cancel", request_->request_id.c_str());
        }
        if (done_) {
            RTP_LLM_LOG_DEBUG("remote store task %s wait done, success %d", request_->request_id.c_str(), all_success_);
            break;
        }
        // sync wait, safe to use this
        if (cond_.wait_for(lock, std::chrono::milliseconds(once_time_ms), [this] { return done_; })) {
            return;
        }
    }
}

bool RemoteStoreTaskImpl::success() const {
    return done_ && all_success_;
}

std::shared_ptr<TransferRequest>
RemoteStoreTaskImpl::makeAvailableRequest(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    if (request_block_buffer == nullptr) {
        return nullptr;
    }

    // event to sync wait compute
    auto event = request_block_buffer->getEvent();
    if (event) {
        event->synchronize();
    }

    if (request_block_buffer->getRequestId() != request_->request_id) {
        RTP_LLM_LOG_WARNING("remote store task make available request failed, request id is %s",
                            request_block_buffer->getRequestId().c_str());
        return nullptr;
    }
    auto transfer_request = std::make_shared<TransferRequest>(request_);
    {
        std::unique_lock<std::shared_mutex> lock(buffers_mutex_);
        if (done_) {
            // 已经完成过了或是已经失败, 不需要继续或是重复发送
            return nullptr;
        }

        auto blocks = request_block_buffer->getBlocks();

        for (auto [key, block] : blocks) {
            auto iter = to_load_buffers_.find(key);
            if (iter == to_load_buffers_.end()) {
                continue;
            }
            transfer_request->buffer_pairs[key] = iter->second;
            loading_buffers_[key]               = iter->second;
            to_load_buffers_.erase(iter);
            RTP_LLM_LOG_DEBUG("remote store task %s found local key %s", request_->request_id.c_str(), key.c_str());
        }
    }

    if (transfer_request->buffer_pairs.empty()) {
        RTP_LLM_LOG_INFO("remote store task make available request %s failed, no key found, to load buffers size is %u",
                         request_block_buffer->getRequestId().c_str(),
                         to_load_buffers_.size());
        return nullptr;
    }

    transfer_request->callback =
        [weak_task = weak_from_this(), request_id = transfer_request->request_id](
            bool success, CacheStoreErrorCode err_code, const std::map<std::string, std::string>& block_keys) {
            auto task = weak_task.lock();
            if (task) {
                task->notifyRequestDone(block_keys, success);
            } else {
                RTP_LLM_LOG_DEBUG("transfer request %s finish after task done", request_id.c_str());
            }
        };

    RTP_LLM_LOG_DEBUG("remote store task make available request success, request id is %s",
                     request_block_buffer->getRequestId().c_str());
    return transfer_request;
}

std::shared_ptr<TransferRequest>
RemoteStoreTaskImpl::makeAvailableRequest(const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    auto transfer_request = std::make_shared<TransferRequest>(request_);
    {
        std::unique_lock<std::shared_mutex> lock(buffers_mutex_);
        if (done_) {
            // 已经完成过了或是已经失败, 不需要继续或是重复发送
            return nullptr;
        }

        if (!blocks.empty()) {
            int block_size  = 0;
            for (auto block : blocks) {
                block_size = block->len;
                break;
            }
            collector_->setBlockSize(block_size * expect_done_buffer_count_);
        }

        for (auto block : blocks) {
            auto key  = block->key;
            auto iter = to_load_buffers_.find(key);
            if (iter == to_load_buffers_.end()) {
                RTP_LLM_LOG_INFO("remote store task %s not found key %s", request_->request_id.c_str(), key.c_str());
                continue;
            }
            transfer_request->buffer_pairs[key] = iter->second;
            loading_buffers_[key]               = iter->second;
            if (to_load_buffers_.size() == expect_done_buffer_count_) {
                // first block ready
                collector_->markFirstBlockReady();
            } 
            if (to_load_buffers_.size() == 1) {
                // all blocks ready
                collector_->markAllBlocksReady();
            }
            to_load_buffers_.erase(iter);
        }
    }

    if (transfer_request->buffer_pairs.empty()) {
        RTP_LLM_LOG_INFO("remote store task make available request %s failed, no key found, to load buffers size is %u",
                         request_->request_id.c_str(),
                         to_load_buffers_.size());
        return nullptr;
    }

    transfer_request->callback =
        [weak_task = weak_from_this(), request_id = transfer_request->request_id](
            bool success, CacheStoreErrorCode err_code, const std::map<std::string, std::string>& block_keys) {
            auto task = weak_task.lock();
            if (task) {
                task->notifyRequestDone(block_keys, success);
            } else {
                RTP_LLM_LOG_INFO("transfer request %s finish after task done", request_id.c_str());
            }
        };

    RTP_LLM_LOG_DEBUG("remote store task make available request success, request id is %s",
                     request_->request_id.c_str());
    return transfer_request;
}

void RemoteStoreTaskImpl::notifyRequestDone(const std::map<std::string, std::string>& block_keys, bool success) {
    {
        std::lock_guard<std::shared_mutex> lock(buffers_mutex_);
        if (done_) {
            return;
        }
        for (auto& [local_key, remote_key] : block_keys) {
            auto iter = loading_buffers_.find(local_key);
            if (iter == loading_buffers_.end()) {
                RTP_LLM_LOG_WARNING("remote store task notify request done, not found local key %s for request %s",
                                    local_key.c_str(),
                                    request_->request_id.c_str());
                continue;
            }
            done_buffers_[local_key] = iter->second;
            loading_buffers_.erase(iter);
        }

        if (!success) {
            // 有部分请求失败, 直接退出
            all_success_    = false;
            done_           = true;
            auto error_code = ErrorCode::UNKNOWN_ERROR;
            error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
            RTP_LLM_LOG_WARNING("remote store task notify request done, some request failed, request id is %s",
                                request_->request_id.c_str());
        } else {
            done_ = done_buffers_.size() == expect_done_buffer_count_;
        }
    }
    if (done_) {
        collector_->markEnd(all_success_);
        cond_.notify_all();
    }
}

}  // namespace rtp_llm
