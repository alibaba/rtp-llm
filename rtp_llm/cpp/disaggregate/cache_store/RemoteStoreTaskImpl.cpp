#include "rtp_llm/cpp/disaggregate/cache_store/RemoteStoreTaskImpl.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <chrono>
#include <algorithm>
#include <thread>

namespace rtp_llm {

RemoteStoreTaskImpl::RemoteStoreTaskImpl(const std::shared_ptr<RemoteStoreRequest>&                    request,
                                         const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>& collector,
                                         CheckCancelFunc check_cancel_func):
    RemoteStoreTask(request, check_cancel_func),
    to_load_buffers_(request->buffer_pairs),
    expect_done_buffer_count_(to_load_buffers_.size()),
    collector_(collector) {
    if (collector_) {
        collector_->markStart();
    }
    if (to_load_buffers_.empty()) {
        finishLocked(ErrorInfo::OkStatus());
    }
}

void RemoteStoreTaskImpl::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!done_) {
        lock.unlock();
        if (check_cancel_func_ && check_cancel_func_()) {
            cancel(ErrorInfo(ErrorCode::CANCELLED, ErrorCodeToString(ErrorCode::CANCELLED)));
        }
        lock.lock();
        cond_.wait_for(lock, std::chrono::milliseconds(5), [this] { return done_; });
    }
}

bool RemoteStoreTaskImpl::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_ && error_info_.ok();
}

bool RemoteStoreTaskImpl::done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return done_;
}

bool RemoteStoreTaskImpl::setDoneCallback(DoneCallback callback) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!callback || done_callback_registered_) {
            return false;
        }
        done_callback_registered_ = true;
        done_callback_            = std::move(callback);
    }
    notifyCallbacks();
    return true;
}

bool RemoteStoreTaskImpl::setDrainedCallback(DrainedCallback callback) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!callback || drained_callback_registered_) {
            return false;
        }
        drained_callback_registered_ = true;
        drained_callback_            = std::move(callback);
    }
    notifyCallbacks();
    return true;
}

bool RemoteStoreTaskImpl::cancel(const ErrorInfo& error) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        finishLocked(error.ok() ? ErrorInfo(ErrorCode::CANCELLED, "remote store cancelled") : error);
    }
    notifyCallbacks();
    return true;
}

void RemoteStoreTaskImpl::finishLocked(const ErrorInfo& error) {
    if (done_) {
        return;
    }
    try {
        error_info_ = error;
    } catch (const std::bad_alloc&) {
        error_info_ = ErrorInfo(error.code(), std::string{});
    }
    done_ = true;
    if (collector_) {
        collector_->markEnd(error.ok());
    }
    cond_.notify_all();
}

void RemoteStoreTaskImpl::notifyCallbacks() {
    // A callback may release the last external task reference or reenter here.
    auto         keep_alive = shared_from_this();
    DoneCallback done_callback;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!done_) {
            return;
        }
        if (done_callback_) {
            done_callback          = std::move(done_callback_);
            done_callback_running_ = true;
        }
    }
    if (done_callback) {
        try {
            done_callback(error_info_);  // Immutable after done_; keep_alive owns it.
        } catch (...) {
            // User notification must not prevent RPC bookkeeping and drain.
            RTP_LLM_LOG_WARNING("remote store done callback threw, request %s", request_->request_id.c_str());
        }
        done_callback = nullptr;  // Release captures outside the mutex too.
        std::lock_guard<std::mutex> lock(mutex_);
        done_callback_running_ = false;
    }
    auto drained_callback = takeDrainedCallback();
    if (drained_callback) {
        try {
            drained_callback();
        } catch (...) {
            RTP_LLM_LOG_WARNING("remote store drained callback threw, request %s", request_->request_id.c_str());
        }
    }
}

RemoteStoreTask::DrainedCallback RemoteStoreTaskImpl::takeDrainedCallback() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Registration may publish a done hook after the first notification lock.
    // Drain must wait for both pending and running done notifications.
    if (!done_callback_ && !done_callback_running_ && outstanding_rpcs_ == 0) {
        return std::move(drained_callback_);
    }
    return {};
}

std::shared_ptr<TransferRequest>
RemoteStoreTaskImpl::makeAvailableRequest(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    RTP_LLM_PROFILE_FUNCTION();
    if (!request_block_buffer || request_block_buffer->getRequestId() != request_->request_id || done()) {
        return nullptr;
    }
    auto event = request_block_buffer->getEvent();
    if (event) {
#if USE_PPU
        while (!event->query()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
#else
        event->synchronize();
#endif
    }
    std::vector<std::shared_ptr<BlockBuffer>> blocks;
    for (const auto& [key, block] : request_block_buffer->getBlocks()) {
        blocks.push_back(block);
    }
    return makeAvailableRequest(blocks);
}

std::shared_ptr<TransferRequest>
RemoteStoreTaskImpl::makeAvailableRequest(const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    RTP_LLM_PROFILE_FUNCTION();
    auto                        transfer = std::make_shared<TransferRequest>(request_);
    std::lock_guard<std::mutex> lock(mutex_);
    if (done_) {
        return nullptr;
    }
    size_t block_size = 0;
    for (const auto& block : blocks) {
        if (!block) {
            continue;
        }
        auto it = to_load_buffers_.find(block->key);
        if (it == to_load_buffers_.end()) {
            continue;
        }
        transfer->buffer_pairs.emplace(*it);
        block_size = block->len;
    }
    if (transfer->buffer_pairs.empty()) {
        return nullptr;
    }
    const auto id = next_rpc_id_++;
    // Strong ownership keeps drain hooks reachable even if the caller releases
    // its task on early failure. Each admitted request must invoke this callback.
    transfer->callback = [task = shared_from_this(),
                          id](bool success, CacheStoreErrorCode, const std::map<std::string, std::string>&) {
        task->completeRpc(id, success);
    };
    pending_rpcs_.emplace(id, transfer->buffer_pairs);
    ++outstanding_rpcs_;
    if (collector_ && to_load_buffers_.size() == expect_done_buffer_count_) {
        collector_->markFirstBlockReady();
        collector_->setBlockSize(block_size * expect_done_buffer_count_);
    }
    for (const auto& [key, remote_key] : transfer->buffer_pairs) {
        to_load_buffers_.erase(key);
    }
    if (collector_ && to_load_buffers_.empty()) {
        collector_->markAllBlocksReady();
    }
    return transfer;
}

void RemoteStoreTaskImpl::completeRpc(uint64_t id, bool success) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = pending_rpcs_.find(id);
        if (it == pending_rpcs_.end()) {
            return;  // Duplicate callback: never settle or decrement twice.
        }
        done_buffer_count_ += it->second.size();
        pending_rpcs_.erase(it);
        if (!success) {
            finishLocked(ErrorInfo(ErrorCode::UNKNOWN_ERROR, ErrorCodeToString(ErrorCode::UNKNOWN_ERROR)));
        } else if (done_buffer_count_ == expect_done_buffer_count_) {
            finishLocked(ErrorInfo::OkStatus());
        }
    }
    notifyCallbacks();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        --outstanding_rpcs_;
    }
    // The callback has finished all work that can use the transfer's buffers.
    notifyCallbacks();
}

void RemoteStoreTaskImpl::notifyRequestDone(const std::map<std::string, std::string>& block_keys, bool success) {
    uint64_t id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = std::find_if(
            pending_rpcs_.begin(), pending_rpcs_.end(), [&](const auto& entry) { return entry.second == block_keys; });
        if (it == pending_rpcs_.end()) {
            return;
        }
        id = it->first;
    }
    completeRpc(id, success);
}

}  // namespace rtp_llm
