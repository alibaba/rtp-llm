#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
// C40 across eight Decode ranks stays below the peer's 1000-entry RPC queue
// even when every rank needs this peer: 40 * 8 * 2 = 640 in flight.
constexpr size_t   kTcpLoadWindow    = 2;
constexpr unsigned kQueueFullRetries = 5;
}  // namespace

SyncContext::SyncContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load):
    cache_store_(cache_store), combine_load_(combine_load) {}

void SyncContext::call(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                       int64_t                                                 timeout_ms,
                       CheckCancelFunc                                         check_cancel_func) {
    if (request_block_buffers.empty()) {
        return;
    }

    auto cache_store = cache_store_.lock();
    if (cache_store == nullptr) {
        error_info_ = ErrorInfo(ErrorCode::UNKNOWN_ERROR, ErrorCodeToString(ErrorCode::UNKNOWN_ERROR));
        RTP_LLM_LOG_WARNING("load failed, cache store is nullptr");
        return;
    }

    start_time_ms_     = autil::TimeUtility::currentTimeInMilliSeconds();
    deadline_ms_       = start_time_ms_ + timeout_ms;
    check_cancel_func_ = check_cancel_func;

    if (combine_load_) {  // for rdma only call rpc once
        auto new_buffer = std::make_shared<RequestBlockBuffer>(request_block_buffers[0]->getRequestId());
        for (auto& request_block_buffer : request_block_buffers) {
            auto blocks = request_block_buffer->getBlocks();
            for (auto& [_, block] : blocks) {
                new_buffer->addBlock(block);
            }
        }
        request_block_buffers_ = {new_buffer};
    } else {
        request_block_buffers_ = request_block_buffers;
    }

    expect_layer_cnt_ = request_block_buffers_.size();

    if (!combine_load_) {
        for (const auto& buffer : request_block_buffers_) {
            pending_calls_.push_back({buffer, start_time_ms_});
        }
        dispatchPendingCalls();
        return;
    }

    for (auto& request_block_buffer : request_block_buffers_) {
        if (!doCall(request_block_buffer, timeout_ms)) {
            updateResult(false, CacheStoreErrorCode::InvalidParams, request_block_buffer);
        }
    }
}

void SyncContext::dispatchPendingCalls() {
    // TCP has one RPC per layer/tag. Bound each request/peer window instead of
    // flooding the remote RPC queue; waitDone drives replenishment and backoff.
    while (true) {
        std::shared_ptr<RequestBlockBuffer> buffer;
        int64_t                             remaining_ms;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto                  now = autil::TimeUtility::currentTimeInMilliSeconds();
            if (stopped_ || pending_calls_.empty() || in_flight_ >= kTcpLoadWindow || now >= deadline_ms_
                || pending_calls_.front().ready_ms > now || (check_cancel_func_ && check_cancel_func_())) {
                return;
            }
            buffer = std::move(pending_calls_.front().buffer);
            pending_calls_.pop_front();
            ++in_flight_;
            remaining_ms = deadline_ms_ - now;
        }
        // A transport is allowed to complete synchronously.
        if (!doCall(buffer, remaining_ms)) {
            updateResult(false, CacheStoreErrorCode::InvalidParams, buffer);
        }
    }
}

void SyncContext::updateResult(bool                                       success,
                               CacheStoreErrorCode                        ec,
                               const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {
        return;
    }
    if (!combine_load_) {
        --in_flight_;
        auto&      retries = retry_counts_[request_block_buffer.get()];
        const auto now     = autil::TimeUtility::currentTimeInMilliSeconds();
        if (!success && ec == CacheStoreErrorCode::PushWorkerItemFailed && retries < kQueueFullRetries
            && now < deadline_ms_) {
            const auto delay_ms = 5LL << retries++;
            pending_calls_.push_back({request_block_buffer, now + delay_ms});
            cond_.notify_all();
            return;
        }
        cond_.notify_all();
    }
    if (!success) {
        auto error_code = transCacheStoreErrorCode(ec);
        error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
        failed_request_block_buffers_.push_back(request_block_buffer);
        RTP_LLM_LOG_WARNING("request %s call finished, state:[%s], error code[%s], cost time %ldms",
                            request_block_buffer->getRequestKey().c_str(),
                            success ? "success" : "failed",
                            CacheStoreErrorCodeToString(ec).c_str(),
                            autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_);
    } else {
        RTP_LLM_LOG_DEBUG("request %s call finished, state:[%s], cost time %ldms",
                          request_block_buffer->getRequestKey().c_str(),
                          success ? "success" : "failed",
                          autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_);
    }

    if (++done_layer_cnt_ == expect_layer_cnt_) {
        cond_.notify_all();
    }
}

void SyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto                         once_time_ms = 30;
    while (true) {
        if (done_layer_cnt_ == expect_layer_cnt_) {
            return;
        }

        if (autil::TimeUtility::currentTimeInMilliSeconds() >= deadline_ms_) {
            auto error_code = ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT;
            error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
            timed_out_      = true;
            stopped_        = true;
            pending_calls_.clear();
            RTP_LLM_LOG_INFO("load context wait done on timeout");
            return;
        }

        if (check_cancel_func_ != nullptr && check_cancel_func_()) {
            auto error_code = ErrorCode::CANCELLED;
            error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
            stopped_        = true;
            pending_calls_.clear();
            RTP_LLM_LOG_INFO("load context wait done on cancelled");
            return;
        }

        if (!combine_load_) {
            lock.unlock();
            dispatchPendingCalls();
            lock.lock();
            if (done_layer_cnt_ == expect_layer_cnt_) {
                return;
            }
        }
        cond_.wait_for(lock, std::chrono::milliseconds(once_time_ms));
    }
}

bool SyncContext::success() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_.ok();
}

std::string SyncContext::getErrorInfoString() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_.ToString();
}

const ErrorInfo& SyncContext::getErrorInfo() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return error_info_;
}

std::vector<std::string> SyncContext::failedBlockDebugInfos() const {
    std::unique_lock<std::mutex> lock(mutex_);
    if (error_info_.ok() || error_info_.code() == ErrorCode::CANCELLED) {
        return {};
    }

    const auto& buffers =
        (timed_out_ || failed_request_block_buffers_.empty()) ? request_block_buffers_ : failed_request_block_buffers_;
    std::vector<std::string> debug_infos;
    debug_infos.reserve(buffers.size());
    for (const auto& buffer : buffers) {
        debug_infos.push_back(buffer == nullptr ? "null" : buffer->debugInfo());
    }
    return debug_infos;
}

LoadContext::LoadContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load):
    SyncContext(cache_store, combine_load) {}

void LoadContext::load(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffer,
                       const std::string&                                      ip,
                       uint32_t                                                port,
                       uint32_t                                                rdma_port,
                       int64_t                                                 timeout_ms,
                       CheckCancelFunc                                         check_cancel_func,
                       int                                                     partition_count,
                       int                                                     partition_id) {
    peer_ip_         = ip;
    port_            = port;
    rdma_port_       = rdma_port;
    partition_count_ = partition_count;
    partition_id_    = partition_id;
    call(request_block_buffer, timeout_ms, check_cancel_func);
}

bool LoadContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) {
    auto cache_store = cache_store_.lock();

    auto load_layer_callback = [request_block_buffer, shared_this = shared_from_this()](bool                success,
                                                                                        CacheStoreErrorCode ec) {
        shared_this->updateResult(success, ec, request_block_buffer);
    };
    cache_store->load(request_block_buffer,
                      load_layer_callback,
                      peer_ip_,
                      port_,
                      rdma_port_,
                      timeout_ms,
                      partition_count_,
                      partition_id_);
    return true;
}

StoreContext::StoreContext(const std::shared_ptr<CacheStore>& cache_store): SyncContext(cache_store, true) {}

void StoreContext::store(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                         int64_t                                                 timeout_ms) {
    call(request_block_buffers, timeout_ms, nullptr);
}

bool StoreContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) {
    auto cache_store = cache_store_.lock();

    auto store_layer_callback = [request_block_buffer, shared_this = shared_from_this()](bool                success,
                                                                                         CacheStoreErrorCode ec) {
        shared_this->updateResult(success, ec, request_block_buffer);
    };
    cache_store->store(request_block_buffer, store_layer_callback);
    return true;
}

}  // namespace rtp_llm
