#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"

namespace rtp_llm {

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

    for (auto& request_block_buffer : request_block_buffers_) {
        if (!doCall(request_block_buffer, timeout_ms)) {
            updateResult(false, CacheStoreErrorCode::InvalidParams, request_block_buffer);
        }
    }
}

void SyncContext::updateResult(bool                                       success,
                               CacheStoreErrorCode                        ec,
                               const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!success) {
        auto error_code = transCacheStoreErrorCode(ec);
        error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
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
            RTP_LLM_LOG_INFO("load context wait done on timeout");
            return;
        }

        if (check_cancel_func_ != nullptr && check_cancel_func_()) {
            auto error_code = ErrorCode::CANCELLED;
            error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
            RTP_LLM_LOG_INFO("load context wait done on cancelled");
            return;
        }

        // sync wait, safe to use this
        if (cond_.wait_for(lock, std::chrono::milliseconds(once_time_ms), [this] {
                return done_layer_cnt_ == expect_layer_cnt_;
            })) {
            return;
        }
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