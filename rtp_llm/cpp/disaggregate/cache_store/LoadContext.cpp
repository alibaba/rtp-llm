#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace rtp_llm {

namespace {

const bool kPdDebugEnabled = []() {
    const char* env = std::getenv("RTP_LLM_PD_DEBUG");
    return env != nullptr && std::string(env) == "1";
}();

bool pdDebugEnabled() {
    return kPdDebugEnabled;
}

std::string summarizeBlocks(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, size_t limit = 3) {
    if (request_block_buffer == nullptr) {
        return "null";
    }
    std::ostringstream oss;
    oss << "request_id=" << request_block_buffer->getRequestId()
        << " request_key=" << request_block_buffer->getRequestKey()
        << " blocks=" << request_block_buffer->getBlocksCount() << " bytes=" << request_block_buffer->getBlocksSize();
    auto   blocks = request_block_buffer->getBlocks();
    size_t idx    = 0;
    oss << " sample_keys=[";
    for (const auto& [key, block] : blocks) {
        if (idx++ >= limit) {
            oss << "...";
            break;
        }
        if (idx > 1) {
            oss << ",";
        }
        oss << key << ":" << (block == nullptr ? 0 : block->len);
    }
    oss << "]";
    return oss.str();
}

std::string summarizeRequestBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                                    size_t                                                  limit = 6) {
    std::ostringstream oss;
    size_t             total_blocks = 0;
    size_t             total_bytes  = 0;
    for (const auto& request_block_buffer : request_block_buffers) {
        if (request_block_buffer == nullptr) {
            continue;
        }
        total_blocks += request_block_buffer->getBlocksCount();
        total_bytes += request_block_buffer->getBlocksSize();
    }
    oss << "buffers=" << request_block_buffers.size() << " total_blocks=" << total_blocks
        << " total_bytes=" << total_bytes << " samples=[";
    for (size_t i = 0; i < request_block_buffers.size() && i < limit; ++i) {
        if (i > 0) {
            oss << " | ";
        }
        oss << summarizeBlocks(request_block_buffers[i], 1);
    }
    if (request_block_buffers.size() > limit) {
        oss << " | ...";
    }
    oss << "]";
    return oss.str();
}

std::string summarizeMissingRequestKeys(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                                        const std::unordered_set<std::string>&                  done_request_keys,
                                        size_t                                                  limit = 16) {
    std::ostringstream oss;
    size_t             missing_count = 0;
    oss << "[";
    for (const auto& request_block_buffer : request_block_buffers) {
        if (request_block_buffer == nullptr) {
            continue;
        }
        const auto& request_key = request_block_buffer->getRequestKey();
        if (done_request_keys.find(request_key) != done_request_keys.end()) {
            continue;
        }
        if (missing_count > 0 && missing_count <= limit) {
            oss << ",";
        }
        if (missing_count < limit) {
            oss << request_key;
        }
        ++missing_count;
    }
    if (missing_count > limit) {
        oss << ",...";
    }
    oss << "] missing_count=" << missing_count;
    return oss.str();
}

}  // namespace

SyncContext::SyncContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load):
    cache_store_(cache_store), combine_load_(combine_load) {}

void SyncContext::setMaxInflightRequestCount(size_t max_inflight_request_count) {
    max_inflight_request_count_ = max_inflight_request_count;
}

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
    timeout_ms_        = timeout_ms;
    check_cancel_func_ = check_cancel_func;
    next_request_idx_  = 0;
    done_request_keys_.clear();
    done_layer_cnt_ = 0;

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
    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][LOAD_CONTEXT_CALL] combine_load=%d timeout_ms=%ld expect_layer_cnt=%d "
                         "max_inflight=%zu %s",
                         static_cast<int>(combine_load_),
                         timeout_ms,
                         expect_layer_cnt_,
                         max_inflight_request_count_,
                         summarizeRequestBuffers(request_block_buffers_).c_str());
    }

    size_t submit_count = request_block_buffers_.size();
    if (max_inflight_request_count_ > 0 && max_inflight_request_count_ < submit_count) {
        submit_count = max_inflight_request_count_;
    }
    next_request_idx_ = submit_count;

    for (size_t i = 0; i < submit_count; ++i) {
        auto& request_block_buffer = request_block_buffers_[i];
        if (!doCall(request_block_buffer, timeout_ms_)) {
            updateResult(false, CacheStoreErrorCode::InvalidParams, request_block_buffer);
        }
    }
}

void SyncContext::updateResult(bool                                       success,
                               CacheStoreErrorCode                        ec,
                               const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    std::shared_ptr<RequestBlockBuffer> next_request_block_buffer;
    int64_t                             next_timeout_ms = 0;
    bool                                should_notify   = false;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pdDebugEnabled()) {
            RTP_LLM_LOG_INFO("[PD_DEBUG][LOAD_CONTEXT_UPDATE] success=%d ec=%s done_before=%d expect=%d cost_ms=%ld %s",
                             static_cast<int>(success),
                             CacheStoreErrorCodeToString(ec).c_str(),
                             done_layer_cnt_.load(),
                             expect_layer_cnt_,
                             autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_,
                             summarizeBlocks(request_block_buffer).c_str());
        }
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

        if (request_block_buffer != nullptr) {
            done_request_keys_.insert(request_block_buffer->getRequestKey());
        }

        const auto done_after = ++done_layer_cnt_;
        const auto now_ms     = autil::TimeUtility::currentTimeInMilliSeconds();
        if (max_inflight_request_count_ > 0 && next_request_idx_ < request_block_buffers_.size()
            && now_ms < deadline_ms_) {
            next_request_block_buffer = request_block_buffers_[next_request_idx_++];
            next_timeout_ms           = std::max<int64_t>(1, deadline_ms_ - now_ms);
        }
        should_notify = done_after == expect_layer_cnt_;
    }

    if (next_request_block_buffer != nullptr) {
        if (pdDebugEnabled()) {
            RTP_LLM_LOG_INFO("[PD_DEBUG][LOAD_CONTEXT_SCHEDULE_NEXT] remaining_timeout_ms=%ld %s",
                             next_timeout_ms,
                             summarizeBlocks(next_request_block_buffer).c_str());
        }
        if (!doCall(next_request_block_buffer, next_timeout_ms)) {
            updateResult(false, CacheStoreErrorCode::InvalidParams, next_request_block_buffer);
        }
    }

    if (should_notify) {
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
            if (pdDebugEnabled()) {
                RTP_LLM_LOG_WARNING(
                    "[PD_DEBUG][LOAD_CONTEXT_TIMEOUT] done=%d expect=%d elapsed_ms=%ld deadline_ms=%ld %s",
                    done_layer_cnt_.load(),
                    expect_layer_cnt_,
                    autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_,
                    deadline_ms_,
                    summarizeRequestBuffers(request_block_buffers_).c_str());
                RTP_LLM_LOG_WARNING("[PD_DEBUG][LOAD_CONTEXT_TIMEOUT_MISSING] %s",
                                    summarizeMissingRequestKeys(request_block_buffers_, done_request_keys_).c_str());
            }
            return;
        }

        if (check_cancel_func_ != nullptr && check_cancel_func_()) {
            auto error_code = ErrorCode::CANCELLED;
            error_info_     = ErrorInfo(error_code, ErrorCodeToString(error_code));
            RTP_LLM_LOG_INFO("load context wait done on cancelled");
            if (pdDebugEnabled()) {
                RTP_LLM_LOG_WARNING("[PD_DEBUG][LOAD_CONTEXT_CANCELLED] done=%d expect=%d elapsed_ms=%ld %s",
                                    done_layer_cnt_.load(),
                                    expect_layer_cnt_,
                                    autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms_,
                                    summarizeRequestBuffers(request_block_buffers_).c_str());
            }
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
    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][LOAD_CONTEXT_LOAD] peer=%s:%u rdma_port=%u timeout_ms=%ld "
                         "partition_count=%d partition_id=%d input_%s",
                         peer_ip_.c_str(),
                         port_,
                         rdma_port_,
                         timeout_ms,
                         partition_count_,
                         partition_id_,
                         summarizeRequestBuffers(request_block_buffer).c_str());
    }
    call(request_block_buffer, timeout_ms, check_cancel_func);
}

bool LoadContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) {
    auto cache_store = cache_store_.lock();
    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][LOAD_CONTEXT_DO_CALL] peer=%s:%u rdma_port=%u timeout_ms=%ld "
                         "partition_count=%d partition_id=%d %s",
                         peer_ip_.c_str(),
                         port_,
                         rdma_port_,
                         timeout_ms,
                         partition_count_,
                         partition_id_,
                         summarizeBlocks(request_block_buffer).c_str());
    }

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
    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][STORE_CONTEXT_STORE] timeout_ms=%ld %s",
                         timeout_ms,
                         summarizeRequestBuffers(request_block_buffers).c_str());
    }
    call(request_block_buffers, timeout_ms, nullptr);
}

bool StoreContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) {
    auto cache_store = cache_store_.lock();
    if (pdDebugEnabled()) {
        RTP_LLM_LOG_INFO("[PD_DEBUG][STORE_CONTEXT_DO_CALL] timeout_ms=%ld %s",
                         timeout_ms,
                         summarizeBlocks(request_block_buffer).c_str());
    }

    auto store_layer_callback = [request_block_buffer, shared_this = shared_from_this()](bool                success,
                                                                                         CacheStoreErrorCode ec) {
        shared_this->updateResult(success, ec, request_block_buffer);
    };
    cache_store->store(request_block_buffer, store_layer_callback);
    return true;
}

}  // namespace rtp_llm
