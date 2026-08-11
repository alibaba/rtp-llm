#include "rtp_llm/cpp/disaggregate/cache_store/LoadContext.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/ErrorCodeUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace rtp_llm {

SyncContext::SyncContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load):
    cache_store_(cache_store), combine_load_(combine_load) {}

CacheStoreLoadDeadline makeCacheStoreLoadDeadline(int64_t timeout_ms, CacheStoreLoadDeadline now) noexcept {
    if (timeout_ms <= 0) {
        return now;
    }

    using FloatMilliseconds = std::chrono::duration<long double, std::milli>;
    const auto max_deadline  = CacheStoreLoadDeadline::max();
    const auto headroom_ms = FloatMilliseconds(max_deadline.time_since_epoch()).count()
                             - FloatMilliseconds(now.time_since_epoch()).count();
    const auto duration_limit_ms = FloatMilliseconds(CacheStoreLoadClock::duration::max()).count();
    if (static_cast<long double>(timeout_ms) >= headroom_ms
        || static_cast<long double>(timeout_ms) >= duration_limit_ms) {
        return max_deadline;
    }
    return now
           + std::chrono::duration_cast<CacheStoreLoadClock::duration>(std::chrono::milliseconds(timeout_ms));
}

bool getCacheStoreLoadRemainingTimeoutMs(CacheStoreLoadDeadline deadline,
                                         CacheStoreLoadDeadline now,
                                         uint32_t&              remaining_timeout_ms) noexcept {
    const auto remaining = deadline - now;
    if (remaining <= CacheStoreLoadClock::duration::zero()) {
        remaining_timeout_ms = 0;
        return false;
    }

    auto rounded_ms = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
    if (std::chrono::duration_cast<CacheStoreLoadClock::duration>(rounded_ms) < remaining) {
        ++rounded_ms;
    }
    constexpr auto kMaxTimeoutMs = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    remaining_timeout_ms = rounded_ms.count() >= kMaxTimeoutMs ?
                               std::numeric_limits<uint32_t>::max() :
                               static_cast<uint32_t>(rounded_ms.count());
    return true;
}

bool isCacheStoreLoadDeadlineReached(CacheStoreLoadDeadline deadline, CacheStoreLoadDeadline now) noexcept {
    return now >= deadline;
}

int64_t SyncContext::makeDeadlineMs(int64_t start_time_ms, int64_t timeout_ms) {
    if (timeout_ms <= 0) {
        return start_time_ms;
    }
    if (start_time_ms > std::numeric_limits<int64_t>::max() - timeout_ms) {
        return std::numeric_limits<int64_t>::max();
    }
    return start_time_ms + timeout_ms;
}

uint32_t SyncContext::normalizeTimeoutMs(int64_t timeout_ms) {
    if (timeout_ms <= 0) {
        return 0;
    }
    return timeout_ms > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max() :
                                                               static_cast<uint32_t>(timeout_ms);
}

void SyncContext::call(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                       int64_t                                                 timeout_ms,
                       CheckCancelFunc                                         check_cancel_func) {
    call(request_block_buffers, makeCacheStoreLoadDeadline(timeout_ms), std::move(check_cancel_func));
}

void SyncContext::call(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                       CacheStoreLoadDeadline                                  deadline,
                       CheckCancelFunc                                         check_cancel_func) {
    if (request_block_buffers.empty()) {
        return;
    }

    start_time_ms_ = autil::TimeUtility::currentTimeInMilliSeconds();
    CheckCancelFuncHolder new_check_cancel_func;
    if (check_cancel_func != nullptr) {
        new_check_cancel_func = std::make_shared<const CheckCancelFunc>(std::move(check_cancel_func));
    }
    CheckCancelFuncHolder retired_check_cancel_func;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        deadline_ = deadline;
        retired_check_cancel_func.swap(check_cancel_func_);
        check_cancel_func_.swap(new_check_cancel_func);
    }
    retired_check_cancel_func.reset();

    auto cache_store = cache_store_.lock();
    if (cache_store == nullptr) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!result_finalized_) {
                finalizeLocked(ErrorCode::UNKNOWN_ERROR, retired_check_cancel_func);
            }
        }
        retired_check_cancel_func.reset();
        RTP_LLM_LOG_WARNING("load failed, cache store is nullptr");
        return;
    }

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
        const auto check_cancel_func = getCheckCancelFuncSnapshot();
        const bool cancellation_requested  = check_cancel_func != nullptr && (*check_cancel_func)();
        bool       stop_submission          = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_submission = result_finalized_
                              || finalizeDeadlineOrCancellationLocked(
                                  cancellation_requested, retired_check_cancel_func);
        }
        retired_check_cancel_func.reset();
        if (stop_submission) {
            break;
        }
        if (!doCall(request_block_buffer, deadline)) {
            updateResult(false, CacheStoreErrorCode::InvalidParams, request_block_buffer);
        }
    }
}

void SyncContext::finalizeLocked(CheckCancelFuncHolder& retired_check_cancel_func) {
    result_finalized_ = true;
    retired_check_cancel_func.swap(check_cancel_func_);
    cond_.notify_all();
}

void SyncContext::finalizeLocked(ErrorCode error_code, CheckCancelFuncHolder& retired_check_cancel_func) {
    error_info_ = ErrorInfo(error_code, ErrorCodeToString(error_code));
    finalizeLocked(retired_check_cancel_func);
}

SyncContext::CheckCancelFuncHolder SyncContext::getCheckCancelFuncSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return result_finalized_ ? nullptr : check_cancel_func_;
}

bool SyncContext::finalizeDeadlineOrCancellationLocked(bool                   cancellation_requested,
                                                       CheckCancelFuncHolder& retired_check_cancel_func,
                                                       CacheStoreLoadDeadline now) {
    if (isCacheStoreLoadDeadlineReached(deadline_, now)) {
        finalizeLocked(ErrorCode::CACHE_STORE_LOAD_BUFFER_TIMEOUT, retired_check_cancel_func);
        RTP_LLM_LOG_INFO("load context reached deadline");
        return true;
    }

    if (cancellation_requested) {
        finalizeLocked(ErrorCode::CANCELLED, retired_check_cancel_func);
        RTP_LLM_LOG_INFO("load context was cancelled");
        return true;
    }
    return false;
}

void SyncContext::updateResult(bool                                       success,
                               CacheStoreErrorCode                        ec,
                               const std::shared_ptr<RequestBlockBuffer>& request_block_buffer) {
    CheckCancelFuncHolder retired_check_cancel_func;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (result_finalized_) {
            return;
        }
        if (finalizeDeadlineOrCancellationLocked(false, retired_check_cancel_func)) {
            // The cancellation target is released below, after mutex_ is unlocked.
        } else {
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
                finalizeLocked(retired_check_cancel_func);
            }
        }
    }
    retired_check_cancel_func.reset();
}

void SyncContext::waitDone() {
    constexpr auto kCancelPollIntervalMs = int64_t{30};
    while (true) {
        const auto check_cancel_func = getCheckCancelFuncSnapshot();
        const bool cancellation_requested = check_cancel_func != nullptr && (*check_cancel_func)();

        CheckCancelFuncHolder retired_check_cancel_func;
        bool                  should_return = false;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (result_finalized_ || expect_layer_cnt_ == 0) {
                return;
            }

            const auto now = CacheStoreLoadClock::now();
            if (finalizeDeadlineOrCancellationLocked(
                    cancellation_requested, retired_check_cancel_func, now)) {
                should_return = true;
            } else {
                const auto wake_deadline = check_cancel_func_ == nullptr ?
                                               deadline_ :
                                               std::min(deadline_, makeCacheStoreLoadDeadline(kCancelPollIntervalMs, now));
                cond_.wait_until(lock, wake_deadline, [this] { return result_finalized_; });
            }
        }

        retired_check_cancel_func.reset();
        if (should_return) {
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

ErrorInfo SyncContext::getErrorInfo() const {
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
    load(request_block_buffer,
         ip,
         port,
         rdma_port,
         makeCacheStoreLoadDeadline(timeout_ms),
         std::move(check_cancel_func),
         partition_count,
         partition_id);
}

void LoadContext::load(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffer,
                       const std::string&                                      ip,
                       uint32_t                                                port,
                       uint32_t                                                rdma_port,
                       CacheStoreLoadDeadline                                  deadline,
                       CheckCancelFunc                                         check_cancel_func,
                       int                                                     partition_count,
                       int                                                     partition_id) {
    peer_ip_         = ip;
    port_            = port;
    rdma_port_       = rdma_port;
    partition_count_ = partition_count;
    partition_id_    = partition_id;
    call(request_block_buffer, deadline, std::move(check_cancel_func));
}

bool LoadContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                         CacheStoreLoadDeadline                     deadline) {
    auto cache_store = cache_store_.lock();

    auto load_layer_callback = [request_block_buffer, shared_this = shared_from_this()](bool                success,
                                                                                        CacheStoreErrorCode ec) {
        shared_this->updateResult(success, ec, request_block_buffer);
    };
    cache_store->loadUntil(request_block_buffer,
                           load_layer_callback,
                           peer_ip_,
                           port_,
                           rdma_port_,
                           deadline,
                           partition_count_,
                           partition_id_);
    return true;
}

StoreContext::StoreContext(const std::shared_ptr<CacheStore>& cache_store): SyncContext(cache_store, true) {}

void StoreContext::store(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                         int64_t                                                 timeout_ms) {
    call(request_block_buffers, timeout_ms, nullptr);
}

bool StoreContext::doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                          CacheStoreLoadDeadline) {
    auto cache_store = cache_store_.lock();

    auto store_layer_callback = [request_block_buffer, shared_this = shared_from_this()](bool                success,
                                                                                         CacheStoreErrorCode ec) {
        shared_this->updateResult(success, ec, request_block_buffer);
    };
    cache_store->store(request_block_buffer, store_layer_callback);
    return true;
}

}  // namespace rtp_llm
