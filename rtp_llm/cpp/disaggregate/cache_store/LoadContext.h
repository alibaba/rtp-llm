#pragma once

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CommonDefine.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class CacheStore;

using CacheStoreLoadClock    = std::chrono::steady_clock;
using CacheStoreLoadDeadline = CacheStoreLoadClock::time_point;

CacheStoreLoadDeadline makeCacheStoreLoadDeadline(int64_t                timeout_ms,
                                                  CacheStoreLoadDeadline now = CacheStoreLoadClock::now()) noexcept;
bool getCacheStoreLoadRemainingTimeoutMs(CacheStoreLoadDeadline deadline,
                                         CacheStoreLoadDeadline now,
                                         uint32_t&              remaining_timeout_ms) noexcept;
bool isCacheStoreLoadDeadlineReached(CacheStoreLoadDeadline deadline, CacheStoreLoadDeadline now) noexcept;

class SyncContext: public std::enable_shared_from_this<SyncContext> {
public:
    SyncContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load);

public:
    typedef std::function<bool()> CheckCancelFunc;
    void                          call(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                                       int64_t                                                 timeout_ms,
                                       CheckCancelFunc                                         check_cancel_func);
    void                          call(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                                       CacheStoreLoadDeadline                                  deadline,
                                       CheckCancelFunc                                         check_cancel_func);

    void waitDone();

    bool        success() const;
    ErrorInfo   getErrorInfo() const;
    std::string getErrorInfoString() const;

    void
    updateResult(bool success, CacheStoreErrorCode ec, const std::shared_ptr<RequestBlockBuffer>& request_block_buffer);

protected:
    using CheckCancelFuncHolder = std::shared_ptr<const CheckCancelFunc>;

    virtual bool doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                        CacheStoreLoadDeadline                     deadline) = 0;

    static int64_t  makeDeadlineMs(int64_t start_time_ms, int64_t timeout_ms);
    static uint32_t normalizeTimeoutMs(int64_t timeout_ms);

    CheckCancelFuncHolder getCheckCancelFuncSnapshot() const;
    bool finalizeDeadlineOrCancellationLocked(bool                   cancellation_requested,
                                               CheckCancelFuncHolder& retired_check_cancel_func,
                                               CacheStoreLoadDeadline now = CacheStoreLoadClock::now());
    void finalizeLocked(CheckCancelFuncHolder& retired_check_cancel_func);
    void finalizeLocked(ErrorCode error_code, CheckCancelFuncHolder& retired_check_cancel_func);

protected:
    std::weak_ptr<CacheStore> cache_store_;
    bool                      combine_load_ = false;

    std::vector<std::shared_ptr<RequestBlockBuffer>> request_block_buffers_;
    ErrorInfo                                        error_info_;

    int64_t                start_time_ms_     = 0;
    CacheStoreLoadDeadline deadline_          = CacheStoreLoadDeadline::min();

    mutable std::mutex      mutex_;
    std::condition_variable cond_;
    int                     expect_layer_cnt_ = 0;
    std::atomic_int         done_layer_cnt_   = 0;
    bool                    result_finalized_ = false;
    // Keep this last so a non-terminal teardown destroys the callable before the synchronization state.
    CheckCancelFuncHolder check_cancel_func_ = nullptr;
};

class LoadContext: public SyncContext {
public:
    LoadContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load);

public:
    void load(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffer,
              const std::string&                                      ip,
              uint32_t                                                port,
              uint32_t                                                rdma_port,
              int64_t                                                 timeout_ms,
              CheckCancelFunc                                         check_cancel_func,
              int                                                     partition_count,
              int                                                     partition_id);
    void load(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffer,
              const std::string&                                      ip,
              uint32_t                                                port,
              uint32_t                                                rdma_port,
              CacheStoreLoadDeadline                                  deadline,
              CheckCancelFunc                                         check_cancel_func,
              int                                                     partition_count,
              int                                                     partition_id);

protected:
    bool doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                CacheStoreLoadDeadline                     deadline) override;

private:
    std::string peer_ip_;
    uint32_t    port_;
    uint32_t    rdma_port_;
    int         partition_count_;
    int         partition_id_;
};

class StoreContext: public SyncContext {
public:
    StoreContext(const std::shared_ptr<CacheStore>& cache_store);

public:
    void store(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffer, int64_t timeout_ms);

protected:
    bool doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                CacheStoreLoadDeadline                     deadline) override;

private:
};

}  // namespace rtp_llm
