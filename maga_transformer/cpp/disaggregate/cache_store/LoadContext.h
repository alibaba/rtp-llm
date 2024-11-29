#pragma once

#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBuffer.h"
#include "maga_transformer/cpp/utils/RpcErrorCode.h"

namespace rtp_llm {

class CacheStore;

class SyncContext: public std::enable_shared_from_this<SyncContext> {
public:
    SyncContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load);

public:
    typedef std::function<bool()> CheckCancelFunc;
    void                          call(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                                       int64_t                                                 timeout_ms,
                                       CheckCancelFunc                                         check_cancel_func);

    void waitDone();

    bool             success() const;
    const ErrorInfo& getErrorInfo() const;
    std::string      getErrorInfoString() const;

    void
    updateResult(bool success, CacheStoreErrorCode ec, const std::shared_ptr<RequestBlockBuffer>& request_block_buffer);

protected:
    virtual bool doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) = 0;

protected:
    std::weak_ptr<CacheStore> cache_store_;
    bool                      combine_load_ = false;

    std::vector<std::shared_ptr<RequestBlockBuffer>> request_block_buffers_;
    ErrorInfo                                        error_info_;

    int64_t         start_time_ms_     = 0;
    int64_t         deadline_ms_       = 0;
    CheckCancelFunc check_cancel_func_ = nullptr;

    mutable std::mutex      mutex_;
    std::condition_variable cond_;
    int                     expect_layer_cnt_ = 0;
    std::atomic_int         done_layer_cnt_   = 0;
};

class LoadContext: public SyncContext {
public:
    LoadContext(const std::shared_ptr<CacheStore>& cache_store, bool combine_load);

public:
    void load(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffer,
              const std::string&                                      ip,
              int64_t                                                 timeout_ms,
              CheckCancelFunc                                         check_cancel_func);

protected:
    bool doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) override;

private:
    std::string peer_ip_;
};

class StoreContext: public SyncContext {
public:
    StoreContext(const std::shared_ptr<CacheStore>& cache_store);

public:
    void store(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffer, int64_t timeout_ms);

protected:
    bool doCall(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer, int64_t timeout_ms) override;

private:
};

}  // namespace rtp_llm