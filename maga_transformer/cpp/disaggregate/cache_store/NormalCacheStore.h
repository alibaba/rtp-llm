#pragma once

#include "maga_transformer/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsReporter.h"
#include "maga_transformer/cpp/disaggregate/cache_store/metrics/CacheStoreMetricsCollector.h"
#include "maga_transformer/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MessagerClient.h"
#include "maga_transformer/cpp/disaggregate/cache_store/MessagerServer.h"
#include "maga_transformer/cpp/disaggregate/cache_store/CacheStore.h"
#include "autil/ThreadPool.h"

#include <memory>

namespace rtp_llm {

class NormalCacheStore: public CacheStore {
private:
    NormalCacheStore() = default;

public:
    ~NormalCacheStore();

public:
    static std::shared_ptr<NormalCacheStore> createNormalCacheStore(const CacheStoreInitParams& params);

    void store(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
               CacheStoreStoreDoneCallback                callback) override;

    void load(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
              CacheStoreLoadDoneCallback                 callback,
              const std::string&                         ip         = "",
              uint32_t                                   timeout_ms = 1000,
              int partition_count = 1,
              int partition_id = 0) override;

    std::shared_ptr<LoadContext>  loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                                       const std::string&                                      ip,
                                       int64_t                                                 timeout_ms,
                                       LoadContext::CheckCancelFunc check_cancel_func,
                                       int partition_count,
                                       int partition_id) override;
    std::shared_ptr<StoreContext> storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                                        int64_t                                                 timeout_ms) override;

    void markRequestEnd(const std::string& requestid);

    void debugInfo();

    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const;

private:
    bool init(const CacheStoreInitParams& params);
    void runStoreTask(const std::shared_ptr<RequestBlockBuffer>&                    value,
                      CacheStoreStoreDoneCallback                                   callback,
                      const std::shared_ptr<CacheStoreClientStoreMetricsCollector>& collector);

    void runLoadTask(const std::shared_ptr<RequestBlockBuffer>&                   value,
                     CacheStoreLoadDoneCallback                                   callback,
                     const std::string&                                           ip,
                     uint32_t                                                     timeout_ms,
                     const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collelctor,
                     int partition_count,
                     int partition_id);

    const std::shared_ptr<RequestBlockBufferStore>& getRequestBlockBufferStore() const;

private:
    CacheStoreInitParams                       params_;
    std::shared_ptr<MemoryUtil>                memory_util_;
    std::shared_ptr<RequestBlockBufferStore>   request_block_buffer_store_;
    std::shared_ptr<CacheStoreMetricsReporter> metrics_reporter_;
    std::unique_ptr<MessagerClient>            messager_client_;
    std::unique_ptr<MessagerServer>            messager_server_;
    autil::ThreadPoolBasePtr                   thread_pool_;  // task executor
    std::shared_ptr<arpc::TimerManager>        timer_manager_;
};

}  // namespace rtp_llm
