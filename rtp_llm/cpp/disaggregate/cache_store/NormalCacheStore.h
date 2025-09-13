#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/CacheStoreMetricsCollector.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBufferStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/Messager.h"
#include "rtp_llm/cpp/disaggregate/cache_store/CacheStore.h"
#include "rtp_llm/cpp/disaggregate/cache_store/RemoteStoreTaskImpl.h"
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
              const std::string&                         ip,
              uint32_t                                   port,
              uint32_t                                   rdma_port,
              uint32_t                                   timeout_ms      = 1000,
              int                                        partition_count = 1,
              int                                        partition_id    = 0) override;

    std::shared_ptr<LoadContext>
    loadBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                const std::string&                                      ip,
                uint32_t                                                port,
                uint32_t                                                rdma_port,
                int64_t                                                 timeout_ms,
                LoadContext::CheckCancelFunc                            check_cancel_func,
                int                                                     partition_count,
                int                                                     partition_id) override;
    std::shared_ptr<StoreContext>
    storeBuffers(const std::vector<std::shared_ptr<RequestBlockBuffer>>& request_block_buffers,
                 int64_t                                                 timeout_ms) override;

    std::shared_ptr<RemoteStoreTask> submitRemoteStoreTask(const std::shared_ptr<RemoteStoreRequest>& request,
                                                           const std::shared_ptr<CacheStoreRemoteStoreMetricsCollector>& collector,
                                                           RemoteStoreTask::CheckCancelFunc check_cancel_func) override;
    void                             releaseRemoteStoreTask(const std::shared_ptr<RemoteStoreTask>& task) override;

    bool                         regUserBuffers(const std::vector<std::shared_ptr<BlockBuffer>>& buffers) override;
    std::shared_ptr<BlockBuffer> findUserBuffer(const std::string& buffer_key) override;

    void markRequestEnd(const std::string& requestid);

    void debugInfo() override;

    const std::shared_ptr<MemoryUtil>& getMemoryUtil() const override;

private:
    bool init(const CacheStoreInitParams& params);
    void runStoreTask(const std::shared_ptr<RequestBlockBuffer>&              value,
                      CacheStoreStoreDoneCallback                             callback,
                      const std::shared_ptr<CacheStoreStoreMetricsCollector>& collector);

    void runLoadTask(const std::shared_ptr<RequestBlockBuffer>&                   value,
                     CacheStoreLoadDoneCallback                                   callback,
                     const std::string&                                           ip,
                     uint32_t                                                     port,
                     uint32_t                                                     rdma_port,
                     uint32_t                                                     timeout_ms,
                     const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collelctor,
                     int                                                          partition_count,
                     int                                                          partition_id);

    const std::shared_ptr<RequestBlockBufferStore>& getRequestBlockBufferStore() const;

private:
    bool                                                                             thread_pool_close_{false};
    CacheStoreInitParams                                                             params_;
    std::shared_ptr<MemoryUtil>                                                      memory_util_;
    std::shared_ptr<RequestBlockBufferStore>                                         request_block_buffer_store_;
    std::shared_ptr<Messager>                                                        messager_;
    autil::ThreadPoolBasePtr                                                         thread_pool_;  // task executor
    kmonitor::MetricsReporterPtr                                                     metrics_reporter_;
    std::shared_mutex                                                                remote_store_tasks_mutex_;
    std::unordered_map<std::string, std::list<std::shared_ptr<RemoteStoreTaskImpl>>> remote_store_tasks_;
    std::shared_mutex                                                                store_tasks_mutex_;
    std::unordered_map<std::shared_ptr<RequestBlockBuffer>, std::pair<CacheStoreStoreDoneCallback, std::function<void()>>> store_tasks_;
};

}  // namespace rtp_llm
