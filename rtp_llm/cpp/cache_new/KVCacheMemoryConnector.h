#pragma once

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/model_rpc/TpBroadcastManager.h"

#include <map>

namespace rtp_llm {

class BlockPool;
class DeviceBase;
class KVCacheAllocator;
class MemoryBlockCache;
class TpBroadcastManager;

class MemoryConnectorAsyncContext: public AsyncContext {
public:
    MemoryConnectorAsyncContext(
        const std::shared_ptr<TPBroadcastResult<CopyCacheRequestPB, CopyCacheResponsePB>>& broadcast_result,
        const std::function<void(bool)>&                                                   done_callback):
        broadcast_result_(broadcast_result), done_callback_(done_callback) {}
    ~MemoryConnectorAsyncContext() override;

public:
    bool done() const override;
    bool success() const override;
    void waitDone();

private:
    std::shared_ptr<TPBroadcastResult<CopyCacheRequestPB, CopyCacheResponsePB>> broadcast_result_;
    std::function<void(bool)>                                                   done_callback_;
    bool                                                                        already_done_{false};
};

class KVCacheMemoryConnector: public KVCacheConnector, public std::enable_shared_from_this<KVCacheMemoryConnector> {
public:
    KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                           const std::shared_ptr<KVCacheAllocator>& allocator,
                           rtp_llm::DeviceBase*                     device,
                           const std::vector<std::string>&          tp_addrs,
                           const kmonitor::MetricsReporterPtr&      metrics_reporter = nullptr);
    ~KVCacheMemoryConnector() override;

public:
    bool init();

    std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                  const std::shared_ptr<Meta>&              meta) override;
    std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                 const std::shared_ptr<Meta>&              meta,
                                                 const std::shared_ptr<AsyncMatchContext>& match_context) override;
    std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                  const std::shared_ptr<Meta>&              meta) override;
    std::shared_ptr<AsyncContext>      asyncWriteByLayer(int                                       layer_id,
                                                         const std::shared_ptr<KVCacheResourceV1>& resource,
                                                         const std::shared_ptr<Meta>&              meta) override {
        throw std::runtime_error("KVCacheMemoryConnector asyncWriteByLayer is not implemented");
    }

    virtual bool copyCache(const MemoryCopyCacheRequestPB& request, MemoryCopyCacheResponsePB& response);
    void         clearCache();

private:
    struct LayerBlock {
        int layer_id;
        int block_id;
    };
    struct CopyInfoPerKey {
        size_t                  cache_key;
        std::vector<LayerBlock> gpu_layer_blocks;
        int                     mem_block_index;
        size_t                  mem_block_size;
    };
    enum class CopyDirection {
        H2D = 0,
        D2H = 1
    };

    std::vector<CopyInfoPerKey> buildCopyPlanForRead(const std::vector<int64_t>& cache_keys,
                                                     const LayerBlockIds&        layer_block_ids,
                                                     size_t                      gpu_matched_num,
                                                     size_t&                     cpu_matched_num);
    std::vector<CopyInfoPerKey> buildCopyPlanForWrite(const std::vector<int64_t>& cache_keys,
                                                      const LayerBlockIds&        layer_block_ids,
                                                      size_t                      cpu_matched_num);
    std::shared_ptr<TPBroadcastResult<CopyCacheRequestPB, CopyCacheResponsePB>>
         sendCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos, CopyDirection direction) const;
    bool prepareCopyBuffers(const std::vector<LayerBlock>& gpu_layer_blocks,
                            int                            mem_block_index,
                            size_t                         mem_block_size,
                            CopyDirection                  direction,
                            std::vector<BufferPtr>&        dst,
                            std::vector<BufferPtr>&        src);
    bool checkKVCacheResource(const std::shared_ptr<KVCacheResourceV1>& resource) const;
    bool mallocBlocks(const std::shared_ptr<BlockPool>& block_pool,
                      size_t                            need_blocks,
                      std::vector<BlockIdxType>&        malloced_blocks);
    bool
    freeBlocks(const std::shared_ptr<BlockPool>& block_pool, const std::vector<int>& blocks, bool cache_free = true);
    void referenceBlocks(const std::shared_ptr<BlockPool>& block_pool, const std::vector<int>& blocks);
    std::shared_ptr<BlockPool> getOrCreateMemoryBlockPool(size_t block_size, bool create = false);
    std::shared_ptr<BlockPool> getBlockPool(size_t block_size) const;
    std::shared_ptr<BlockPool> createBlockPool(size_t block_size);
    bool                       ensureEnoughFreeBlocks(const std::shared_ptr<BlockPool>& block_pool, size_t need_blocks);
    void                       printCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos) const;
    void                       waitContextDoneAsync(const std::shared_ptr<MemoryConnectorAsyncContext>& context);

    void reportReadMetrics(
        bool success, int64_t latency_us, int64_t input_block_num, int64_t matched_block_num, int64_t read_block_num);
    void reportWriteMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t write_block_num);
    void reportCopyMetrics(bool success, int64_t latency_us, CopyDirection direction);
    void reportMetricsLoop();

private:
    const CacheConfig&                cache_config_;
    std::shared_ptr<KVCacheAllocator> allocator_;
    rtp_llm::DeviceBase*              device_{nullptr};
    std::vector<std::string>          tp_addrs_;

    // cache key wise block size -> BlockPool
    std::map<size_t, std::shared_ptr<BlockPool>> block_pools_;
    std::shared_ptr<MemoryBlockCache>            block_cache_;
    std::shared_ptr<TpBroadcastManager>          tp_broadcast_manager_;
    std::shared_ptr<autil::LockFreeThreadPool>   wait_done_thread_pool_;

    // metrics reporter
    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::shared_ptr<std::thread> metrics_reporter_thread_{nullptr};
    std::atomic<bool>            stop_{false};
};

}  // namespace rtp_llm
