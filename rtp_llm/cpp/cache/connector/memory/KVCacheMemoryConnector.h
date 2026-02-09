#pragma once

#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/memory/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class BlockPool;
class BroadcastManager;
class DeviceBase;
class KVCacheAllocator;
class MemoryAsyncContext;

class KVCacheMemoryConnector: public KVCacheConnector, public std::enable_shared_from_this<KVCacheMemoryConnector> {
public:
    KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                           const KVCacheConfig&                     kv_cache_config,
                           const std::shared_ptr<KVCacheAllocator>& allocator,
                           rtp_llm::DeviceBase*                     device,
                           const std::vector<std::string>&          tp_addrs,
                           const kmonitor::MetricsReporterPtr&      metrics_reporter = nullptr);
    ~KVCacheMemoryConnector() override;

public:
    bool init();

    std::shared_ptr<AsyncMatchContext> asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::shared_ptr<Meta>&            meta) override;
    std::shared_ptr<AsyncContext>      asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                 const std::shared_ptr<Meta>&              meta,
                                                 const std::shared_ptr<AsyncMatchContext>& match_context,
                                                 int                                       start_read_block_index,
                                                 int                                       read_block_num) override;
    std::shared_ptr<AsyncContext>      asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                  const std::shared_ptr<Meta>&            meta) override;
    std::shared_ptr<AsyncContext>      asyncWriteByLayer(int                                     layer_id,
                                                         const std::shared_ptr<KVCacheResource>& resource,
                                                         const std::shared_ptr<Meta>&            meta) override {
        RTP_LLM_FAIL("KVCacheMemoryConnector asyncWriteByLayer is not implemented");
    }

    // virtual for test
    virtual bool              copyCache(const MemoryOperationRequestPB& request, MemoryOperationResponsePB& response);
    std::vector<CacheKeyType> cacheKeys() const;

private:
    struct LayerBlock {
        int          layer_id;
        BlockIdxType block_id;
    };
    struct CopyInfoPerKey {
        CacheKeyType            cache_key;
        std::vector<LayerBlock> gpu_layer_blocks;
        BlockIdxType            mem_block_index;
        size_t                  mem_block_size;
    };
    enum class CopyDirection {
        H2D = 0,
        D2H = 1
    };
    struct CopyPlan {
        std::vector<CopyInfoPerKey> copy_infos;
        CopyDirection               direction;
    };

    std::shared_ptr<CopyPlan> buildCopyPlanForRead(const CacheKeysType& cache_keys,
                                                   const LayerBlockIds& layer_block_ids,
                                                   int                  start_index,
                                                   int                  read_num);
    std::shared_ptr<CopyPlan> buildCopyPlanForWrite(const CacheKeysType& cache_keys,
                                                    const LayerBlockIds& layer_block_ids,
                                                    int                  start_index,
                                                    int                  write_num);
    bool startCopyAsync(const std::shared_ptr<MemoryAsyncContext>& context, const std::shared_ptr<CopyPlan>& copy_plan);
    std::shared_ptr<BroadcastResult<FunctionRequestPB, FunctionResponsePB>>
         sendCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const;
    void printCopyPlan(const std::shared_ptr<CopyPlan>& copy_plan) const;

    bool prepareCopyBuffers(const std::vector<LayerBlock>& gpu_layer_blocks,
                            BlockIdxType                   mem_block_index,
                            size_t                         mem_block_size,
                            CopyDirection                  direction,
                            std::vector<BufferPtr>&        dst,
                            std::vector<BufferPtr>&        src);
    bool appendCopyBytesToBuffers(const BlockInfo&        mem_block,
                                  const BlockInfo&        gpu_block,
                                  size_t                  byte_off,
                                  CopyDirection           direction,
                                  std::vector<BufferPtr>& dst,
                                  std::vector<BufferPtr>& src);

    bool checkLayerBlocks(const LayerBlockIds& layer_block_ids, size_t required_len) const;
    bool mallocBlocks(const std::shared_ptr<BlockPool>& block_pool,
                      size_t                            need_blocks,
                      std::vector<BlockIdxType>&        malloced_blocks);
    bool freeBlocks(const std::shared_ptr<BlockPool>& block_pool,
                    const std::vector<BlockIdxType>&  blocks,
                    bool                              cache_free = true);
    void referenceBlocks(const std::shared_ptr<BlockPool>& block_pool, const std::vector<BlockIdxType>& blocks);
    bool ensureEnoughFreeBlocks(const std::shared_ptr<BlockPool>& block_pool, size_t need_blocks);

    void                       initBlockPool();
    std::shared_ptr<BlockPool> getBlockPool(size_t block_size) const;
    std::shared_ptr<BlockPool> createBlockPool(size_t block_size, size_t pool_size_mb) const;
    std::string                blockPoolDebugString() const;
    void                       putToCache(const MemoryBlockCache::CacheItem& item);

    void reportMatchMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t matched_block_num);
    void reportReadMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t read_block_num);
    void reportWriteMetrics(bool success, int64_t latency_us, int64_t input_block_num, int64_t write_block_num);
    void reportCopyMetrics(bool success, int64_t latency_us, CopyDirection direction);
    void reportMetricsLoop();

private:
    const CacheConfig&                cache_config_;
    const KVCacheConfig&              kv_cache_config_;
    std::shared_ptr<KVCacheAllocator> allocator_;
    rtp_llm::DeviceBase*              device_{nullptr};
    const std::vector<std::string>    tp_addrs_;

    // cache key wise block size -> BlockPool
    std::map<size_t, std::shared_ptr<BlockPool>> block_pools_;
    mutable std::shared_mutex                    pool_mutex_;
    std::shared_ptr<MemoryBlockCache>            block_cache_;
    std::shared_ptr<BroadcastManager>            broadcast_manager_;
    std::shared_ptr<autil::LockFreeThreadPool>   wait_done_thread_pool_;

    // metrics reporter
    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::shared_ptr<std::thread> metrics_reporter_thread_{nullptr};
    std::atomic<bool>            stop_{false};
};

}  // namespace rtp_llm
