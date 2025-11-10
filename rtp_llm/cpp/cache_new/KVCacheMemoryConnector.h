#pragma once

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

#include <map>

namespace rtp_llm {

class BlockPool;
class DeviceBase;
class KVCacheAllocator;
class MemoryBlockCache;
class TpBroadcastManager;
class TPBroadcastResult;

class MemoryConnectorAsyncContext: public KVCacheConnector::AsyncContext {
public:
    MemoryConnectorAsyncContext(const std::shared_ptr<TPBroadcastResult>& broadcast_result,
                                const std::function<void(bool)>&          done_callback):
        broadcast_result_(broadcast_result), done_callback_(done_callback) {}
    MemoryConnectorAsyncContext(bool already_done, bool success): already_done_(already_done), success_(success) {}
    ~MemoryConnectorAsyncContext() override = default;

public:
    bool success() const override;
    void cancel() override;
    void waitDone() override;

private:
    bool allResponseSuccess() const;

private:
    std::shared_ptr<TPBroadcastResult> broadcast_result_;
    std::function<void(bool)>          done_callback_;
    bool                               already_done_{false};
    bool                               success_{false};
};

class KVCacheMemoryConnector final:
    public KVCacheConnector,
    public std::enable_shared_from_this<KVCacheMemoryConnector> {
public:
    KVCacheMemoryConnector(const CacheConfig&                       cache_config,
                           const std::shared_ptr<KVCacheAllocator>& allocator,
                           rtp_llm::DeviceBase*                     device,
                           const std::vector<std::string>&          tp_addrs);
    ~KVCacheMemoryConnector() override;

public:
    bool                          init() override;
    std::shared_ptr<AsyncContext> asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                            const std::shared_ptr<Meta>&              meta) override;
    std::shared_ptr<AsyncContext> asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource,
                                             const std::shared_ptr<Meta>&              meta) override;
    std::shared_ptr<AsyncContext> asyncWriteByLayer(int                                       layer_id,
                                                    const std::shared_ptr<KVCacheResourceV1>& resource,
                                                    const std::shared_ptr<Meta>&              meta) override {
        throw std::runtime_error("KVCacheMemoryConnector asyncWriteByLayer is not implemented");
    }

    // 同步拷贝KVCache(单TP)
    void copyCache(const MemoryBroadcastTpRequestPB& request, MemoryBroadcastTpResponsePB& response);

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

    std::vector<CopyInfoPerKey>        buildCopyPlanForRead(const std::vector<size_t>& cache_keys,
                                                            const LayerBlockIds&       layer_block_ids,
                                                            size_t                     gpu_reuse_len) const;
    std::vector<CopyInfoPerKey>        buildCopyPlanForWrite(const std::vector<size_t>& cache_keys,
                                                             const LayerBlockIds&       layer_block_ids,
                                                             size_t                     match_len);
    std::shared_ptr<TPBroadcastResult> sendCopyPlan(const std::vector<CopyInfoPerKey>& copy_infos,
                                                    CopyDirection                      direction) const;
    bool                               prepareCopyBuffers(const std::vector<LayerBlock>& gpu_layer_blocks,
                                                          int                            mem_block_index,
                                                          size_t                         mem_block_size,
                                                          CopyDirection                  direction,
                                                          std::vector<BufferPtr>&        dst,
                                                          std::vector<BufferPtr>&        src);
    bool                               mallocMemoryBlocks(const std::shared_ptr<BlockPool>& block_pool,
                                                          size_t                            need_blocks,
                                                          std::vector<BlockIdxType>&        malloced_blocks) const;
    bool freeMemoryBlocks(const std::shared_ptr<BlockPool>& block_pool, const std::vector<int>& blocks);
    std::shared_ptr<BlockPool> getOrCreateMemoryBlockPool(size_t block_size, bool create = false);
    bool ensureEnoughFreeBlocks(const std::shared_ptr<BlockPool>& block_pool, size_t need_blocks) const;

private:
    const CacheConfig&                cache_config_;
    std::shared_ptr<KVCacheAllocator> allocator_;
    rtp_llm::DeviceBase*              device_{nullptr};
    std::vector<std::string>          tp_addrs_;

    // cache key wise block size -> BlockPool
    std::map<size_t, std::shared_ptr<BlockPool>> block_pools_;
    std::shared_ptr<MemoryBlockCache>            block_cache_;
    std::shared_ptr<TpBroadcastManager>          tp_broadcast_manager_;
};

}  // namespace rtp_llm
