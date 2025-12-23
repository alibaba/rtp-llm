#pragma once

#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/types.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class KVCacheConnectorCoordinator;
class KVCacheConnectorReadWriteContext;

class KVCacheManager {
public:
    bool               init();
    const CacheConfig& cacheConfig() const;

    CacheLayerLayout layerCacheBase() const;

    KVCacheManager(const CacheConfig&                 config,
                   rtp_llm::DeviceBase*               device,
                   bool                               warmup             = false,
                   const kmonitor::MetricsReporterPtr metrics_reporter   = nullptr,
                   const KVCacheConfig&               kv_cache_config    = KVCacheConfig{},
                   const ParallelismConfig&           parallelism_config = ParallelismConfig{},
                   const RuntimeConfig&               runtime_config     = RuntimeConfig{});
    ~KVCacheManager();

    size_t      freeBlocksNum() const;
    size_t      availableBlocksNum() const;
    size_t      availableTokensNum() const;
    size_t      totalBlocksNum() const;
    size_t      maxAvailableTokensNum() const;
    KVCacheInfo getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const;

    // For backward compatibility with old code
    KVCacheBuffer kvCacheBuffer() const;

    MallocResult malloc(const MallocInfo& malloc_info);
    void         free(const FreeInfo& free_info);
    void         insertIntoCache(const InsertInfo& insert_info);

    void blockCopy(int src_block_index, int dest_block_index);
    void blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping);
    void blockBatchCopy(const rtp_llm::Buffer& copy_mapping);
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end);

    BlockAddrInfo convertIndexToAddr(int block_index, int layer_id) const;

    void regUserMr(size_t model_id);

    bool updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                       const std::vector<int>&        block_src_batch,
                       bool                           copy_last_block,
                       std::vector<BlockIdPair>&      block_update_mapping);

    // Write one KV block (optionally per-layer) from host/device buffers for test
    virtual bool setKVBlockValue(int block_index, int layer_id, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer);
    virtual bool setKVBlockValue(int block_index, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer);

    // async load cache from connector to gpu, for all rank
    std::shared_ptr<AsyncContext>
    asyncLoadCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context);

    // async store cache from gpu to connector, for all rank
    std::shared_ptr<AsyncContext>
    asyncStoreCache(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context);

    // broadcast tp for single rank
    bool broadcastTp(const BroadcastTpRequestPB& request, BroadcastTpResponsePB& response);

private:
    bool initConnectorCoordinator();

private:
    void allocateAndSync();
    void reportMetricsLoop();

    CacheConfig          config_;
    rtp_llm::DeviceBase* device_;
    KVCacheAllocatorPtr  allocator_;

    const kmonitor::MetricsReporterPtr metrics_reporter_;
    const KVCacheConfig                kv_cache_config_;
    const ParallelismConfig            parallelism_config_;
    const RuntimeConfig                runtime_config_;

    std::atomic<bool> stop_{false};
    std::thread       metrics_reporter_thread_;

    std::shared_ptr<KVCacheConnectorCoordinator> connector_coordinator_;
};

}  // namespace rtp_llm