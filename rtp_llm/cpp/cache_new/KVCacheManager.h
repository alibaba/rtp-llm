#pragma once

#include <cassert>
#include <vector>

#include "autil/LockFreeThreadPool.h"
#include "rtp_llm/cpp/cache_new/AsyncContext.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class KVCacheConnector;

class KVCacheManager {
public:
    bool               init();
    const CacheConfig& cacheConfig() const;

    CacheLayerLayout layerCacheBase() const;

    KVCacheManager(const CacheConfig&                 config,
                   rtp_llm::DeviceBase*               device,
                   bool                               warmup           = false,
                   const kmonitor::MetricsReporterPtr metrics_reporter = nullptr,
                   const GptInitParameter&            params           = GptInitParameter{});
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

    // async load cache from connector to gpu, for all tp
    std::shared_ptr<AsyncContext> asyncLoadCache(const BatchKVCacheResourcePtr& batch_resource);

private:
    bool initMemoryConnector();

private:
    CacheConfig          config_;
    rtp_llm::DeviceBase* device_;
    KVCacheAllocatorPtr  allocator_;

    const kmonitor::MetricsReporterPtr metrics_reporter_;
    const GptInitParameter&            params_;

    std::shared_ptr<KVCacheConnector>          memory_connector_;
    std::shared_ptr<autil::LockFreeThreadPool> wait_cache_thread_pool_;
};

}  // namespace rtp_llm