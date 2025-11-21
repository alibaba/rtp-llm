#pragma once

#include <cassert>
#include <mutex>
#include <set>
#include <vector>

#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class KVCacheManager {
public:
    bool               init();
    const CacheConfig& cacheConfig() const;
    // size_t availableBlocksNum() const;
    // size_t freeBlocksNum() const;

    size_t availableTokenNums() const;

    CacheLayerLayout layerCacheBase() const;

    KVCacheManager(const CacheConfig&                 config,
                   rtp_llm::DeviceBase*               device,
                   bool                               warmup           = false,
                   const kmonitor::MetricsReporterPtr metrics_reporter = nullptr,
                   const GptInitParameter&            params           = GptInitParameter{});
    ~KVCacheManager();

    size_t      freeBlocksNum() const;
    size_t      availableBlocksNum() const;
    size_t      totalBlocksNum() const;
    size_t      maxSeqLen() const;
    KVCacheInfo getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const;

    // For backward compatibility with old code
    KVCacheBuffer kvCacheBuffer() const;

    // Write one KV block (optionally per-layer) from host/device buffers for test
    virtual void setKVBlockValue(int block_index, int layer_id, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer);
    virtual void setKVBlockValue(int block_index, rtp_llm::Buffer& k_buffer, rtp_llm::Buffer& v_buffer);

    MallocResult malloc(const MallocInfo& malloc_info);
    FreeResult   free(const FreeInfo& free_info);
    InsertResult insertIntoCache(const InsertInfo& insert_info);

    void blockCopy(int src_block_index, int dest_block_index);
    void blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping);
    void blockBatchCopy(const rtp_llm::Buffer& copy_mapping);
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end);

    BlockAddrInfo convertIndexToAddr(int block_index, int layer_id) const;

    void regUserMr(size_t model_id);

    // Distributed cache methods not implemented yet, delete when refactor ready
    bool getCacheForRank(const CacheKeysType&                      cache_keys,
                         const BlockIndicesType&                   block_indices,
                         size_t                                    ignore_block_num,
                         int64_t                                   request_id,
                         const std::map<std::string, std::string>& extra_metas) const;

    bool putCacheForRank(const CacheKeysType&                      cache_keys,
                         const BlockIndicesType&                   block_indices,
                         size_t                                    ignore_block_num,
                         int64_t                                   request_id,
                         const std::map<std::string, std::string>& extra_metas) const;

    bool                                    updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                          const std::vector<int>&        block_src_batch,
                                                          bool                           copy_last_block,
                                                          std::vector<BlockIdPair>&      block_update_mapping);
    std::shared_ptr<class MemoryBlockCache> memoryBlockCache() const;

private:
    CacheConfig          config_;
    rtp_llm::DeviceBase* device_;
    KVCacheAllocatorPtr  allocator_;

    const kmonitor::MetricsReporterPtr metrics_reporter_;
    const GptInitParameter&            params_;
};

}  // namespace rtp_llm