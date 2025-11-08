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
    bool init();
    const CacheConfig& cacheConfig() const;
    // size_t availableBlockNums() const;
    // size_t freeBlockNums() const;

    size_t availableTokenNums() const;
    
    CacheLayerLayout layerCacheBase() const;

    KVCacheManager(const CacheConfig&      config,
        rtp_llm::DeviceBase*               device,
        bool                               warmup           = false,
        const kmonitor::MetricsReporterPtr metrics_reporter = nullptr,
        const GptInitParameter&            params           = GptInitParameter{});
    ~KVCacheManager();

    size_t freeBlocksNums() const;
    size_t availableBlocksNums() const;
    size_t totalBlocksNums() const;
    size_t maxSeqLen() const;
    KVCacheInfo getKVCacheInfo(int64_t latest_version, bool need_cache_keys) const;
    
    // For backward compatibility with old code
    KVCacheBuffer kvCacheBuffer() const;

    MallocResult malloc(const MallocInfo& malloc_info);
    FreeResult free(const FreeInfo& free_info);
    InsertResult insertIntoCache(const InsertInfo& insert_info);
    
    void blockCopy(int src_block_index, int dest_block_index);
    void blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping);
    void blockBatchCopy(const rtp_llm::Buffer& copy_mapping);
    void blockBatchCopy(const BlockIdPair* copy_mapping_begin, const BlockIdPair* copy_mapping_end);

    BlockAddrInfo convertIndexToAddr(int block_index, int layer_id) const;

    void regUserMr(size_t model_id);

    // Distributed cache methods not implemented yet, delete when refactor ready
    bool getCacheForRank(const std::vector<size_t>&                cache_keys,
                        const std::vector<int32_t>&               block_indices,
                        size_t                                    ignore_block_num,
                        int64_t                                   request_id,
                        const std::map<std::string, std::string>& extra_metas) const;
    
    bool putCacheForRank(const std::vector<size_t>&                cache_keys,
                        const std::vector<int32_t>&               block_indices,
                        size_t                                    ignore_block_num,
                        int64_t                                   request_id,
                        const std::map<std::string, std::string>& extra_metas) const;
    
    std::shared_ptr<class MemoryBlockCache> memoryBlockCache() const;

    // TODO: InsertInfo 考虑  hicache distkvcache 的logic
private:
    CacheConfig config_;
    rtp_llm::DeviceBase* device_;
    KVCacheAllocatorPtr allocator_; 

    const kmonitor::MetricsReporterPtr metrics_reporter_;
    const GptInitParameter& params_;
};

}  // namespace rtp_llm