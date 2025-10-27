#pragma once

#include <cassert>
#include <mutex>
#include <set>
#include <vector>

#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/MemoryBlockCache.h"
#include "rtp_llm/cpp/cache_new/Notifier.h"

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

    MallocResult malloc(const MallocInfo& malloc_info);
    FreeResult free(const FreeInfo& free_info);
    InsertResult insertIntoCache(const InsertInfo& insert_info); 

    // connector need kvCacheGroups to transfer caches
    std::vector<std::shared_ptr<KVCacheGroup>> kvCacheGroups() const {
        return allocator_->kvCacheGroups();
    }

    // groupInfo groupInfo()
    // TODO: InsertInfo 考虑  hicache distkvcache 的逻辑
private:
    CacheConfig config_;
    KVCacheAllocatorPtr allocator_; 
    std::shared_ptr<MemoryBlockCache> memory_block_cache_;
    std::shared_ptr<KVCacheReaderWriter> kv_cache_reader_writer_;
    std::shared_ptr<Notifier> notifier_;
    // HashUtil hash_util_;
w}