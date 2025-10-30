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
    // size_t availableBlockNums() const;
    // size_t freeBlockNums() const;

    size_t availableTokenNums() const;

    CacheLayerLayout layerCacheBase() const;

    KVCacheManager(const CacheConfig&                 config,
                   rtp_llm::DeviceBase*               device,
                   bool                               warmup           = false,
                   const kmonitor::MetricsReporterPtr metrics_reporter = nullptr,
                   const GptInitParameter&            params           = GptInitParameter{});
    ~KVCacheManager();

    MallocResult malloc(const MallocInfo& malloc_info);
    FreeResult   free(const FreeInfo& free_info);
    InsertResult insertIntoCache(const InsertInfo& insert_info);

    // connector need kvCacheGroups to transfer caches
    std::vector<std::shared_ptr<KVCacheGroup>> kvCacheGroups() const {
        return allocator_->kvCacheGroups();
    }

    // groupInfo groupInfo()
    // TODO: InsertInfo 考虑  hicache distkvcache 的逻辑

private:
    InsertResult insertIntoMemoryCache(const InsertInfo& insert_info);

private:
    CacheConfig config_;
    rtp_llm::DeviceBase* device_;
    KVCacheAllocatorPtr allocator_; 

    const kmonitor::MetricsReporterPtr metrics_reporter_;
    const GptInitParameter& params_;
};

    HashUtil hash_util_;

    std::shared_ptr<KVCacheReaderWriter> mem_reader_writer_;
    std::shared_ptr<KVCacheReaderWriter> remote_reader_writer_;
};
