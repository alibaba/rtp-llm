#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/types.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"

namespace rtp_llm {



class KVCacheAllocator {
public:
    KVCacheAllocator(const CacheConfig& config, rtp_llm::DeviceBase* device, AllocationType atype = AllocationType::DEVICE)
        : config_(config), device_(device), atype_(atype) {}
    
    virtual ~KVCacheAllocator() = default;

    virtual bool init() = 0;
    virtual MallocResult malloc(const MallocInfo& malloc_info) = 0;
    virtual FreeResult free(const FreeInfo& free_info) = 0;
    virtual InsertResult insertIntoCache(const InsertInfo& insert_info) = 0;
    virtual CacheLayerLayout layerCacheBase() const = 0;

protected:
    CacheConfig config_;
    rtp_llm::DeviceBase* device_;
    AllocationType atype_;
    std::vector<std::shared_ptr<KVCacheGroup>> kv_cache_groups_;
};

using KVCacheAllocatorPtr = std::shared_ptr<KVCacheAllocator>;

}  // namespace rtp_llm
