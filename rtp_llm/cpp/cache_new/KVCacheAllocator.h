#pragma once

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/cache_new/types.h"

namespace rtp_llm {

class KVCacheAllocator {
public:
    KVCacheAllocator(const CacheConfig& config, rtp_llm::DeviceBase* device, AllocationType atype = AllocationType::DEVICE);
    bool init() {}

    MallocResult malloc(const MallocInfo& malloc_info) {}

    FreeResult free(const FreeInfo& free_info) {
        // only consider the scenario of full fallback.
    }
    InsertResult insertIntoCache(const InsertInfo& insert_info){
        // insert blocks in stream that have been cached in block_cache into block_cache   
    }

    CacheLayerLayout layerCacheBase() const {}

private:
    std::vector<std::shared_ptr<KVCacheGroup>> kv_cache_groups_;
};

}  // namespace rtp_llm
