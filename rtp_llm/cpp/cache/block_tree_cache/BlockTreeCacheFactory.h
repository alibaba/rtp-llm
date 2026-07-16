#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/allocator/KVCacheAllocator.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class BroadcastManager;
class StorageBackend;

bool shouldPinHostBlockPool();

size_t computeHostUsableBlockCount(size_t memory_cache_size_bytes, size_t stride_bytes);

std::string resolveDiskMountPath(const std::string& disk_paths_csv, int64_t local_world_size, int64_t local_rank);

BlockTreeCachePtr createBlockTreeCache(const CacheConfig&                       cache_config,
                                       const KVCacheConfig&                     kv_cache_config,
                                       const std::shared_ptr<KVCacheAllocator>& allocator,
                                       const ParallelismConfig&                 parallelism_config = {},
                                       std::shared_ptr<StorageBackend>          storage_backend    = nullptr,
                                       std::shared_ptr<BroadcastManager>        broadcast_manager  = nullptr);

}  // namespace rtp_llm
