#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class BroadcastManager;
class StorageBackend;

size_t computeHostUsableBlockCount(size_t capacity_bytes, size_t stride_bytes);

std::string resolveDiskMountPath(const std::string& disk_paths_csv, int64_t local_world_size, int64_t local_rank);

BlockTreeCachePtr createBlockTreeCache(const CacheDomainConfig&                        cache_config,
                                       const KVCacheConfig&                            kv_cache_config,
                                       const std::shared_ptr<CoordinatorCacheManager>& coordinator_manager,
                                       const ParallelismConfig&                        parallelism_config = {},
                                       std::shared_ptr<StorageBackend>                 storage_backend    = nullptr,
                                       std::shared_ptr<BroadcastManager>               broadcast_manager  = nullptr,
                                       std::shared_ptr<kmonitor::MetricsReporter>      metrics_reporter   = nullptr);

inline BlockTreeCachePtr createBlockTreeCache(const CacheConfig&                              config,
                                              const KVCacheConfig&                            kv_config,
                                              const std::shared_ptr<CoordinatorCacheManager>& coordinator,
                                              const ParallelismConfig&                        parallelism = {},
                                              std::shared_ptr<StorageBackend>                 storage     = nullptr,
                                              std::shared_ptr<BroadcastManager>               broadcast   = nullptr,
                                              std::shared_ptr<kmonitor::MetricsReporter>      metrics     = nullptr) {
    return createBlockTreeCache(CacheDomainConfig(config),
                                kv_config,
                                coordinator,
                                parallelism,
                                std::move(storage),
                                std::move(broadcast),
                                std::move(metrics));
}

}  // namespace rtp_llm
