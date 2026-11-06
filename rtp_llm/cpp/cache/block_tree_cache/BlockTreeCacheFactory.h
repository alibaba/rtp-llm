#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheCapacityNegotiator.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"
#include "rtp_llm/cpp/cache/CoordinatorCacheManager.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class BroadcastManager;
class StorageBackend;

/** Group-set packing plan, built once before PP negotiation and then consumed read-only by the factory. */
struct BlockTreeCacheLayout {
    std::vector<std::vector<std::string>> group_members;
    std::vector<BackingLayout>            backing_layouts;
    BlockTreeCacheSizing                   sizing;
    std::optional<BlockTreeCacheBlockNums> block_nums;
};

using BlockTreeCacheLayoutPtr = std::shared_ptr<BlockTreeCacheLayout>;

BlockTreeCacheLayoutPtr planBlockTreeCacheLayout(const CacheConfig& cache_config, const KVCacheConfig& kv_cache_config);

std::string resolveDiskMountPath(const std::string& disk_paths_csv, int64_t local_world_size, int64_t local_rank);

BlockTreeCachePtr createBlockTreeCache(const CacheConfig&                              cache_config,
                                       const KVCacheConfig&                            kv_cache_config,
                                       const std::shared_ptr<CoordinatorCacheManager>& coordinator_manager,
                                       const ParallelismConfig&                        parallelism_config = {},
                                       std::shared_ptr<StorageBackend>                 storage_backend    = nullptr,
                                       std::shared_ptr<BroadcastManager>               broadcast_manager  = nullptr,
                                       std::shared_ptr<kmonitor::MetricsReporter>      metrics_reporter   = nullptr,
                                       std::shared_ptr<const BlockTreeCacheLayout>     layout             = nullptr);

}  // namespace rtp_llm
