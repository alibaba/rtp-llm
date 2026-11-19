#pragma once

#include <cstddef>
#include <memory>
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

/** GroupSet membership, local geometry and budgets. HOST/DISK block counts are finalized before pool creation. */
struct BlockTreeCachePlan {
    std::vector<std::vector<std::string>> group_members;
    std::vector<BackingLayout>            backing_layouts;
    BlockTreeCacheSizing                   sizing;
    BlockTreeCacheBlockNums               block_counts;

    BlockTreeCacheBlockNums computeBlockCounts(const CacheConfig& cache_config) const;
};

using BlockTreeCachePlanPtr = std::shared_ptr<BlockTreeCachePlan>;

BlockTreeCachePlanPtr planBlockTreeCacheLayout(const CacheConfig& cache_config, const KVCacheConfig& kv_cache_config);

std::string resolveDiskMountPath(const std::string& disk_paths_csv, int64_t local_world_size, int64_t local_rank);

/** A supplied plan carries final counts for each enabled tier. Callers without a plan use local sizing. */
BlockTreeCachePtr createBlockTreeCache(const CacheConfig&                              cache_config,
                                       const KVCacheConfig&                            kv_cache_config,
                                       const std::shared_ptr<CoordinatorCacheManager>& coordinator_manager,
                                       const ParallelismConfig&                        parallelism_config = {},
                                       std::shared_ptr<StorageBackend>                 storage_backend    = nullptr,
                                       std::shared_ptr<BroadcastManager>               broadcast_manager  = nullptr,
                                       std::shared_ptr<kmonitor::MetricsReporter>      metrics_reporter   = nullptr,
                                       std::shared_ptr<const BlockTreeCachePlan>       plan               = nullptr);

}  // namespace rtp_llm
