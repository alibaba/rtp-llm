#pragma once

#include <memory>
#include <unordered_map>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/allocator/KVCacheAllocator.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

class BroadcastManager;
class StorageBackend;

// Optional per-group SWA configuration.
// Key: group_id, Value: sliding_window_size (in tokens).
// Groups not present in this map use default (0 = no window).
using SWAGroupConfig = std::unordered_map<int, int>;

// Reads RTP_LLM_PIN_HOST_BLOCK_POOL to decide whether the L2 host block pool uses pinned
// memory. Returns true when unset; returns false for "0"/"false"/"FALSE"/"off"/"OFF".
bool shouldPinHostBlockPool();

// Number of tree-usable host blocks for a given L2 memory-cache budget and page-aligned
// stride. The reserved block 0 is counted *within* the budget, so this returns
// max(0, memory_cache_size_bytes / stride_bytes - 1). Returns 0 when the budget cannot
// hold at least two blocks (one reserved + one usable) or when stride_bytes is 0.
size_t computeHostUsableBlockCount(size_t memory_cache_size_bytes, size_t stride_bytes);

// Factory function: create a BlockTreeCache from existing CacheConfig + KVCacheConfig.
//
// This function:
// 1. Creates a BlockTree with group_slot_count = cache_config.groupNums()
// 2. Creates a ComponentGroup for each group based on CacheGroupType (FULL/SWA/LINEAR)
// 3. Optionally creates Host BlockPool if kv_cache_config.enable_memory_cache
// 4. Optionally creates DiskBlockPool if kv_cache_config.enable_memory_cache_disk
// 5. Assembles and returns a BlockTreeCache
//
// Parameters:
// - cache_config: cache topology (groups, layers, block sizes)
// - kv_cache_config: tier enable flags and sizing
// - allocator: existing KVCacheAllocator (used to derive device pool info)
// - world_rank / local_rank: used to build per-rank unique disk pool file paths
// - swa_configs: optional sliding_window_size per SWA group
// - storage_backend: optional remote storage backend
// - broadcast_manager: optional multi-node broadcast manager
BlockTreeCachePtr createBlockTreeCache(const CacheConfig&                       cache_config,
                                       const KVCacheConfig&                     kv_cache_config,
                                       const std::shared_ptr<KVCacheAllocator>& allocator,
                                       int64_t                                  world_rank        = 0,
                                       int64_t                                  local_rank        = 0,
                                       const SWAGroupConfig&                    swa_configs       = {},
                                       std::shared_ptr<StorageBackend>          storage_backend   = nullptr,
                                       std::shared_ptr<BroadcastManager>        broadcast_manager = nullptr);

}  // namespace rtp_llm
