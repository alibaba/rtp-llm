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

// Reads RTP_LLM_PIN_HOST_BLOCK_POOL to decide whether the L2 host block pool uses pinned
// memory. Returns true when unset; returns false for "0"/"false"/"FALSE"/"off"/"OFF".
bool shouldPinHostBlockPool();

// Number of tree-usable host blocks for a given L2 memory-cache budget and page-aligned
// stride. The reserved block 0 is counted *within* the budget, so this returns
// max(0, memory_cache_size_bytes / stride_bytes - 1). Returns 0 when the budget cannot
// hold at least two blocks (one reserved + one usable) or when stride_bytes is 0.
size_t computeHostUsableBlockCount(size_t memory_cache_size_bytes, size_t stride_bytes);

// Selects paths[local_rank] from a comma-separated list whose entry count must equal
// local_world_size. Aborts via RTP_LLM_CHECK_WITH_INFO on count/range mismatch.
std::string resolveDiskMountPath(const std::string& disk_paths_csv, int64_t local_world_size, int64_t local_rank);

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
// - parallelism_config: rank info (world_rank / local_rank / local_world_size) used to
//   select this rank's disk mount path and build per-rank unique backing files; SWA
//   sliding_window_size is now model-derived and carried per group in CacheGroupPolicy
// - storage_backend: optional remote storage backend
// - broadcast_manager: optional multi-node broadcast manager
BlockTreeCachePtr createBlockTreeCache(const CacheConfig&                       cache_config,
                                       const KVCacheConfig&                     kv_cache_config,
                                       const std::shared_ptr<KVCacheAllocator>& allocator,
                                       const ParallelismConfig&                 parallelism_config = {},
                                       std::shared_ptr<StorageBackend>          storage_backend    = nullptr,
                                       std::shared_ptr<BroadcastManager>        broadcast_manager  = nullptr);

}  // namespace rtp_llm
