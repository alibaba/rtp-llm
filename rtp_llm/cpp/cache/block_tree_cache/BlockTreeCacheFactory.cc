#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

#include <cstdlib>
#include <limits>
#include <string>

#include "rtp_llm/cpp/utils/StringUtil.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {

namespace {

// Create a ComponentGroup based on CacheGroupType.
ComponentGroupPtr
createComponentGroup(int group_id, CacheGroupType type, int seq_size_per_block, const SWAGroupConfig& swa_configs) {
    ComponentGroupPtr group;

    switch (type) {
        case CacheGroupType::FULL: {
            group = std::make_shared<FullComponentGroup>();
            break;
        }
        case CacheGroupType::SWA: {
            int  sliding_window_size = 0;
            auto it                  = swa_configs.find(group_id);
            if (it != swa_configs.end()) {
                sliding_window_size = it->second;
            }
            RTP_LLM_CHECK_WITH_INFO(
                sliding_window_size >= 0, "SWA sliding_window_size must be non-negative, got %d", sliding_window_size);
            RTP_LLM_CHECK_WITH_INFO(
                seq_size_per_block > 0, "SWA seq_size_per_block must be positive, got %d", seq_size_per_block);
            group = std::make_shared<SWAComponentGroup>(sliding_window_size, seq_size_per_block);
            break;
        }
        case CacheGroupType::LINEAR: {
            group = std::make_shared<LinearComponentGroup>();
            break;
        }
        default:
            RTP_LLM_CHECK_WITH_INFO(false, "Unknown CacheGroupType: %d", static_cast<int>(type));
    }

    group->component_group_id = group_id;
    group->group_type         = type;
    return group;
}

// Alignment for host/disk pool block strides (page-aligned for pinned host + O_DIRECT disk).
constexpr size_t kBlockTreeCachePoolAlignment = 4096;

size_t alignUp(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

int checkedTransferTimeoutMs(int64_t timeout_ms, const char* config_name) {
    RTP_LLM_CHECK_WITH_INFO(timeout_ms > 0 && timeout_ms <= std::numeric_limits<int>::max(),
                            "%s must be in range (0, %d], got %ld",
                            config_name,
                            std::numeric_limits<int>::max(),
                            timeout_ms);
    return static_cast<int>(timeout_ms);
}

// Create v4 HostBlockPool for L2 memory cache.
// usable_block_count is the number of blocks usable by the tree (excluding the
// reserved block 0); physical_block_count is usable_block_count + 1.
std::shared_ptr<HostBlockPool> createHostPool(size_t payload_bytes, size_t usable_block_count) {
    if (payload_bytes == 0 || usable_block_count == 0) {
        return nullptr;
    }

    auto config                     = std::make_shared<HostBlockPoolConfig>();
    config->pool_type               = BlockPoolType::HOST;
    config->pool_name               = "block_tree_host";
    config->physical_block_count    = usable_block_count + 1;
    config->payload_bytes           = payload_bytes;
    config->stride_bytes            = alignUp(payload_bytes, kBlockTreeCachePoolAlignment);
    config->enable_pinned           = shouldPinHostBlockPool();
    config->alignment               = kBlockTreeCachePoolAlignment;

    auto pool = std::make_shared<HostBlockPool>(config);
    if (!pool->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize HostBlockPool for BlockTreeCache");
        return nullptr;
    }
    return pool;
}

// Create v4 DiskBlockPool for L3 disk cache.
// physical_block_count is derived inside DiskBlockPool::normalizeConfig from
// disk_size_bytes / stride_bytes, so it is not set here.
std::shared_ptr<DiskBlockPool> createDiskPool(const KVCacheConfig& kv_cache_config,
                                              size_t               payload_bytes,
                                              int64_t              world_rank,
                                              int64_t              local_rank,
                                              int64_t              local_world_size) {
    RTP_LLM_CHECK_WITH_INFO(!kv_cache_config.memory_cache_disk_paths.empty(),
                            "disk cache enabled but memory_cache_disk_paths is empty");
    RTP_LLM_CHECK_WITH_INFO(payload_bytes > 0, "disk cache enabled but payload_bytes is 0");

    const std::string mount_path =
        resolveDiskMountPath(kv_cache_config.memory_cache_disk_paths, local_world_size, local_rank);

    auto config                     = std::make_shared<DiskBlockPoolConfig>();
    config->pool_type               = BlockPoolType::DISK;
    config->pool_name               = "block_tree_disk";
    config->work_dir                = mount_path;
    config->manage_mount            = true;
    config->local_rank              = local_rank;
    config->world_rank              = world_rank;
    config->disk_size_bytes         = static_cast<size_t>(kv_cache_config.memory_cache_disk_size_mb) * 1024UL * 1024UL;
    config->payload_bytes           = payload_bytes;
    config->stride_bytes            = alignUp(payload_bytes, kBlockTreeCachePoolAlignment);
    config->buffered_io             = kv_cache_config.memory_cache_disk_buffered_io;

    auto pool = std::make_shared<DiskBlockPool>(config);
    if (!pool->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize DiskBlockPool for BlockTreeCache");
        return nullptr;
    }
    return pool;
}

}  // namespace

bool shouldPinHostBlockPool() {
    const char* value = std::getenv("RTP_LLM_PIN_HOST_BLOCK_POOL");
    if (value == nullptr) {
        return true;
    }
    const std::string flag(value);
    return flag != "0" && flag != "false" && flag != "FALSE" && flag != "off" && flag != "OFF";
}

size_t computeHostUsableBlockCount(size_t memory_cache_size_bytes, size_t stride_bytes) {
    if (stride_bytes == 0) {
        return 0;
    }
    const size_t total_block_count = memory_cache_size_bytes / stride_bytes;
    return total_block_count > 0 ? total_block_count - 1 : 0;
}

std::string resolveDiskMountPath(const std::string& disk_paths_csv, int64_t local_world_size, int64_t local_rank) {
    const auto paths = split(disk_paths_csv, ',');
    RTP_LLM_CHECK_WITH_INFO(paths.size() == static_cast<size_t>(local_world_size),
                            "disk cache path count must equal local_world_size, paths=%zu, local_world_size=%ld",
                            paths.size(),
                            local_world_size);
    RTP_LLM_CHECK_WITH_INFO(local_rank >= 0 && local_rank < local_world_size,
                            "disk cache invalid local_rank=%ld, local_world_size=%ld",
                            local_rank,
                            local_world_size);
    return paths[static_cast<size_t>(local_rank)];
}

BlockTreeCachePtr createBlockTreeCache(const CacheConfig&                       cache_config,
                                       const KVCacheConfig&                     kv_cache_config,
                                       const std::shared_ptr<KVCacheAllocator>& allocator,
                                       int64_t                                  world_rank,
                                       int64_t                                  local_rank,
                                       int64_t                                  local_world_size,
                                       const SWAGroupConfig&                    swa_configs,
                                       std::shared_ptr<StorageBackend>          storage_backend,
                                       std::shared_ptr<BroadcastManager>        broadcast_manager) {
    const int group_num = cache_config.groupNums();
    RTP_LLM_CHECK_WITH_INFO(group_num > 0, "cache_config must have at least one group");

    // 1. Create BlockTree with proper group_slot_count.
    auto tree = std::make_unique<BlockTree>(group_num);

    // 2. Create ComponentGroup for each group.
    std::vector<ComponentGroupPtr> component_groups;
    component_groups.reserve(static_cast<size_t>(group_num));

    for (int gid = 0; gid < group_num; ++gid) {
        const auto type               = cache_config.typeForGroup(static_cast<size_t>(gid));
        int        seq_size_per_block = static_cast<int>(cache_config.seq_size_per_block);
        // Use group-specific seq_size_per_block if available.
        if (static_cast<size_t>(gid) < cache_config.group_seq_size_per_block.size()
            && cache_config.group_seq_size_per_block[gid] > 0) {
            seq_size_per_block = static_cast<int>(cache_config.group_seq_size_per_block[gid]);
        }

        auto group = createComponentGroup(gid, type, seq_size_per_block, swa_configs);
        component_groups.push_back(std::move(group));
    }

    // 3. Create Components (empty for now; will be populated when CopyEngine needs them).
    std::vector<Component> components;

    // Block payload size (raw KV bytes per block); stride is page-aligned inside the pools.
    const size_t payload_bytes = cache_config.kv_block_size_bytes;

    // 4. Compute Host pool block count and create pool if memory cache enabled.
    std::shared_ptr<HostBlockPool> host_pool = nullptr;
    if (kv_cache_config.enable_memory_cache && kv_cache_config.memory_cache_size_mb > 0 && payload_bytes > 0) {
        // usable_block_count = memory_cache_size_bytes / stride_bytes - 1: the reserved
        // block 0 is counted within the configured budget so backing never exceeds it.
        const size_t stride_bytes = alignUp(payload_bytes, kBlockTreeCachePoolAlignment);
        const size_t memory_cache_size_bytes =
            static_cast<size_t>(kv_cache_config.memory_cache_size_mb) * 1024 * 1024;
        const size_t usable_block_count = computeHostUsableBlockCount(memory_cache_size_bytes, stride_bytes);
        RTP_LLM_CHECK_WITH_INFO(usable_block_count > 0,
                                "L2 memory cache enabled but memory_cache_size_mb=%ld is too small for block "
                                "stride=%zu bytes (need at least 2 blocks including the reserved block 0)",
                                kv_cache_config.memory_cache_size_mb,
                                stride_bytes);
        host_pool = createHostPool(payload_bytes, usable_block_count);
    }

    // 5. Create DiskBlockPool if disk cache enabled.
    std::shared_ptr<DiskBlockPool> disk_pool = nullptr;
    if (kv_cache_config.enable_memory_cache_disk && kv_cache_config.memory_cache_disk_size_mb > 0) {
        disk_pool = createDiskPool(kv_cache_config, payload_bytes, world_rank, local_rank, local_world_size);
    }

    // 6. Set host_pool and disk_pool on each ComponentGroup.
    for (auto& group : component_groups) {
        group->setHostPool(host_pool);
        group->setDiskPool(disk_pool);
    }

    // 6b. Inject allocator-owned device (GPU) block pools into each ComponentGroup.
    // Independent multi-pool allocators expose per-group pools via groupBlockPools(); the
    // single-pool allocators expose one pool via getDeviceBlockPool().
    if (allocator) {
        const auto& group_pools = allocator->groupBlockPools();
        if (!group_pools.empty()) {
            for (auto& group : component_groups) {
                group->setDevicePools(group_pools);
            }
        } else if (auto device_pool = allocator->getDeviceBlockPool()) {
            for (auto& group : component_groups) {
                group->setDevicePools(std::vector<DeviceBlockPoolPtr>{device_pool});
            }
        }
    }

    // 7. Build BlockTreeCacheConfig.
    BlockTreeCacheConfig config;
    config.enable_device_cache = kv_cache_config.enable_device_cache;
    config.enable_memory_cache = kv_cache_config.enable_memory_cache;
    config.enable_disk_cache   = kv_cache_config.enable_memory_cache_disk;
    config.enable_remote_cache = kv_cache_config.enable_remote_cache;
    config.memory_cache_sync_timeout_ms =
        checkedTransferTimeoutMs(kv_cache_config.memory_cache_sync_timeout_ms, "memory_cache_sync_timeout_ms");
    if (config.enable_disk_cache) {
        config.memory_cache_disk_sync_timeout_ms = checkedTransferTimeoutMs(
            kv_cache_config.memory_cache_disk_sync_timeout_ms, "memory_cache_disk_sync_timeout_ms");
    } else {
        config.memory_cache_disk_sync_timeout_ms = config.memory_cache_sync_timeout_ms;
    }

    // 8. Assemble and return BlockTreeCache.
    auto cache = std::make_shared<BlockTreeCache>(std::move(tree),
                                                  std::move(component_groups),
                                                  std::move(components),
                                                  std::move(config),
                                                  storage_backend,
                                                  broadcast_manager);

    RTP_LLM_LOG_INFO("Created BlockTreeCache: groups=%d, host_pool=%s, disk_pool=%s, "
                     "device=%d, memory=%d, disk=%d, remote=%d",
                     group_num,
                     host_pool ? "enabled" : "disabled",
                     disk_pool ? "enabled" : "disabled",
                     kv_cache_config.enable_device_cache,
                     kv_cache_config.enable_memory_cache,
                     kv_cache_config.enable_memory_cache_disk,
                     kv_cache_config.enable_remote_cache);

    return cache;
}

}  // namespace rtp_llm
