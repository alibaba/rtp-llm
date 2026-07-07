#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/copy_engine/DiskBlockPool.h"
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

// Create Host BlockPool for L2 memory cache.
BlockPoolPtr createHostPool(const CacheConfig& cache_config, size_t host_block_count) {
    if (host_block_count == 0) {
        return nullptr;
    }

    // Use BlockPoolConfigHelper to create a proper host pool config.
    // The host pool mirrors the device pool layout but uses HOST allocation.
    auto config      = BlockPoolConfigHelper::createConfig(cache_config);
    config.pool_name = "block_tree_cache_host";
    config.block_num = static_cast<uint32_t>(host_block_count);

    // Scale pool sizes proportionally to block count ratio.
    const double ratio = static_cast<double>(host_block_count) / static_cast<double>(cache_config.block_num);
    for (auto& layout : config.memory_layouts) {
        layout.block_num                = static_cast<uint32_t>(host_block_count);
        layout.kv_block_pool_size_bytes = static_cast<size_t>(layout.kv_block_pool_size_bytes * ratio);
        layout.kv_scale_pool_size_bytes = static_cast<size_t>(layout.kv_scale_pool_size_bytes * ratio);
        layout.total_size_bytes         = layout.kv_block_pool_size_bytes + layout.kv_scale_pool_size_bytes;
    }
    config.total_size_bytes = 0;
    for (const auto& layout : config.memory_layouts) {
        config.total_size_bytes += layout.total_size_bytes;
    }

    auto pool = std::make_shared<BlockPool>(config, AllocationType::HOST);
    if (!pool->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize host BlockPool for BlockTreeCache");
        return nullptr;
    }
    return pool;
}

// Create DiskBlockPool for L3 disk cache.
std::shared_ptr<DiskBlockPool> createDiskPool(const KVCacheConfig& kv_cache_config, size_t block_size_bytes) {
    if (kv_cache_config.memory_cache_disk_size_mb <= 0 || kv_cache_config.memory_cache_disk_paths.empty()) {
        return nullptr;
    }

    DiskBlockPoolConfig disk_config;
    disk_config.work_dir         = kv_cache_config.memory_cache_disk_paths;
    disk_config.disk_size_bytes  = static_cast<size_t>(kv_cache_config.memory_cache_disk_size_mb) * 1024 * 1024;
    disk_config.block_size_bytes = block_size_bytes;
    disk_config.buffered_io      = kv_cache_config.memory_cache_disk_buffered_io;

    auto pool = std::make_shared<DiskBlockPool>(std::move(disk_config));
    if (!pool->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize DiskBlockPool for BlockTreeCache");
        return nullptr;
    }
    return pool;
}

}  // namespace

BlockTreeCachePtr createBlockTreeCache(const CacheConfig&                       cache_config,
                                       const KVCacheConfig&                     kv_cache_config,
                                       const std::shared_ptr<KVCacheAllocator>& allocator,
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

    // 4. Compute Host pool block count and create pool if memory cache enabled.
    BlockPoolPtr host_pool = nullptr;
    if (kv_cache_config.enable_memory_cache && kv_cache_config.memory_cache_size_mb > 0) {
        // Compute host block count: memory_cache_size_mb * 1MB / block_size_bytes.
        const size_t block_size_bytes = cache_config.kv_block_size_bytes;
        if (block_size_bytes > 0) {
            const size_t host_block_count =
                static_cast<size_t>(kv_cache_config.memory_cache_size_mb) * 1024 * 1024 / block_size_bytes;
            host_pool = createHostPool(cache_config, host_block_count);
        }
    }

    // 5. Create DiskBlockPool if disk cache enabled.
    std::shared_ptr<DiskBlockPool> disk_pool = nullptr;
    if (kv_cache_config.enable_memory_cache_disk && kv_cache_config.memory_cache_disk_size_mb > 0) {
        disk_pool = createDiskPool(kv_cache_config, cache_config.kv_block_size_bytes);
    }

    // 6. Set host_pool and disk_pool on each ComponentGroup.
    for (auto& group : component_groups) {
        group->setHostPool(host_pool);
        group->setDiskPool(disk_pool);
    }

    // 7. Determine eviction thread pool size (default = 2).
    const int eviction_thread_pool_size = 2;

    // 8. Assemble and return BlockTreeCache.
    auto cache = std::make_shared<BlockTreeCache>(std::move(tree),
                                                  std::move(component_groups),
                                                  std::move(components),
                                                  eviction_thread_pool_size,
                                                  storage_backend,
                                                  kv_cache_config.enable_device_cache,
                                                  kv_cache_config.enable_memory_cache,
                                                  kv_cache_config.enable_memory_cache_disk,
                                                  kv_cache_config.enable_remote_cache,
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
