#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/utils/StringUtil.h"

#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceFullKVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceLinearKVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/device_group/DeviceSWAKVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"

namespace rtp_llm {

namespace {

// Create a ComponentGroup based on CacheGroupType.
ComponentGroupPtr
createComponentGroup(int group_id, CacheGroupType type, int seq_size_per_block, int sliding_window_size) {
    ComponentGroupPtr group;

    switch (type) {
        case CacheGroupType::FULL: {
            group = std::make_shared<FullComponentGroup>();
            break;
        }
        case CacheGroupType::SWA: {
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

// Create the single-pool DeviceKVCacheGroup for one per-tag cache group. Mirrors the group
// construction previously done in the allocators' doInit(): derives layer_ids / spec /
// policy from cache_config (indexed by the per-tag gid) and wraps exactly one device pool.
// The per-tag gid is retained as the DeviceKVCacheGroup's own group id because allocators
// address blocks by per-tag gid.
DeviceKVCacheGroupPtr
createDeviceKVCacheGroup(int type, int tag_gid, const CacheConfig& cache_config, DeviceBlockPoolPtr device_pool) {
    const auto ids    = cache_config.layerIdsForGroup(static_cast<size_t>(tag_gid));
    auto       spec   = cache_config.specForGroup(static_cast<size_t>(tag_gid));
    const auto policy = cache_config.policyForGroup(static_cast<size_t>(tag_gid));

    DeviceKVCacheGroupPtr group;
    switch (static_cast<CacheGroupType>(type)) {
        case CacheGroupType::LINEAR:
            group = std::make_shared<DeviceLinearKVCacheGroup>(
                ids, spec, device_pool, tag_gid, cache_config.linear_step, nullptr, policy);
            break;
        case CacheGroupType::SWA:
            group = std::make_shared<DeviceSWAKVCacheGroup>(
                ids, spec, device_pool, tag_gid, cache_config.linear_step, nullptr, policy);
            break;
        case CacheGroupType::FULL:
        default:
            group = std::make_shared<DeviceFullKVCacheGroup>(ids, spec, device_pool, tag_gid, nullptr, policy);
            break;
    }
    RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize DeviceKVCacheGroup (tag_gid %d)", tag_gid);
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
std::shared_ptr<HostBlockPool>
createHostPool(const std::string& pool_name, size_t payload_bytes, size_t usable_block_count) {
    if (payload_bytes == 0 || usable_block_count == 0) {
        return nullptr;
    }

    auto config                  = std::make_shared<HostBlockPoolConfig>();
    config->pool_type            = BlockPoolType::HOST;
    config->pool_name            = pool_name;
    config->physical_block_count = usable_block_count + 1;
    config->payload_bytes        = payload_bytes;
    config->stride_bytes         = alignUp(payload_bytes, kBlockTreeCachePoolAlignment);
    config->enable_pinned        = shouldPinHostBlockPool();
    config->alignment            = kBlockTreeCachePoolAlignment;

    auto pool = std::make_shared<HostBlockPool>(config);
    if (!pool->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize HostBlockPool %s for BlockTreeCache", pool_name.c_str());
        return nullptr;
    }
    return pool;
}

std::shared_ptr<DiskMountGuard> createDiskMountGuard(const KVCacheConfig& kv_cache_config,
                                                     int64_t              local_rank,
                                                     int64_t              local_world_size) {
    RTP_LLM_CHECK_WITH_INFO(!kv_cache_config.memory_cache_disk_paths.empty(),
                            "disk cache enabled but memory_cache_disk_paths is empty");
    const std::string mount_path =
        resolveDiskMountPath(kv_cache_config.memory_cache_disk_paths, local_world_size, local_rank);
    auto guard = std::make_shared<DiskMountGuard>();
    if (!guard->init(mount_path)) {
        RTP_LLM_LOG_ERROR("Failed to init DiskMountGuard on mount [%s] for BlockTreeCache", mount_path.c_str());
        return nullptr;
    }
    return guard;
}

std::shared_ptr<DiskBlockPool> createDiskPool(const KVCacheConfig&                   kv_cache_config,
                                              const std::shared_ptr<DiskMountGuard>& mount_guard,
                                              const std::string&                     pool_name,
                                              size_t                                 payload_bytes,
                                              size_t                                 disk_size_bytes,
                                              int64_t                                world_rank,
                                              int64_t                                local_rank) {
    RTP_LLM_CHECK_WITH_INFO(mount_guard != nullptr, "disk cache enabled but mount guard is null");
    RTP_LLM_CHECK_WITH_INFO(payload_bytes > 0, "disk cache enabled but payload_bytes is 0");

    auto config             = std::make_shared<DiskBlockPoolConfig>();
    config->pool_type       = BlockPoolType::DISK;
    config->pool_name       = pool_name;
    config->work_dir        = mount_guard->workDir();
    config->mount_guard     = mount_guard;
    config->local_rank      = local_rank;
    config->world_rank      = world_rank;
    config->disk_size_bytes = disk_size_bytes;
    config->payload_bytes   = payload_bytes;
    config->stride_bytes    = alignUp(payload_bytes, kBlockTreeCachePoolAlignment);
    config->buffered_io     = kv_cache_config.memory_cache_disk_buffered_io;

    auto pool = std::make_shared<DiskBlockPool>(config);
    if (!pool->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize DiskBlockPool %s for BlockTreeCache", pool_name.c_str());
        return nullptr;
    }
    return pool;
}

// Per-group_type aggregation plan. component_group_id space is per-group_type (compact,
// first-seen order over REUSABLE tags); the per-tag pool/component space stays per-tag.
struct AggregationPlan {
    std::vector<CacheGroupType>   cg_types;            // indexed by component_group_id
    std::vector<std::vector<int>> cg_member_tags;      // component_group_id -> per-tag gids (local pool order)
    std::vector<int>              per_tag_cg;          // per-tag gid -> component_group_id (-1 = NON_REUSABLE)
    std::vector<int>              per_tag_local_pool;  // per-tag gid -> local pool index (-1 = NON_REUSABLE)
};

// Aggregate REUSABLE per-tag groups by group_type. NON_REUSABLE tags (skip_prefix_reuse)
// are excluded from the tree/eviction model but still get their own DeviceKVCacheGroup for
// device allocation (mapped as per_tag_cg = -1). Asserts that all members of one
// group_type share the same group_seq_size_per_block.
AggregationPlan buildAggregationPlan(const CacheConfig& cache_config) {
    const int       group_num = cache_config.groupNums();
    AggregationPlan plan;
    plan.per_tag_cg.assign(static_cast<size_t>(group_num), -1);
    plan.per_tag_local_pool.assign(static_cast<size_t>(group_num), -1);

    auto seqSizeForTag = [&](int gid) -> size_t {
        if (static_cast<size_t>(gid) < cache_config.group_seq_size_per_block.size()
            && cache_config.group_seq_size_per_block[static_cast<size_t>(gid)] > 0) {
            return cache_config.group_seq_size_per_block[static_cast<size_t>(gid)];
        }
        return cache_config.seq_size_per_block;
    };

    std::unordered_map<int, int> type_to_cg;
    for (int gid = 0; gid < group_num; ++gid) {
        const auto policy = cache_config.policyForGroup(static_cast<size_t>(gid));
        if (policy.reuse_policy == CacheReusePolicy::NON_REUSABLE) {
            continue;  // excluded from the aggregated tree model
        }
        const int type_key = static_cast<int>(policy.group_type);
        auto      it       = type_to_cg.find(type_key);
        int       cg_id;
        if (it == type_to_cg.end()) {
            cg_id                = static_cast<int>(plan.cg_types.size());
            type_to_cg[type_key] = cg_id;
            plan.cg_types.push_back(policy.group_type);
            plan.cg_member_tags.emplace_back();
        } else {
            cg_id = it->second;
            RTP_LLM_CHECK_WITH_INFO(
                seqSizeForTag(gid) == seqSizeForTag(plan.cg_member_tags[cg_id].front()),
                "group_type %d aggregates tags with inconsistent group_seq_size_per_block (%zu vs %zu)",
                type_key,
                seqSizeForTag(gid),
                seqSizeForTag(plan.cg_member_tags[cg_id].front()));
        }
        plan.per_tag_cg[static_cast<size_t>(gid)]         = cg_id;
        plan.per_tag_local_pool[static_cast<size_t>(gid)] = static_cast<int>(plan.cg_member_tags[cg_id].size());
        plan.cg_member_tags[cg_id].push_back(gid);
    }
    return plan;
}

bool validateBlockTreeCacheParameters(
    const std::vector<ComponentGroupPtr>&             component_groups,
    const std::vector<DeviceKVCacheGroupPtr>&         per_tag_device_groups,
    const std::vector<BlockTreeCache::PerTagMapping>& per_tag_mapping) {
    if (per_tag_mapping.empty()) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: per_tag_mapping must not be empty");
        return false;
    }
    if (per_tag_mapping.size() != per_tag_device_groups.size()) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: per-tag mapping and device group counts must match, mapping=%zu, "
                          "device_groups=%zu",
                          per_tag_mapping.size(),
                          per_tag_device_groups.size());
        return false;
    }
    for (size_t component_group_index = 0; component_group_index < component_groups.size(); ++component_group_index) {
        const ComponentGroupPtr& component_group = component_groups[component_group_index];
        if (component_group == nullptr
            || component_group->component_group_id != static_cast<int>(component_group_index)) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: component group must be non-null and indexed by id, index=%zu",
                              component_group_index);
            return false;
        }
    }
    for (size_t tag_group_index = 0; tag_group_index < per_tag_mapping.size(); ++tag_group_index) {
        const BlockTreeCache::PerTagMapping& mapping = per_tag_mapping[tag_group_index];
        const bool non_reusable_mapping =
            mapping.component_group_id == -1 && mapping.local_pool_index == -1;
        if (non_reusable_mapping) {
            continue;
        }
        if (mapping.component_group_id < 0
            || static_cast<size_t>(mapping.component_group_id) >= component_groups.size()
            || mapping.local_pool_index < 0
            || static_cast<size_t>(mapping.local_pool_index)
                   >= component_groups[static_cast<size_t>(mapping.component_group_id)]->devicePoolCount()) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: invalid per-tag mapping, tag_group_index=%zu, "
                              "component_group_id=%d, local_pool_index=%d",
                              tag_group_index,
                              mapping.component_group_id,
                              mapping.local_pool_index);
            return false;
        }
    }
    return true;
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
                                       const ParallelismConfig&                 parallelism_config,
                                       std::shared_ptr<StorageBackend>          storage_backend,
                                       std::shared_ptr<BroadcastManager>        broadcast_manager) {
    const int group_num = cache_config.groupNums();
    RTP_LLM_CHECK_WITH_INFO(group_num > 0, "cache_config must have at least one group");
    if (kv_cache_config.enable_memory_cache_disk && !kv_cache_config.enable_memory_cache) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: enable_memory_cache_disk requires enable_memory_cache = true");
        return nullptr;
    }

    // 1. Aggregate per-tag groups by group_type (REUSABLE only). component_group_id is
    // per-group_type; the per-tag pool/component space is preserved for the allocator.
    const AggregationPlan plan   = buildAggregationPlan(cache_config);
    const int             cg_num = static_cast<int>(plan.cg_types.size());
    RTP_LLM_CHECK_WITH_INFO(cg_num > 0, "cache_config produced no reusable component groups");

    // 2. Create BlockTree sized by the aggregated component group count.
    auto tree = std::make_unique<BlockTree>(cg_num);

    // 3. Create one ComponentGroup per aggregated group_type.
    std::vector<ComponentGroupPtr> component_groups;
    component_groups.reserve(static_cast<size_t>(cg_num));
    for (int cg_id = 0; cg_id < cg_num; ++cg_id) {
        const auto type               = plan.cg_types[static_cast<size_t>(cg_id)];
        const int  first_tag          = plan.cg_member_tags[static_cast<size_t>(cg_id)].front();
        int        seq_size_per_block = static_cast<int>(cache_config.seq_size_per_block);
        if (static_cast<size_t>(first_tag) < cache_config.group_seq_size_per_block.size()
            && cache_config.group_seq_size_per_block[static_cast<size_t>(first_tag)] > 0) {
            seq_size_per_block =
                static_cast<int>(cache_config.group_seq_size_per_block[static_cast<size_t>(first_tag)]);
        }
        const int sliding_window_size =
            (type == CacheGroupType::SWA) ? cache_config.slidingWindowForGroup(static_cast<size_t>(first_tag)) : 0;
        component_groups.push_back(createComponentGroup(cg_id, type, seq_size_per_block, sliding_window_size));
    }

    // 4. Resolve the per-tag device pools from the allocator.
    std::vector<DeviceBlockPoolPtr> per_tag_pools(static_cast<size_t>(group_num));
    if (allocator) {
        const auto& group_pools = allocator->groupBlockPools();
        for (int gid = 0; gid < group_num; ++gid) {
            if (!group_pools.empty()) {
                per_tag_pools[static_cast<size_t>(gid)] =
                    static_cast<size_t>(gid) < group_pools.size() ? group_pools[static_cast<size_t>(gid)] : nullptr;
            } else {
                per_tag_pools[static_cast<size_t>(gid)] = allocator->getDeviceBlockPool();
            }
        }
    }

    // 5. Build the global Component vector, per-tag DeviceKVCacheGroup registry, per-tag
    // mapping, and inject device pools / kv groups / component_indices / host_block_size
    // into each aggregated ComponentGroup.
    std::vector<Component>                     components;
    std::vector<DeviceKVCacheGroupPtr>         per_tag_device_groups(static_cast<size_t>(group_num));
    std::vector<BlockTreeCache::PerTagMapping> per_tag_mapping(static_cast<size_t>(group_num));
    std::vector<size_t>                        cg_host_block_size(static_cast<size_t>(cg_num), 0);

    for (int cg_id = 0; cg_id < cg_num; ++cg_id) {
        const auto&                     member_tags = plan.cg_member_tags[static_cast<size_t>(cg_id)];
        const auto                      type        = plan.cg_types[static_cast<size_t>(cg_id)];
        std::vector<DeviceBlockPoolPtr> cg_pools;
        std::vector<int>                comp_indices;
        size_t                          host_block_size = 0;

        for (size_t local = 0; local < member_tags.size(); ++local) {
            const int  tag_gid = member_tags[local];
            const auto pool    = per_tag_pools[static_cast<size_t>(tag_gid)];
            cg_pools.push_back(pool);

            // Component descriptor (one per REUSABLE tag): host packing layout is one slot
            // per layer, each slot sized by that tag's per-layer device stride.
            Component comp;
            comp.component_id       = static_cast<int>(components.size());
            comp.component_group_id = cg_id;
            comp.type               = type;
            comp.device_pool_index  = static_cast<int>(local);
            const size_t stride     = cache_config.kvBlockStrideBytesForGroup(static_cast<size_t>(tag_gid))
                                  + cache_config.kvScaleStrideBytesForGroup(static_cast<size_t>(tag_gid));
            const auto& tag = cache_config.tagForGroup(static_cast<size_t>(tag_gid));
            for (int layer_id : cache_config.layerIdsForGroup(static_cast<size_t>(tag_gid))) {
                comp.memory_block_layer_tag_slots.push_back(MemoryBlockLayerTagSlot{layer_id, tag, stride});
                host_block_size += stride;
            }
            comp_indices.push_back(comp.component_id);
            components.push_back(std::move(comp));

            // Per-tag DeviceKVCacheGroup (wraps this pool; keeps the per-tag gid as its id).
            DeviceKVCacheGroupPtr dkv =
                pool ? createDeviceKVCacheGroup(static_cast<int>(type), tag_gid, cache_config, pool) : nullptr;
            per_tag_device_groups[static_cast<size_t>(tag_gid)] = dkv;
            per_tag_mapping[static_cast<size_t>(tag_gid)]       = {cg_id, static_cast<int>(local)};
        }

        component_groups[static_cast<size_t>(cg_id)]->setDevicePools(cg_pools);
        component_groups[static_cast<size_t>(cg_id)]->component_indices = comp_indices;
        component_groups[static_cast<size_t>(cg_id)]->host_block_size   = host_block_size;
        cg_host_block_size[static_cast<size_t>(cg_id)]                  = host_block_size;
    }

    // 5b. NON_REUSABLE tags are not part of any ComponentGroup but still need a
    // DeviceKVCacheGroup so the allocator can allocate/query them by per-tag gid.
    for (int gid = 0; gid < group_num; ++gid) {
        if (plan.per_tag_cg[static_cast<size_t>(gid)] >= 0) {
            continue;
        }
        const auto pool = per_tag_pools[static_cast<size_t>(gid)];
        if (pool) {
            const auto type = cache_config.typeForGroup(static_cast<size_t>(gid));
            per_tag_device_groups[static_cast<size_t>(gid)] =
                createDeviceKVCacheGroup(static_cast<int>(type), gid, cache_config, pool);
        }
        per_tag_mapping[static_cast<size_t>(gid)] = {-1, -1};
    }

    if (!validateBlockTreeCacheParameters(component_groups, per_tag_device_groups, per_tag_mapping)) {
        return nullptr;
    }

    // 6. Create per-ComponentGroup Host pools (L2). All groups share a unified usable block
    // count N so a host-resident tree node occupies one block in every group's pool:
    // N = memory_cache_size_bytes / Sum(alignUp(host_block_size_g)) - 1 (reserved block 0).
    if (kv_cache_config.enable_memory_cache && kv_cache_config.memory_cache_size_mb > 0) {
        size_t sum_aligned = 0;
        for (int cg_id = 0; cg_id < cg_num; ++cg_id) {
            sum_aligned += alignUp(cg_host_block_size[static_cast<size_t>(cg_id)], kBlockTreeCachePoolAlignment);
        }
        const size_t memory_cache_size_bytes = static_cast<size_t>(kv_cache_config.memory_cache_size_mb) * 1024 * 1024;
        const size_t usable_block_count      = computeHostUsableBlockCount(memory_cache_size_bytes, sum_aligned);
        RTP_LLM_CHECK_WITH_INFO(usable_block_count > 0,
                                "L2 memory cache enabled but memory_cache_size_mb=%ld is too small for the "
                                "aggregated host block stride sum=%zu bytes (need at least 2 blocks including the "
                                "reserved block 0)",
                                kv_cache_config.memory_cache_size_mb,
                                sum_aligned);
        for (int cg_id = 0; cg_id < cg_num; ++cg_id) {
            const size_t payload = cg_host_block_size[static_cast<size_t>(cg_id)];
            auto host_pool = createHostPool("block_tree_host_g" + std::to_string(cg_id), payload, usable_block_count);
            component_groups[static_cast<size_t>(cg_id)]->setHostPool(host_pool);
        }
    }

    // 7. Create per-ComponentGroup Disk pools (L3) with the same unified usable count.
    if (kv_cache_config.enable_memory_cache_disk && kv_cache_config.memory_cache_disk_size_mb > 0) {
        size_t sum_aligned = 0;
        for (int cg_id = 0; cg_id < cg_num; ++cg_id) {
            sum_aligned += alignUp(cg_host_block_size[static_cast<size_t>(cg_id)], kBlockTreeCachePoolAlignment);
        }
        const size_t disk_size_bytes = static_cast<size_t>(kv_cache_config.memory_cache_disk_size_mb) * 1024UL * 1024UL;
        const size_t usable_block_count =
            sum_aligned > 0 ? computeHostUsableBlockCount(disk_size_bytes, sum_aligned) : 0;
        auto disk_mount_guard = createDiskMountGuard(
            kv_cache_config, parallelism_config.local_rank, parallelism_config.local_world_size);
        if (disk_mount_guard == nullptr) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: failed to create disk mount guard");
            return nullptr;
        }
        for (int cg_id = 0; cg_id < cg_num; ++cg_id) {
            const size_t payload = cg_host_block_size[static_cast<size_t>(cg_id)];
            if (payload == 0 || usable_block_count == 0) {
                continue;
            }
            // Bound each pool to N usable blocks (+1 reserved) at its own aligned stride.
            const size_t pool_disk_bytes = (usable_block_count + 1) * alignUp(payload, kBlockTreeCachePoolAlignment);
            auto         disk_pool       = createDiskPool(kv_cache_config,
                                            disk_mount_guard,
                                            "block_tree_disk_g" + std::to_string(cg_id),
                                            payload,
                                            pool_disk_bytes,
                                            parallelism_config.world_rank,
                                            parallelism_config.local_rank);
            component_groups[static_cast<size_t>(cg_id)]->setDiskPool(disk_pool);
        }
    }

    // 8. Build BlockTreeCacheConfig.
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

    // 9. Assemble and return BlockTreeCache with the per-tag <-> component_group mapping.
    auto cache = std::make_shared<BlockTreeCache>(std::move(tree),
                                                  std::move(component_groups),
                                                  std::move(components),
                                                  std::move(config),
                                                  storage_backend,
                                                  broadcast_manager,
                                                  std::move(per_tag_device_groups),
                                                  std::move(per_tag_mapping));
    if (!cache->init()) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: failed to initialize BlockTreeCache");
        return nullptr;
    }
    RTP_LLM_LOG_INFO("Created BlockTreeCache: per_tag_groups=%d, component_groups=%d, "
                     "device=%d, memory=%d, disk=%d, remote=%d",
                     group_num,
                     cg_num,
                     kv_cache_config.enable_device_cache,
                     kv_cache_config.enable_memory_cache,
                     kv_cache_config.enable_memory_cache_disk,
                     kv_cache_config.enable_remote_cache);

    return cache;
}

}  // namespace rtp_llm
