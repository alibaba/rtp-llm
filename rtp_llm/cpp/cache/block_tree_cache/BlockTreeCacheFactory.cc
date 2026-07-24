#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

#include <cstdlib>
#include <algorithm>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockCacheTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LinearComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAComponentGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/host/HostBlockPool.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {
namespace {

constexpr size_t kPoolAlignment = 4096;

constexpr double kDefaultDeviceWatermarkRatio = 0.9;
constexpr double kDefaultHostWatermarkRatio   = 0.9;
constexpr double kDefaultDiskWatermarkRatio   = 0.9;

size_t alignUp(size_t value, size_t alignment) {
    RTP_LLM_CHECK_WITH_INFO(alignment > 0 && value <= std::numeric_limits<size_t>::max() - (alignment - 1),
                            "BlockTreeCache pool stride overflow: value=%zu alignment=%zu",
                            value,
                            alignment);
    return ((value + alignment - 1) / alignment) * alignment;
}

int checkedTimeout(int64_t timeout_ms, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(timeout_ms > 0 && timeout_ms <= std::numeric_limits<int>::max(),
                            "%s must be in range (0, %d], got %ld",
                            name,
                            std::numeric_limits<int>::max(),
                            timeout_ms);
    return static_cast<int>(timeout_ms);
}

int slidingWindowSize(const GroupBase& group) {
    RTP_LLM_CHECK_WITH_INFO(group.policy.group_type == CacheGroupType::SWA,
                            "sliding window requested for non-SWA tag %s",
                            group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "SWA tag %s has null cache spec", group.tag.c_str());
    RTP_LLM_CHECK_WITH_INFO(group.policy.sliding_window_size >= 0,
                            "SWA tag %s has invalid sliding window=%d",
                            group.tag.c_str(),
                            group.policy.sliding_window_size);
    return group.policy.sliding_window_size;
}

ComponentGroupPtr createComponentGroup(int id, const GroupBase& group) {
    ComponentGroupPtr result;
    switch (group.policy.group_type) {
        case CacheGroupType::FULL:
            result = std::make_shared<FullComponentGroup>();
            break;
        case CacheGroupType::LINEAR:
            result = std::make_shared<LinearComponentGroup>();
            break;
        case CacheGroupType::SWA: {
            const auto seq_size = group.seq_size_per_block;
            RTP_LLM_CHECK_WITH_INFO(seq_size > 0 && seq_size <= static_cast<size_t>(std::numeric_limits<int>::max()),
                                    "SWA tag %s has invalid seq_size_per_block=%zu",
                                    group.tag.c_str(),
                                    seq_size);
            result = std::make_shared<SWAComponentGroup>(slidingWindowSize(group), static_cast<int>(seq_size));
            break;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(result != nullptr, "unsupported cache group type for tag %s", group.tag.c_str());
    result->component_group_id = id;
    result->group_type         = group.policy.group_type;
    result->evict_policy       = group.policy.evict_policy;
    return result;
}

std::vector<KVCacheGroupPtr> alignAllocatorGroups(const CacheConfig&         cache_config,
                                                  const KVCacheAllocatorPtr& allocator) {
    if (!allocator) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: allocator is null");
        return {};
    }
    const auto allocator_groups = allocator->cacheGroups();
    const auto group_count      = static_cast<size_t>(cache_config.groupNums());
    if (allocator_groups.size() != group_count) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: allocator/topology group count mismatch, allocator=%zu topology=%zu",
                          allocator_groups.size(),
                          group_count);
        return {};
    }

    std::unordered_map<std::string, KVCacheGroupPtr> by_tag;
    by_tag.reserve(allocator_groups.size());
    for (const auto& group : allocator_groups) {
        if (!group || group->tag().empty() || !group->blockPool()) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: allocator group/tag/direct pool must be non-null");
            return {};
        }
        if (!by_tag.emplace(group->tag(), group).second) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: duplicate allocator group tag %s", group->tag().c_str());
            return {};
        }
    }

    std::vector<KVCacheGroupPtr> aligned;
    aligned.reserve(group_count);
    std::unordered_set<std::string> topology_tags;
    topology_tags.reserve(group_count);
    for (const auto& declared : cache_config.topology().groups()) {
        if (declared.tag.empty() || !topology_tags.emplace(declared.tag).second) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: topology contains an empty or duplicate stable tag");
            return {};
        }
        const auto it = by_tag.find(declared.tag);
        if (it == by_tag.end()) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: allocator is missing topology tag %s", declared.tag.c_str());
            return {};
        }
        const auto& actual = it->second->config();
        if (actual.tag != declared.tag || actual.spec != declared.spec
            || !CacheConfig::samePolicy(actual.policy, declared.policy) || actual.layer_ids != declared.layer_ids
            || actual.block_num != declared.block_num || actual.local_kv_head_num != declared.local_kv_head_num
            || actual.seq_size_per_block != declared.seq_size_per_block
            || actual.kernel_seq_size_per_block != declared.kernel_seq_size_per_block
            || actual.kv_block_stride_bytes != declared.kv_block_stride_bytes
            || actual.kv_scale_stride_bytes != declared.kv_scale_stride_bytes) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: allocator group does not exactly match topology tag %s",
                              declared.tag.c_str());
            return {};
        }
        aligned.push_back(it->second);
    }
    if (topology_tags.size() != by_tag.size()) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: allocator contains an unknown stable tag");
        return {};
    }
    return aligned;
}

std::shared_ptr<HostBlockPool> createHostPool(const std::string& name, size_t payload_bytes, size_t usable_blocks) {
    if (payload_bytes == 0 || usable_blocks == 0) {
        return nullptr;
    }
    auto config                  = std::make_shared<HostBlockPoolConfig>();
    config->pool_type            = BlockPoolType::HOST;
    config->pool_name            = name;
    config->physical_block_count = usable_blocks + 1;
    config->payload_bytes        = payload_bytes;
    config->stride_bytes         = alignUp(payload_bytes, kPoolAlignment);
    config->enable_pinned        = shouldPinHostBlockPool();
    config->alignment            = kPoolAlignment;
    auto pool                    = std::make_shared<HostBlockPool>(config);
    return pool->init() ? pool : nullptr;
}

std::shared_ptr<BlockTreeDiskMountGuard>
createDiskMountGuard(const KVCacheConfig& config, int64_t local_world_size, int64_t local_rank) {
    if (config.memory_cache_disk_paths.empty()) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: disk cache paths are empty");
        return nullptr;
    }
    auto       guard = std::make_shared<BlockTreeDiskMountGuard>();
    const auto path  = resolveDiskMountPath(config.memory_cache_disk_paths, local_world_size, local_rank);
    return guard->init(path) ? guard : nullptr;
}

BlockTreeDiskBlockPoolPtr createDiskPool(const KVCacheConfig&                            kv_config,
                                         const std::shared_ptr<BlockTreeDiskMountGuard>& guard,
                                         const std::string&                              name,
                                         size_t                                          payload_bytes,
                                         size_t                                          usable_blocks,
                                         int64_t                                         world_rank,
                                         int64_t                                         local_rank) {
    if (!guard || payload_bytes == 0 || usable_blocks == 0) {
        return nullptr;
    }
    auto config                  = std::make_shared<BlockTreeDiskBlockPoolConfig>();
    config->pool_type            = BlockPoolType::DISK;
    config->pool_name            = name;
    config->work_dir             = guard->workDir();
    config->local_rank           = local_rank;
    config->world_rank           = world_rank;
    config->payload_bytes        = payload_bytes;
    config->stride_bytes         = alignUp(payload_bytes, kPoolAlignment);
    config->physical_block_count = usable_blocks + 1;
    config->disk_size_bytes      = config->physical_block_count * config->stride_bytes;
    config->buffered_io          = kv_config.memory_cache_disk_buffered_io;
    config->mount_guard          = guard;
    auto pool                    = std::make_shared<BlockTreeDiskBlockPool>(config);
    return pool->init() ? pool : nullptr;
}

struct AggregationPlan {
    std::vector<std::vector<int>> members;
};

bool aggregationCompatible(const CacheConfig& cache_config, int lhs_gid, int rhs_gid) {
    const auto& lhs = cache_config.topology().groupById(static_cast<size_t>(lhs_gid));
    const auto& rhs = cache_config.topology().groupById(static_cast<size_t>(rhs_gid));
    if (lhs.policy.evict_policy != CacheEvictPolicy::CHAIN || rhs.policy.evict_policy != CacheEvictPolicy::CHAIN
        || !CacheConfig::samePolicy(lhs.policy, rhs.policy) || lhs.block_num != rhs.block_num
        || lhs.local_kv_head_num != rhs.local_kv_head_num || lhs.seq_size_per_block != rhs.seq_size_per_block
        || lhs.kernel_seq_size_per_block != rhs.kernel_seq_size_per_block
        || lhs.kv_block_stride_bytes != rhs.kv_block_stride_bytes
        || lhs.kv_scale_stride_bytes != rhs.kv_scale_stride_bytes || (lhs.spec == nullptr) != (rhs.spec == nullptr)
        || (lhs.spec != nullptr && lhs.spec->type != rhs.spec->type)) {
        return false;
    }
    if (lhs.policy.group_type == CacheGroupType::SWA) {
        return slidingWindowSize(lhs) == slidingWindowSize(rhs);
    }
    return true;
}

AggregationPlan buildAggregationPlan(const CacheConfig& cache_config) {
    AggregationPlan plan;
    for (int gid = 0; gid < cache_config.groupNums(); ++gid) {
        const auto& group = cache_config.topology().groupById(static_cast<size_t>(gid));
        if (!group.policy.enable_prefix_reuse) {
            continue;
        }
        auto it = std::find_if(plan.members.begin(), plan.members.end(), [&](const std::vector<int>& members) {
            return aggregationCompatible(cache_config, members.front(), gid);
        });
        if (it == plan.members.end()) {
            plan.members.push_back({gid});
        } else {
            it->push_back(gid);
        }
    }
    return plan;
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

size_t computeHostUsableBlockCount(size_t capacity_bytes, size_t combined_stride_bytes) {
    if (combined_stride_bytes == 0) {
        return 0;
    }
    const size_t physical_blocks = capacity_bytes / combined_stride_bytes;
    return physical_blocks > 0 ? physical_blocks - 1 : 0;
}

std::string resolveDiskMountPath(const std::string& paths_csv, int64_t local_world_size, int64_t local_rank) {
    const auto paths = split(paths_csv, ',');
    RTP_LLM_CHECK_WITH_INFO(paths.size() == static_cast<size_t>(local_world_size),
                            "disk cache path count must equal local_world_size, paths=%zu local_world_size=%ld",
                            paths.size(),
                            local_world_size);
    RTP_LLM_CHECK_WITH_INFO(local_rank >= 0 && local_rank < local_world_size,
                            "disk cache invalid local_rank=%ld local_world_size=%ld",
                            local_rank,
                            local_world_size);
    return paths[static_cast<size_t>(local_rank)];
}

BlockTreeCachePtr createBlockTreeCache(const CacheConfig&                cache_config,
                                       const KVCacheConfig&              kv_cache_config,
                                       const KVCacheAllocatorPtr&        allocator,
                                       const ParallelismConfig&          parallelism_config,
                                       std::shared_ptr<StorageBackend>   storage_backend,
                                       std::shared_ptr<BroadcastManager> broadcast_manager) {
    const int group_count = cache_config.groupNums();
    if (group_count <= 0) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: topology must contain at least one group");
        return nullptr;
    }
    const auto groups = alignAllocatorGroups(cache_config, allocator);
    if (groups.size() != static_cast<size_t>(group_count)) {
        return nullptr;
    }
    std::vector<DeviceBlockPoolPtr> per_tag_pools(static_cast<size_t>(group_count));
    const auto&                     independent_pools = allocator->groupBlockPools();
    if (!independent_pools.empty() && independent_pools.size() != static_cast<size_t>(group_count)) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: independent pool/topology count mismatch, pools=%zu topology=%d",
                          independent_pools.size(),
                          group_count);
        return nullptr;
    }
    for (int gid = 0; gid < group_count; ++gid) {
        auto pool =
            independent_pools.empty() ? allocator->getDeviceBlockPool() : independent_pools[static_cast<size_t>(gid)];
        if (!pool || groups[static_cast<size_t>(gid)]->blockPool() != pool) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: allocator/group direct pool mismatch for gid %d", gid);
            return nullptr;
        }
        per_tag_pools[static_cast<size_t>(gid)] = std::move(pool);
    }

    const bool host_enabled = kv_cache_config.enable_tiered_memory_cache && kv_cache_config.enable_memory_cache;
    const bool disk_enabled = host_enabled && kv_cache_config.enable_memory_cache_disk;
    if (kv_cache_config.enable_tiered_memory_cache && !kv_cache_config.enable_memory_cache) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: tiered memory requires enable_memory_cache");
        return nullptr;
    }
    if (kv_cache_config.enable_memory_cache_disk && !host_enabled) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: disk cache requires tiered host cache");
        return nullptr;
    }
    if (host_enabled && kv_cache_config.memory_cache_size_mb <= 0) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: host cache size must be positive");
        return nullptr;
    }
    if (disk_enabled && kv_cache_config.memory_cache_disk_size_mb <= 0) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: disk cache size must be positive");
        return nullptr;
    }

    std::vector<std::string>                   tags;
    std::vector<DeviceKVCacheGroupPtr>         device_groups;
    std::vector<BlockTreeCache::PerTagMapping> mappings(static_cast<size_t>(group_count), {-1, -1});
    std::vector<ComponentGroupPtr>             component_groups;
    std::vector<Component>                     components;
    std::vector<std::vector<std::string>>      device_group_tags;
    tags.reserve(static_cast<size_t>(group_count));
    device_groups.reserve(static_cast<size_t>(group_count));
    for (int gid = 0; gid < group_count; ++gid) {
        tags.push_back(cache_config.topology().groupById(static_cast<size_t>(gid)).tag);
        device_groups.push_back(groups[static_cast<size_t>(gid)]);
    }

    const auto plan = buildAggregationPlan(cache_config);
    component_groups.reserve(plan.members.size());
    device_group_tags.reserve(plan.members.size());
    for (size_t aggregate_index = 0; aggregate_index < plan.members.size(); ++aggregate_index) {
        const auto&                     members = plan.members[aggregate_index];
        const auto&                     first = cache_config.topology().groupById(static_cast<size_t>(members.front()));
        const int                       component_group_id = static_cast<int>(aggregate_index);
        auto                            component_group    = createComponentGroup(component_group_id, first);
        std::vector<DeviceBlockPoolPtr> device_pools;
        std::vector<std::string>        member_tags;
        std::vector<int>                component_indices;
        std::vector<std::vector<size_t>> component_layer_bytes;
        device_pools.reserve(members.size());
        member_tags.reserve(members.size());
        component_indices.reserve(members.size());
        component_layer_bytes.reserve(members.size());

        for (size_t local_pool = 0; local_pool < members.size(); ++local_pool) {
            const int   gid      = members[local_pool];
            const auto& declared = cache_config.topology().groupById(static_cast<size_t>(gid));
            device_pools.push_back(per_tag_pools[static_cast<size_t>(gid)]);
            member_tags.push_back(declared.tag);

            Component component;
            component.component_id                 = static_cast<int>(components.size());
            component.tag                          = declared.tag;
            component.component_group_id           = component_group_id;
            component.type                         = declared.policy.group_type;
            const size_t group_stride              = declared.kv_block_stride_bytes + declared.kv_scale_stride_bytes;
            const bool   legacy_shared_single_pool = !cache_config.use_independent_block_pools && group_count == 1
                                                   && cache_config.mtp_sub_configs.empty()
                                                   && cache_config.layer_to_block_stride_bytes.empty();
            if (group_stride == 0 || declared.layer_ids.empty()) {
                RTP_LLM_LOG_ERROR("createBlockTreeCache: tag %s has an invalid layer layout", declared.tag.c_str());
                return nullptr;
            }
            if (!cache_config.use_independent_block_pools && cache_config.layer_to_block_stride_bytes.empty()
                && !legacy_shared_single_pool) {
                RTP_LLM_LOG_ERROR("createBlockTreeCache: shared pools require a complete physical stride table");
                return nullptr;
            }
            for (const int layer_id : declared.layer_ids) {
                size_t physical_stride = group_stride;
                if (!cache_config.use_independent_block_pools && !legacy_shared_single_pool) {
                    if (layer_id < 0 || static_cast<size_t>(layer_id) >= cache_config.layer_to_block_stride_bytes.size()
                        || cache_config.layer_to_block_stride_bytes[static_cast<size_t>(layer_id)] <= 0) {
                        RTP_LLM_LOG_ERROR("createBlockTreeCache: invalid shared-pool physical stride for layer %d",
                                          layer_id);
                        return nullptr;
                    }
                    physical_stride =
                        static_cast<size_t>(cache_config.layer_to_block_stride_bytes[static_cast<size_t>(layer_id)]);
                }
                component.model_layer_ids.push_back(layer_id);
                component.layer_bytes.push_back(physical_stride);
            }
            component_indices.push_back(component.component_id);
            component_layer_bytes.push_back(component.layer_bytes);
            components.push_back(std::move(component));
            mappings[static_cast<size_t>(gid)] = {component_group_id, static_cast<int>(local_pool)};
        }

        component_group->setDevicePools(std::move(device_pools), std::move(member_tags));
        auto layout = ComponentGroupLayout::create(component_layer_bytes);
        RTP_LLM_CHECK_WITH_INFO(layout.has_value(),
                                "createBlockTreeCache: failed to finalize layout for component group %d",
                                component_group_id);
        RTP_LLM_CHECK_WITH_INFO(component_group->setLayout(std::move(component_indices), std::move(*layout)),
                                "createBlockTreeCache: failed to seal layout for component group %d",
                                component_group_id);
        RTP_LLM_LOG_INFO("createBlockTreeCache: group[%d] layout sealed: payload_bytes=%zu",
                         component_group_id,
                         component_group->layout().payloadBytes());
        device_group_tags.push_back(component_group->tags());
        component_groups.push_back(std::move(component_group));
    }

    auto   tree            = std::make_unique<BlockTree>(static_cast<int>(component_groups.size()));
    size_t combined_stride = 0;
    for (const auto& component_group : component_groups) {
        combined_stride += alignUp(component_group->layout().payloadBytes(), kPoolAlignment);
    }

    if (host_enabled && !component_groups.empty()) {
        const size_t bytes  = static_cast<size_t>(kv_cache_config.memory_cache_size_mb) * 1024UL * 1024UL;
        const size_t usable = computeHostUsableBlockCount(bytes, combined_stride);
        if (usable == 0) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: host budget is too small for one complete tree coordinate");
            return nullptr;
        }
        for (size_t i = 0; i < component_groups.size(); ++i) {
            const size_t payload = component_groups[i]->layout().payloadBytes();
            auto         pool    = createHostPool("block_tree_host_g" + std::to_string(i), payload, usable);
            if (!pool) {
                return nullptr;
            }
            component_groups[i]->setHostPool(std::move(pool));
        }
    }

    if (disk_enabled && !component_groups.empty()) {
        const size_t bytes  = static_cast<size_t>(kv_cache_config.memory_cache_disk_size_mb) * 1024UL * 1024UL;
        const size_t usable = computeHostUsableBlockCount(bytes, combined_stride);
        if (usable == 0) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: disk budget is too small for one complete tree coordinate");
            return nullptr;
        }
        auto guard =
            createDiskMountGuard(kv_cache_config, parallelism_config.local_world_size, parallelism_config.local_rank);
        if (!guard) {
            return nullptr;
        }
        for (size_t i = 0; i < component_groups.size(); ++i) {
            auto pool = createDiskPool(kv_cache_config,
                                       guard,
                                       "block_tree_disk_g" + std::to_string(i),
                                       component_groups[i]->layout().payloadBytes(),
                                       usable,
                                       parallelism_config.world_rank,
                                       parallelism_config.local_rank);
            if (!pool) {
                return nullptr;
            }
            component_groups[i]->setDiskPool(std::move(pool));
        }
    }

    BlockTreeCacheConfig config;
    config.enable_device_cache    = kv_cache_config.enable_device_cache;
    config.enable_memory_cache    = host_enabled;
    config.enable_disk_cache      = disk_enabled;
    config.enable_remote_cache    = kv_cache_config.enable_remote_cache && storage_backend != nullptr;
    config.enable_load_back       = host_enabled;
    config.device_min_free_blocks = kv_cache_config.device_cache_min_free_blocks > 0 ?
                                        static_cast<size_t>(kv_cache_config.device_cache_min_free_blocks) :
                                        0;
    if (config.enable_device_cache) {
        config.watermark_device = {kDefaultDeviceWatermarkRatio, 0};
    }
    if (disk_enabled) {
        config.watermark_host = {kDefaultHostWatermarkRatio, 0};
        config.watermark_disk = {kDefaultDiskWatermarkRatio, 0};
    }
    config.memory_cache_size_mb          = kv_cache_config.memory_cache_size_mb;
    config.memory_cache_disk_size_mb     = kv_cache_config.memory_cache_disk_size_mb;
    config.memory_cache_disk_buffered_io = kv_cache_config.memory_cache_disk_buffered_io;
    config.memory_cache_sync_timeout_ms =
        checkedTimeout(kv_cache_config.memory_cache_sync_timeout_ms, "memory_cache_sync_timeout_ms");
    config.memory_cache_disk_sync_timeout_ms =
        disk_enabled ?
            checkedTimeout(kv_cache_config.memory_cache_disk_sync_timeout_ms, "memory_cache_disk_sync_timeout_ms") :
            config.memory_cache_sync_timeout_ms;

    auto components_ptr  = std::make_shared<const std::vector<Component>>(std::move(components));
    auto per_rank_engine = std::make_shared<PerRankBlockTransferEngine>(component_groups, components_ptr);
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine;
    if (broadcast_manager != nullptr) {
        multi_rank_engine =
            std::make_shared<MultiRankBlockTransferEngine>(component_groups, std::move(broadcast_manager));
    }
    auto transfer_dispatcher =
        std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine), std::move(multi_rank_engine));
    auto task_pool = std::make_unique<BlockCacheTaskPool>(
        static_cast<size_t>(config.eviction_thread_pool_size), 1000, "BlockTreeEvictionPool");

    auto result = std::make_shared<BlockTreeCache>(std::move(tree),
                                                   std::move(component_groups),
                                                   std::move(components_ptr),
                                                   std::move(config),
                                                   std::move(storage_backend),
                                                   std::move(transfer_dispatcher),
                                                   std::move(task_pool),
                                                   std::move(tags),
                                                   std::move(device_groups),
                                                   std::move(mappings),
                                                   std::move(device_group_tags));
    if (!result->init()) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: BlockTreeCache init failed");
        return nullptr;
    }
    return result;
}

}  // namespace rtp_llm
