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
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/SWAGroupSet.h"
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

GroupSetPtr createGroupSet(const GroupBase& group) {
    GroupSetPtr result;
    switch (group.policy.group_type) {
        case CacheGroupType::FULL:
            result = std::make_shared<FullGroupSet>();
            break;
        case CacheGroupType::LINEAR:
            result = std::make_shared<LinearGroupSet>();
            break;
        case CacheGroupType::SWA: {
            const auto seq_size = group.seq_size_per_block;
            RTP_LLM_CHECK_WITH_INFO(seq_size > 0 && seq_size <= static_cast<size_t>(std::numeric_limits<int>::max()),
                                    "SWA tag %s has invalid seq_size_per_block=%zu",
                                    group.tag.c_str(),
                                    seq_size);
            result = std::make_shared<SWAGroupSet>(slidingWindowSize(group), static_cast<int>(seq_size));
            break;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(result != nullptr, "unsupported cache group type for tag %s", group.tag.c_str());
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
    for (int group_id = 0; group_id < cache_config.groupNums(); ++group_id) {
        const auto& group = cache_config.topology().groupById(static_cast<size_t>(group_id));
        if (!group.policy.enable_prefix_reuse) {
            continue;
        }
        auto it = std::find_if(plan.members.begin(), plan.members.end(), [&](const std::vector<int>& members) {
            return aggregationCompatible(cache_config, members.front(), group_id);
        });
        if (it == plan.members.end()) {
            plan.members.push_back({group_id});
        } else {
            it->push_back(group_id);
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
    std::vector<DeviceBlockPoolPtr> group_pools(static_cast<size_t>(group_count));
    const auto&                     independent_pools = allocator->groupBlockPools();
    if (!independent_pools.empty() && independent_pools.size() != static_cast<size_t>(group_count)) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: independent pool/topology count mismatch, pools=%zu topology=%d",
                          independent_pools.size(),
                          group_count);
        return nullptr;
    }
    for (int group_id = 0; group_id < group_count; ++group_id) {
        auto pool = independent_pools.empty() ? allocator->getDeviceBlockPool() :
                                                independent_pools[static_cast<size_t>(group_id)];
        if (!pool || groups[static_cast<size_t>(group_id)]->blockPool() != pool) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: allocator/group direct pool mismatch for group_id %d", group_id);
            return nullptr;
        }
        group_pools[static_cast<size_t>(group_id)] = std::move(pool);
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

    std::vector<GroupSetPtr>   group_sets;
    const auto                 plan = buildAggregationPlan(cache_config);
    std::unordered_set<size_t> planned_group_ids;
    for (const auto& members : plan.members) {
        for (int group_id : members) {
            RTP_LLM_CHECK_WITH_INFO(group_id >= 0 && planned_group_ids.emplace(static_cast<size_t>(group_id)).second,
                                    "BlockTreeCache aggregation plan contains invalid or duplicate group_id=%d",
                                    group_id);
        }
    }
    for (size_t group_id = 0; group_id < cache_config.topology().groups().size(); ++group_id) {
        const bool reusable = cache_config.topology().groupById(group_id).policy.enable_prefix_reuse;
        RTP_LLM_CHECK_WITH_INFO(reusable == (planned_group_ids.find(group_id) != planned_group_ids.end()),
                                "BlockTreeCache aggregation plan coverage mismatch for group_id=%zu reusable=%d",
                                group_id,
                                static_cast<int>(reusable));
    }
    group_sets.reserve(plan.members.size());
    for (size_t aggregate_index = 0; aggregate_index < plan.members.size(); ++aggregate_index) {
        const auto&                     members = plan.members[aggregate_index];
        const auto&                     first = cache_config.topology().groupById(static_cast<size_t>(members.front()));
        const size_t                    group_set_id = aggregate_index;
        auto                            group_set    = createGroupSet(first);
        std::vector<DeviceBlockPoolPtr> device_pools;
        std::vector<size_t>             group_ids;
        device_pools.reserve(members.size());
        group_ids.reserve(members.size());

        for (size_t local_pool = 0; local_pool < members.size(); ++local_pool) {
            const int group_id = members[local_pool];
            device_pools.push_back(group_pools[static_cast<size_t>(group_id)]);
            group_ids.push_back(static_cast<size_t>(group_id));
        }

        group_set->initialize(group_set_id, cache_config.topologyPtr(), std::move(group_ids), std::move(device_pools));
        RTP_LLM_LOG_INFO("createBlockTreeCache: group[%zu] membership sealed: payload_bytes=%zu",
                         group_set_id,
                         group_set->payloadBytes());
        group_sets.push_back(std::move(group_set));
    }

    auto   tree            = std::make_unique<BlockTree>(group_sets.size());
    size_t combined_stride = 0;
    for (const auto& group_set : group_sets) {
        const size_t stride = alignUp(group_set->payloadBytes(), kPoolAlignment);
        RTP_LLM_CHECK_WITH_INFO(stride <= std::numeric_limits<size_t>::max() - combined_stride,
                                "BlockTreeCache combined lower-tier stride overflow");
        combined_stride += stride;
    }

    if (host_enabled && !group_sets.empty()) {
        const size_t bytes  = static_cast<size_t>(kv_cache_config.memory_cache_size_mb) * 1024UL * 1024UL;
        const size_t usable = computeHostUsableBlockCount(bytes, combined_stride);
        if (usable == 0) {
            RTP_LLM_LOG_ERROR("createBlockTreeCache: host budget is too small for one complete tree coordinate");
            return nullptr;
        }
        for (size_t i = 0; i < group_sets.size(); ++i) {
            const size_t payload = group_sets[i]->payloadBytes();
            auto         pool    = createHostPool("block_tree_host_g" + std::to_string(i), payload, usable);
            if (!pool) {
                return nullptr;
            }
            group_sets[i]->setHostPool(std::move(pool));
        }
    }

    if (disk_enabled && !group_sets.empty()) {
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
        for (size_t i = 0; i < group_sets.size(); ++i) {
            auto pool = createDiskPool(kv_cache_config,
                                       guard,
                                       "block_tree_disk_g" + std::to_string(i),
                                       group_sets[i]->payloadBytes(),
                                       usable,
                                       parallelism_config.world_rank,
                                       parallelism_config.local_rank);
            if (!pool) {
                return nullptr;
            }
            group_sets[i]->setDiskPool(std::move(pool));
        }
    }

    BlockTreeCacheConfig config;
    config.enable_device_cache    = kv_cache_config.enable_device_cache;
    config.enable_memory_cache    = host_enabled;
    config.enable_disk_cache      = disk_enabled;
    config.enable_remote_cache    = kv_cache_config.enable_remote_cache && storage_backend != nullptr;
    config.enable_load            = host_enabled;
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

    auto per_rank_engine = std::make_shared<PerRankBlockTransferEngine>(group_sets);
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine;
    if (broadcast_manager != nullptr) {
        multi_rank_engine = std::make_shared<MultiRankBlockTransferEngine>(group_sets, std::move(broadcast_manager));
    }
    auto transfer_dispatcher =
        std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine), std::move(multi_rank_engine));
    auto task_pool = std::make_unique<BlockTreeTaskPool>(
        static_cast<size_t>(config.eviction_thread_pool_size), 1000, "BlockTreeEvictionPool");

    auto result = std::make_shared<BlockTreeCache>(std::move(tree),
                                                   std::move(group_sets),
                                                   std::move(config),
                                                   std::move(storage_backend),
                                                   std::move(transfer_dispatcher),
                                                   std::move(task_pool));
    if (!result->init()) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: BlockTreeCache init failed");
        return nullptr;
    }
    return result;
}

}  // namespace rtp_llm
