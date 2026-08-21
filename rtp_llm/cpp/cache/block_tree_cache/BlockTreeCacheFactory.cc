#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeTaskPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTree.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/LinearGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/SWAGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/DiskBlockPool.h"
#include "rtp_llm/cpp/cache/block_tree_cache/block_pool/HostBlockPool.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {
namespace {

constexpr size_t kPoolAlignment = 4096;

std::optional<EvictionPolicy> parseEvictionPolicy(const std::string& value) {
    std::string normalized = value;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (normalized == "lru") {
        return EvictionPolicy::LRU;
    }
    if (normalized == "lfu") {
        return EvictionPolicy::LFU;
    }
    if (normalized == "fifo") {
        return EvictionPolicy::FIFO;
    }
    return std::nullopt;
}

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

int slidingWindowSize(const GroupBase& group, size_t group_id) {
    RTP_LLM_CHECK_WITH_INFO(
        group.policy.group_type == CacheGroupType::SWA, "sliding window requested for non-SWA group_id=%zu", group_id);
    RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "SWA group_id=%zu has null cache spec", group_id);
    RTP_LLM_CHECK_WITH_INFO(group.policy.sliding_window_size >= 0,
                            "SWA group_id=%zu has invalid sliding window=%d",
                            group_id,
                            group.policy.sliding_window_size);
    return group.policy.sliding_window_size;
}

GroupSetPtr createGroupSet(const GroupBase&                group,
                           size_t                          group_id,
                           std::vector<DeviceBlockPoolPtr> device_pools,
                           std::shared_ptr<HostBlockPool>  host_pool,
                           BlockTreeDiskBlockPoolPtr       disk_pool) {
    GroupSetPtr result;
    switch (group.policy.group_type) {
        case CacheGroupType::FULL:
            result =
                std::make_shared<FullGroupSet>(std::move(device_pools), std::move(host_pool), std::move(disk_pool));
            break;
        case CacheGroupType::LINEAR:
            result =
                std::make_shared<LinearGroupSet>(std::move(device_pools), std::move(host_pool), std::move(disk_pool));
            break;
        case CacheGroupType::SWA: {
            const auto seq_size = group.seq_size_per_block;
            RTP_LLM_CHECK_WITH_INFO(seq_size > 0 && seq_size <= static_cast<size_t>(std::numeric_limits<int>::max()),
                                    "SWA group_id=%zu has invalid seq_size_per_block=%zu",
                                    group_id,
                                    seq_size);
            result = std::make_shared<SWAGroupSet>(slidingWindowSize(group, group_id),
                                                   static_cast<int>(seq_size),
                                                   std::move(device_pools),
                                                   std::move(host_pool),
                                                   std::move(disk_pool));
            break;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(result != nullptr, "unsupported cache group type for group_id=%zu", group_id);
    return result;
}

std::vector<KVCacheGroupPtr> alignAllocatorGroups(const CacheConfig&         cache_config,
                                                  const KVCacheAllocatorPtr& allocator) {
    if (!allocator) {
        RTP_LLM_LOG_ERROR("allocator is null");
        return {};
    }
    const auto allocator_groups = allocator->cacheGroups();
    const auto group_count      = static_cast<size_t>(cache_config.groupNums());
    if (allocator_groups.size() != group_count) {
        RTP_LLM_LOG_ERROR("allocator/topology group count mismatch, allocator=%zu topology=%zu",
                          allocator_groups.size(),
                          group_count);
        return {};
    }

    std::vector<KVCacheGroupPtr> aligned(group_count);
    for (const auto& group : allocator_groups) {
        if (!group || !group->blockPool()) {
            RTP_LLM_LOG_ERROR("allocator group/direct pool must be non-null");
            return {};
        }
        const int group_id = group->group_id();
        if (group_id < 0 || static_cast<size_t>(group_id) >= group_count) {
            RTP_LLM_LOG_ERROR("allocator group_id=%d out of range [0, %zu)", group_id, group_count);
            return {};
        }
        auto& aligned_group = aligned[static_cast<size_t>(group_id)];
        if (aligned_group != nullptr) {
            RTP_LLM_LOG_ERROR("duplicate allocator group_id=%d", group_id);
            return {};
        }
        aligned_group = group;
    }

    for (size_t group_id = 0; group_id < group_count; ++group_id) {
        const auto& group = aligned[group_id];
        if (group == nullptr) {
            RTP_LLM_LOG_ERROR("allocator is missing group_id=%zu", group_id);
            return {};
        }
        const auto& actual   = group->config();
        const auto& declared = cache_config.topology().groupById(group_id);
        if (actual.spec != declared.spec || !CacheConfig::samePolicy(actual.policy, declared.policy)
            || actual.layer_ids != declared.layer_ids || actual.block_num != declared.block_num
            || actual.local_kv_head_num != declared.local_kv_head_num
            || actual.seq_size_per_block != declared.seq_size_per_block
            || actual.kernel_seq_size_per_block != declared.kernel_seq_size_per_block
            || actual.kv_block_stride_bytes != declared.kv_block_stride_bytes
            || actual.kv_scale_stride_bytes != declared.kv_scale_stride_bytes) {
            RTP_LLM_LOG_ERROR("allocator group_id=%zu does not exactly match topology", group_id);
            return {};
        }
    }
    return aligned;
}

std::shared_ptr<HostBlockPool>
createHostPool(const std::string& name, size_t payload_bytes, size_t usable_blocks, bool enable_pinned) {
    if (payload_bytes == 0 || usable_blocks == 0) {
        return nullptr;
    }
    auto config                  = std::make_shared<HostBlockPoolConfig>();
    config->pool_type            = BlockPoolType::HOST;
    config->pool_name            = name;
    config->physical_block_count = usable_blocks + 1;
    config->payload_bytes        = payload_bytes;
    config->stride_bytes         = alignUp(payload_bytes, kPoolAlignment);
    config->enable_pinned        = enable_pinned;
    config->alignment            = kPoolAlignment;
    auto pool                    = std::make_shared<HostBlockPool>(config);
    return pool->init() ? pool : nullptr;
}

std::shared_ptr<BlockTreeDiskMountGuard>
createDiskMountGuard(const KVCacheConfig& config, int64_t local_world_size, int64_t local_rank) {
    if (config.disk_cache_paths.empty()) {
        RTP_LLM_LOG_ERROR("disk cache paths are empty");
        return nullptr;
    }
    auto       guard = std::make_shared<BlockTreeDiskMountGuard>();
    const auto path  = resolveDiskMountPath(config.disk_cache_paths, local_world_size, local_rank);
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
    config->buffered_io          = kv_config.disk_cache_buffered_io;
    config->mount_guard          = guard;
    auto pool                    = std::make_shared<BlockTreeDiskBlockPool>(config);
    return pool->init() ? pool : nullptr;
}

struct AggregationPlan {
    std::vector<std::vector<int>> members;
};

bool aggregationCompatible(const CacheConfig& cache_config, int lhs_group_id, int rhs_group_id) {
    const auto& lhs = cache_config.topology().groupById(static_cast<size_t>(lhs_group_id));
    const auto& rhs = cache_config.topology().groupById(static_cast<size_t>(rhs_group_id));
    if (!CacheConfig::samePolicy(lhs.policy, rhs.policy) || lhs.block_num != rhs.block_num
        || lhs.local_kv_head_num != rhs.local_kv_head_num || lhs.seq_size_per_block != rhs.seq_size_per_block
        || lhs.kernel_seq_size_per_block != rhs.kernel_seq_size_per_block
        || lhs.kv_block_stride_bytes != rhs.kv_block_stride_bytes
        || lhs.kv_scale_stride_bytes != rhs.kv_scale_stride_bytes || (lhs.spec == nullptr) != (rhs.spec == nullptr)
        || (lhs.spec != nullptr && lhs.spec->type != rhs.spec->type)) {
        return false;
    }
    if (lhs.policy.group_type == CacheGroupType::SWA) {
        return slidingWindowSize(lhs, static_cast<size_t>(lhs_group_id))
               == slidingWindowSize(rhs, static_cast<size_t>(rhs_group_id));
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
size_t computeGroupSetPayloadBytes(const CacheConfig& cache_config, const std::vector<int>& members) {
    size_t payload_bytes = 0;
    for (int group_id : members) {
        RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid group_id=%d", group_id);
        const size_t group_bytes = cache_config.blockSizeBytesForGroup(static_cast<size_t>(group_id));
        RTP_LLM_CHECK_WITH_INFO(group_bytes > 0, "group_id=%d has zero payload", group_id);
        RTP_LLM_CHECK_WITH_INFO(group_bytes <= std::numeric_limits<size_t>::max() - payload_bytes,
                                "group set payload overflow at group_id=%d",
                                group_id);
        payload_bytes += group_bytes;
    }
    return payload_bytes;
}

std::vector<BlockInfo> resolveStorageBuffers(const CacheTopology&                   topology,
                                             const std::vector<DeviceBlockPoolPtr>& group_pools,
                                             int                                    layer_id,
                                             int                                    group_id,
                                             int                                    block_id) {
    RTP_LLM_CHECK_WITH_INFO(
        group_id >= 0 && static_cast<size_t>(group_id) < group_pools.size(), "invalid storage group_id=%d", group_id);
    const auto& group = topology.groupById(static_cast<size_t>(group_id));
    const auto  layer = std::find(group.layer_ids.begin(), group.layer_ids.end(), layer_id);
    RTP_LLM_CHECK_WITH_INFO(
        layer != group.layer_ids.end(), "layer_id=%d does not belong to storage group_id=%d", layer_id, group_id);
    auto buffers = group_pools[static_cast<size_t>(group_id)]->convertIndexToBuffer(
        static_cast<int>(std::distance(group.layer_ids.begin(), layer)), block_id);
    RTP_LLM_CHECK_WITH_INFO(!buffers.empty(), "storage group_id=%d returned no block buffers", group_id);
    RTP_LLM_CHECK_WITH_INFO(buffers[0].size_bytes >= group.kv_block_stride_bytes,
                            "storage group_id=%d physical kv block is smaller than logical block",
                            group_id);
    buffers[0].size_bytes = group.kv_block_stride_bytes;
    if (group.kv_scale_stride_bytes == 0) {
        buffers.resize(1);
        return buffers;
    }
    RTP_LLM_CHECK_WITH_INFO(buffers.size() >= 2 && buffers[1].size_bytes >= group.kv_scale_stride_bytes,
                            "storage group_id=%d has an invalid scale block buffer",
                            group_id);
    buffers[1].size_bytes = group.kv_scale_stride_bytes;
    buffers.resize(2);
    return buffers;
}

}  // namespace

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
    const auto device_eviction_policy = parseEvictionPolicy(kv_cache_config.device_eviction_policy);
    const auto host_eviction_policy   = parseEvictionPolicy(kv_cache_config.host_eviction_policy);
    const auto disk_eviction_policy   = parseEvictionPolicy(kv_cache_config.disk_eviction_policy);
    if (!device_eviction_policy.has_value() || !host_eviction_policy.has_value() || !disk_eviction_policy.has_value()) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: unsupported eviction policy, device=%s host=%s disk=%s",
                          kv_cache_config.device_eviction_policy.c_str(),
                          kv_cache_config.host_eviction_policy.c_str(),
                          kv_cache_config.disk_eviction_policy.c_str());
        return nullptr;
    }
    const int group_count = cache_config.groupNums();
    if (group_count <= 0) {
        RTP_LLM_LOG_ERROR("topology must contain at least one group");
        return nullptr;
    }
    const auto groups = alignAllocatorGroups(cache_config, allocator);
    if (groups.size() != static_cast<size_t>(group_count)) {
        return nullptr;
    }
    std::vector<DeviceBlockPoolPtr> group_pools(static_cast<size_t>(group_count));
    const auto&                     independent_pools = allocator->groupBlockPools();
    if (!independent_pools.empty() && independent_pools.size() != static_cast<size_t>(group_count)) {
        RTP_LLM_LOG_ERROR(
            "independent pool/topology count mismatch, pools=%zu topology=%d", independent_pools.size(), group_count);
        return nullptr;
    }
    for (int group_id = 0; group_id < group_count; ++group_id) {
        auto pool = independent_pools.empty() ? allocator->getDeviceBlockPool() :
                                                independent_pools[static_cast<size_t>(group_id)];
        if (!pool || groups[static_cast<size_t>(group_id)]->blockPool() != pool) {
            RTP_LLM_LOG_ERROR("allocator/group direct pool mismatch for group_id %d", group_id);
            return nullptr;
        }
        group_pools[static_cast<size_t>(group_id)] = std::move(pool);
    }

    const bool host_enabled = kv_cache_config.enable_host_cache;
    const bool disk_enabled = kv_cache_config.enable_disk_cache;
    if (host_enabled && kv_cache_config.host_cache_size_mb <= 0) {
        RTP_LLM_LOG_ERROR("host cache size must be positive");
        return nullptr;
    }
    if (disk_enabled && kv_cache_config.disk_cache_size_mb <= 0) {
        RTP_LLM_LOG_ERROR("disk cache size must be positive");
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
    if (disk_enabled) {
        for (const auto& members : plan.members) {
            RTP_LLM_CHECK_WITH_INFO(!members.empty(), "BlockTreeCache aggregation plan contains an empty group set");
            const auto group_id = static_cast<size_t>(members.front());
            if (cache_config.topology().groupById(group_id).policy.group_type == CacheGroupType::LINEAR) {
                RTP_LLM_LOG_ERROR(
                    "disk cache does not support reusable LINEAR group sets, group_id=%zu", group_id);
                return nullptr;
            }
        }
    }

    std::vector<size_t> group_set_payload_bytes;
    group_set_payload_bytes.reserve(plan.members.size());
    size_t combined_stride = 0;
    for (const auto& members : plan.members) {
        const size_t payload_bytes = computeGroupSetPayloadBytes(cache_config, members);
        group_set_payload_bytes.push_back(payload_bytes);
        const size_t stride = alignUp(payload_bytes, kPoolAlignment);
        RTP_LLM_CHECK_WITH_INFO(stride <= std::numeric_limits<size_t>::max() - combined_stride,
                                "BlockTreeCache combined lower-tier stride overflow");
        combined_stride += stride;
    }

    std::vector<std::shared_ptr<HostBlockPool>> host_pools(plan.members.size());
    if (host_enabled && !plan.members.empty()) {
        const size_t bytes  = static_cast<size_t>(kv_cache_config.host_cache_size_mb) * 1024UL * 1024UL;
        const size_t usable = computeHostUsableBlockCount(bytes, combined_stride);
        if (usable == 0) {
            RTP_LLM_LOG_ERROR("host budget is too small for one complete tree coordinate");
            return nullptr;
        }
        for (size_t group_set_id = 0; group_set_id < plan.members.size(); ++group_set_id) {
            host_pools[group_set_id] = createHostPool("block_tree_host_g" + std::to_string(group_set_id),
                                                      group_set_payload_bytes[group_set_id],
                                                      usable,
                                                      kv_cache_config.enable_host_cache_pinned);
            if (!host_pools[group_set_id]) {
                return nullptr;
            }
        }
    }

    std::vector<BlockTreeDiskBlockPoolPtr> disk_pools(plan.members.size());
    if (disk_enabled && !plan.members.empty()) {
        const size_t bytes  = static_cast<size_t>(kv_cache_config.disk_cache_size_mb) * 1024UL * 1024UL;
        const size_t usable = computeHostUsableBlockCount(bytes, combined_stride);
        if (usable == 0) {
            RTP_LLM_LOG_ERROR("disk budget is too small for one complete tree coordinate");
            return nullptr;
        }
        auto guard =
            createDiskMountGuard(kv_cache_config, parallelism_config.local_world_size, parallelism_config.local_rank);
        if (!guard) {
            return nullptr;
        }
        for (size_t group_set_id = 0; group_set_id < plan.members.size(); ++group_set_id) {
            disk_pools[group_set_id] = createDiskPool(kv_cache_config,
                                                      guard,
                                                      "block_tree_disk_g" + std::to_string(group_set_id),
                                                      group_set_payload_bytes[group_set_id],
                                                      usable,
                                                      parallelism_config.world_rank,
                                                      parallelism_config.local_rank);
            if (!disk_pools[group_set_id]) {
                return nullptr;
            }
        }
    }

    group_sets.reserve(plan.members.size());
    for (size_t group_set_id = 0; group_set_id < plan.members.size(); ++group_set_id) {
        const auto&                     members = plan.members[group_set_id];
        std::vector<DeviceBlockPoolPtr> device_pools;
        std::vector<size_t>             group_ids;
        device_pools.reserve(members.size());
        group_ids.reserve(members.size());
        for (int group_id : members) {
            device_pools.push_back(group_pools[static_cast<size_t>(group_id)]);
            group_ids.push_back(static_cast<size_t>(group_id));
        }
        const auto& first     = cache_config.topology().groupById(group_ids.front());
        auto        group_set = createGroupSet(first,
                                        group_ids.front(),
                                        std::move(device_pools),
                                        std::move(host_pools[group_set_id]),
                                        std::move(disk_pools[group_set_id]));
        group_set->initialize(group_set_id, cache_config.topologyPtr(), std::move(group_ids));
        RTP_LLM_LOG_INFO(
            "group_set[%zu] membership sealed: payload_bytes=%zu", group_set_id, group_set->payloadBytes());
        group_sets.push_back(std::move(group_set));
    }

    BlockTreeCacheConfig config;
    config.enable_device_cache = kv_cache_config.enable_device_cache;
    config.enable_host_cache   = host_enabled;
    config.enable_disk_cache   = disk_enabled;
    config.enable_remote_cache = kv_cache_config.enable_remote_cache && storage_backend != nullptr;
    if (!config.enable_remote_cache) {
        storage_backend = nullptr;
    }
    config.device_eviction_policy = *device_eviction_policy;
    config.host_eviction_policy   = *host_eviction_policy;
    config.disk_eviction_policy   = *disk_eviction_policy;
    if (config.enable_device_cache) {
        config.watermark_device.ratio = kDefaultDeviceWatermarkRatio;
    }
    if (host_enabled) {
        config.watermark_host.ratio = kDefaultHostWatermarkRatio;
    }
    if (disk_enabled) {
        config.watermark_disk.ratio = kDefaultDiskWatermarkRatio;
    }
    config.host_cache_sync_timeout_ms =
        checkedTimeout(kv_cache_config.host_cache_sync_timeout_ms, "host_cache_sync_timeout_ms");
    config.disk_cache_sync_timeout_ms =
        disk_enabled ? checkedTimeout(kv_cache_config.disk_cache_sync_timeout_ms, "disk_cache_sync_timeout_ms") :
                       config.host_cache_sync_timeout_ms;

    if (disk_enabled) {
        const int64_t staging_block_count = kv_cache_config.disk_cache_staging_block_count;
        if (staging_block_count < 2 || staging_block_count % 2 != 0
            || static_cast<uint64_t>(staging_block_count) > std::numeric_limits<size_t>::max()) {
            RTP_LLM_LOG_ERROR("disk_cache_staging_block_count must be even and >= 2, got %ld", staging_block_count);
            return nullptr;
        }
        config.device_disk_staging_block_count = static_cast<size_t>(staging_block_count);
    }

    const int64_t max_batch_descriptors = kv_cache_config.memory_cache_max_descriptors_per_transfer_batch;
    if (max_batch_descriptors <= 0
        || static_cast<uint64_t>(max_batch_descriptors) > std::numeric_limits<size_t>::max()) {
        RTP_LLM_LOG_ERROR("memory_cache_max_descriptors_per_transfer_batch must be > 0, got %ld",
                          max_batch_descriptors);
        return nullptr;
    }
    config.max_descriptors_per_transfer_batch = static_cast<size_t>(max_batch_descriptors);

    const int64_t scan_interval_ms = kv_cache_config.block_tree_full_prefix_scan_interval_ms;
    if (scan_interval_ms < 0 || (scan_interval_ms > 0 && scan_interval_ms < 1000)) {
        RTP_LLM_LOG_ERROR("block_tree_full_prefix_scan_interval_ms must be 0 or >= 1000, got %ld", scan_interval_ms);
        return nullptr;
    }
    config.full_prefix_scan_interval_ms = scan_interval_ms;
    config.world_rank                   = static_cast<int>(parallelism_config.world_rank);
    config.local_rank                   = static_cast<int>(parallelism_config.local_rank);

    auto per_rank_engine = std::make_shared<PerRankBlockTransferEngine>(group_sets,
                                                                        DeviceHostCopyOptions{},
                                                                        config.device_disk_staging_block_count,
                                                                        config.max_descriptors_per_transfer_batch);
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine;
    if (broadcast_manager != nullptr) {
        multi_rank_engine = std::make_shared<MultiRankBlockTransferEngine>(group_sets, std::move(broadcast_manager));
    }
    auto transfer_dispatcher =
        std::make_unique<BlockTransferDispatcher>(std::move(per_rank_engine), std::move(multi_rank_engine));
    auto task_pool =
        std::make_unique<BlockTreeTaskPool>(static_cast<size_t>(config.task_pool_size), 1000, "BlockTreeCacheTaskPool");

    auto tree = std::make_unique<BlockTree>(std::move(group_sets));

    auto result = std::make_shared<BlockTreeCache>(std::move(tree),
                                                   std::move(config),
                                                   std::move(storage_backend),
                                                   std::move(transfer_dispatcher),
                                                   std::move(task_pool));
    if (result->isRemoteCacheEnabled()) {
        const auto storage_topology = cache_config.topologyPtr();
        const auto resolver_pools   = group_pools;
        RTP_LLM_CHECK_WITH_INFO(result->storageBackend()->init(
                                    storage_topology,
                                    group_pools,
                                    [storage_topology, resolver_pools](int layer_id, int group_id, int block_id) {
                                        return resolveStorageBuffers(
                                            *storage_topology, resolver_pools, layer_id, group_id, block_id);
                                    }),
                                "StorageBackend init failed");
    }
    if (!result->init()) {
        RTP_LLM_LOG_ERROR("BlockTreeCache init failed");
        return nullptr;
    }
    return result;
}

}  // namespace rtp_llm
