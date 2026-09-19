#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCacheFactory.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
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

size_t checkedSize(int64_t value, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(value > 0, "%s must be > 0, got %ld", name, value);
    return static_cast<size_t>(value);
}

size_t checkedQueueSize(int64_t value, const char* name) {
    RTP_LLM_CHECK_WITH_INFO(value >= 0, "%s must be >= 0, got %ld", name, value);
    return static_cast<size_t>(value);
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
            const auto seq_size = group.cacheKeyTokenStride();
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
    const auto&                  topology_groups = cache_config.topology().groups();
    for (const auto& group : allocator_groups) {
        if (!group || !group->blockPool()) {
            RTP_LLM_LOG_ERROR("allocator group/direct pool must be non-null");
            return {};
        }
        const auto& tag   = group->tag();
        const auto  found = std::find_if(topology_groups.begin(),
                                        topology_groups.end(),
                                        [&tag](const GroupBase& declared) { return declared.tag == tag; });
        if (found == topology_groups.end()) {
            RTP_LLM_LOG_ERROR("allocator has unknown group tag=%s", tag.c_str());
            return {};
        }
        const size_t group_id      = static_cast<size_t>(std::distance(topology_groups.begin(), found));
        auto&        aligned_group = aligned[group_id];
        if (aligned_group != nullptr) {
            RTP_LLM_LOG_ERROR("duplicate allocator group tag=%s", tag.c_str());
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
        const auto& declared = topology_groups[group_id];
        if (actual.spec != declared.spec || !CacheConfig::samePolicy(actual.policy, declared.policy)
            || actual.block_num != declared.block_num) {
            RTP_LLM_LOG_ERROR("allocator group_id=%zu does not exactly match topology", group_id);
            return {};
        }
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

bool groupSetSemanticsCompatible(const CacheConfig& cache_config, std::string_view lhs_tag, std::string_view rhs_tag) {
    const GroupBase& lhs = cache_config.topology().group(lhs_tag);
    const GroupBase& rhs = cache_config.topology().group(rhs_tag);
    if (lhs.policy.group_type != rhs.policy.group_type || lhs.seqSizePerBlock() != rhs.seqSizePerBlock()
        || lhs.policy.cp_mapping != rhs.policy.cp_mapping) {
        return false;
    }
    if (lhs.policy.group_type == CacheGroupType::SWA) {
        if (slidingWindowSize(lhs, cache_config.topology().groupIdForTag(lhs_tag))
            != slidingWindowSize(rhs, cache_config.topology().groupIdForTag(rhs_tag))) {
            return false;
        }
    }
    if (lhs.policy.group_type == CacheGroupType::SWA || lhs.policy.group_type == CacheGroupType::LINEAR) {
        if (lhs.policy.active_tail_blocks != rhs.policy.active_tail_blocks) {
            return false;
        }
    }
    return true;
}

bool buildGroupMembers(const CacheConfig& cache_config, std::vector<std::vector<std::string>>& group_members) {
    group_members.clear();
    for (const GroupBase& group : cache_config.topology().groups()) {
        if (!group.policy.enable_prefix_reuse) {
            continue;
        }

        const CacheGroupType group_type = group.policy.group_type;
        if (group_type != CacheGroupType::FULL && group_type != CacheGroupType::SWA
            && group_type != CacheGroupType::LINEAR) {
            RTP_LLM_LOG_ERROR("unsupported reusable cache group type for tag=%s", group.tag.c_str());
            return false;
        }

        auto it =
            std::find_if(group_members.begin(), group_members.end(), [&](const std::vector<std::string>& members) {
                const GroupBase& first = cache_config.topology().group(members.front());
                return first.policy.group_type == group.policy.group_type;
            });
        if (it == group_members.end()) {
            group_members.push_back({group.tag});
        } else {
            if (!groupSetSemanticsCompatible(cache_config, it->front(), group.tag)) {
                RTP_LLM_LOG_ERROR("incompatible BlockTree reuse coordinates for tags=%s and %s",
                                  it->front().c_str(),
                                  group.tag.c_str());
                return false;
            }
            it->push_back(group.tag);
        }
    }
    return true;
}

size_t computeGroupSetPayloadBytes(const CacheConfig& cache_config, const std::vector<std::string>& members) {
    size_t payload_bytes = 0;
    for (const auto& tag : members) {
        const size_t group_bytes = cache_config.blockSizeBytesForGroup(tag);
        RTP_LLM_CHECK_WITH_INFO(group_bytes > 0, "tag=%s has zero payload", tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group_bytes <= std::numeric_limits<size_t>::max() - payload_bytes,
                                "group set payload overflow at tag=%s",
                                tag.c_str());
        payload_bytes += group_bytes;
    }
    return payload_bytes;
}

std::vector<BlockInfo> resolveStorageBuffers(const CacheConfig&                     cache_config,
                                             const std::vector<DeviceBlockPoolPtr>& group_pools,
                                             int                                    layer_id,
                                             const std::string&                     tag,
                                             int                                    block_id) {
    const auto& topology = cache_config.topology();
    const auto  group_id = topology.groupIdForTag(tag);
    RTP_LLM_CHECK_WITH_INFO(group_id < group_pools.size(), "invalid storage tag=%.*s", (int)tag.size(), tag.data());
    const auto layer_ids = topology.layerIdsForGroup(tag);
    const auto layer     = std::find(layer_ids.begin(), layer_ids.end(), layer_id);
    RTP_LLM_CHECK_WITH_INFO(
        layer != layer_ids.end(), "layer_id=%d does not belong to storage group_id=%d", layer_id, group_id);
    // Pools are laid out in group-local layer order. This is the same model-global -> pool-layer mapping used by
    // KVCacheGroup::convertIndexToBuffer; model layer IDs cannot be passed
    // directly to these physical pools.
    auto buffers = group_pools[group_id]->convertIndexToBuffer(
        static_cast<int>(std::distance(layer_ids.begin(), layer)), block_id);
    const auto& physical_group = cache_config.physicalGroupForLayer(layer_id, tag);
    RTP_LLM_CHECK_WITH_INFO(!buffers.empty(), "storage group_id=%d returned no block buffers", group_id);
    RTP_LLM_CHECK_WITH_INFO(buffers[0].size_bytes >= physical_group.kvBlockStrideBytes(),
                            "storage group_id=%d physical kv block is smaller than logical block",
                            group_id);
    buffers[0].size_bytes = physical_group.kvBlockStrideBytes();
    if (physical_group.kvScaleStrideBytes() == 0) {
        buffers.resize(1);
        return buffers;
    }
    RTP_LLM_CHECK_WITH_INFO(buffers.size() >= 2 && buffers[1].size_bytes >= physical_group.kvScaleStrideBytes(),
                            "storage group_id=%d has an invalid scale block buffer",
                            group_id);
    buffers[1].size_bytes = physical_group.kvScaleStrideBytes();
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

BlockTreeCachePtr createBlockTreeCache(const CacheConfig&                         cache_config,
                                       const KVCacheConfig&                       kv_cache_config,
                                       const KVCacheAllocatorPtr&                 allocator,
                                       const ParallelismConfig&                   parallelism_config,
                                       std::shared_ptr<StorageBackend>            storage_backend,
                                       std::shared_ptr<BroadcastManager>          broadcast_manager,
                                       std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter) {
    const auto device_eviction_policy = parseEvictionPolicy(kv_cache_config.device_eviction_policy);
    const auto host_eviction_policy   = parseEvictionPolicy(kv_cache_config.memory_eviction_policy);
    const auto disk_eviction_policy   = parseEvictionPolicy(kv_cache_config.disk_eviction_policy);
    if (!device_eviction_policy.has_value() || !host_eviction_policy.has_value() || !disk_eviction_policy.has_value()) {
        RTP_LLM_LOG_ERROR("createBlockTreeCache: unsupported eviction policy, device=%s memory=%s disk=%s",
                          kv_cache_config.device_eviction_policy.c_str(),
                          kv_cache_config.memory_eviction_policy.c_str(),
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
    if (independent_pools.size() != static_cast<size_t>(group_count)) {
        RTP_LLM_LOG_ERROR(
            "independent pool/topology count mismatch, pools=%zu topology=%d", independent_pools.size(), group_count);
        return nullptr;
    }
    std::unordered_set<const DeviceBlockPool*> unique_pools;
    for (const auto& pool : independent_pools) {
        if (!pool || !unique_pools.insert(pool.get()).second) {
            RTP_LLM_LOG_ERROR("each cache group must own a distinct non-null device pool");
            return nullptr;
        }
    }
    for (int group_id = 0; group_id < group_count; ++group_id) {
        auto        pool = groups[static_cast<size_t>(group_id)]->blockPool();
        const auto& tag  = cache_config.topology().groups()[static_cast<size_t>(group_id)].tag;
        if (!pool || pool->poolName() != tag || unique_pools.erase(pool.get()) != 1) {
            RTP_LLM_LOG_ERROR("allocator/group direct pool mismatch for tag=%s", tag.c_str());
            return nullptr;
        }
        group_pools[static_cast<size_t>(group_id)] = std::move(pool);
    }

    const bool       host_enabled = kv_cache_config.enable_memory_cache;
    const bool       disk_enabled = kv_cache_config.enable_disk_cache;
    constexpr size_t bytes_per_mb = 1024UL * 1024UL;
    const auto       valid_budget = [](int64_t mb) {
        return mb > 0 && static_cast<uint64_t>(mb) <= std::numeric_limits<size_t>::max() / bytes_per_mb;
    };
    if (host_enabled && !valid_budget(kv_cache_config.memory_cache_size_mb)) {
        RTP_LLM_LOG_ERROR("memory cache size must be positive and fit in bytes");
        return nullptr;
    }
    if (disk_enabled && !valid_budget(kv_cache_config.disk_cache_size_mb)) {
        RTP_LLM_LOG_ERROR("disk cache size must be positive and fit in bytes");
        return nullptr;
    }

    std::vector<GroupSetPtr>              group_sets;
    std::vector<std::vector<std::string>> group_members;
    if (!buildGroupMembers(cache_config, group_members)) {
        return nullptr;
    }
    if (disk_enabled) {
        for (const std::vector<std::string>& members : group_members) {
            RTP_LLM_CHECK_WITH_INFO(!members.empty(), "BlockTreeCache aggregation plan contains an empty group set");
            if (cache_config.topology().group(members.front()).policy.group_type == CacheGroupType::LINEAR) {
                RTP_LLM_LOG_ERROR("disk cache does not support reusable LINEAR group sets, tag=%s",
                                  members.front().c_str());
                return nullptr;
            }
        }
    }

    std::vector<size_t> group_set_payload_bytes;
    group_set_payload_bytes.reserve(group_members.size());
    size_t combined_stride = 0;
    for (const std::vector<std::string>& members : group_members) {
        const size_t payload_bytes = computeGroupSetPayloadBytes(cache_config, members);
        group_set_payload_bytes.push_back(payload_bytes);
        const size_t stride = alignUp(payload_bytes, kPoolAlignment);
        RTP_LLM_CHECK_WITH_INFO(stride <= std::numeric_limits<size_t>::max() - combined_stride,
                                "BlockTreeCache combined lower-tier stride overflow");
        combined_stride += stride;
    }

    std::vector<std::shared_ptr<HostBlockPool>> host_pools(group_members.size());
    if (host_enabled && !group_members.empty()) {
        const size_t bytes  = static_cast<size_t>(kv_cache_config.memory_cache_size_mb) * 1024UL * 1024UL;
        const size_t usable = computeHostUsableBlockCount(bytes, combined_stride);
        if (usable == 0) {
            RTP_LLM_LOG_ERROR("host budget is too small for one complete tree coordinate");
            return nullptr;
        }
        for (size_t group_set_id = 0; group_set_id < group_members.size(); ++group_set_id) {
            const std::vector<std::string>& members = group_members[group_set_id];
            const GroupBase&                first   = cache_config.topology().group(members.front());
            const std::string               pool_name =
                "block_tree_host_" + std::string(metricCacheGroupTypeName(first.policy.group_type));
            host_pools[group_set_id] = createHostPool(pool_name, group_set_payload_bytes[group_set_id], usable);
            if (!host_pools[group_set_id]) {
                return nullptr;
            }
        }
    }

    std::vector<BlockTreeDiskBlockPoolPtr> disk_pools(group_members.size());
    if (disk_enabled && !group_members.empty()) {
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
        for (size_t group_set_id = 0; group_set_id < group_members.size(); ++group_set_id) {
            const std::vector<std::string>& members = group_members[group_set_id];
            const GroupBase&                first   = cache_config.topology().group(members.front());
            const std::string               pool_name =
                "block_tree_disk_" + std::string(metricCacheGroupTypeName(first.policy.group_type));
            disk_pools[group_set_id] = createDiskPool(kv_cache_config,
                                                      guard,
                                                      pool_name,
                                                      group_set_payload_bytes[group_set_id],
                                                      usable,
                                                      parallelism_config.world_rank,
                                                      parallelism_config.local_rank);
            if (!disk_pools[group_set_id]) {
                return nullptr;
            }
        }
    }

    // Canonical CP keys cover a virtual token block. Keep the physical layout
    // unchanged and publish the key stride only to cache matching/remote storage.
    auto cache_topology = cache_config.topologyPtr();
    if (const auto mapper = allocator->cpSlotMapper(); mapper && mapper->isSharded()) {
        auto         groups  = cache_topology->groups();
        const size_t cp_size = static_cast<size_t>(mapper->cpSize());
        RTP_LLM_CHECK_WITH_INFO(cache_config.seq_size_per_block <= std::numeric_limits<size_t>::max() / cp_size,
                                "canonical CP cache key stride overflow");
        for (auto& group : groups) {
            auto spec                    = group.spec->clone();
            spec->cache_key_token_stride = cache_config.seq_size_per_block * cp_size;
            group.spec                   = std::move(spec);
        }
        cache_topology = CacheTopology::create(std::move(groups), cache_topology->layers());
    }
    group_sets.reserve(group_members.size());
    for (size_t group_set_id = 0; group_set_id < group_members.size(); ++group_set_id) {
        const std::vector<std::string>& members = group_members[group_set_id];
        std::vector<DeviceBlockPoolPtr> device_pools;
        device_pools.reserve(members.size());
        for (const auto& tag : members) {
            const size_t group_id = cache_config.topology().groupIdForTag(tag);
            device_pools.push_back(group_pools[group_id]);
        }
        const auto&  first     = cache_topology->group(members.front());
        const size_t first_id  = cache_topology->groupIdForTag(first.tag);
        auto         group_set = createGroupSet(first,
                                        first_id,
                                        std::move(device_pools),
                                        std::move(host_pools[group_set_id]),
                                        std::move(disk_pools[group_set_id]));
        group_set->initialize(group_set_id, cache_topology, members, group_set_payload_bytes[group_set_id]);
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
        config.watermark_device = {kv_cache_config.block_tree_device_evict_low_watermark_ratio,
                                   kv_cache_config.block_tree_device_evict_high_watermark_ratio};
    }
    if (host_enabled) {
        config.watermark_host = {kv_cache_config.block_tree_memory_evict_low_watermark_ratio,
                                 kv_cache_config.block_tree_memory_evict_high_watermark_ratio};
    }
    if (disk_enabled) {
        config.watermark_disk = {kv_cache_config.block_tree_disk_evict_low_watermark_ratio,
                                 kv_cache_config.block_tree_disk_evict_high_watermark_ratio};
    }
    const auto valid_watermark = [](const TierWatermark& watermark) {
        return (watermark.low_ratio == 0.0 && watermark.high_ratio == 0.0)
               || (watermark.low_ratio > 0.0 && watermark.low_ratio < watermark.high_ratio
                   && watermark.high_ratio <= 1.0);
    };
    for (const auto& tier_watermark : {std::pair<Tier, TierWatermark>{Tier::DEVICE, config.watermark_device},
                                       std::pair<Tier, TierWatermark>{Tier::HOST, config.watermark_host},
                                       std::pair<Tier, TierWatermark>{Tier::DISK, config.watermark_disk}}) {
        if (!valid_watermark(tier_watermark.second)) {
            RTP_LLM_LOG_ERROR("invalid cache watermark: tier=%s low_ratio=%f high_ratio=%f",
                              tierName(tier_watermark.first),
                              tier_watermark.second.low_ratio,
                              tier_watermark.second.high_ratio);
            return nullptr;
        }
    }
    config.host_cache_sync_timeout_ms =
        checkedTimeout(kv_cache_config.memory_cache_sync_timeout_ms, "memory_cache_sync_timeout_ms");
    config.disk_cache_sync_timeout_ms =
        disk_enabled ? checkedTimeout(kv_cache_config.disk_cache_sync_timeout_ms, "disk_cache_sync_timeout_ms") :
                       config.host_cache_sync_timeout_ms;
    config.transfer_worker_count =
        checkedSize(kv_cache_config.block_tree_transfer_worker_count, "block_tree_transfer_worker_count");
    config.business_queue_max_size =
        checkedQueueSize(kv_cache_config.block_tree_business_queue_max_size, "block_tree_business_queue_max_size");
    config.transfer_queue_max_size =
        checkedQueueSize(kv_cache_config.block_tree_transfer_queue_max_size, "block_tree_transfer_queue_max_size");

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

    const int64_t configured_scan_interval_ms = kv_cache_config.block_tree_full_prefix_scan_interval_ms;
    if (configured_scan_interval_ms < 0 || (configured_scan_interval_ms > 0 && configured_scan_interval_ms < 1000)
        || configured_scan_interval_ms > std::numeric_limits<int64_t>::max() / 1000) {
        RTP_LLM_LOG_ERROR("block_tree_full_prefix_scan_interval_ms must be 0 or >= 1000, got %ld",
                          configured_scan_interval_ms);
        return nullptr;
    }
    // Only tp_rank 0 of a non-FFN service drives normal request scheduling, so it is the
    // single rank per DP group that owns a mutable BlockTree worth scanning.
    const bool owns_mutable_block_tree =
        parallelism_config.tp_rank == 0 && !parallelism_config.ffn_disaggregate_config.is_ffn_service();
    config.full_prefix_scan_interval_ms = owns_mutable_block_tree ? configured_scan_interval_ms : 0;

    auto cache_metrics_reporter = std::make_shared<BlockTreeCacheMetricsReporter>(std::move(metrics_reporter));
    auto per_rank_engine        = std::make_shared<PerRankBlockTransferEngine>(group_sets,
                                                                        kv_cache_config.enable_disk_cache,
                                                                        DeviceHostCopyOptions{},
                                                                        config.device_disk_staging_block_count,
                                                                        config.max_descriptors_per_transfer_batch,
                                                                        config.transfer_worker_count,
                                                                        config.transfer_queue_max_size,
                                                                        cache_metrics_reporter);
    std::shared_ptr<MultiRankBlockTransferEngine> multi_rank_engine;
    if (broadcast_manager != nullptr) {
        multi_rank_engine = std::make_shared<MultiRankBlockTransferEngine>(group_sets, std::move(broadcast_manager));
    }
    auto transfer_dispatcher = std::make_unique<BlockTransferDispatcher>(
        std::move(per_rank_engine), std::move(multi_rank_engine), config.max_descriptors_per_transfer_batch);
    const size_t business_queue_size = config.business_queue_max_size == 0 ?
                                           0 :
                                           config.business_queue_max_size + BlockTreeTaskPool::kLoadReservedSlots;
    auto         task_pool           = std::make_unique<BlockTreeTaskPool>(
        static_cast<size_t>(config.task_pool_size), business_queue_size, "BlockTreeCacheTaskPool");

    auto tree = std::make_unique<BlockTree>(std::move(group_sets));

    auto result = std::make_shared<BlockTreeCache>(std::move(tree),
                                                   std::move(config),
                                                   std::move(storage_backend),
                                                   std::move(transfer_dispatcher),
                                                   std::move(task_pool),
                                                   std::move(cache_metrics_reporter));
    if (result->isRemoteCacheEnabled()) {
        const std::shared_ptr<const CacheTopology> storage_topology = cache_topology;
        const CacheConfig                          storage_config   = cache_config;
        const std::vector<DeviceBlockPoolPtr>      resolver_pools   = group_pools;
        RTP_LLM_CHECK_WITH_INFO(
            result->storageBackend()->init(
                storage_topology,
                group_pools,
                [storage_config, resolver_pools](int layer_id, const std::string& tag, int block_id) {
                    return resolveStorageBuffers(storage_config, resolver_pools, layer_id, tag, block_id);
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
