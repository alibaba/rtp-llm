#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {
namespace {

inline int cpEffectiveSeqLenForReserve(const std::shared_ptr<CPSlotMapper>& mapper,
                                       const CacheConfig&                   config,
                                       size_t                               group_index,
                                       int                                  seq_len) {
    return (mapper && mapper->isSharded()) ?
               mapper->effectiveSeqLenForAlloc(config, config.topology().groups().at(group_index).tag, seq_len) :
               seq_len;
}

void appendPoolSummary(std::ostringstream&    os,
                       bool&                  has_any,
                       int                    group_index,
                       const std::string&     tag,
                       CacheGroupType         group_type,
                       const BlockPoolConfig& pool_config) {
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    if (has_any) {
        os << "; ";
    }
    has_any = true;
    os << "pool_name=" << pool_config.pool_name << ", group_index=" << group_index << ", tag=" << tag
       << ", type=" << cacheGroupTypeName(group_type) << ", size=" << pool_config.total_size_bytes << " bytes("
       << std::fixed << std::setprecision(2) << static_cast<double>(pool_config.total_size_bytes) / kBytesPerMB
       << " MB)"
       << ", blocks=" << pool_config.block_num;
}

AllocationType allocationTypeForPlacement(CacheMemoryPlacement placement, AllocationType fallback) {
    if (placement == CacheMemoryPlacement::HOST) {
        return AllocationType::HOST;
    }
    return fallback;
}

bool pinnedCpuBackingForPlacement(CacheMemoryPlacement placement) {
    return placement == CacheMemoryPlacement::HOST_PINNED;
}

}  // namespace

HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio,
                                                       RoleType                           role_type):
    HybridKVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio), role_type_(role_type) {}

bool HybridPoolKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "no cache groups found in CacheConfig");

    const int group_nums = config_.groupNums();
    group_block_pools_.reserve(static_cast<size_t>(group_nums));
    kv_cache_groups_.reserve(static_cast<size_t>(group_nums));

    SharedBlockCache*       shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;
    static constexpr double kBytesPerMB      = 1024.0 * 1024.0;
    std::ostringstream      pool_summary;
    size_t                  pool_total_bytes  = 0;
    size_t                  pool_total_blocks = 0;
    bool                    has_pool          = false;

    std::vector<BlockPoolConfig> group_pool_configs;
    group_pool_configs.reserve(static_cast<size_t>(group_nums));
    for (int group_index = 0; group_index < group_nums; ++group_index) {
        const auto& group       = config_.topology().groups().at(static_cast<size_t>(group_index));
        auto        pool_config = BlockPoolConfigHelper::createConfigForGroup(config_, group.tag);
        const auto& tag         = group.tag;
        const auto  group_type  = group.policy.group_type;
        appendPoolSummary(pool_summary, has_pool, group_index, tag, group_type, pool_config);
        pool_total_bytes += pool_config.total_size_bytes;
        pool_total_blocks += pool_config.block_num;
        group_pool_configs.push_back(std::move(pool_config));
    }

    if (has_pool) {
        const auto summary = pool_summary.str();
        RTP_LLM_LOG_INFO("HybridPool pool summary: pools=[%s], total_size=%zu bytes total_size_mb=%.2f "
                         "total_blocks=%zu",
                         summary.c_str(),
                         pool_total_bytes,
                         static_cast<double>(pool_total_bytes) / kBytesPerMB,
                         pool_total_blocks);
    }

    for (int group_index = 0; group_index < group_nums; ++group_index) {
        const auto& pool_config = group_pool_configs[static_cast<size_t>(group_index)];
        const auto& cache_group = config_.topology().groups().at(static_cast<size_t>(group_index));
        const auto  group_type  = cache_group.policy.group_type;
        const auto  policy      = cache_group.policy;

        auto group_pool =
            std::make_shared<BlockPool>(pool_config,
                                        allocationTypeForPlacement(policy.memory_placement, allocation_type_),
                                        pinnedCpuBackingForPlacement(policy.memory_placement),
                                        use_cuda_malloc_block_pool_);
        RTP_LLM_CHECK_WITH_INFO(group_pool->init(),
                                "Failed to initialize block pool %s(group %d)",
                                pool_config.pool_name.c_str(),
                                group_index);

        KVCacheGroupPtr group;
        if (group_type == CacheGroupType::LINEAR) {
            group = std::make_shared<LinearKVCacheGroup>(
                cache_group, group_pool, config_.linear_step, shared_cache_raw, metrics_reporter_);
            linear_group_indices_.push_back(group_index);
        } else if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(
                cache_group, group_pool, config_.linear_step, shared_cache_raw, metrics_reporter_);
            swa_group_indices_.push_back(group_index);
        } else {
            group = std::make_shared<FullKVCacheGroup>(cache_group, group_pool, shared_cache_raw, metrics_reporter_);
            full_group_indices_.push_back(group_index);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(),
                                "Failed to initialize KVCacheGroup %s(group_index %d)",
                                pool_config.pool_name.c_str(),
                                group_index);
        group_block_pools_.push_back(group_pool);
        kv_cache_groups_.push_back(group);
    }

    if (shared_block_cache_) {
        SharedBlockCache::TaggedBlockPools group_pools;
        for (size_t group_index = 0; group_index < kv_cache_groups_.size(); ++group_index) {
            group_pools.emplace_back(groupTag(static_cast<int>(group_index)), group_block_pools_[group_index]);
        }
        shared_block_cache_->init(group_pools);
    }

    RTP_LLM_LOG_INFO("HybridPoolKVCacheAllocator init success, group pools=%zu", group_block_pools_.size());
    return true;
}

const BlockPoolPtr& HybridPoolKVCacheAllocator::groupPool(std::string_view tag) const {
    return group_block_pools_.at(groupIndex(tag));
}

void HybridPoolKVCacheAllocator::referenceBlocksInGroup(int                     group_index,
                                                        const BlockIndicesType& blocks,
                                                        bool                    is_connector) const {
    if (is_connector) {
        group_block_pools_[static_cast<size_t>(group_index)]->connectorReference(blocks);
    } else {
        group_block_pools_[static_cast<size_t>(group_index)]->requestReference(blocks);
    }
}

void HybridPoolKVCacheAllocator::freeBlocksInGroup(int group_index, const BlockIndicesType& blocks, bool is_connector) {
    if (is_connector) {
        group_block_pools_[static_cast<size_t>(group_index)]->connectorFree(blocks);
    } else {
        group_block_pools_[static_cast<size_t>(group_index)]->requestFree(blocks);
    }
}

GroupedCacheLayerLayout HybridPoolKVCacheAllocator::allLayerCacheBase() const {
    const auto topology = config_.topologyPtr();
    RTP_LLM_CHECK_WITH_INFO(kv_cache_groups_.size() == topology->groups().size(),
                            "cache group count=%zu topology count=%zu",
                            kv_cache_groups_.size(),
                            topology->groups().size());

    GroupedCacheLayerLayout::GroupLayouts groups;
    for (size_t group_index = 0; group_index < kv_cache_groups_.size(); ++group_index) {
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        const auto                      layer_tensors = kv_cache_groups_[group_index]->allLayerCacheBase();
        const auto                      scale_tensors = kv_cache_groups_[group_index]->allLayerScaleCacheBase();
        for (const auto& [layer_id, tensor] : layer_tensors) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                    "layer_id %d out of group kv layout range %zu",
                                    layer_id,
                                    layers.size());
            layers[static_cast<size_t>(layer_id)].kv_addr = tensor;
        }
        for (const auto& [layer_id, tensor] : scale_tensors) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers.size(),
                                    "layer_id %d out of group scale layout range %zu",
                                    layer_id,
                                    layers.size());
            layers[static_cast<size_t>(layer_id)].kv_scale_addr = tensor;
        }
        groups.emplace(groupTag(static_cast<int>(group_index)), CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(groups));
}

BlockAddrInfo HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    const auto& tag = config_.topology().soleGroupForLayer(layer_id).tag;
    return cacheGroup(tag)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    const auto& tag = config_.topology().soleGroupForLayer(layer_id).tag;
    return cacheGroup(tag)->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    const auto& tag = config_.topology().soleGroupForLayer(layer_id).tag;
    return cacheGroup(tag)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo
HybridPoolKVCacheAllocator::convertIndexToAddrByTag(int layer_id, const std::string& tag, int block_id) const {
    (void)config_.groupForLayer(layer_id, tag);
    return cacheGroup(tag)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBufferByTag(int layer_id, const std::string& tag, int block_id) const {
    (void)config_.groupForLayer(layer_id, tag);
    return cacheGroup(tag)->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBufferByTag(
    int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const {
    (void)config_.groupForLayer(layer_id, tag);
    return cacheGroup(tag)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

void HybridPoolKVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    if (end_ptr == begin_ptr) {
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(config_.topology().hasOneGroupPerLayer(),
                            "legacy layer-only block copy requires exactly one cache group per layer");
    std::vector<TaggedBlockIdPair> tagged_mappings;
    tagged_mappings.reserve(static_cast<size_t>(end_ptr - begin_ptr) * config_.topology().groups().size());
    for (const auto& group : config_.topology().groups()) {
        for (auto it = begin_ptr; it != end_ptr; ++it) {
            tagged_mappings.push_back({group.tag, it->src, it->dst});
        }
    }
    blockBatchCopyByTag(tagged_mappings);
}

void HybridPoolKVCacheAllocator::blockBatchCopyByTag(const std::vector<TaggedBlockIdPair>& copy_mapping) {
    if (copy_mapping.empty()) {
        return;
    }

    size_t copy_nums[BatchCopyParams::TYPE_SIZE] = {};
    for (const auto& mapping : copy_mapping) {
        const size_t group_index       = groupIndex(mapping.tag);
        const auto&  group             = config_.topology().groups().at(group_index);
        const auto&  pool              = group_block_pools_.at(group_index);
        const auto   copy_type         = BatchCopyParams::get_copy_type(pool->where(), pool->where());
        const size_t buffers_per_layer = group.kv_scale_stride_bytes > 0 ? 2 : 1;
        copy_nums[copy_type] += group.layer_ids.size() * buffers_per_layer;
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (const auto& mapping : copy_mapping) {
        const size_t group_index         = groupIndex(mapping.tag);
        const auto&  group               = config_.topology().groups().at(group_index);
        const auto&  pool                = group_block_pools_.at(group_index);
        const auto&  cache_group         = kv_cache_groups_.at(group_index);
        const size_t kv_block_size_bytes = group.kv_block_stride_bytes;
        const size_t scale_block_bytes   = group.kv_scale_stride_bytes;
        const auto   copy_type           = BatchCopyParams::get_copy_type(pool->where(), pool->where());

        for (int layer_id : group.layer_ids) {
            auto src_addr_info = cache_group->convertIndexToAddr(layer_id, mapping.src);
            auto dst_addr_info = cache_group->convertIndexToAddr(layer_id, mapping.dst);

            if (!src_addr_info.kv_addr || !dst_addr_info.kv_addr) {
                RTP_LLM_LOG_ERROR("Failed to get block address for pool %s(tag %s) layer %d, src_block %d, "
                                  "dst_block %d",
                                  pool->poolName().c_str(),
                                  mapping.tag.c_str(),
                                  layer_id,
                                  mapping.src,
                                  mapping.dst);
                continue;
            }

            copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);
            if (scale_block_bytes > 0 && src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                copy_params.add(dst_addr_info.kv_scale_addr, src_addr_info.kv_scale_addr, scale_block_bytes, copy_type);
            }
        }
    }

    execBatchCopy(copy_params);
}

size_t HybridPoolKVCacheAllocator::freeBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->freeBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::availableBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->availableBlocksNum();
    }
    return total;
}

BatchKVCacheResourcePtr HybridPoolKVCacheAllocator::popBlocksFromCache(size_t min_blocks_to_free) {
    if (min_blocks_to_free == 0 || !shared_block_cache_) {
        return nullptr;
    }

    auto evict_result = shared_block_cache_->selectAndEvict(min_blocks_to_free);
    if (evict_result.evicted_keys.empty()) {
        return nullptr;
    }
    if (metrics_reporter_) {
        for (const auto& [cache_key, lifetime_ms] : evict_result.evicted_lifetime_ms) {
            RtpLLMCacheEvictionMetricsCollector collector;
            collector.lifetime_ms = lifetime_ms;
            kmonitor::MetricsTags tags("scope", "gpu");
            tags.AddTag("evict_policy",
                        evict_result.evicted_independent_group.count(cache_key) ? "independent" : "chain");
            tags.AddTag("backing", "device");
            metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(&tags,
                                                                                                       &collector);
        }
    }

    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(1);
    batch_resource->initGroups(config_.topologyPtr());
    batch_resource->setLastBlockAligned(true);

    for (int group_index = 0; group_index < config_.groupNums(); ++group_index) {
        batch_resource->mutableBlockIdsByIndex(0, static_cast<size_t>(group_index))
            .resize(evict_result.evicted_keys.size(), NULL_BLOCK_IDX);
    }

    CacheKeysType         evicted_keys;
    BlockDependenciesType evicted_dependencies;
    evicted_keys.reserve(evict_result.evicted_keys.size());
    evicted_dependencies.reserve(evict_result.evicted_keys.size());
    for (size_t evicted_idx = 0; evicted_idx < evict_result.evicted_keys.size(); ++evicted_idx) {
        const auto  cache_key       = evict_result.evicted_keys[evicted_idx];
        const auto& group_block_ids = evict_result.evicted_group_block_ids.at(cache_key);
        evicted_keys.push_back(cache_key);
        auto dep_it = evict_result.evicted_dependencies.find(cache_key);
        if (dep_it != evict_result.evicted_dependencies.end()) {
            evicted_dependencies.push_back(dep_it->second);
        } else {
            BlockDependency dependency;
            dependency.ordinal = static_cast<uint32_t>(evicted_idx);
            if (evicted_idx > 0) {
                dependency.has_parent = true;
                dependency.parent_key = evict_result.evicted_keys[evicted_idx - 1];
            }
            evicted_dependencies.push_back(dependency);
        }
        for (const auto& [tag, block_id] : group_block_ids) {
            if (!isNullBlockIdx(block_id)) {
                batch_resource->mutableBlockIds(0, tag).setAt(evicted_idx, block_id);
            }
        }
    }
    batch_resource->cacheResource(0).setCacheKeys(std::move(evicted_keys));
    batch_resource->cacheResource(0).setBlockDependencies(std::move(evicted_dependencies));
    // Evicted keys already come from the GPU cache's actual key namespace.
    // Under CP this can be a mixed batch of canonical paged keys and logical
    // state/SWA keys, so coordinator must not remap the whole batch again.
    batch_resource->cacheResource(0).setCacheKeysAreCpCanonical(true);
    return batch_resource;
}

void HybridPoolKVCacheAllocator::blockCacheFree(const BatchKVCacheResourcePtr& batch_kv_cache_resource) {
    if (!batch_kv_cache_resource) {
        return;
    }
    for (int batch_id = 0; batch_id < batch_kv_cache_resource->batchSize(); ++batch_id) {
        const auto& group_blocks = batch_kv_cache_resource->groupBlocks(batch_id);
        for (size_t group_index = 0; group_index < group_blocks.size(); ++group_index) {
            const auto&                      block_ids = group_blocks[group_index];
            BlockIndicesType                 blocks_to_free;
            std::unordered_set<BlockIdxType> seen_blocks;
            for (auto block_idx : block_ids->blocks()) {
                if (isNullBlockIdx(block_idx) || !seen_blocks.insert(block_idx).second) {
                    continue;
                }
                blocks_to_free.push_back(block_idx);
            }
            if (!blocks_to_free.empty()) {
                group_block_pools_.at(group_index)->blockCacheFree(blocks_to_free);
            }
        }
    }
}

size_t HybridPoolKVCacheAllocator::requestRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->requestRefBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::connectorRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->connectorRefBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::blockCacheRefBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->blockCacheRefBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::notInUseBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->notInUseBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::minTokenCapacity(bool use_available_blocks, bool full_groups_only) const {
    if (group_block_pools_.empty()) {
        return 0;
    }

    auto calculate = [&](bool only_full_groups) {
        size_t min_tokens = std::numeric_limits<size_t>::max();
        bool   saw_group  = false;
        for (size_t group_index = 0; group_index < group_block_pools_.size(); ++group_index) {
            const auto& group = config_.topology().groups().at(group_index);
            if (only_full_groups && group.policy.group_type != CacheGroupType::FULL) {
                continue;
            }
            if (!group_block_pools_[group_index]) {
                continue;
            }
            saw_group        = true;
            const auto block = use_available_blocks ? group_block_pools_[group_index]->availableBlocksNum() :
                                                      group_block_pools_[group_index]->totalBlocksNum();
            min_tokens       = std::min(min_tokens, block * logicalSeqSizePerBlockForCapacity(group.tag));
        }
        return std::make_pair(saw_group, min_tokens);
    };

    if (full_groups_only) {
        const auto [saw_full_group, min_tokens] = calculate(/*only_full_groups=*/true);
        if (saw_full_group) {
            return min_tokens;
        }
    }

    const auto [saw_group, min_tokens] = calculate(/*only_full_groups=*/false);
    return saw_group ? min_tokens : 0;
}

size_t HybridPoolKVCacheAllocator::availableTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/true, /*full_groups_only=*/true);
}

size_t HybridPoolKVCacheAllocator::totalTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/false, /*full_groups_only=*/true);
}

size_t HybridPoolKVCacheAllocator::totalBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->totalBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::maxAvailableTokensNum() const {
    return minTokenCapacity(/*use_available_blocks=*/false, /*full_groups_only=*/true);
}

KVCacheTokenCapacity HybridPoolKVCacheAllocator::tokenCapacity(size_t default_seq_size_per_block) const {
    (void)default_seq_size_per_block;
    if (group_block_pools_.empty()) {
        return {};
    }
    size_t total_tokens     = std::numeric_limits<size_t>::max();
    size_t available_tokens = std::numeric_limits<size_t>::max();
    bool   has_pool         = false;
    for (size_t group_index = 0; group_index < group_block_pools_.size(); ++group_index) {
        const auto& pool = group_block_pools_[group_index];
        if (!pool) {
            continue;
        }
        const size_t seq_size = config_.topology().groups().at(group_index).seq_size_per_block;
        total_tokens          = std::min(total_tokens, pool->totalBlocksNum() * seq_size);
        available_tokens      = std::min(available_tokens, pool->availableBlocksNum() * seq_size);
        has_pool              = true;
    }
    return has_pool ? KVCacheTokenCapacity{total_tokens, available_tokens} : KVCacheTokenCapacity{};
}

std::vector<KVCachePoolMetricsSnapshot> HybridPoolKVCacheAllocator::poolMetricsSnapshots() const {
    std::vector<KVCachePoolMetricsSnapshot> snapshots;
    snapshots.reserve(group_block_pools_.size());
    const size_t reserve_blocks                    = reserveBlocksNum();
    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();
    for (size_t group_index = 0; group_index < group_block_pools_.size(); ++group_index) {
        const auto& pool = group_block_pools_[group_index];
        if (!pool) {
            continue;
        }
        KVCachePoolMetricsSnapshot snapshot;
        snapshot.pool_index           = group_index;
        snapshot.pool_name            = pool->poolName();
        snapshot.total_blocks         = pool->totalBlocksNum();
        snapshot.available_blocks     = pool->availableBlocksNum();
        snapshot.free_blocks          = pool->freeBlocksNum();
        snapshot.request_ref_blocks   = pool->requestRefBlocksNum();
        snapshot.connector_ref_blocks = pool->connectorRefBlocksNum();
        snapshot.reserve_blocks = reserveBlocksForPool(group_index, reserve_blocks, total_reservable_available_blocks);
        snapshot.used_ratio     = (snapshot.total_blocks == 0) ?
                                      0.0f :
                                      static_cast<float>(100.0 * (snapshot.total_blocks - snapshot.available_blocks)
                                                     / static_cast<double>(snapshot.total_blocks));
        snapshots.push_back(snapshot);
    }
    return snapshots;
}

void HybridPoolKVCacheAllocator::regUserMr(size_t model_id, std::shared_ptr<CacheStore> cache_store) {
    for (auto& pool : group_block_pools_) {
        pool->regUserMr(model_id, cache_store);
    }
}

int64_t HybridPoolKVCacheAllocator::getMrCostTimeMs() const {
    int64_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->getMrCostTimeMs();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::totalReservableAvailableBlocks() const {
    size_t total = 0;
    for (size_t group_index = 0; group_index < group_block_pools_.size(); ++group_index) {
        if (!group_block_pools_[group_index]
            || config_.usesExplicitIndependentBlocks(config_.topology().groups().at(group_index).tag)) {
            continue;
        }
        total += group_block_pools_[group_index]->availableBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::reservableAvailableBlocksNum() const {
    return totalReservableAvailableBlocks();
}

size_t HybridPoolKVCacheAllocator::reserveBlocksForPool(size_t group_index,
                                                        size_t reserve_blocks,
                                                        size_t total_reservable_available_blocks) const {
    if (group_index >= group_block_pools_.size() || !group_block_pools_[group_index]
        || config_.usesExplicitIndependentBlocks(config_.topology().groups().at(group_index).tag)
        || total_reservable_available_blocks == 0) {
        return 0;
    }
    return reserve_blocks * group_block_pools_[group_index]->availableBlocksNum() / total_reservable_available_blocks;
}

bool HybridPoolKVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info,
                                                              size_t            reserve_blocks) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return true;
    }
    const auto& cp_mapper          = cp_slot_mapper_;
    const int   batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int   total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int   raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int   raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool  reuse_enabled      = malloc_info.reuse_cache;

    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();

    for (int group_index = 0; group_index < static_cast<int>(kv_cache_groups_.size()); ++group_index) {
        const int group_common_seq =
            cpEffectiveSeqLenForReserve(cp_mapper, config_, static_cast<size_t>(group_index), raw_common_seq_len);
        const int group_seq_len =
            cpEffectiveSeqLenForReserve(cp_mapper, config_, static_cast<size_t>(group_index), raw_seq_len);
        const int group_reuse_blocks_len =
            reuse_enabled ? malloc_info.batch_kv_cache_resource->blocksNumByIndex(0, static_cast<size_t>(group_index)) :
                            0;
        const auto need = kv_cache_groups_[static_cast<size_t>(group_index)]->getNeedBlocks(
            group_common_seq, group_seq_len, reserve_step, group_reuse_blocks_len, reuse_enabled);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        if (need_blocks <= 0) {
            continue;
        }
        const auto&  pool             = group_block_pools_[static_cast<size_t>(group_index)];
        const size_t available_blocks = pool->availableBlocksNum();
        const size_t total_blocks     = pool->totalBlocksNum();
        const size_t group_reserve_blocks =
            reserveBlocksForPool(static_cast<size_t>(group_index), reserve_blocks, total_reservable_available_blocks);
        if (available_blocks < static_cast<size_t>(need_blocks) + group_reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc rejected by reserve blocks: request_id=%ld pool_name=%s "
                                 "group=%d need_blocks=%d total_blocks=%zu available_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 group_index,
                                 need_blocks,
                                 total_blocks,
                                 available_blocks,
                                 reserve_blocks,
                                 group_reserve_blocks);
            }
            return false;
        }
    }
    return true;
}

}  // namespace rtp_llm
