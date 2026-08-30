#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <map>
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
                                       std::string_view                     tag,
                                       int                                  seq_len) {
    return (mapper && mapper->isSharded()) ? mapper->effectiveSeqLenForAlloc(config, tag, seq_len) : seq_len;
}

void appendPoolSummary(std::ostringstream&    os,
                       bool&                  has_any,
                       size_t                 idx,
                       const std::string&     tag,
                       CacheGroupType         group_type,
                       const BlockPoolConfig& pool_config) {
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    if (has_any) {
        os << "; ";
    }
    has_any = true;
    os << "pool_name=" << pool_config.pool_name << ", idx=" << idx << ", tag=" << tag
       << ", type=" << cacheGroupTypeName(group_type) << ", size=" << pool_config.total_size_bytes << " bytes("
       << std::fixed << std::setprecision(2) << static_cast<double>(pool_config.total_size_bytes) / kBytesPerMB
       << " MB)"
       << ", blocks=" << pool_config.block_num;
}

}  // namespace

HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio,
                                                       RoleType                           role_type):
    KVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio), role_type_(role_type) {}

BlockPoolPtr HybridPoolKVCacheAllocator::soleGroupBlockPool() const {
    RTP_LLM_CHECK_WITH_INFO(group_block_pools_.size() == 1,
                            "sole group block pool requires exactly one initialized group pool, got %zu",
                            group_block_pools_.size());
    return group_block_pools_[0];
}

size_t HybridPoolKVCacheAllocator::storageIdxForTag(std::string_view tag) const {
    const auto it = tag_to_idx_.find(std::string(tag));
    RTP_LLM_CHECK_WITH_INFO(it != tag_to_idx_.end(), "missing allocator cache group tag=%s", std::string(tag).c_str());
    return it->second;
}

const KVCacheGroupPtr& HybridPoolKVCacheAllocator::groupStrategy(std::string_view tag) const {
    const auto idx = storageIdxForTag(tag);
    RTP_LLM_CHECK_WITH_INFO(
        idx < kv_cache_groups_.size(), "missing cache group strategy for tag=%s", std::string(tag).c_str());
    return kv_cache_groups_[idx];
}

bool HybridPoolKVCacheAllocator::initGroup(const KVCacheGroupPtr& group) {
    return group->init();
}

BlockPoolPtr HybridPoolKVCacheAllocator::blockPool(std::string_view tag) const {
    const auto idx = storageIdxForTag(tag);
    RTP_LLM_CHECK_WITH_INFO(idx < group_block_pools_.size(), "missing block pool for tag=%s", std::string(tag).c_str());
    return group_block_pools_[idx];
}

bool HybridPoolKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "no cache groups found in CacheConfig");

    const int                               group_nums = config_.groupNums();
    std::vector<BlockPoolPtr>               staged_group_block_pools;
    std::vector<KVCacheGroupPtr>            staged_kv_cache_groups;
    std::unordered_map<std::string, size_t> staged_tag_to_idx;
    std::vector<std::string>                staged_full_group_tags;
    std::vector<std::string>                staged_linear_group_tags;
    std::vector<std::string>                staged_swa_group_tags;
    staged_group_block_pools.reserve(static_cast<size_t>(group_nums));
    staged_kv_cache_groups.reserve(static_cast<size_t>(group_nums));
    staged_tag_to_idx.reserve(static_cast<size_t>(group_nums));
    staged_full_group_tags.reserve(static_cast<size_t>(group_nums));
    staged_linear_group_tags.reserve(static_cast<size_t>(group_nums));
    staged_swa_group_tags.reserve(static_cast<size_t>(group_nums));

    SharedBlockCache*       shared_cache_raw = shared_block_cache_ ? shared_block_cache_.get() : nullptr;
    static constexpr double kBytesPerMB      = 1024.0 * 1024.0;
    std::ostringstream      pool_summary;
    size_t                  pool_total_bytes  = 0;
    size_t                  pool_total_blocks = 0;
    bool                    has_pool          = false;

    std::vector<BlockPoolConfig> group_pool_configs;
    group_pool_configs.reserve(static_cast<size_t>(group_nums));
    for (size_t idx = 0; idx < config_.groups().size(); ++idx) {
        const auto& group       = config_.groups()[idx];
        auto        pool_config = BlockPoolConfigHelper::createConfigForGroup(config_, group.tag);
        appendPoolSummary(pool_summary, has_pool, idx, group.tag, group.policy.group_type, pool_config);
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

    for (size_t idx = 0; idx < config_.groups().size(); ++idx) {
        const auto& pool_config = group_pool_configs[idx];
        const auto& cache_group = config_.groups()[idx];
        const auto  group_type  = cache_group.policy.group_type;
        auto        group_pool =
            std::make_shared<BlockPool>(pool_config, allocation_type_, false, use_cuda_malloc_block_pool_);
        RTP_LLM_CHECK_WITH_INFO(
            group_pool->init(), "Failed to initialize block pool %s", pool_config.pool_name.c_str());

        RTP_LLM_CHECK_WITH_INFO(staged_tag_to_idx.emplace(cache_group.tag, idx).second,
                                "duplicate allocator cache group tag=%s",
                                cache_group.tag.c_str());

        KVCacheGroupPtr group;
        if (group_type == CacheGroupType::LINEAR) {
            group = std::make_shared<LinearKVCacheGroup>(cache_group,
                                                         config_.groupLayerIds(cache_group.tag),
                                                         group_pool,
                                                         config_.linear_step,
                                                         shared_cache_raw,
                                                         metrics_reporter_);
            staged_linear_group_tags.push_back(cache_group.tag);
        } else if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(cache_group,
                                                      config_.groupLayerIds(cache_group.tag),
                                                      group_pool,
                                                      config_.linear_step,
                                                      shared_cache_raw,
                                                      metrics_reporter_);
            staged_swa_group_tags.push_back(cache_group.tag);
        } else {
            group = std::make_shared<FullKVCacheGroup>(
                cache_group, config_.groupLayerIds(cache_group.tag), group_pool, shared_cache_raw, metrics_reporter_);
            staged_full_group_tags.push_back(cache_group.tag);
        }

        RTP_LLM_CHECK_WITH_INFO(
            initGroup(group), "Failed to initialize KVCacheGroup %s", pool_config.pool_name.c_str());
        staged_group_block_pools.push_back(group_pool);
        staged_kv_cache_groups.push_back(group);
    }

    group_block_pools_.swap(staged_group_block_pools);
    kv_cache_groups_.swap(staged_kv_cache_groups);
    tag_to_idx_.swap(staged_tag_to_idx);
    full_group_tags_.swap(staged_full_group_tags);
    linear_group_tags_.swap(staged_linear_group_tags);
    swa_group_tags_.swap(staged_swa_group_tags);

    if (shared_block_cache_) {
        std::map<std::string, BlockPoolPtr> tagged_group_pools;
        for (size_t idx = 0; idx < config_.groups().size(); ++idx) {
            tagged_group_pools.emplace(config_.groups()[idx].tag, group_block_pools_[idx]);
        }
        shared_block_cache_->init(config_, tagged_group_pools);
    }

    RTP_LLM_LOG_INFO("HybridPoolKVCacheAllocator init success, group pools=%zu", group_block_pools_.size());
    return true;
}

const CacheGroup& HybridPoolKVCacheAllocator::defaultGroupForLayer(int layer_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("invalid layer_id=%d", layer_id);
    }
    const auto& group = config_.soleGroupForLayer(layer_id);
    (void)groupStrategy(group.tag);
    return group;
}

const CacheGroup& HybridPoolKVCacheAllocator::validateGroupForLayer(int layer_id, std::string_view tag) const {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config_.layer_all_num,
                            "invalid layer id %d for layer_all_num=%u",
                            layer_id,
                            config_.layer_all_num);
    const auto& group = config_.groupForLayer(layer_id, tag);
    (void)groupStrategy(group.tag);
    return group;
}

void HybridPoolKVCacheAllocator::referenceBlocks(std::string_view        tag,
                                                 const BlockIndicesType& blocks,
                                                 bool                    is_connector) const {
    if (is_connector) {
        blockPool(tag)->connectorReference(blocks);
    } else {
        blockPool(tag)->requestReference(blocks);
    }
}

void HybridPoolKVCacheAllocator::freeBlocks(std::string_view tag, const BlockIndicesType& blocks, bool is_connector) {
    if (is_connector) {
        blockPool(tag)->connectorFree(blocks);
    } else {
        blockPool(tag)->requestFree(blocks);
    }
}

GroupedCacheLayerLayout HybridPoolKVCacheAllocator::allLayerCacheBase() const {
    const auto topology = std::make_shared<const CacheConfig>(config_.groups(), config_.layers(), config_.layer_num);
    RTP_LLM_CHECK_WITH_INFO(kv_cache_groups_.size() == topology->groups().size(),
                            "cache group count=%zu topology count=%zu",
                            kv_cache_groups_.size(),
                            topology->groups().size());

    GroupedCacheLayerLayout::GroupLayouts groups;
    for (const auto& group_config : topology->groups()) {
        const auto&                     strategy = groupStrategy(group_config.tag);
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        const auto                      layer_tensors = strategy->allLayerCacheBase();
        const auto                      scale_tensors = strategy->allLayerScaleCacheBase();
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
        groups.emplace(group_config.tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(*topology, std::move(groups));
}

BlockAddrInfo HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    const auto& group = defaultGroupForLayer(layer_id);
    return groupStrategy(group.tag)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    const auto& group = defaultGroupForLayer(layer_id);
    return groupStrategy(group.tag)->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    const auto& group = defaultGroupForLayer(layer_id);
    return groupStrategy(group.tag)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo
HybridPoolKVCacheAllocator::convertIndexToAddrByTag(int layer_id, std::string_view tag, int block_id) const {
    const auto& group = validateGroupForLayer(layer_id, tag);
    return groupStrategy(group.tag)->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBufferByTag(int layer_id, std::string_view tag, int block_id) const {
    const auto& group = validateGroupForLayer(layer_id, tag);
    return groupStrategy(group.tag)->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBufferByTag(
    int layer_id, std::string_view tag, int block_id, int partition_count, int partition_id) const {
    const auto& group = validateGroupForLayer(layer_id, tag);
    return groupStrategy(group.tag)->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

void HybridPoolKVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    if (end_ptr == begin_ptr) {
        return;
    }

    RTP_LLM_CHECK_WITH_INFO(config_.hasOneGroupPerLayer(),
                            "legacy layer-only block copy requires exactly one cache group per layer");
    std::vector<TaggedBlockIdPair> tagged_mappings;
    tagged_mappings.reserve(static_cast<size_t>(end_ptr - begin_ptr) * config_.groups().size());
    for (const auto& group : config_.groups()) {
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
        const auto&  pool              = blockPool(mapping.tag);
        const auto   copy_type         = BatchCopyParams::get_copy_type(pool->where(), pool->where());
        const auto&  group             = config_.group(mapping.tag);
        const size_t buffers_per_layer = group.kv_scale_stride_bytes > 0 ? 2 : 1;
        copy_nums[copy_type] += config_.groupLayerIds(group.tag).size() * buffers_per_layer;
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (const auto& mapping : copy_mapping) {
        const auto&  group               = config_.group(mapping.tag);
        const size_t kv_block_size_bytes = group.kv_block_stride_bytes;
        const size_t scale_block_bytes   = group.kv_scale_stride_bytes;
        const auto&  pool                = blockPool(mapping.tag);
        const auto   copy_type           = BatchCopyParams::get_copy_type(pool->where(), pool->where());

        for (int layer_id : config_.groupLayerIds(group.tag)) {
            auto src_addr_info = groupStrategy(mapping.tag)->convertIndexToAddr(layer_id, mapping.src);
            auto dst_addr_info = groupStrategy(mapping.tag)->convertIndexToAddr(layer_id, mapping.dst);

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
    if (evict_result.evictions.empty()) {
        return nullptr;
    }
    if (metrics_reporter_) {
        for (const auto& eviction : evict_result.evictions) {
            RtpLLMCacheEvictionMetricsCollector collector;
            collector.lifetime_ms = eviction.lifetime_ms;
            kmonitor::MetricsTags tags("scope", "gpu");
            tags.AddTag("evict_policy", eviction.kind == EvictionKind::IndependentGroup ? "independent" : "chain");
            tags.AddTag("backing", "device");
            metrics_reporter_->report<RtpLLMCacheEvictionMetrics, RtpLLMCacheEvictionMetricsCollector>(&tags,
                                                                                                       &collector);
        }
    }

    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(1);
    batch_resource->initGroups(config_);
    batch_resource->setLastBlockAligned(true);

    for (const auto& group : config_.groups()) {
        batch_resource->mutableBlockIds(0, group.tag).resize(evict_result.evictions.size(), NULL_BLOCK_IDX);
    }

    CacheKeysType         evicted_keys;
    BlockDependenciesType evicted_dependencies;
    evicted_keys.reserve(evict_result.evictions.size());
    evicted_dependencies.reserve(evict_result.evictions.size());
    for (size_t evicted_idx = 0; evicted_idx < evict_result.evictions.size(); ++evicted_idx) {
        const auto& eviction = evict_result.evictions[evicted_idx];
        evicted_keys.push_back(eviction.cache_key);
        if (eviction.has_dependency) {
            evicted_dependencies.push_back(eviction.dependency);
        } else {
            BlockDependency dependency;
            dependency.ordinal = static_cast<uint32_t>(evicted_idx);
            if (evicted_idx > 0) {
                dependency.has_parent = true;
                dependency.parent_key = evict_result.evictions[evicted_idx - 1].cache_key;
            }
            evicted_dependencies.push_back(dependency);
        }
        for (const auto& [tag, block_id] : eviction.blocks_by_group) {
            batch_resource->mutableBlockIds(0, tag).setAt(evicted_idx, block_id);
        }
    }
    batch_resource->cacheResource(0).setCacheKeysAndBlockDependencies(std::move(evicted_keys),
                                                                      std::move(evicted_dependencies));
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
        const auto& resource = batch_kv_cache_resource->cacheResource(batch_id);
        for (const auto& group : config_.groups()) {
            const auto&                      tag       = group.tag;
            const auto&                      block_ids = resource.blockIds(tag);
            BlockIndicesType                 blocks_to_free;
            std::unordered_set<BlockIdxType> seen_blocks;
            for (auto block_idx : block_ids.blocks()) {
                if (isNullBlockIdx(block_idx) || !seen_blocks.insert(block_idx).second) {
                    continue;
                }
                blocks_to_free.push_back(block_idx);
            }
            if (!blocks_to_free.empty()) {
                blockPool(tag)->blockCacheFree(blocks_to_free);
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
        for (const auto& group : config_.groups()) {
            if (only_full_groups && group.policy.group_type != CacheGroupType::FULL) {
                continue;
            }
            const auto& pool = blockPool(group.tag);
            if (!pool) {
                continue;
            }
            saw_group        = true;
            const auto block = use_available_blocks ? pool->availableBlocksNum() : pool->totalBlocksNum();
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
    for (const auto& group : config_.groups()) {
        const auto& pool = blockPool(group.tag);
        if (!pool) {
            continue;
        }
        const size_t seq_size = group.seqSizePerBlock();
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
    for (const auto& group : config_.groups()) {
        const auto& pool = blockPool(group.tag);
        if (!pool) {
            continue;
        }
        KVCachePoolMetricsSnapshot snapshot;
        snapshot.tag                  = group.tag;
        snapshot.pool_name            = pool->poolName();
        snapshot.total_blocks         = pool->totalBlocksNum();
        snapshot.available_blocks     = pool->availableBlocksNum();
        snapshot.free_blocks          = pool->freeBlocksNum();
        snapshot.request_ref_blocks   = pool->requestRefBlocksNum();
        snapshot.connector_ref_blocks = pool->connectorRefBlocksNum();
        snapshot.reserve_blocks = reserveBlocksForPool(group.tag, reserve_blocks, total_reservable_available_blocks);
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
    for (const auto& group : config_.groups()) {
        const auto& pool = blockPool(group.tag);
        if (!pool || config_.usesExplicitIndependentBlocks(group.tag)) {
            continue;
        }
        total += pool->availableBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::reservableAvailableBlocksNum() const {
    return totalReservableAvailableBlocks();
}

size_t HybridPoolKVCacheAllocator::reserveBlocksForPool(std::string_view tag,
                                                        size_t           reserve_blocks,
                                                        size_t           total_reservable_available_blocks) const {
    const auto& pool = blockPool(tag);
    if (!pool || config_.usesExplicitIndependentBlocks(tag) || total_reservable_available_blocks == 0) {
        return 0;
    }
    return reserve_blocks * pool->availableBlocksNum() / total_reservable_available_blocks;
}

MallocStatus HybridPoolKVCacheAllocator::evaluateInitCapacity(const MallocInfo& malloc_info,
                                                              size_t            reserve_blocks,
                                                              InitCapacityMode  mode) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return MallocStatus::NONE;
    }
    const auto& cp_mapper          = cp_slot_mapper_;
    const int   batch_size         = malloc_info.batch_kv_cache_resource->batchSize();
    const int   total_seq_len      = malloc_info.complete_token_ids->totalSeqLength();
    const int   raw_common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int   raw_seq_len        = malloc_info.complete_token_ids->seqLength();
    const int   reserve_step       = malloc_info.complete_token_ids->getReserveStep();
    const bool  reuse_enabled      = malloc_info.reuse_cache;

    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();
    // The "can this ever fit" verdict must not depend on transient availability,
    // so the reserve share for the total-capacity test is prorated by each pool's
    // total size instead.
    size_t total_reservable_blocks = 0;
    for (const auto& group : config_.groups()) {
        const auto& reservable_pool = blockPool(group.tag);
        if (!reservable_pool || config_.usesExplicitIndependentBlocks(group.tag)) {
            continue;
        }
        total_reservable_blocks += reservable_pool->totalBlocksNum();
    }

    MallocStatus status = MallocStatus::NONE;
    const auto&  groups = config_.groups();
    for (size_t idx = 0; idx < groups.size(); ++idx) {
        const auto& group = groups[idx];
        // Diagnostic-only positional column in the rejection logs; never an identity.
        const int group_common_seq = cpEffectiveSeqLenForReserve(cp_mapper, config_, group.tag, raw_common_seq_len);
        const int group_seq_len    = cpEffectiveSeqLenForReserve(cp_mapper, config_, group.tag, raw_seq_len);
        const int group_reuse_blocks_len =
            reuse_enabled ? malloc_info.batch_kv_cache_resource->blocksNum(0, group.tag) : 0;
        const auto need = groupStrategy(group.tag)->getNeedBlocks(
            group_common_seq, group_seq_len, reserve_step, group_reuse_blocks_len, reuse_enabled);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        if (need_blocks <= 0) {
            continue;
        }
        const auto&  pool             = blockPool(group.tag);
        const size_t available_blocks = pool->availableBlocksNum();
        const size_t total_blocks     = pool->totalBlocksNum();
        const size_t required_blocks  = static_cast<size_t>(need_blocks);

        const size_t total_reserve_blocks =
            (config_.usesExplicitIndependentBlocks(group.tag) || total_reservable_blocks == 0) ?
                0 :
                reserve_blocks * total_blocks / total_reservable_blocks;
        if (required_blocks > total_blocks || total_reserve_blocks > total_blocks - required_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc permanently rejected: request_id=%ld pool_name=%s "
                                 "idx=%zu tag=%s need_blocks=%d total_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 idx,
                                 group.tag.c_str(),
                                 need_blocks,
                                 total_blocks,
                                 reserve_blocks,
                                 total_reserve_blocks);
            }
            return MallocStatus::PERMANENT_RESOURCE_EXHAUSTED;
        }

        if (mode != InitCapacityMode::TOTAL_AND_AVAILABLE || status != MallocStatus::NONE) {
            continue;
        }
        const size_t group_reserve_blocks =
            reserveBlocksForPool(group.tag, reserve_blocks, total_reservable_available_blocks);
        if (available_blocks < required_blocks + group_reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc rejected by reserve blocks: request_id=%ld pool_name=%s "
                                 "idx=%zu tag=%s need_blocks=%d total_blocks=%zu available_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 idx,
                                 group.tag.c_str(),
                                 need_blocks,
                                 total_blocks,
                                 available_blocks,
                                 reserve_blocks,
                                 group_reserve_blocks);
            }
            // Keep scanning: a later pool may turn this into a permanent verdict.
            status = MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED;
        }
    }
    return status;
}

bool HybridPoolKVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info,
                                                              size_t            reserve_blocks) const {
    return evaluateInitCapacity(malloc_info, reserve_blocks, InitCapacityMode::TOTAL_AND_AVAILABLE)
           == MallocStatus::NONE;
}

// Per-pool KV-exhaustion record. This is the primary field-debug tool for
// KV-exhaustion incidents: one aggregate line plus one line per pool carrying
// the demand, the reserve share, the shortfall and the pool's ref-count split.
void HybridPoolKVCacheAllocator::logMallocFailure(const MallocInfo& malloc_info,
                                                  const char*       phase,
                                                  int               failed_batch,
                                                  std::string_view  failed_tag,
                                                  bool              incremental,
                                                  int               failed_need_blocks) const {
    if (!malloc_info.verbose || !malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return;
    }

    const auto& resource       = malloc_info.batch_kv_cache_resource;
    const auto& cp_mapper      = cp_slot_mapper_;
    const int   batch_size     = resource->batchSize();
    const int   raw_seq_len    = incremental ? malloc_info.incrSeqLen() : malloc_info.complete_token_ids->seqLength();
    const int   raw_common_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), raw_seq_len);
    const int   total_seq_len  = malloc_info.complete_token_ids->totalSeqLength();
    const int   request_reserve_step = malloc_info.complete_token_ids->getReserveStep();
    const bool  reserve_admission    = !incremental && failed_tag.empty();
    const int   reserve_step         = incremental || reserve_admission ? request_reserve_step : 0;
    const int   planning_raw_seq_len = !incremental && !reserve_admission ? raw_common_len : raw_seq_len;
    const auto  reserve_blocks       = reserveBlocksNum();

    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();

    RTP_LLM_LOG_WARNING("HybridPool malloc failure: error_code=602 request_id=%ld phase=%s failed_batch=%d "
                        "failed_tag=%.*s incremental=%d batch_size=%d seq_len=%d common_seq_len=%d total_seq_len=%d "
                        "planning_seq_len=%d request_reserve_step=%d planning_reserve_step=%d "
                        "failed_need_blocks=%d reserve_blocks=%zu snapshot=best_effort_at_failure",
                        malloc_info.request_id,
                        phase,
                        failed_batch,
                        static_cast<int>(failed_tag.size()),
                        failed_tag.data(),
                        incremental,
                        batch_size,
                        raw_seq_len,
                        raw_common_len,
                        total_seq_len,
                        planning_raw_seq_len,
                        request_reserve_step,
                        reserve_step,
                        failed_need_blocks,
                        reserve_blocks);

    size_t idx               = 0;
    bool   before_failed_tag = !failed_tag.empty();
    for (const auto& group_config : config_.groups()) {
        const auto& tag        = group_config.tag;
        const auto  group_type = group_config.policy.group_type;
        if (tag == failed_tag) {
            before_failed_tag = false;
        }
        // Diagnostic-only positional column in the failure log; never an identity.
        const int group_seq_len = cpEffectiveSeqLenForReserve(cp_mapper, config_, tag, planning_raw_seq_len);

        int    need_blocks          = 0;
        int    need_slots           = 0;
        size_t current_slots        = 0;
        size_t current_valid_blocks = 0;
        for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
            const auto& blocks = resource->blocks(batch_id, tag);
            current_slots += blocks.size();
            current_valid_blocks += static_cast<size_t>(std::count_if(
                blocks.begin(), blocks.end(), [](auto block) { return !isNullBlockIdx(block) && block > 0; }));
            need_slots +=
                groupStrategy(tag)->needBlocksNum(group_seq_len, static_cast<int>(blocks.size()), reserve_step);
        }
        if (incremental) {
            // Dense groups materialize every logical slot. Sparse groups
            // (LINEAR / SWA) skip slots, so their exact physical request is the
            // value the group allocator reported immediately before this snapshot.
            need_blocks = groupStrategy(tag)->hasSparseSlots() ? -1 : need_slots;
        } else if (!reserve_admission && before_failed_tag) {
            // These groups already completed their initial allocation before
            // a later group failed.
            need_blocks = 0;
            need_slots  = 0;
        } else {
            const int  group_common_len = cpEffectiveSeqLenForReserve(cp_mapper, config_, tag, raw_common_len);
            const int  reuse_blocks_len = malloc_info.reuse_cache ? resource->blocksNum(0, tag) : 0;
            const auto need             = groupStrategy(tag)->getNeedBlocks(
                group_common_len, group_seq_len, reserve_step, reuse_blocks_len, malloc_info.reuse_cache);
            need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        }
        if (tag == failed_tag && failed_need_blocks >= 0) {
            need_blocks = failed_need_blocks;
        }

        const auto&  pool      = blockPool(tag);
        const size_t available = pool->availableBlocksNum();
        const size_t group_reserve =
            reserve_admission ? reserveBlocksForPool(tag, reserve_blocks, total_reservable_available_blocks) : 0;
        const long long required_available = need_blocks < 0 ? -1 : static_cast<long long>(need_blocks + group_reserve);
        const long long shortfall =
            required_available < 0 ? -1 : std::max(required_available - static_cast<long long>(available), 0LL);

        RTP_LLM_LOG_WARNING("HybridPool malloc failure pool: error_code=602 request_id=%ld idx=%zu "
                            "pool_name=%s "
                            "group_type=%s tag=%s failed=%d need_blocks=%d need_slots=%d "
                            "group_reserve_blocks=%zu required_available_blocks=%lld shortfall_blocks=%lld "
                            "current_slots=%zu "
                            "current_valid_blocks=%zu total_blocks=%zu available_blocks=%zu free_blocks=%zu "
                            "request_ref_blocks=%zu connector_ref_blocks=%zu block_cache_ref_blocks=%zu "
                            "layer_count=%zu block_bytes=%zu seq_size_per_block=%zu",
                            malloc_info.request_id,
                            idx,
                            pool->poolName().c_str(),
                            cacheGroupTypeName(group_type),
                            tag.c_str(),
                            tag == failed_tag,
                            need_blocks,
                            need_slots,
                            group_reserve,
                            required_available,
                            shortfall,
                            current_slots,
                            current_valid_blocks,
                            pool->totalBlocksNum(),
                            available,
                            pool->freeBlocksNum(),
                            pool->requestRefBlocksNum(),
                            pool->connectorRefBlocksNum(),
                            pool->blockCacheRefBlocksNum(),
                            config_.groupLayerIds(tag).size(),
                            config_.blockSizeBytes(tag),
                            group_config.seqSizePerBlock());
        ++idx;
    }
}

}  // namespace rtp_llm
