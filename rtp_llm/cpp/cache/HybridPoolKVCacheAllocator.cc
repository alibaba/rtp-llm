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
                                       size_t                               gid,
                                       int                                  seq_len) {
    return (mapper && mapper->isSharded()) ? mapper->effectiveSeqLenForAlloc(config, gid, seq_len) : seq_len;
}

void appendPoolSummary(std::ostringstream&    os,
                       bool&                  has_any,
                       int                    gid,
                       const std::string&     tag,
                       CacheGroupType         group_type,
                       const BlockPoolConfig& pool_config) {
    static constexpr double kBytesPerMB = 1024.0 * 1024.0;
    if (has_any) {
        os << "; ";
    }
    has_any = true;
    os << "pool_name=" << pool_config.pool_name << ", gid=" << gid << ", tag=" << tag
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
    for (int gid = 0; gid < group_nums; ++gid) {
        auto       pool_config = BlockPoolConfigHelper::createConfigForGroup(config_, static_cast<size_t>(gid));
        const auto tag         = config_.tagForGroup(static_cast<size_t>(gid));
        const auto group_type  = config_.typeForGroup(static_cast<size_t>(gid));
        appendPoolSummary(pool_summary, has_pool, gid, tag, group_type, pool_config);
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

    for (int gid = 0; gid < group_nums; ++gid) {
        const auto& pool_config = group_pool_configs[static_cast<size_t>(gid)];
        const auto  group_type  = config_.typeForGroup(static_cast<size_t>(gid));
        const auto  policy      = config_.policyForGroup(static_cast<size_t>(gid));

        auto group_pool =
            std::make_shared<BlockPool>(pool_config,
                                        allocationTypeForPlacement(policy.memory_placement, allocation_type_),
                                        pinnedCpuBackingForPlacement(policy.memory_placement),
                                        use_cuda_malloc_block_pool_);
        RTP_LLM_CHECK_WITH_INFO(
            group_pool->init(), "Failed to initialize block pool %s(group %d)", pool_config.pool_name.c_str(), gid);

        const auto& cache_group = config_.topology().groupBySlot(static_cast<size_t>(gid));

        KVCacheGroupPtr group;
        if (group_type == CacheGroupType::LINEAR) {
            group = std::make_shared<LinearKVCacheGroup>(
                cache_group, group_pool, gid, config_.linear_step, shared_cache_raw, metrics_reporter_);
            linear_group_ids_.push_back(gid);
        } else if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(
                cache_group, group_pool, gid, config_.linear_step, shared_cache_raw, metrics_reporter_);
            swa_group_ids_.push_back(gid);
        } else {
            group =
                std::make_shared<FullKVCacheGroup>(cache_group, group_pool, gid, shared_cache_raw, metrics_reporter_);
            full_group_ids_.push_back(gid);
        }

        RTP_LLM_CHECK_WITH_INFO(
            group->init(), "Failed to initialize KVCacheGroup %s(gid %d)", pool_config.pool_name.c_str(), gid);
        group_block_pools_.push_back(group_pool);
        kv_cache_groups_.push_back(group);
    }

    if (shared_block_cache_) {
        shared_block_cache_->init(group_nums, group_block_pools_);
    }

    RTP_LLM_LOG_INFO("HybridPoolKVCacheAllocator init success, group pools=%zu", group_block_pools_.size());
    return true;
}

int HybridPoolKVCacheAllocator::defaultGroupIdForLayer(int layer_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_all_num) {
        RTP_LLM_FAIL("invalid layer_id=%d", layer_id);
    }
    const int gid = config_.groupIdFor(layer_id);
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()),
                            "invalid default group id %d for layer %d",
                            gid,
                            layer_id);
    return gid;
}

int HybridPoolKVCacheAllocator::validateGroupIdForLayer(int layer_id, int group_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0 && group_id < static_cast<int>(kv_cache_groups_.size()),
                            "invalid group id %d for layer %d",
                            group_id,
                            layer_id);
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < config_.layer_all_num,
                            "invalid layer id %d for layer_all_num=%u",
                            layer_id,
                            config_.layer_all_num);
    const auto& group_ids = config_.groupIdsForLayer(layer_id);
    RTP_LLM_CHECK_WITH_INFO(std::find(group_ids.begin(), group_ids.end(), group_id) != group_ids.end(),
                            "layer %d does not own cache group %d",
                            layer_id,
                            group_id);
    return group_id;
}

void HybridPoolKVCacheAllocator::referenceBlocksInGroup(int                     gid,
                                                        const BlockIndicesType& blocks,
                                                        bool                    is_connector) const {
    if (is_connector) {
        group_block_pools_[static_cast<size_t>(gid)]->connectorReference(blocks);
    } else {
        group_block_pools_[static_cast<size_t>(gid)]->requestReference(blocks);
    }
}

void HybridPoolKVCacheAllocator::freeBlocksInGroup(int gid, const BlockIndicesType& blocks, bool is_connector) {
    if (is_connector) {
        group_block_pools_[static_cast<size_t>(gid)]->connectorFree(blocks);
    } else {
        group_block_pools_[static_cast<size_t>(gid)]->requestFree(blocks);
    }
}

GroupedCacheLayerLayout HybridPoolKVCacheAllocator::allLayerCacheBase() const {
    const auto topology = config_.topologyPtr();
    RTP_LLM_CHECK_WITH_INFO(kv_cache_groups_.size() == topology->groups().size(),
                            "cache group count=%zu topology count=%zu",
                            kv_cache_groups_.size(),
                            topology->groups().size());

    GroupedCacheLayerLayout::GroupLayouts groups;
    for (size_t gid = 0; gid < kv_cache_groups_.size(); ++gid) {
        std::vector<BlockBufferPtrInfo> layers(topology->layers().size());
        const auto                      layer_tensors = kv_cache_groups_[gid]->allLayerCacheBase();
        const auto                      scale_tensors = kv_cache_groups_[gid]->allLayerScaleCacheBase();
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
        groups.emplace(topology->groupBySlot(gid).tag, CacheLayerLayout(std::move(layers)));
    }
    return GroupedCacheLayerLayout(topology, std::move(groups));
}

BlockAddrInfo HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    const int gid = defaultGroupIdForLayer(layer_id);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

BlockAddrInfo HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, int group_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology slot=%d", group_id);
    return convertIndexToAddrByTag(
        layer_id, config_.topology().groupBySlot(static_cast<size_t>(group_id)).tag, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, int group_id, int block_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology slot=%d", group_id);
    return convertIndexToBufferByTag(
        layer_id, config_.topology().groupBySlot(static_cast<size_t>(group_id)).tag, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(
    int layer_id, int group_id, int block_id, int partition_count, int partition_id) const {
    RTP_LLM_CHECK_WITH_INFO(group_id >= 0, "invalid cache topology slot=%d", group_id);
    return convertIndexToBufferByTag(layer_id,
                                     config_.topology().groupBySlot(static_cast<size_t>(group_id)).tag,
                                     block_id,
                                     partition_count,
                                     partition_id);
}

BlockAddrInfo
HybridPoolKVCacheAllocator::convertIndexToAddrByTag(int layer_id, const std::string& tag, int block_id) const {
    const auto gid = static_cast<int>(config_.topology().slotForTag(tag));
    validateGroupIdForLayer(layer_id, gid);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBufferByTag(int layer_id, const std::string& tag, int block_id) const {
    const auto gid = static_cast<int>(config_.topology().slotForTag(tag));
    validateGroupIdForLayer(layer_id, gid);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBufferByTag(
    int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const {
    const auto gid = static_cast<int>(config_.topology().slotForTag(tag));
    validateGroupIdForLayer(layer_id, gid);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

void HybridPoolKVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    if (end_ptr == begin_ptr) {
        return;
    }

    size_t copy_nums[BatchCopyParams::TYPE_SIZE] = {};
    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(gid) < group_block_pools_.size(), "missing block pool for group %d", gid);
        const auto   copy_type = BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(gid)]->where(),
                                                              group_block_pools_[static_cast<size_t>(gid)]->where());
        const auto&  spec      = config_.specForGroup(static_cast<size_t>(gid));
        const size_t buffers_per_layer = spec->scale_block_size_bytes() > 0 ? 2 : 1;
        copy_nums[copy_type] += config_.layerIdsForGroup(static_cast<size_t>(gid)).size()
                                * static_cast<size_t>(end_ptr - begin_ptr) * buffers_per_layer;
    }

    BatchCopyParams copy_params;
    for (size_t i = 0; i < BatchCopyParams::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<BatchCopyParams::CopyType>(i), copy_nums[i]);
    }

    for (auto it = begin_ptr; it != end_ptr; ++it) {
        auto [src_block_index, dest_block_index] = *it;

        for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
            const auto&  spec                = config_.specForGroup(static_cast<size_t>(gid));
            const size_t kv_block_size_bytes = spec->block_size_bytes();
            const size_t scale_block_bytes   = spec->scale_block_size_bytes();
            const auto   copy_type =
                BatchCopyParams::get_copy_type(group_block_pools_[static_cast<size_t>(gid)]->where(),
                                               group_block_pools_[static_cast<size_t>(gid)]->where());

            for (int layer_id : config_.layerIdsForGroup(static_cast<size_t>(gid))) {
                auto src_addr_info =
                    kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, src_block_index);
                auto dst_addr_info =
                    kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, dest_block_index);

                if (!src_addr_info.kv_addr || !dst_addr_info.kv_addr) {
                    RTP_LLM_LOG_ERROR("Failed to get block address for pool %s(group %d) layer %d, src_block %d, "
                                      "dst_block %d",
                                      group_block_pools_[static_cast<size_t>(gid)]->poolName().c_str(),
                                      gid,
                                      layer_id,
                                      src_block_index,
                                      dest_block_index);
                    continue;
                }

                copy_params.add(dst_addr_info.kv_addr, src_addr_info.kv_addr, kv_block_size_bytes, copy_type);

                if (scale_block_bytes > 0 && src_addr_info.kv_scale_addr && dst_addr_info.kv_scale_addr) {
                    copy_params.add(
                        dst_addr_info.kv_scale_addr, src_addr_info.kv_scale_addr, scale_block_bytes, copy_type);
                }
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
    batch_resource->initGroups(config_.groupNums(),
                               static_cast<int>(config_.layer_all_num),
                               config_.layerGroupIdsSnapshot(),
                               config_.kernelBlocksPerKvBlock(),
                               config_.groupTypesSnapshot(),
                               config_.groupKernelBlocksPerKvBlockSnapshot());
    batch_resource->setLastBlockAligned(true);

    for (int gid = 0; gid < config_.groupNums(); ++gid) {
        batch_resource->mutableBlockIds(0, gid).resize(evict_result.evicted_keys.size(), NULL_BLOCK_IDX);
    }

    CacheKeysType         evicted_keys;
    BlockDependenciesType evicted_dependencies;
    evicted_keys.reserve(evict_result.evicted_keys.size());
    evicted_dependencies.reserve(evict_result.evicted_keys.size());
    for (size_t evicted_idx = 0; evicted_idx < evict_result.evicted_keys.size(); ++evicted_idx) {
        const auto  cache_key = evict_result.evicted_keys[evicted_idx];
        const auto& slots     = evict_result.evicted_slots.at(cache_key);
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
        for (int gid = 0; gid < static_cast<int>(slots.size()) && gid < config_.groupNums(); ++gid) {
            if (!isNullBlockIdx(slots[gid])) {
                batch_resource->mutableBlockIds(0, gid).setAt(evicted_idx, slots[gid]);
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
        for (int gid = 0; gid < batch_kv_cache_resource->groupNums(); ++gid) {
            BlockIndicesType                 blocks_to_free;
            std::unordered_set<BlockIdxType> seen_blocks;
            for (auto block_idx : batch_kv_cache_resource->blocks(batch_id, gid)) {
                if (isNullBlockIdx(block_idx) || !seen_blocks.insert(block_idx).second) {
                    continue;
                }
                blocks_to_free.push_back(block_idx);
            }
            if (!blocks_to_free.empty()) {
                group_block_pools_[static_cast<size_t>(gid)]->blockCacheFree(blocks_to_free);
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
        for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
            if (only_full_groups && config_.typeForGroup(gid) != CacheGroupType::FULL) {
                continue;
            }
            if (!group_block_pools_[gid]) {
                continue;
            }
            saw_group        = true;
            const auto block = use_available_blocks ? group_block_pools_[gid]->availableBlocksNum() :
                                                      group_block_pools_[gid]->totalBlocksNum();
            min_tokens       = std::min(min_tokens, block * logicalSeqSizePerBlockForCapacity(gid));
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
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        const auto& pool = group_block_pools_[gid];
        if (!pool) {
            continue;
        }
        const size_t seq_size = config_.seqSizePerBlockForGroup(gid);
        total_tokens          = std::min(total_tokens, pool->totalBlocksNum() * seq_size);
        available_tokens      = std::min(available_tokens, pool->availableBlocksNum() * seq_size);
        has_pool              = true;
    }
    return has_pool ? KVCacheTokenCapacity{total_tokens, available_tokens} : KVCacheTokenCapacity{};
}

std::vector<KVCachePoolMetricsSnapshot> HybridPoolKVCacheAllocator::poolMetricsSnapshots() const {
    std::vector<KVCachePoolMetricsSnapshot> snapshots;
    snapshots.reserve(group_block_pools_.size());
    const size_t reserve_blocks                    = reserveBlockNum();
    const size_t total_reservable_available_blocks = totalReservableAvailableBlocks();
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        const auto& pool = group_block_pools_[gid];
        if (!pool) {
            continue;
        }
        KVCachePoolMetricsSnapshot snapshot;
        snapshot.pool_index           = gid;
        snapshot.pool_name            = pool->poolName();
        snapshot.total_blocks         = pool->totalBlocksNum();
        snapshot.available_blocks     = pool->availableBlocksNum();
        snapshot.free_blocks          = pool->freeBlocksNum();
        snapshot.request_ref_blocks   = pool->requestRefBlocksNum();
        snapshot.connector_ref_blocks = pool->connectorRefBlocksNum();
        snapshot.reserve_blocks       = reserveBlocksForPool(gid, reserve_blocks, total_reservable_available_blocks);
        snapshot.used_ratio           = (snapshot.total_blocks == 0) ?
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
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        if (!group_block_pools_[gid] || config_.usesExplicitIndependentBlocks(gid)) {
            continue;
        }
        total += group_block_pools_[gid]->availableBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::reserveBlocksForPool(size_t gid,
                                                        size_t reserve_blocks,
                                                        size_t total_reservable_available_blocks) const {
    if (gid >= group_block_pools_.size() || !group_block_pools_[gid] || config_.usesExplicitIndependentBlocks(gid)
        || total_reservable_available_blocks == 0) {
        return 0;
    }
    return reserve_blocks * group_block_pools_[gid]->availableBlocksNum() / total_reservable_available_blocks;
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

    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const int group_common_seq =
            cpEffectiveSeqLenForReserve(cp_mapper, config_, static_cast<size_t>(gid), raw_common_seq_len);
        const int group_seq_len =
            cpEffectiveSeqLenForReserve(cp_mapper, config_, static_cast<size_t>(gid), raw_seq_len);
        const int  group_reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->blocksNum(0, gid) : 0;
        const auto need                   = kv_cache_groups_[static_cast<size_t>(gid)]->getNeedBlocks(
            group_common_seq, group_seq_len, reserve_step, group_reuse_blocks_len, reuse_enabled);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        if (need_blocks <= 0) {
            continue;
        }
        const auto&  pool             = group_block_pools_[static_cast<size_t>(gid)];
        const size_t available_blocks = pool->availableBlocksNum();
        const size_t total_blocks     = pool->totalBlocksNum();
        const size_t group_reserve_blocks =
            reserveBlocksForPool(static_cast<size_t>(gid), reserve_blocks, total_reservable_available_blocks);
        if (available_blocks < static_cast<size_t>(need_blocks) + group_reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc rejected by reserve blocks: request_id=%ld pool_name=%s "
                                 "group=%d need_blocks=%d total_blocks=%zu available_blocks=%zu "
                                 "reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 pool->poolName().c_str(),
                                 gid,
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
