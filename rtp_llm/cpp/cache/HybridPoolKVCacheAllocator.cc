#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio):
    HybridKVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

bool HybridPoolKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(!config_.cache_specs.empty(), "no cache_specs found in CacheConfig");
    RTP_LLM_CHECK_WITH_INFO(config_.cache_specs.size() == config_.global_layer_ids.size(),
                            "cache_specs size %zu != global_layer_ids size %zu",
                            config_.cache_specs.size(),
                            config_.global_layer_ids.size());

    const int group_nums = static_cast<int>(config_.cache_specs.size());
    group_block_pools_.reserve(static_cast<size_t>(group_nums));
    kv_cache_groups_.reserve(static_cast<size_t>(group_nums));

    for (int gid = 0; gid < group_nums; ++gid) {
        auto pool_config = BlockPoolConfigHelper::createConfigForGroup(config_, static_cast<size_t>(gid));
        auto group_pool  = std::make_shared<BlockPool>(pool_config, allocation_type_);
        RTP_LLM_CHECK_WITH_INFO(group_pool->init(), "Failed to initialize block pool for group %d", gid);

        const auto& ids  = config_.global_layer_ids[static_cast<size_t>(gid)];
        auto        spec = config_.cache_specs[static_cast<size_t>(gid)];

        KVCacheGroupPtr group;
        RTP_LLM_CHECK_WITH_INFO(gid < static_cast<int>(config_.group_types.size()),
                                "missing group type for group %d in HybridPoolKVCacheAllocator",
                                gid);
        const auto group_type = config_.group_types[static_cast<size_t>(gid)];
        if (group_type == CacheGroupType::LINEAR) {
            group = std::make_shared<LinearKVCacheGroup>(ids, spec, group_pool, gid, config_.linear_step);
            linear_group_ids_.push_back(gid);
        } else if (group_type == CacheGroupType::SWA) {
            group = std::make_shared<SWAKVCacheGroup>(ids, spec, group_pool, gid);
            swa_group_ids_.push_back(gid);
        } else {
            group = std::make_shared<FullKVCacheGroup>(ids, spec, group_pool, gid);
            full_group_ids_.push_back(gid);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup gid %d", gid);
        group_block_pools_.push_back(group_pool);
        kv_cache_groups_.push_back(group);
    }

    // HybridPool owns one BlockPool per group; do not read pool stats from block_pool_ in HybridPool mode.
    block_pool_ = group_block_pools_.empty() ? nullptr : group_block_pools_[0];
    RTP_LLM_LOG_INFO("HybridPoolKVCacheAllocator init success, group pools=%zu", group_block_pools_.size());
    return true;
}

int HybridPoolKVCacheAllocator::defaultGroupIdForLayer(int layer_id) const {
    if (layer_id < 0 || static_cast<size_t>(layer_id) >= config_.layer_to_group_id.size()) {
        RTP_LLM_FAIL("invalid layer_id=%d", layer_id);
    }
    const int gid = config_.layer_to_group_id[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()),
                            "invalid default group id %d for layer %d",
                            gid,
                            layer_id);
    return gid;
}

int HybridPoolKVCacheAllocator::groupIdForLayerRegion(int layer_id, KVCacheRegionName region_name) const {
    const size_t attn_id = static_cast<size_t>(region_name);
    if (layer_id >= 0 && static_cast<size_t>(layer_id) < config_.layer_region_to_group_id.size()) {
        const auto& dense = config_.layer_region_to_group_id[static_cast<size_t>(layer_id)];
        if (attn_id < dense.size() && dense[attn_id] >= 0) {
            const int gid = dense[attn_id];
            RTP_LLM_CHECK_WITH_INFO(gid < static_cast<int>(kv_cache_groups_.size()),
                                    "invalid group id %d for layer %d region %zu",
                                    gid,
                                    layer_id,
                                    attn_id);
            return gid;
        }
    }
    if (region_name == KVCacheRegionName::DEFAULT) {
        return defaultGroupIdForLayer(layer_id);
    }
    RTP_LLM_FAIL("missing group mapping for layer_id=%d region=%zu", layer_id, attn_id);
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

CacheLayerLayout HybridPoolKVCacheAllocator::allLayerCacheBase() const {
    CacheLayerLayout layout;
    layout.layer_to_groups          = config_.layer_to_group_id;
    layout.layer_to_group_ids       = config_.layer_to_group_ids;
    layout.layer_region_to_group_id = config_.layer_region_to_group_id;
    layout.group_types              = config_.group_types;
    layout.group_region_names       = config_.group_region_names;
    layout.layer_group_types        = config_.layer_group_types;

    const bool has_typed_mapping = !config_.layer_region_to_group_id.empty();
    if (has_typed_mapping) {
        RTP_LLM_CHECK_WITH_INFO(config_.group_region_names.size() == kv_cache_groups_.size(),
                                "group_region_names size %zu != group num %zu for typed layer-region mapping",
                                config_.group_region_names.size(),
                                kv_cache_groups_.size());
    }

    layout.layers_to_kv_buffer_ptrs.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs.resize(config_.layer_all_num);
    const size_t region_name_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    layout.layers_to_kv_buffer_ptrs_by_attn.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs_by_attn.resize(config_.layer_all_num);
    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_all_num); ++layer_id) {
        layout.layers_to_kv_buffer_ptrs_by_attn[layer_id].resize(region_name_count);
        layout.layers_to_scale_buffer_ptrs_by_attn[layer_id].resize(region_name_count);
    }

    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_all_num); ++layer_id) {
        const int  gid           = defaultGroupIdForLayer(static_cast<int>(layer_id));
        const auto layer_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerCacheBase();
        const auto scale_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerScaleCacheBase();
        auto       it            = layer_tensors.find(static_cast<int>(layer_id));
        if (it != layer_tensors.end()) {
            layout.layers_to_kv_buffer_ptrs[layer_id] = it->second;
        }
        auto scale_it = scale_tensors.find(static_cast<int>(layer_id));
        if (scale_it != scale_tensors.end()) {
            layout.layers_to_scale_buffer_ptrs[layer_id] = scale_it->second;
        }
    }

    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const auto layer_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerCacheBase();
        const auto scale_tensors = kv_cache_groups_[static_cast<size_t>(gid)]->allLayerScaleCacheBase();
        const auto region_name   = static_cast<size_t>(gid < static_cast<int>(config_.group_region_names.size()) ?
                                                         config_.group_region_names[static_cast<size_t>(gid)] :
                                                         KVCacheRegionName::DEFAULT);
        RTP_LLM_CHECK_WITH_INFO(
            region_name < region_name_count, "group %d has invalid region id %zu", gid, region_name);
        for (const auto& [layer_id, tensor] : layer_tensors) {
            RTP_LLM_CHECK_WITH_INFO(
                layer_id >= 0 && static_cast<size_t>(layer_id) < layout.layers_to_kv_buffer_ptrs_by_attn.size(),
                "layer_id %d out of typed kv layout range %zu",
                layer_id,
                layout.layers_to_kv_buffer_ptrs_by_attn.size());
            layout.layers_to_kv_buffer_ptrs_by_attn[static_cast<size_t>(layer_id)][region_name] = tensor;
        }
        for (const auto& [layer_id, tensor] : scale_tensors) {
            RTP_LLM_CHECK_WITH_INFO(
                layer_id >= 0 && static_cast<size_t>(layer_id) < layout.layers_to_scale_buffer_ptrs_by_attn.size(),
                "layer_id %d out of typed scale layout range %zu",
                layer_id,
                layout.layers_to_scale_buffer_ptrs_by_attn.size());
            layout.layers_to_scale_buffer_ptrs_by_attn[static_cast<size_t>(layer_id)][region_name] = tensor;
        }
    }
    return layout;
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

BlockAddrInfo
HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, KVCacheRegionName region_name, int block_id) const {
    const int gid = groupIdForLayerRegion(layer_id, region_name);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, KVCacheRegionName region_name, int block_id) const {
    const int gid = groupIdForLayerRegion(layer_id, region_name);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(
    int layer_id, KVCacheRegionName region_name, int block_id, int partition_count, int partition_id) const {
    const int gid = groupIdForLayerRegion(layer_id, region_name);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

void HybridPoolKVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    if (end_ptr == begin_ptr) {
        return;
    }

    auto copy_type = BatchCopyParams::get_copy_type(
        allocation_type_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU,
        allocation_type_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU);

    size_t copy_num = 0;
    for (const auto& layers : config_.global_layer_ids) {
        copy_num += layers.size();
    }
    copy_num *= static_cast<size_t>(end_ptr - begin_ptr);

    BatchCopyParams copy_params;
    copy_params.reserve(copy_type, copy_num);

    for (auto it = begin_ptr; it != end_ptr; ++it) {
        auto [src_block_index, dest_block_index] = *it;

        for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
            RTP_LLM_CHECK_WITH_INFO(
                static_cast<size_t>(gid) < config_.cache_specs.size(), "missing cache spec for group %d", gid);
            RTP_LLM_CHECK_WITH_INFO(
                static_cast<size_t>(gid) < config_.global_layer_ids.size(), "missing layer ids for group %d", gid);

            const size_t kv_block_size_bytes = config_.cache_specs[static_cast<size_t>(gid)]->block_size_bytes();
            const size_t scale_block_bytes   = config_.cache_specs[static_cast<size_t>(gid)]->scale_block_size_bytes();

            for (int layer_id : config_.global_layer_ids[static_cast<size_t>(gid)]) {
                auto src_addr_info =
                    kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, src_block_index);
                auto dst_addr_info =
                    kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, dest_block_index);

                if (!src_addr_info.kv_addr || !dst_addr_info.kv_addr) {
                    RTP_LLM_LOG_ERROR("Failed to get block address for group %d layer %d, src_block %d, dst_block %d",
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
    if (min_blocks_to_free == 0 || group_block_pools_.empty()) {
        return nullptr;
    }

    std::vector<CacheKeyType>                                            evicted_keys;
    std::unordered_set<CacheKeyType>                                     seen_keys;
    std::unordered_map<CacheKeyType, std::vector<BlockCache::CacheItem>> evicted_items;

    size_t evicted_blocks = 0;
    for (const auto& pool : group_block_pools_) {
        if (!pool || !pool->blockCache()) {
            continue;
        }
        const size_t group_need         = min_blocks_to_free > evicted_blocks ? min_blocks_to_free - evicted_blocks : 1;
        auto         group_evict_result = pool->blockCache()->selectAndEvict(group_need);
        for (const auto cache_key : group_evict_result.evicted_keys) {
            auto items_it = group_evict_result.evicted_items.find(cache_key);
            if (items_it == group_evict_result.evicted_items.end() || items_it->second.empty()) {
                continue;
            }
            if (seen_keys.insert(cache_key).second) {
                evicted_keys.push_back(cache_key);
            }
            auto& items_for_key = evicted_items[cache_key];
            evicted_blocks += items_it->second.size();
            items_for_key.insert(items_for_key.end(), items_it->second.begin(), items_it->second.end());
        }
        if (evicted_blocks >= min_blocks_to_free) {
            break;
        }
    }

    if (evicted_keys.empty()) {
        return nullptr;
    }

    auto batch_resource = std::make_shared<BatchKVCacheResource>();
    batch_resource->resetBatchSize(1);
    batch_resource->initGroups(config_.groupNums(),
                               static_cast<int>(config_.layer_all_num),
                               config_.layer_to_group_id,
                               config_.kernelBlocksPerKvBlock(),
                               config_.group_types,
                               config_.layer_region_to_group_id);
    batch_resource->setLastBlockAligned(true);

    for (int gid = 0; gid < config_.groupNums(); ++gid) {
        batch_resource->mutableBlockIds(0, gid).resize(evicted_keys.size(), NULL_BLOCK_IDX);
    }

    for (size_t evicted_idx = 0; evicted_idx < evicted_keys.size(); ++evicted_idx) {
        const auto cache_key = evicted_keys[evicted_idx];
        batch_resource->pushBackCacheKey(0, cache_key);
        for (const auto& item : evicted_items[cache_key]) {
            RTP_LLM_CHECK_WITH_INFO(item.group_id >= 0 && item.group_id < config_.groupNums(),
                                    "invalid evicted group id=%d for cache key=%ld",
                                    item.group_id,
                                    cache_key);
            batch_resource->mutableBlockIds(0, item.group_id).setAt(evicted_idx, item.block_index);
        }
    }
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

size_t HybridPoolKVCacheAllocator::availableTokensNum() const {
    if (group_block_pools_.empty()) {
        return 0;
    }
    size_t min_tokens = std::numeric_limits<size_t>::max();
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        const size_t seq_size =
            (gid < config_.group_seq_size_per_block.size() && config_.group_seq_size_per_block[gid] > 0) ?
                config_.group_seq_size_per_block[gid] :
                config_.seq_size_per_block;
        min_tokens = std::min(min_tokens, group_block_pools_[gid]->availableBlocksNum() * seq_size);
    }
    return min_tokens;
}

size_t HybridPoolKVCacheAllocator::totalBlocksNum() const {
    size_t total = 0;
    for (const auto& pool : group_block_pools_) {
        total += pool->totalBlocksNum();
    }
    return total;
}

size_t HybridPoolKVCacheAllocator::maxAvailableTokensNum() const {
    if (group_block_pools_.empty()) {
        return 0;
    }
    size_t min_tokens     = std::numeric_limits<size_t>::max();
    bool   saw_full_group = false;
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        if (gid < config_.group_types.size() && config_.group_types[gid] != CacheGroupType::FULL) {
            continue;
        }
        saw_full_group = true;
        const size_t seq_size =
            (gid < config_.group_seq_size_per_block.size() && config_.group_seq_size_per_block[gid] > 0) ?
                config_.group_seq_size_per_block[gid] :
                config_.seq_size_per_block;
        min_tokens = std::min(min_tokens, group_block_pools_[gid]->totalBlocksNum() * seq_size);
    }
    if (!saw_full_group) {
        for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
            const size_t seq_size =
                (gid < config_.group_seq_size_per_block.size() && config_.group_seq_size_per_block[gid] > 0) ?
                    config_.group_seq_size_per_block[gid] :
                    config_.seq_size_per_block;
            min_tokens = std::min(min_tokens, group_block_pools_[gid]->totalBlocksNum() * seq_size);
        }
    }
    return min_tokens;
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

bool HybridPoolKVCacheAllocator::hasAvailableBlocksForReserve(const MallocInfo& malloc_info,
                                                              size_t            reserve_blocks) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return true;
    }
    const int  batch_size     = malloc_info.batch_kv_cache_resource->batchSize();
    const int  total_seq_len  = malloc_info.complete_token_ids->totalSeqLength();
    const int  common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int  seq_len        = malloc_info.complete_token_ids->seqLength();
    const int  reserve_step   = malloc_info.complete_token_ids->getReserveStep();
    const bool reuse_enabled  = malloc_info.reuse_cache;

    size_t total_available_blocks = 0;
    for (const auto& pool : group_block_pools_) {
        total_available_blocks += pool->availableBlocksNum();
    }

    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const int  group_reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->blocksNum(0, gid) : 0;
        const auto need                   = kv_cache_groups_[static_cast<size_t>(gid)]->getNeedBlocks(
            common_seq_len, seq_len, reserve_step, group_reuse_blocks_len, reuse_enabled);
        const int need_blocks = need.common_blocks + batch_size * need.extra_blocks;
        if (need_blocks <= 0) {
            continue;
        }
        const size_t available_blocks = group_block_pools_[static_cast<size_t>(gid)]->availableBlocksNum();
        const size_t group_reserve_blocks =
            total_available_blocks > 0 ? reserve_blocks * available_blocks / total_available_blocks : 0;
        if (available_blocks < static_cast<size_t>(need_blocks) + group_reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("HybridPool initMalloc rejected by reserve blocks: request_id=%ld group=%d "
                                 "need_blocks=%d available_blocks=%zu reserve_blocks=%zu group_reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 gid,
                                 need_blocks,
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
