#include "rtp_llm/cpp/cache/HybridPoolKVCacheAllocator.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

HybridPoolKVCacheAllocator::HybridPoolKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio):
    KVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

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
        const auto      group_type = (gid < static_cast<int>(config_.group_types.size())) ?
                                         config_.group_types[static_cast<size_t>(gid)] :
                                         CacheGroupType::FULL;
        if (group_type == CacheGroupType::LINEAR) {
            group = std::make_shared<LinearKVCacheGroup>(ids, spec, group_pool, gid, config_.linear_step);
            linear_group_ids_.push_back(gid);
        } else {
            group = std::make_shared<FullKVCacheGroup>(ids, spec, group_pool, gid);
            full_group_ids_.push_back(gid);
        }

        RTP_LLM_CHECK_WITH_INFO(group->init(), "Failed to initialize KVCacheGroup gid %d", gid);
        group_block_pools_.push_back(group_pool);
        kv_cache_groups_.push_back(group);
    }

    // Keep block_pool_ non-null for legacy callers of getBlockPool(). It is not authoritative for this allocator.
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

int HybridPoolKVCacheAllocator::groupIdForLayerAttn(int layer_id, KVCacheAttnType attn_type) const {
    const size_t attn_id = static_cast<size_t>(attn_type);
    if (layer_id >= 0 && static_cast<size_t>(layer_id) < config_.layer_attn_to_group_id.size()) {
        const auto& dense = config_.layer_attn_to_group_id[static_cast<size_t>(layer_id)];
        if (attn_id < dense.size() && dense[attn_id] >= 0) {
            const int gid = dense[attn_id];
            RTP_LLM_CHECK_WITH_INFO(gid < static_cast<int>(kv_cache_groups_.size()),
                                    "invalid group id %d for layer %d attn_type %zu",
                                    gid,
                                    layer_id,
                                    attn_id);
            return gid;
        }
    }
    if (attn_type == KVCacheAttnType::DEFAULT) {
        return defaultGroupIdForLayer(layer_id);
    }
    RTP_LLM_FAIL("missing group mapping for layer_id=%d attn_type=%zu", layer_id, attn_id);
}

void HybridPoolKVCacheAllocator::referenceValidBlocks(int                     gid,
                                                      const BlockIndicesType& blocks,
                                                      bool                    is_connector) const {
    BlockIndicesType valid;
    valid.reserve(blocks.size());
    for (auto b : blocks) {
        if (!isNullBlockIdx(b)) {
            valid.push_back(b);
        }
    }
    if (valid.empty()) {
        return;
    }
    if (is_connector) {
        group_block_pools_[static_cast<size_t>(gid)]->connectorReference(valid);
    } else {
        group_block_pools_[static_cast<size_t>(gid)]->requestReference(valid);
    }
}

int HybridPoolKVCacheAllocator::reuseCache(const CacheKeysType& cache_keys, BatchKVCacheResource& kv_resource) {
    int                           min_full_reuse_blocks = static_cast<int>(cache_keys.size());
    std::vector<BlockIndicesType> full_matched_blocks(kv_cache_groups_.size());

    for (int gid : full_group_ids_) {
        auto match_result     = kv_cache_groups_[static_cast<size_t>(gid)]->match(cache_keys);
        min_full_reuse_blocks = std::min(min_full_reuse_blocks, static_cast<int>(match_result.reuse_blocks));
        full_matched_blocks[static_cast<size_t>(gid)] = std::move(match_result.block_indices);
    }

    int                       pos = min_full_reuse_blocks - 1;
    std::vector<BlockIdxType> linear_tail_blocks(linear_group_ids_.size(), NULL_BLOCK_IDX);
    for (; pos >= 0; --pos) {
        bool all_linear_matched = true;
        for (size_t i = 0; i < linear_group_ids_.size(); ++i) {
            const int gid      = linear_group_ids_[i];
            auto* linear_group = dynamic_cast<LinearKVCacheGroup*>(kv_cache_groups_[static_cast<size_t>(gid)].get());
            RTP_LLM_CHECK_WITH_INFO(linear_group != nullptr, "group %d is not LinearKVCacheGroup", gid);
            auto result = linear_group->matchSingleKey(cache_keys[static_cast<size_t>(pos)]);
            if (result.block_indices.empty()) {
                all_linear_matched = false;
                break;
            }
            linear_tail_blocks[i] = result.block_indices[0];
        }
        if (all_linear_matched) {
            break;
        }
    }

    const int reuse_blocks_len = std::max(pos + 1, 0);
    if (reuse_blocks_len <= 0) {
        return 0;
    }

    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        kv_resource.mutableBlockIds(0, gid).assign(
            BlockIndicesType(static_cast<size_t>(reuse_blocks_len), NULL_BLOCK_IDX));
    }

    for (int gid : full_group_ids_) {
        BlockIndicesType full_blocks = full_matched_blocks[static_cast<size_t>(gid)];
        if (static_cast<int>(full_blocks.size()) > reuse_blocks_len) {
            full_blocks.resize(static_cast<size_t>(reuse_blocks_len));
        }
        kv_resource.mutableBlockIds(0, gid).assign(std::move(full_blocks));
    }

    for (size_t i = 0; i < linear_group_ids_.size(); ++i) {
        const int gid = linear_group_ids_[i];
        kv_resource.mutableBlockIds(0, gid).setAt(static_cast<size_t>(reuse_blocks_len - 1), linear_tail_blocks[i]);
    }
    return reuse_blocks_len;
}

MallocResult HybridPoolKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();

    const int seq_len        = malloc_info.complete_token_ids->seqLength();
    const int common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);

    const auto&  cache_keys         = kv_resource->cacheKeys(0);
    int64_t      match_cost_time_us = 0;
    const size_t reserve_blocks     = reserveBlockNum();
    int          reuse_blocks       = 0;

    if (malloc_info.enable_device_cache) {
        CacheKeysType match_keys(cache_keys.begin(), cache_keys.empty() ? cache_keys.end() : cache_keys.end() - 1);
        auto          begin_us = currentTimeUs();
        reuse_blocks           = reuseCache(match_keys, *kv_resource);
        match_cost_time_us     = currentTimeUs() - begin_us;

        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            referenceValidBlocks(gid, kv_resource->blocks(0, gid));
        }
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(reuse_blocks);
    }

    const int need_blocks = (reserve_blocks > 0) ? getNeedBlocks(malloc_info) : 0;
    if (reserve_blocks > 0 && need_blocks > 0
        && availableBlocksNum() < static_cast<size_t>(need_blocks) + reserve_blocks) {
        return {false, 0};
    }

    for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
        auto& block_ids_0 = kv_resource->mutableBlockIds(0, gid);
        if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(
                block_ids_0, common_seq_len, malloc_info.reuse_cache, 0)) {
            return {false, 0};
        }
    }

    for (int b = 1; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            kv_cache_groups_[static_cast<size_t>(gid)]->reference(kv_resource->mutableBlockIds(b, gid),
                                                                  kv_resource->blocks(0, gid));
        }
    }
    return {true, reuse_blocks * seqSizePerBlock(), match_cost_time_us};
}

MallocResult HybridPoolKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto&     kv_resource  = malloc_info.batch_kv_cache_resource;
    const int batch_size   = kv_resource->batchSize();
    const int seq_len      = malloc_info.complete_token_ids->seqLength();
    const int reserve_step = malloc_info.complete_token_ids->getReserveStep();

    std::vector<std::vector<size_t>> original_sizes(static_cast<size_t>(batch_size));
    for (int b = 0; b < batch_size; ++b) {
        original_sizes[static_cast<size_t>(b)].resize(static_cast<size_t>(kv_resource->groupNums()));
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            original_sizes[static_cast<size_t>(b)][static_cast<size_t>(gid)] = kv_resource->blocksNum(b, gid);
        }
    }

    bool all_success  = true;
    int  failed_batch = -1;
    int  failed_group = -1;
    for (int b = 0; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            auto& block_ids = kv_resource->mutableBlockIds(b, gid);
            if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(
                    block_ids, seq_len, malloc_info.reuse_cache, reserve_step)) {
                all_success  = false;
                failed_batch = b;
                failed_group = gid;
                break;
            }
        }
        if (!all_success) {
            break;
        }
    }

    if (all_success) {
        for (int b = 0; b < batch_size; ++b) {
            for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
                kv_cache_groups_[static_cast<size_t>(gid)]->removeSkippedBlocks(
                    kv_resource->mutableBlockIds(b, gid), malloc_info.reuse_cache, reserve_step);
            }
        }
        return {true, 0};
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            auto&  block_ids    = kv_resource->mutableBlockIds(b, gid);
            size_t original_num = original_sizes[static_cast<size_t>(b)][static_cast<size_t>(gid)];
            if (block_ids.blocksNum() <= original_num) {
                continue;
            }
            BlockIndicesType blocks_to_free;
            const auto&      blk = block_ids.blocks();
            for (size_t i = original_num; i < blk.size(); ++i) {
                if (!isNullBlockIdx(blk[i])) {
                    blocks_to_free.push_back(blk[i]);
                }
            }
            block_ids.resize(original_num);
            if (!blocks_to_free.empty()) {
                group_block_pools_[static_cast<size_t>(gid)]->requestFree(blocks_to_free);
            }
        }
        if (b > failed_batch) {
            break;
        }
    }
    RTP_LLM_LOG_WARNING("HybridPool incrMalloc failed at batch=%d group=%d", failed_batch, failed_group);
    return {false, 0};
}

void HybridPoolKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;
    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }
    for (int batch_id = 0; batch_id < kv_cache_resource->batchSize(); ++batch_id) {
        for (int gid = 0; gid < kv_cache_resource->groupNums(); ++gid) {
            kv_cache_groups_[static_cast<size_t>(gid)]->free(kv_cache_resource->blocks(batch_id, gid));
        }
    }
    kv_cache_resource->clearBlocks();
}

void HybridPoolKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);

    const int batch_size = kv_cache_resource->batchSize();
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        const auto& cache_keys = kv_cache_resource->cacheKeys(batch_id);
        auto        token_ids  = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1 || cache_keys.empty()) {
            continue;
        }
        const size_t token_len = token_ids.size() - 1;
        for (int gid = 0; gid < kv_cache_resource->groupNums(); ++gid) {
            const int    group_seq_size  = kv_cache_groups_[static_cast<size_t>(gid)]->seqSizePerBlock();
            const size_t full_blocks_num = token_len / static_cast<size_t>(group_seq_size);
            const size_t n               = std::min(cache_keys.size(), full_blocks_num);
            if (n == 0) {
                continue;
            }
            CacheKeysType    put_cache_keys(cache_keys.begin(), cache_keys.begin() + n);
            const auto&      blocks = kv_cache_resource->blocks(batch_id, gid);
            BlockIndicesType put_blocks;
            put_blocks.reserve(n);
            for (size_t i = 0; i < n && i < blocks.size(); ++i) {
                put_blocks.push_back(blocks[i]);
            }
            kv_cache_groups_[static_cast<size_t>(gid)]->insertIntoCache(
                put_cache_keys, put_blocks, insert_info.is_resident);
        }
    }
}

CacheLayerLayout HybridPoolKVCacheAllocator::allLayerCacheBase() const {
    CacheLayerLayout layout;
    layout.layer_to_groups        = config_.layer_to_group_id;
    layout.layer_to_group_ids     = config_.layer_to_group_ids;
    layout.layer_attn_to_group_id = config_.layer_attn_to_group_id;
    layout.group_types            = config_.group_types;
    layout.group_attn_types       = config_.group_attn_types;
    layout.layer_attn_types       = config_.layer_attn_types;

    layout.layers_to_kv_buffer_ptrs.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs.resize(config_.layer_all_num);
    const size_t attn_type_count = static_cast<size_t>(KVCacheAttnType::TYPE_COUNT);
    layout.layers_to_kv_buffer_ptrs_by_attn.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs_by_attn.resize(config_.layer_all_num);
    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_all_num); ++layer_id) {
        layout.layers_to_kv_buffer_ptrs_by_attn[layer_id].resize(attn_type_count);
        layout.layers_to_scale_buffer_ptrs_by_attn[layer_id].resize(attn_type_count);
        // Initialize all slots with empty tensors so pybind11 can serialize
        // the vector<vector<Tensor>> without hitting undefined tensors.
        // DSV4: 7 pools, each layer has up to 8 attn_type slots.
        for (size_t j = 0; j < attn_type_count; ++j) {
            layout.layers_to_kv_buffer_ptrs_by_attn[layer_id][j]    = torch::empty({0});
            layout.layers_to_scale_buffer_ptrs_by_attn[layer_id][j] = torch::empty({0});
        }
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
        const auto attn_type     = static_cast<size_t>(gid < static_cast<int>(config_.group_attn_types.size()) ?
                                                       config_.group_attn_types[static_cast<size_t>(gid)] :
                                                       KVCacheAttnType::DEFAULT);
        RTP_LLM_CHECK_WITH_INFO(attn_type < attn_type_count, "group %d has invalid attn type id %zu", gid, attn_type);
        for (const auto& [layer_id, tensor] : layer_tensors) {
            RTP_LLM_CHECK_WITH_INFO(
                layer_id >= 0 && static_cast<size_t>(layer_id) < layout.layers_to_kv_buffer_ptrs_by_attn.size(),
                "layer_id %d out of typed kv layout range %zu",
                layer_id,
                layout.layers_to_kv_buffer_ptrs_by_attn.size());
            layout.layers_to_kv_buffer_ptrs_by_attn[static_cast<size_t>(layer_id)][attn_type] = tensor;
        }
        for (const auto& [layer_id, tensor] : scale_tensors) {
            RTP_LLM_CHECK_WITH_INFO(
                layer_id >= 0 && static_cast<size_t>(layer_id) < layout.layers_to_scale_buffer_ptrs_by_attn.size(),
                "layer_id %d out of typed scale layout range %zu",
                layer_id,
                layout.layers_to_scale_buffer_ptrs_by_attn.size());
            layout.layers_to_scale_buffer_ptrs_by_attn[static_cast<size_t>(layer_id)][attn_type] = tensor;
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
HybridPoolKVCacheAllocator::convertIndexToAddr(int layer_id, KVCacheAttnType attn_type, int block_id) const {
    const int gid = groupIdForLayerAttn(layer_id, attn_type);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo>
HybridPoolKVCacheAllocator::convertIndexToBuffer(int layer_id, KVCacheAttnType attn_type, int block_id) const {
    const int gid = groupIdForLayerAttn(layer_id, attn_type);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridPoolKVCacheAllocator::convertIndexToBuffer(
    int layer_id, KVCacheAttnType attn_type, int block_id, int partition_count, int partition_id) const {
    const int gid = groupIdForLayerAttn(layer_id, attn_type);
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

std::shared_ptr<KVCacheResource> HybridPoolKVCacheAllocator::incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                                            const CacheKeysType&   cache_keys,
                                                                            bool                   is_connector) {
    if (cache_keys.empty() || kvcache_resource.groupNums() <= 0) {
        return nullptr;
    }

    std::unordered_map<CacheKeyType, size_t> key_to_pos;
    const auto&                              resource_keys = kvcache_resource.cacheKeys();
    for (size_t i = 0; i < resource_keys.size(); ++i) {
        key_to_pos.emplace(resource_keys[i], i);
    }

    auto selected_resource_ptr = new KVCacheResource(kvcache_resource);
    auto deleter               = [self = shared_from_this(), is_connector](KVCacheResource* resource) {
        self->decrKVCacheRef(*resource, is_connector);
        delete resource;
    };
    std::shared_ptr<KVCacheResource> selected_resource(selected_resource_ptr, deleter);
    selected_resource->initGroups(kvcache_resource.groupNums(),
                                  static_cast<int>(config_.layer_all_num),
                                  config_.layer_to_group_id,
                                  config_.kernelBlocksPerKvBlock(),
                                  config_.group_types,
                                  config_.layer_attn_to_group_id);

    CacheKeysType&                selected_keys = selected_resource->cacheKeys();
    std::vector<BlockIndicesType> selected_blocks(static_cast<size_t>(kvcache_resource.groupNums()));

    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t pos = it->second;
        for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
            const auto& src_blocks = kvcache_resource.blocks(gid);
            if (pos >= src_blocks.size()) {
                continue;
            }
            selected_blocks[static_cast<size_t>(gid)].push_back(src_blocks[pos]);
        }
    }

    selected_keys.assign(cache_keys.begin(), cache_keys.end());
    for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
        referenceValidBlocks(gid, selected_blocks[static_cast<size_t>(gid)], is_connector);
        selected_resource->mutableBlockIds(gid).assign(std::move(selected_blocks[static_cast<size_t>(gid)]));
    }
    return selected_resource;
}

void HybridPoolKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector) {
    for (int gid = 0; gid < kvcache_resource.groupNums(); ++gid) {
        BlockIndicesType valid;
        for (auto b : kvcache_resource.blocks(gid)) {
            if (!isNullBlockIdx(b) && b > 0) {
                valid.push_back(b);
            }
        }
        if (valid.empty()) {
            continue;
        }
        if (is_connector) {
            group_block_pools_[static_cast<size_t>(gid)]->connectorFree(valid);
        } else {
            group_block_pools_[static_cast<size_t>(gid)]->requestFree(valid);
        }
    }
}

bool HybridPoolKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                               const std::vector<int>&        block_src_batch,
                                               bool                           copy_last_block,
                                               std::vector<BlockIdPair>&      block_update_mapping) {
    (void)batch_kv_cache_resource;
    (void)block_src_batch;
    (void)copy_last_block;
    (void)block_update_mapping;
    return true;
}

int HybridPoolKVCacheAllocator::seqSizePerBlock() const {
    return static_cast<int>(config_.seq_size_per_block);
}

int HybridPoolKVCacheAllocator::getNeedBlocks(const MallocInfo& malloc_info) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return 0;
    }
    const int  batch_size       = malloc_info.batch_kv_cache_resource->batchSize();
    const int  total_seq_len    = malloc_info.complete_token_ids->totalSeqLength();
    const int  common_seq_len   = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);
    const int  seq_len          = malloc_info.complete_token_ids->seqLength();
    const int  reserve_step     = malloc_info.complete_token_ids->getReserveStep();
    const bool reuse_enabled    = malloc_info.reuse_cache;
    const int  reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->curBlocksNum() : 0;

    int common_blocks_total = 0;
    int extra_blocks_total  = 0;
    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        const auto need = kv_cache_groups_[static_cast<size_t>(gid)]->getNeedBlocks(
            common_seq_len, seq_len, reserve_step, reuse_blocks_len, reuse_enabled);
        common_blocks_total += need.common_blocks;
        extra_blocks_total += need.extra_blocks;
    }
    return common_blocks_total + batch_size * extra_blocks_total;
}

int HybridPoolKVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                      int                            seq_len,
                                                      int                            reserve_step) const {
    int need_blocks = 0;
    for (int gid = 0; gid < batch_kv_cache_resource->groupNums(); ++gid) {
        const int cur_blocks = batch_kv_cache_resource->blocksNum(0, gid);
        need_blocks += kv_cache_groups_[static_cast<size_t>(gid)]->needBlocksNum(seq_len, cur_blocks, reserve_step);
    }
    return need_blocks;
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
    (void)min_blocks_to_free;
    return nullptr;
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
    size_t min_tokens = std::numeric_limits<size_t>::max();
    for (size_t gid = 0; gid < group_block_pools_.size(); ++gid) {
        const size_t seq_size =
            (gid < config_.group_seq_size_per_block.size() && config_.group_seq_size_per_block[gid] > 0) ?
                config_.group_seq_size_per_block[gid] :
                config_.seq_size_per_block;
        min_tokens = std::min(min_tokens, group_block_pools_[gid]->totalBlocksNum() * seq_size);
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

}  // namespace rtp_llm
