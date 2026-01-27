#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {
HybridLayerKVCacheAllocator::HybridLayerKVCacheAllocator(const CacheConfig&                 config,
                                                         rtp_llm::DeviceBase*               device,
                                                         AllocationType                     allocation_type,
                                                         const kmonitor::MetricsReporterPtr metrics_reporter,
                                                         RoleType                           role_type,
                                                         int64_t                            reserve_block_ratio):
    KVCacheAllocator(config, device, allocation_type, metrics_reporter, reserve_block_ratio), role_type_(role_type) {}

bool HybridLayerKVCacheAllocator::doInit() {
    if (config_.cache_specs.empty()) {
        RTP_LLM_LOG_ERROR("no cache_specs found in CacheConfig");
        return false;
    }

    auto pool_config = BlockPoolConfigHelper::createConfig(config_);
    block_pool_      = std::make_shared<BlockPool>(pool_config, device_, allocation_type_);
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for HybridLayerKVCacheAllocator");
        return false;
    }

    const auto& layer_groups = config_.layer_ids;
    const int   group_nums   = static_cast<int>(layer_groups.size());
    kv_cache_groups_.clear();
    kv_cache_groups_.reserve(group_nums);
    full_group_ids_.clear();
    linear_group_ids_.clear();

    // global layer id -> group id mapping (for address lookup APIs)
    layer_to_group_id_.assign(static_cast<size_t>(config_.layer_num), -1);

    for (int gid = 0; gid < group_nums; ++gid) {
        KVCacheSpecPtr spec = config_.cache_specs[static_cast<size_t>(gid)];
        const auto&    ids  = layer_groups[static_cast<size_t>(gid)];

        KVCacheGroupPtr group;
        if (spec && spec->type == KVCacheSpecType::LinearAttention) {
            group = std::make_shared<LinearKVCacheGroup>(ids, spec, block_pool_, gid, config_.linear_step, role_type_);
            linear_group_ids_.push_back(gid);
        } else {
            group = std::make_shared<FullKVCacheGroup>(ids, spec, block_pool_, gid);
            full_group_ids_.push_back(gid);
        }

        if (!group->init()) {
            RTP_LLM_LOG_ERROR("Failed to initialize KVCacheGroup gid=%d", gid);
            return false;
        }
        kv_cache_groups_.push_back(group);

        for (int layer_id : ids) {
            layer_to_group_id_[static_cast<size_t>(layer_id)] = gid;
        }
    }

    global_layer_to_local_id_.assign(static_cast<size_t>(config_.layer_num), -1);
    for (const auto& group_layers : config_.layer_ids) {
        for (size_t local = 0; local < group_layers.size(); ++local) {
            const int global_layer = group_layers[local];
            if (global_layer >= 0 && static_cast<size_t>(global_layer) < global_layer_to_local_id_.size()) {
                global_layer_to_local_id_[static_cast<size_t>(global_layer)] = static_cast<int>(local);
            }
        }
    }

    RTP_LLM_LOG_INFO("HybridLayerKVCacheAllocator init success");
    return true;
}

void HybridLayerKVCacheAllocator::referenceValidBlocks(const BlockIndicesType& blocks) const {
    BlockIndicesType valid;
    valid.reserve(blocks.size());
    for (auto b : blocks) {
        if (!isNullBlockIdx(b)) {
            valid.push_back(b);
        }
    }
    if (!valid.empty()) {
        block_pool_->requestReference(valid);
    }
}

int HybridLayerKVCacheAllocator::reuseCache(const CacheKeysType& cache_keys, BatchKVCacheResource& kv_resource) {
    // 1) Prefix match on all full-attn groups, take the shortest prefix.
    int                           min_full_reuse_blocks = static_cast<int>(cache_keys.size());
    std::vector<BlockIndicesType> full_matched_blocks(kv_cache_groups_.size());

    for (int gid : full_group_ids_) {
        auto match_result     = kv_cache_groups_[static_cast<size_t>(gid)]->match(cache_keys);
        min_full_reuse_blocks = std::min(min_full_reuse_blocks, static_cast<int>(match_result.reuse_blocks));
        full_matched_blocks[static_cast<size_t>(gid)] = std::move(match_result.block_indices);
    }

    // 2) Right-to-left joint check for all linear groups (single-key match).
    int                       pos = min_full_reuse_blocks - 1;
    std::vector<BlockIdxType> linear_tail_blocks;  // per linear group
    linear_tail_blocks.resize(linear_group_ids_.size(), NULL_BLOCK_IDX);

    for (; pos >= 0; --pos) {
        bool all_linear_matched = true;
        for (size_t i = 0; i < linear_group_ids_.size(); ++i) {
            const int gid      = linear_group_ids_[i];
            auto* linear_group = dynamic_cast<LinearKVCacheGroup*>(kv_cache_groups_[static_cast<size_t>(gid)].get());
            auto  result       = linear_group->matchSingleKey(cache_keys[static_cast<size_t>(pos)]);
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

    // Write matched blocks into batch 0 blocks, per group.
    // NOTE: for linear groups we only reuse the tail block; other slots are set to NULL_BLOCK_IDX.
    for (int gid = 0; gid < static_cast<int>(kv_cache_groups_.size()); ++gid) {
        auto& blocks = kv_resource.mutableBlocks(0, gid);
        blocks.clear();
        blocks.resize(static_cast<size_t>(reuse_blocks_len), NULL_BLOCK_IDX);
    }

    for (int gid : full_group_ids_) {
        auto& blocks = kv_resource.mutableBlocks(0, gid);
        blocks       = full_matched_blocks[static_cast<size_t>(gid)];
        if (static_cast<int>(blocks.size()) > reuse_blocks_len) {
            blocks.resize(static_cast<size_t>(reuse_blocks_len));
        }
    }

    for (size_t i = 0; i < linear_group_ids_.size(); ++i) {
        const int gid                                     = linear_group_ids_[i];
        auto&     blocks                                  = kv_resource.mutableBlocks(0, gid);
        blocks[static_cast<size_t>(reuse_blocks_len - 1)] = linear_tail_blocks[i];
    }

    return reuse_blocks_len;
}

MallocResult HybridLayerKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto&     kv_resource  = malloc_info.batch_kv_cache_resource;
    const int batch_size   = kv_resource->batchSize();
    const int seq_len      = malloc_info.complete_token_ids->seqLength();
    const int reserve_step = malloc_info.complete_token_ids->getReserveStep();

    // Record original sizes for rollback in case any subsequent allocation fails
    std::vector<std::vector<size_t>> original_sizes(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        original_sizes[b].resize(static_cast<size_t>(kv_resource->groupNums()));
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            original_sizes[b][static_cast<size_t>(gid)] = kv_resource->blocksNum(b, gid);
        }
    }

    bool all_success  = true;
    int  failed_batch = -1;
    int  failed_group = -1;

    for (int b = 0; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            auto& blocks = kv_resource->mutableBlocks(b, gid);

            if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(
                    blocks, seq_len, malloc_info.enable_device_cache, reserve_step)) {
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
        // Decode-time memory saving for linear groups (apply after we know allocations succeeded).
        for (int b = 0; b < batch_size; ++b) {
            for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
                kv_cache_groups_[static_cast<size_t>(gid)]->removeSkippedBlocks(
                    kv_resource->mutableBlocks(b, gid), malloc_info.enable_device_cache, reserve_step);
            }
        }
        return {true, 0};
    }

    // rollback kvcache blocks
    BlockIndicesType blocks_to_free;

    for (int b = 0; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            auto&  blocks       = kv_resource->mutableBlocks(b, gid);
            size_t original_num = original_sizes[b][static_cast<size_t>(gid)];
            if (blocks.size() > original_num) {
                for (size_t i = original_num; i < blocks.size(); ++i) {
                    if (!isNullBlockIdx(blocks[i])) {
                        blocks_to_free.push_back(blocks[i]);
                    }
                }
                blocks.resize(original_num);
            }
        }
        if (b > failed_batch) {
            break;
        }
    }
    if (!blocks_to_free.empty()) {
        // All groups share the same block pool; free directly.
        block_pool_->requestFree(blocks_to_free);
    }
    RTP_LLM_LOG_WARNING("Hybrid incrMalloc failed at batch=%d group=%d role_type=%d",
                        failed_batch,
                        failed_group,
                        static_cast<int>(role_type_));
    return {false, 0};
}

MallocResult HybridLayerKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();

    const int seq_len        = malloc_info.complete_token_ids->seqLength();
    const int common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);

    const auto&  cache_keys         = kv_resource->cacheKeys(0);
    int64_t      match_cost_time_us = 0;
    const size_t reserve_blocks     = reserveBlockNum();
    int          reuse_blocks       = 0;

    // If role_type is DECODE, we skip match since the kv cache block of linear groups should always be transfered from
    // prefill node
    if (malloc_info.enable_device_cache && role_type_ != RoleType::DECODE) {
        // Drop last key of partial block (same rationale as SingleType).
        CacheKeysType match_keys(cache_keys.begin(), cache_keys.empty() ? cache_keys.end() : cache_keys.end() - 1);
        auto          begin_us = currentTimeUs();
        reuse_blocks           = reuseCache(match_keys, *kv_resource);
        match_cost_time_us     = currentTimeUs() - begin_us;

        // Reference reused blocks in batch 0 (filter NULL_BLOCK_IDX).
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            const auto& blocks = kv_resource->blocks(0, gid);
            referenceValidBlocks(blocks);
        }
    }

    const int need_blocks = (reserve_blocks > 0) ? getNeedBlocks(malloc_info) : 0;
    // Reserve blocks check (best-effort, similar to SingleType).
    if (reserve_blocks > 0 && need_blocks > 0) {
        const size_t available_blocks = availableBlocksNum();
        if (available_blocks < static_cast<size_t>(need_blocks) + reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("Hybrid initMalloc rejected by reserve blocks: request_id=%ld role_type=%d "
                                 "need_blocks=%d available_blocks=%zu "
                                 "reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 static_cast<int>(role_type_),
                                 need_blocks,
                                 available_blocks,
                                 reserve_blocks);
            }
            return {false, 0};
        }
    }

    // Allocate common blocks on batch 0.
    for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
        auto& blocks_0 = kv_resource->mutableBlocks(0, gid);
        if (role_type_ == RoleType::DECODE) {
            if (dynamic_cast<LinearKVCacheGroup*>(kv_cache_groups_[static_cast<size_t>(gid)].get()) != nullptr) {
                continue;
            }
        }

        // Common blocks are shared across batches; reserve_step is per-batch extra and will be handled in incrMalloc.
        if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(
                blocks_0, common_seq_len, malloc_info.enable_device_cache, 0)) {
            return {false, 0};
        }
    }

    // Other batches reference batch 0's common blocks.
    for (int b = 1; b < batch_size; ++b) {
        for (int gid = 0; gid < kv_resource->groupNums(); ++gid) {
            kv_cache_groups_[static_cast<size_t>(gid)]->reference(kv_resource->mutableBlocks(b, gid),
                                                                  kv_resource->blocks(0, gid));
        }
    }

    return {true, reuse_blocks * seqSizePerBlock(), match_cost_time_us};
}

void HybridLayerKVCacheAllocator::free(const FreeInfo& free_info) {
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

void HybridLayerKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);

    int batch_size         = kv_cache_resource->batchSize();
    int seq_size_per_block = seqSizePerBlock();

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        const auto& cache_keys = kv_cache_resource->cacheKeys(batch_id);

        auto token_ids = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1 || cache_keys.empty()) {
            continue;
        }

        // Only insert full blocks.
        const size_t token_len       = token_ids.size() - 1;
        const size_t full_blocks_num = token_len / static_cast<size_t>(seq_size_per_block);
        const size_t n               = std::min(cache_keys.size(), full_blocks_num);
        if (n == 0) {
            continue;
        }

        CacheKeysType put_cache_keys(cache_keys.begin(), cache_keys.begin() + n);
        for (int gid = 0; gid < kv_cache_resource->groupNums(); ++gid) {
            auto&            blocks = kv_cache_resource->mutableBlocks(batch_id, gid);
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

CacheLayerLayout HybridLayerKVCacheAllocator::allLayerCacheBase() const {
    CacheLayerLayout layout;
    const auto       layer_tensors = block_pool_->allLayerCacheBase();
    const auto       scale_tensors = block_pool_->allLayerScaleCacheBase();

    layout.layer_to_groups = layer_to_group_id_;
    layout.layers_to_buffer_ptrs.assign(config_.layer_num, nullptr);
    layout.layers_to_scale_buffer_ptrs.assign(config_.layer_num, nullptr);

    for (size_t layer_id = 0; layer_id < static_cast<size_t>(config_.layer_num); ++layer_id) {
        int32_t      local     = global_layer_to_local_id_[layer_id];
        const size_t local_idx = static_cast<size_t>(local);

        if (local_idx < layer_tensors.size() && layer_tensors[local_idx].defined()
            && layer_tensors[local_idx].numel() > 0) {
            layout.layers_to_buffer_ptrs[layer_id] = torchTensor2Buffer(layer_tensors[local_idx]);
        }
        if (!scale_tensors.empty() && local_idx < scale_tensors.size() && scale_tensors[local_idx].defined()
            && scale_tensors[local_idx].numel() > 0) {
            layout.layers_to_scale_buffer_ptrs[layer_id] = torchTensor2Buffer(scale_tensors[local_idx]);
        }
    }
    return layout;
}

BlockAddrInfo HybridLayerKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    if (layer_id < 0 || layer_id >= static_cast<int>(layer_to_group_id_.size())) {
        RTP_LLM_FAIL("convertIndexToAddr invalid layer_id=%d", layer_id);
    }
    const int gid = layer_to_group_id_[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()), "invalid group id mapping");
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> HybridLayerKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    if (layer_id < 0 || layer_id >= static_cast<int>(layer_to_group_id_.size())) {
        RTP_LLM_FAIL("convertIndexToBuffer invalid layer_id=%d", layer_id);
    }
    const int gid = layer_to_group_id_[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()), "invalid group id mapping");
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> HybridLayerKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                         int block_id,
                                                                         int partition_count,
                                                                         int partition_id) const {
    if (layer_id < 0 || layer_id >= static_cast<int>(layer_to_group_id_.size())) {
        RTP_LLM_FAIL("convertIndexToBuffer(partition) invalid layer_id=%d", layer_id);
    }
    const int gid = layer_to_group_id_[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()), "invalid group id mapping");
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(
        layer_id, block_id, partition_count, partition_id);
}

std::shared_ptr<KVCacheResource> HybridLayerKVCacheAllocator::incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                                             const CacheKeysType&   cache_keys) {
    if (cache_keys.empty()) {
        return nullptr;
    }

    const int group_nums = kvcache_resource.groupNums();
    if (group_nums <= 0) {
        return nullptr;
    }

    std::unordered_map<CacheKeyType, size_t> key_to_pos;
    const auto&                              resource_keys = kvcache_resource.cacheKeys();
    key_to_pos.reserve(resource_keys.size());
    for (size_t i = 0; i < resource_keys.size(); ++i) {
        key_to_pos.emplace(resource_keys[i], i);
    }

    auto selected_resource_ptr = new KVCacheResource(kvcache_resource);
    auto deleter               = [self = shared_from_this()](KVCacheResource* resource) {
        self->decrKVCacheRef(*resource);
        delete resource;
    };
    std::shared_ptr<KVCacheResource> selected_resource(selected_resource_ptr, deleter);
    selected_resource->initGroups(group_nums, static_cast<int>(config_.layer_all_num), config_.layer_to_group_id);

    CacheKeysType&                selected_keys = selected_resource->cacheKeys();
    std::vector<BlockIndicesType> selected_blocks(static_cast<size_t>(group_nums));

    BlockIndicesType blocks_to_reference;
    blocks_to_reference.reserve(cache_keys.size());

    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t pos = it->second;
        for (int gid = 0; gid < group_nums; ++gid) {
            auto& src_blocks = kvcache_resource.blocks(gid);
            if (pos >= src_blocks.size()) {
                continue;
            }
            const auto block = src_blocks[pos];
            selected_blocks[static_cast<size_t>(gid)].push_back(block);
            if (!isNullBlockIdx(block) && block > 0) {
                blocks_to_reference.push_back(block);
            }
        }
    }

    selected_keys.assign(cache_keys.begin(), cache_keys.end());
    block_pool_->requestReference(blocks_to_reference);

    for (int gid = 0; gid < group_nums; ++gid) {
        selected_resource->blocks(gid) = std::move(selected_blocks[static_cast<size_t>(gid)]);
    }

    return selected_resource;
}

void HybridLayerKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource) {
    const int        group_nums = kvcache_resource.groupNums();
    std::vector<int> blocks_to_free;
    for (int gid = 0; gid < group_nums; ++gid) {
        const auto& blocks = kvcache_resource.blocks(gid);
        for (auto b : blocks) {
            if (!isNullBlockIdx(b) && b > 0) {
                blocks_to_free.push_back(b);
            }
        }
    }
    block_pool_->requestFree(blocks_to_free);
}

int HybridLayerKVCacheAllocator::seqSizePerBlock() const {
    return static_cast<int>(config_.seq_size_per_block);
}

bool HybridLayerKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                const std::vector<int>&        block_src_batch,
                                                bool                           copy_last_block,
                                                std::vector<BlockIdPair>&      block_update_mapping) {
    // TODO(chanyin): may be implemented in Base class in future
    return true;
}

int HybridLayerKVCacheAllocator::getNeedBlocks(const MallocInfo& malloc_info) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return 0;
    }
    const int batch_size     = malloc_info.batch_kv_cache_resource->batchSize();
    const int total_seq_len  = malloc_info.complete_token_ids->totalSeqLength();
    const int common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);

    const int seq_len      = malloc_info.complete_token_ids->seqLength();
    const int reserve_step = malloc_info.complete_token_ids->getReserveStep();

    const bool reuse_enabled    = malloc_info.enable_device_cache;
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

int HybridLayerKVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                       int                            seq_len,
                                                       int                            reserve_step) const {
    int need_blocks = 0;
    for (int gid = 0; gid < batch_kv_cache_resource->groupNums(); ++gid) {
        const int cur_blocks = batch_kv_cache_resource->blocksNum(0, gid);
        need_blocks += kv_cache_groups_[static_cast<size_t>(gid)]->needBlocksNum(seq_len, cur_blocks, reserve_step);
    }

    return need_blocks;
}

}  // namespace rtp_llm
