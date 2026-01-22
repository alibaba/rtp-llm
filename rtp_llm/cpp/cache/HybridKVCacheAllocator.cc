#include "rtp_llm/cpp/cache/HybridKVCacheAllocator.h"

#include <algorithm>
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
                                                         const kmonitor::MetricsReporterPtr metrics_reporter):
    KVCacheAllocator(config, device, allocation_type, metrics_reporter) {}

bool HybridLayerKVCacheAllocator::init() {
    if (config_.cache_specs.empty()) {
        RTP_LLM_LOG_ERROR("no cache_specs found in CacheConfig");
        return false;
    }

    auto pool_config = BlockPoolConfigHelper::createLayerFirstConfig(config_);
    block_pool_ = std::make_shared<BlockPool>(pool_config, device_, allocation_type_);
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for HybridLayerKVCacheAllocator");
        return false;
    }

    const auto& layer_groups = config_.layer_ids;
    const int group_nums = static_cast<int>(layer_groups.size());
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
        if (spec && spec->type == KVCacheType::LinearAttention) {
            group = std::make_shared<LinearKVCacheGroup>(ids, spec, block_pool_, gid, config_.linear_step);
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
            const int gid    = linear_group_ids_[i];
            auto*     linear_group = dynamic_cast<LinearKVCacheGroup*>(kv_cache_groups_[static_cast<size_t>(gid)].get());
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
        const int gid    = linear_group_ids_[i];
        auto&     blocks = kv_resource.mutableBlocks(0, gid);
        blocks[static_cast<size_t>(reuse_blocks_len - 1)] = linear_tail_blocks[i];
    }


    return reuse_blocks_len;
}

MallocResult HybridLayerKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();
    const int seq_len     = malloc_info.complete_token_ids->totalSeqLength();

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
            auto* linear_group =
                dynamic_cast<LinearKVCacheGroup*>(kv_cache_groups_[static_cast<size_t>(gid)].get());
            if (linear_group) {
                if (!linear_group->malloc(blocks, seq_len, kv_resource->enable_reuse_cache)) {
                    all_success  = false;
                    failed_batch = b;
                    failed_group = gid;
                    break;
                }
            } else if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(blocks, seq_len)) {
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
                kv_cache_groups_[static_cast<size_t>(gid)]->removeSkippedBlocks(kv_resource->mutableBlocks(b, gid));
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
    RTP_LLM_LOG_WARNING("Hybrid incrMalloc failed at batch=%d group=%d", failed_batch, failed_group);
    return {false, 0};
}

MallocResult HybridLayerKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto&     kv_resource = malloc_info.batch_kv_cache_resource;
    const int batch_size  = kv_resource->batchSize();

    const int total_seq_len  = malloc_info.complete_token_ids->totalSeqLength();
    const int common_seq_len = std::min(malloc_info.complete_token_ids->commonSeqLength(), total_seq_len);

    const auto& cache_keys         = kv_resource->cacheKeys(0);
    int64_t     match_cost_time_us = 0;
    const size_t reserve_blocks   = reserveBlockNum();
    int          reuse_blocks     = 0;

    // Drop last key of partial block (same rationale as SingleType).
    if (kv_resource->enable_reuse_cache) {
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
                RTP_LLM_LOG_INFO("Hybrid initMalloc rejected by reserve blocks: request_id=%ld "
                                 "need_blocks=%d available_blocks=%zu "
                                 "reserve_blocks=%zu",
                                 malloc_info.request_id,
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
        auto* linear_group = dynamic_cast<LinearKVCacheGroup*>(kv_cache_groups_[static_cast<size_t>(gid)].get());
        if (!linear_group) {
            if (!kv_cache_groups_[static_cast<size_t>(gid)]->malloc(blocks_0, common_seq_len)) {
                return {false, 0};
            }
            continue;
        }

        // For linear-attn groups:
        // - reuse_cache=true: allocate blocks at linear-step intervals over the whole common length and the tail block.
        // - reuse_cache=false: only keep the tail block.
        const int common_slots = linear_group->needBlocksNum(common_seq_len, /*current_blocks=*/0);
        if (common_slots <= 0) {
            blocks_0.clear();
            continue;
        }
        if (blocks_0.size() < static_cast<size_t>(common_slots)) {
            blocks_0.resize(static_cast<size_t>(common_slots), NULL_BLOCK_IDX);
        }

        const int step = std::max(1, config_.linear_step);
        if (kv_resource->enable_reuse_cache) {
            for (int i = reuse_blocks; i < common_slots; ++i) {
                const bool should_alloc = (((i + 1) % step) == 0) || (i == common_slots - 1);
                if (!should_alloc) {
                    continue;
                }
                if (!isNullBlockIdx(blocks_0[static_cast<size_t>(i)])) {
                    continue;  // already reused
                }
                auto result = block_pool_->malloc(1);
                if (result.empty()) {
                    return {false, 0};
                }
                blocks_0[static_cast<size_t>(i)] = result[0];
            }
        } else {
            auto& tail = blocks_0[static_cast<size_t>(common_slots - 1)];
            if (isNullBlockIdx(tail)) {
                auto result = block_pool_->malloc(1);
                if (result.empty()) {
                    return {false, 0};
                }
                tail = result[0];
            }
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
            const auto&      blocks = kv_cache_resource->blocks(batch_id, gid);
            BlockIndicesType put_blocks;
            put_blocks.reserve(n);
            for (size_t i = 0; i < n && i < blocks.size(); ++i) {
                put_blocks.push_back(blocks[i]);
            }
            kv_cache_groups_[static_cast<size_t>(gid)]->insertIntoCache(
                put_cache_keys, put_blocks, insert_info.is_resident);
            // Prefill memory reclaim for linear groups.
            kv_cache_groups_[static_cast<size_t>(gid)]->removeSkippedBlocks(
                kv_cache_resource->mutableBlocks(batch_id, gid));
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
        if (layer_id < layer_tensors.size() && layer_tensors[layer_id].defined()
            && layer_tensors[layer_id].numel() > 0) {
            layout.layers_to_buffer_ptrs[layer_id] = torchTensor2Buffer(layer_tensors[layer_id]);
        }
        if (!scale_tensors.empty() && layer_id < scale_tensors.size() && scale_tensors[layer_id].defined()
            && scale_tensors[layer_id].numel() > 0) {
            layout.layers_to_scale_buffer_ptrs[layer_id] = torchTensor2Buffer(scale_tensors[layer_id]);
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

BlockBufferPtrInfo HybridLayerKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    if (layer_id < 0 || layer_id >= static_cast<int>(layer_to_group_id_.size())) {
        RTP_LLM_FAIL("convertIndexToBuffer invalid layer_id=%d", layer_id);
    }
    const int gid = layer_to_group_id_[static_cast<size_t>(layer_id)];
    RTP_LLM_CHECK_WITH_INFO(gid >= 0 && gid < static_cast<int>(kv_cache_groups_.size()), "invalid group id mapping");
    return kv_cache_groups_[static_cast<size_t>(gid)]->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BufferPtr> HybridLayerKVCacheAllocator::convertIndexToBuffer(int layer_id,
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

std::shared_ptr<KVCacheResource> HybridLayerKVCacheAllocator::incrKVCacheRef(KVCacheResource&     kvcache_resource,
                                                                             const CacheKeysType& cache_keys) {
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

    auto selected = std::make_shared<KVCacheResource>();
    selected->initGroups(group_nums);

    CacheKeysType& selected_keys = selected->cacheKeys();
    std::vector<BlockIndicesType> selected_blocks(static_cast<size_t>(group_nums));

    BlockIndicesType blocks_to_reference;
    blocks_to_reference.reserve(cache_keys.size());

    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t pos       = it->second;
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
    block_pool_->blockCacheReference(blocks_to_reference);

    for (int gid = 0; gid < group_nums; ++gid) {
        selected->blocks(gid) = std::move(selected_blocks[static_cast<size_t>(gid)]);
    }

    return selected;
}

void HybridLayerKVCacheAllocator::decrKVCacheRef(KVCacheResource& kvcache_resource) {
    const int group_nums = kvcache_resource.groupNums();
    std::vector<int> blocks_to_free;
    for (int gid = 0; gid < group_nums; ++gid) {
        const auto& blocks = kvcache_resource.blocks(gid);
        for (auto b : blocks) {
            if (!isNullBlockIdx(b) && b > 0) {
                blocks_to_free.push_back(b);
            }
        }
    }
    block_pool_->blockCacheFree(blocks_to_free);
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

    const bool reuse_enabled   = malloc_info.batch_kv_cache_resource->enable_reuse_cache;
    const int  reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->curBlocksNum() : 0;

    const int linear_step = std::max(1, config_.linear_step);
    // calculate the number of blocks in the range (begin, end]
    auto count_linear_sparse_range = [&](int begin, int end) -> int {
        // We only allocate linear-step blocks when reuse_cache is enabled; otherwise only tail is kept.
        if (!reuse_enabled) {
            return 0;
        }
        const int eligible = (end + 1) / linear_step - (begin + 1) / linear_step;
        const int tail = ((end + 1) % linear_step == 0) ? 0 : 1;
        return eligible + tail;
    };

    int common_blocks_total = 0;
    int extra_blocks_total  = 0;

    const int common_slots = kv_cache_groups_[0]->needBlocksNum(common_seq_len, /*current_blocks=*/0);
    const int total_slots  = kv_cache_groups_[0]->needBlocksNum(total_seq_len, /*current_blocks=*/0);

    common_blocks_total += count_linear_sparse_range(reuse_blocks_len, common_slots) * config_.linear_group_num;
    extra_blocks_total += count_linear_sparse_range(common_slots, total_slots) * config_.linear_group_num;

    common_blocks_total += std::max(common_slots - reuse_blocks_len, 0) * config_.full_group_num;
    extra_blocks_total += std::max(total_slots - common_slots, 0) * config_.full_group_num;

    return common_blocks_total + batch_size * extra_blocks_total;
}

int HybridLayerKVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                       int                            seq_len) const {
    // blocks number in each group are equal 
    const int cur_blocks = batch_kv_cache_resource->blocksNum(0, 0);
    return kv_cache_groups_[0]->needBlocksNum(seq_len, cur_blocks) * batch_kv_cache_resource->groupNums();
}

}  // namespace rtp_llm
