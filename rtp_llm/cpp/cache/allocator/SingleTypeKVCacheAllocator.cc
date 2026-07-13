#include "rtp_llm/cpp/cache/allocator/SingleTypeKVCacheAllocator.h"

#include <algorithm>
#include <unordered_map>

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/block_tree_cache/DeviceBlockPool.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

int SingleTypeKVCacheAllocator::getNeedBlocks(const MallocInfo& malloc_info) const {
    if (!malloc_info.batch_kv_cache_resource || !malloc_info.complete_token_ids) {
        return 0;
    }
    const bool reuse_enabled    = malloc_info.reuse_cache;
    const int  reuse_blocks_len = reuse_enabled ? malloc_info.batch_kv_cache_resource->curBlocksNum() : 0;
    const int  batch_size       = malloc_info.batch_kv_cache_resource->batchSize();
    int        seq_len          = malloc_info.complete_token_ids->seqLength();
    const int  reserve_step     = malloc_info.complete_token_ids->getReserveStep();
    int        common_seq_len   = std::min(malloc_info.complete_token_ids->commonSeqLength(), seq_len);

    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        seq_len        = cp_slot_mapper_->effectiveSeqLenForAlloc(seq_len);
        common_seq_len = cp_slot_mapper_->effectiveSeqLenForAlloc(common_seq_len);
    }

    const auto need =
        fullGroup()->getNeedBlocks(common_seq_len, seq_len, reserve_step, reuse_blocks_len, reuse_enabled);
    return (batch_size <= 0) ? 0 : (need.common_blocks + batch_size * need.extra_blocks);
}

void SingleTypeKVCacheAllocator::checkCPShardedMallocResult(const MallocInfo& malloc_info) const {
    if (!cp_slot_mapper_ || !cp_slot_mapper_->isSharded()) {
        return;
    }

    const auto& kv_resource       = malloc_info.batch_kv_cache_resource;
    const int   seq_len           = malloc_info.incrSeqLen();
    const int   reserve_step      = malloc_info.complete_token_ids->getReserveStep();
    const int   effective_seq_len = cp_slot_mapper_->effectiveSeqLenForAlloc(seq_len);
    const int   expected_blocks   = fullGroup()->needBlocksNum(effective_seq_len, 0, reserve_step);

    for (int batch_id = 0; batch_id < kv_resource->batchSize(); ++batch_id) {
        const int actual_blocks = kv_resource->blocksNum(batch_id);
        RTP_LLM_CHECK_WITH_INFO(actual_blocks == expected_blocks,
                                "CP invariant violated: batch=%d blocks=%d != expected_local_blocks=%d "
                                "(seq_len=%d, effective_seq_len=%d, reserve_step=%d, cp_size=%d, "
                                "block_size=%d, cacheKeys=%zu)",
                                batch_id,
                                actual_blocks,
                                expected_blocks,
                                seq_len,
                                effective_seq_len,
                                reserve_step,
                                cp_slot_mapper_->cpSize(),
                                cp_slot_mapper_->blockSize(),
                                kv_resource->cacheKeys(batch_id).size());
    }
}

SingleTypeKVCacheAllocator::SingleTypeKVCacheAllocator(const CacheConfig&                 config,
                                                       AllocationType                     allocation_type,
                                                       const kmonitor::MetricsReporterPtr metrics_reporter,
                                                       int64_t                            reserve_block_ratio):
    KVCacheAllocator(config, allocation_type, metrics_reporter, reserve_block_ratio) {}

bool SingleTypeKVCacheAllocator::doInit() {
    RTP_LLM_CHECK_WITH_INFO(config_.groupNums() > 0, "cache groups must not be empty");
    auto& spec = config_.specForGroup(0);
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "cache spec[0] is null");
    RTP_LLM_CHECK_WITH_INFO(spec->type == rtp_llm::KVCacheSpecType::MultiHeadAttention
                                || spec->type == rtp_llm::KVCacheSpecType::MultiHeadLatentAttention,
                            "SingleTypeKVCacheAllocator only support Full Attention");

    BlockPoolConfig pool_config = BlockPoolConfigHelper::createConfig(config_);

    auto device_config                     = std::make_shared<DeviceBlockPoolConfig>();
    device_config->pool_type               = BlockPoolType::DEVICE;
    device_config->pool_name               = pool_config.pool_name;
    device_config->physical_block_count    = pool_config.block_num;
    device_config->total_size_bytes        = pool_config.total_size_bytes;
    device_config->memory_layouts          = pool_config.memory_layouts;
    device_config->allocation_type         = allocation_type_;
    device_config->use_cuda_malloc_backing = use_cuda_malloc_block_pool_;

    std::shared_ptr<const DeviceBlockPoolConfig> const_config = device_config;
    block_pool_                                               = std::make_shared<DeviceBlockPool>(const_config);
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for SingleTypeKVCacheAllocator");
        return false;
    }

    // The DeviceKVCacheGroup is created and owned by BlockTreeCache (see
    // BlockTreeCacheFactory::createBlockTreeCache); this allocator only owns the pool.
    RTP_LLM_LOG_INFO("SingleTypeKVCacheAllocator initialized successfully");
    return true;
}

DeviceKVCacheGroupPtr SingleTypeKVCacheAllocator::fullGroup() const {
    RTP_LLM_CHECK_WITH_INFO(block_tree_cache_ != nullptr,
                            "SingleTypeKVCacheAllocator: BlockTreeCache not injected before group access");
    auto group = block_tree_cache_->deviceKVGroup(0);
    RTP_LLM_CHECK_WITH_INFO(group != nullptr, "SingleTypeKVCacheAllocator: DeviceKVCacheGroup(0) is null");
    return group;
}

MallocResult SingleTypeKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto& kv_resource = malloc_info.batch_kv_cache_resource;
    int   reuse_len   = 0;
    int   common_seq_len =
        std::min(malloc_info.complete_token_ids->commonSeqLength(), malloc_info.complete_token_ids->totalSeqLength());

    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        common_seq_len = cp_slot_mapper_->effectiveSeqLenForAlloc(common_seq_len);
    }

    const auto& cache_keys         = kv_resource->cacheKeys(0);
    auto&       block_ids_0        = kv_resource->mutableBlockIds(0);
    int64_t     match_cost_time_us = 0;

    const size_t reserve_blocks   = reserveBlockNum();
    const int    estimated_blocks = (reserve_blocks > 0) ? getNeedBlocks(malloc_info) : 0;
    int          reuse_blocks     = 0;

    // drop the last cache key of the partial block to avoid reuse it for two reasons:
    // 1. if the last block is partial, it actually cannot be reused, because only full blocks will be inserted into the
    // cache.
    // 2. if the last block is full and matched, the reuse length will be equal to the seq_len, which causes core dump
    // in computing ops.
    if (malloc_info.enable_device_cache) {
        CacheKeysType match_keys;
        if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
            // Drop the last virtual-block key (same reasoning as non-CP) to avoid
            // a full-len reuse / empty-block crash. Use last-rank stride so all
            // ranks share one canonical key namespace.
            int  cp_size     = cp_slot_mapper_->cpSize();
            auto vblock_keys = kv_resource->cacheResource(0).localCacheKeys(cp_size - 1, cp_size);
            match_keys.assign(vblock_keys.begin(), vblock_keys.empty() ? vblock_keys.end() : vblock_keys.end() - 1);
        } else {
            match_keys.assign(cache_keys.begin(), cache_keys.empty() ? cache_keys.end() : cache_keys.end() - 1);
        }
        auto match_begin_time_us = currentTimeUs();
        // Whole-sequence match on the shared BlockTreeCache. match() references the
        // matched device blocks internally; we take the group's indices and release
        // those refs immediately, leaving KVCacheGroup::reference() as the sole owner.
        BlockTreeMatchResult tree_match =
            block_tree_cache_ ? block_tree_cache_->match(match_keys) : BlockTreeMatchResult{};
        const int        group_id = fullGroup()->group_id();
        BlockIndicesType matched_block_indices;
        auto             it = tree_match.group_block_indices.find(group_id);
        if (it != tree_match.group_block_indices.end()) {
            matched_block_indices = it->second;
        }
        reuse_blocks = static_cast<int>(tree_match.matched_blocks);
        if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
            // virtual block ⇒ reuse_length covers cp_size physical blocks of
            // tokens; reuse_blocks counts virtual blocks.
            reuse_len = reuse_blocks * cp_slot_mapper_->virtualBlockSize();
        } else {
            reuse_len = reuse_blocks * fullGroup()->seqSizePerBlock();
        }
        match_cost_time_us = currentTimeUs() - match_begin_time_us;
        kv_resource->cacheResource(0).setDeviceReuseBlockNum(reuse_blocks);
        fullGroup()->reference(block_ids_0, matched_block_indices);
        if (block_tree_cache_) {
            block_tree_cache_->releaseMatchedBlocks(tree_match.matched_block_sets);
        }
    }

    // Check if available blocks are enough for the request.
    if (reserve_blocks > 0 && estimated_blocks > 0) {
        const size_t available_blocks = availableBlocksNum();
        const int    actual_blocks    = std::max(estimated_blocks - reuse_blocks, 0);
        if (actual_blocks > 0 && available_blocks < static_cast<size_t>(actual_blocks) + reserve_blocks) {
            if (malloc_info.verbose) {
                RTP_LLM_LOG_INFO("SingleTypeKVCacheAllocator initMalloc rejected by reserve blocks: request_id=%ld "
                                 "need_blocks=%d reuse_blocks=%d adjusted_need_blocks=%d available_blocks=%zu "
                                 "reserve_blocks=%zu",
                                 malloc_info.request_id,
                                 estimated_blocks,
                                 reuse_blocks,
                                 actual_blocks,
                                 available_blocks,
                                 reserve_blocks);
            }
            return {false, 0};
        }
    }

    if (!fullGroup()->malloc(block_ids_0, common_seq_len)) {
        return {false, 0};
    }

    // other batches reference batch 0's blocks
    for (int batch_id = 1; batch_id < kv_resource->batchSize(); ++batch_id) {
        fullGroup()->reference(kv_resource->mutableBlockIds(batch_id), block_ids_0.blocks());
    }

    return {true, reuse_len, match_cost_time_us};
}

MallocResult SingleTypeKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto& kv_resource    = malloc_info.batch_kv_cache_resource;
    int   batch_size     = kv_resource->batchSize();
    int   current_blocks = kv_resource->curBlocksNum();
    int   seq_len        = malloc_info.incrSeqLen();
    int   reserve_step   = malloc_info.complete_token_ids->getReserveStep();

    if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
        seq_len = cp_slot_mapper_->effectiveSeqLenForAlloc(seq_len);
    }

    auto need_blocks = fullGroup()->needBlocksNum(seq_len, current_blocks, reserve_step);
    if (need_blocks == 0) {
        return {true, 0};
    }

    // Record original sizes for rollback in case any subsequent allocation fails
    std::vector<size_t> original_blocks_num;
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        original_blocks_num.push_back(kv_resource->blocksNum(batch_id));
    }

    bool all_success   = true;
    int  current_batch = 0;
    for (; current_batch < batch_size; ++current_batch) {
        auto& block_ids = kv_resource->mutableBlockIds(current_batch);
        if (!fullGroup()->malloc(block_ids, seq_len, false, reserve_step)) {
            all_success = false;
            break;
        }
    }

    if (all_success) {
        return {true, 0};
    }

    // rollback kvcache blocks
    BlockIndicesType blocks_to_free;
    for (int batch_id = 0; batch_id <= current_batch; ++batch_id) {
        auto& block_ids    = kv_resource->mutableBlockIds(batch_id);
        auto  original_num = original_blocks_num[batch_id];
        if (block_ids.blocksNum() > original_num) {
            const auto& blk = block_ids.blocks();
            blocks_to_free.insert(blocks_to_free.end(), blk.begin() + original_num, blk.end());
            block_ids.resize(original_num);
        }
    }
    if (!blocks_to_free.empty()) {
        fullGroup()->free(blocks_to_free);
    }
    return {false, 0};
}

void SingleTypeKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;

    if (kv_cache_resource->curBlocksNum() == 0) {
        return;
    }

    auto all_blocks = kv_cache_resource->getAllBatchBlocks();
    for (const auto& blocks : all_blocks) {
        fullGroup()->free(blocks);
    }
    kv_cache_resource->clearBlocks();
}

void SingleTypeKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_resource = insert_info.batch_kv_cache_resource;
    if (!block_tree_cache_) {
        return;
    }

    int batch_size = kv_resource->batchSize();
    batch_size     = 1;

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        kv_resource->cacheResource(batch_id).ensureLinearBlockDependencies();
        const auto& blocks = kv_resource->blocks(batch_id);

        // Under CP sharding, use the same last-rank-key canonical namespace as match()
        // (see initMallocForCommonLen) so the device cache stays consistent across ranks
        // without any cross-rank coordination; otherwise use the full key sequence.
        CacheKeysType insert_keys;
        if (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) {
            int cp_size = cp_slot_mapper_->cpSize();
            insert_keys = kv_resource->cacheResource(batch_id).localCacheKeys(cp_size - 1, cp_size);
        } else {
            insert_keys = kv_resource->cacheKeys(batch_id);
        }

        const size_t block_num = std::min(size_t(insert_keys.size()), size_t(blocks.size()));
        if (block_num == 0) {
            continue;
        }
        insert_keys.resize(block_num);

        // One component group (group 0). slots[i][0].device_blocks holds the block
        // for cache_key i; parent-child dependencies are implicit in key ordering.
        std::vector<std::vector<GroupSlot>> slots(block_num, std::vector<GroupSlot>(1));
        bool                                any_block = false;
        for (size_t i = 0; i < block_num; ++i) {
            if (isNullBlockIdx(blocks[i])) {
                continue;
            }
            slots[i][0].device_blocks = {blocks[i]};
            any_block                 = true;
        }
        if (any_block) {
            block_tree_cache_->insert(/*parent=*/nullptr, insert_keys, slots);
        }
    }
}

CacheLayerLayout SingleTypeKVCacheAllocator::allLayerCacheBase() const {
    CacheLayerLayout layout;
    auto             layer_tensors = fullGroup()->allLayerCacheBase();
    auto             scale_tensors = fullGroup()->allLayerScaleCacheBase();

    layout.layers_to_kv_buffer_ptrs.resize(config_.layer_all_num);
    layout.layers_to_scale_buffer_ptrs.resize(config_.layer_all_num);

    for (int layer_id = 0; layer_id < config_.layer_all_num; ++layer_id) {
        if (layer_tensors[layer_id].defined() && layer_tensors[layer_id].numel() > 0) {
            layout.layers_to_kv_buffer_ptrs[layer_id] = layer_tensors[layer_id];
        }
        if (scale_tensors[layer_id].defined() && scale_tensors[layer_id].numel() > 0) {
            layout.layers_to_scale_buffer_ptrs[layer_id] = scale_tensors[layer_id];
        }
    }
    layout.layer_to_group_ids.resize(config_.layer_all_num);
    const int group_id = fullGroup()->group_id();
    for (int layer_id = 0; layer_id < config_.layer_all_num; ++layer_id) {
        layout.layer_to_group_ids[static_cast<size_t>(layer_id)] = {group_id};
    }
    return layout;
}

BlockAddrInfo SingleTypeKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    return fullGroup()->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> SingleTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    return fullGroup()->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo> SingleTypeKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                        int block_id,
                                                                        int partition_count,
                                                                        int partition_id) const {
    return fullGroup()->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

std::shared_ptr<KVCacheResource> SingleTypeKVCacheAllocator::incrKVCacheRef(const KVCacheResource& kvcache_resource,
                                                                            const CacheKeysType&   cache_keys,
                                                                            bool                   is_connector) {
    if (cache_keys.empty()) {
        return nullptr;
    }

    RTP_LLM_CHECK_WITH_INFO(
        kvcache_resource.groupNums() == 1, "incrKVCacheRef expects groupNums==1, got %d", kvcache_resource.groupNums());

    std::unordered_map<CacheKeyType, size_t> key_to_pos;
    const auto&                              resource_keys = kvcache_resource.cacheKeys();
    key_to_pos.reserve(resource_keys.size());
    for (size_t i = 0; i < resource_keys.size(); ++i) {
        key_to_pos.emplace(resource_keys[i], i);
    }

    auto selected_resource_ptr = new KVCacheResource(kvcache_resource);
    auto deleter               = [self = shared_from_this(), is_connector](KVCacheResource* resource) {
        self->decrKVCacheRef(*resource, is_connector);
        delete resource;
    };
    std::shared_ptr<KVCacheResource> selected_resource(selected_resource_ptr, deleter);
    selected_resource->initGroups(
        1, config_.layer_all_num, config_.layerGroupIdsSnapshot(), config_.kernelBlocksPerKvBlock());

    CacheKeysType         selected_cache_keys;
    BlockDependenciesType selected_dependencies;
    BlockIndicesType      selected_blocks;
    BlockIndicesType      referenced_blocks;

    const auto& src_blocks          = kvcache_resource.blocks(0);
    const auto& source_dependencies = kvcache_resource.blockDependencies();

    for (auto key : cache_keys) {
        auto it = key_to_pos.find(key);
        if (it == key_to_pos.end()) {
            continue;
        }
        const size_t pos                     = it->second;
        const bool   preserve_connector_tail = is_connector && !kvcache_resource.lastBlockAligned()
                                             && pos + 1 == resource_keys.size() && !selected_cache_keys.empty();
        if (pos >= src_blocks.size() && !preserve_connector_tail) {
            continue;
        }
        const auto block = pos < src_blocks.size() ? src_blocks[pos] : NULL_BLOCK_IDX;
        if ((block > 0 && !isNullBlockIdx(block)) || preserve_connector_tail) {
            selected_cache_keys.push_back(key);
            selected_dependencies.push_back(
                pos < source_dependencies.size() ?
                    source_dependencies[pos] :
                    BlockDependency{false, 0, static_cast<uint32_t>(selected_dependencies.size())});
            selected_blocks.push_back(block);
            if (block > 0 && !isNullBlockIdx(block)) {
                referenced_blocks.push_back(block);
            }
        }
    }

    if (referenced_blocks.empty()) {
        return nullptr;
    }

    // Single-count pool: request and connector holders share one reference category
    // (is_connector still distinguishes the release path via the deleter above).
    block_pool_->incRef(referenced_blocks);
    selected_resource->mutableBlockIds(0).assign(std::move(selected_blocks));
    selected_resource->setCacheKeys(std::move(selected_cache_keys));
    selected_resource->setBlockDependencies(std::move(selected_dependencies));

    return selected_resource;
}

void SingleTypeKVCacheAllocator::decrKVCacheRef(const KVCacheResource& kvcache_resource, bool is_connector) {
    RTP_LLM_CHECK_WITH_INFO(
        kvcache_resource.groupNums() == 1, "decrKVCacheRef expects groupNums==1, got %d", kvcache_resource.groupNums());

    BlockIndicesType blocks_to_free;
    for (auto block : kvcache_resource.blocks(0)) {
        if (block > 0 && !isNullBlockIdx(block)) {
            blocks_to_free.push_back(block);
        }
    }
    if (!blocks_to_free.empty()) {
        // Single-count pool: both request and connector releases decrement one holder.
        (void)is_connector;
        block_pool_->releaseRef(blocks_to_free);
    }
}

// Update kv blocks for beam search or multi-return sequences.
// - batch_kv_cache_resource: in/out, batch blocks and cache_keys will be rearranged based on block_src_batch
// - block_src_batch: new batch i forks from old batch block_src_batch[i]
// - copy_last_block: whether to copy the last block for each forked batch (instead of sharing)
// - block_update_mapping: out, mapping from old block to new block for batch copy
bool SingleTypeKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr& kv_cache_resource,
                                               const std::vector<int>&        block_src_batch,
                                               bool                           copy_last_block,
                                               std::vector<BlockIdPair>&      block_update_mapping) {
    block_update_mapping.clear();
    if (block_src_batch.empty()) {
        return true;
    }

    const int        old_batch_size = kv_cache_resource->batchSize();
    const int        new_batch_size = static_cast<int>(block_src_batch.size());
    std::vector<int> batch_fork_count(old_batch_size, 0);
    for (const int old_batch_idx : block_src_batch) {
        RTP_LLM_CHECK_WITH_INFO(old_batch_idx < old_batch_size,
                                "try to reuse an old batch %d that out of range %d",
                                old_batch_idx,
                                old_batch_size);
        ++batch_fork_count[old_batch_idx];
    }

    std::vector<int> disused_kv_blocks;
    uint32_t         new_blocks_num = 0;
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        const int fork_count = batch_fork_count[old_batch_idx];
        if (fork_count == 0) {
            const auto& blocks = kv_cache_resource->blocks(old_batch_idx);
            disused_kv_blocks.insert(disused_kv_blocks.end(), blocks.begin(), blocks.end());
        } else if (fork_count > 1 && copy_last_block) {
            new_blocks_num += static_cast<uint32_t>(fork_count - 1);
        }
    }

    // free disused first to reclaim capacity
    if (!disused_kv_blocks.empty()) {
        fullGroup()->free(disused_kv_blocks);
    }

    // ensure there are enough free blocks for last-block copies
    if (new_blocks_num > 0) {
        if (!fullGroup()->ensureFreeBlocks(static_cast<int>(new_blocks_num))) {
            RTP_LLM_LOG_WARNING("ensure free blocks failed for kv cache update, need %u", new_blocks_num);
            return false;
        }
    }

    // rebuild batch_kv_cache_resource and generate mapping
    std::vector<KVCacheResource> old_resources;
    kv_cache_resource->resetAndReturnOldResources(new_batch_size, old_resources);

    // init for all batch
    kv_cache_resource->initGroups(
        1, config_.layer_all_num, config_.layerGroupIdsSnapshot(), config_.kernelBlocksPerKvBlock());

    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        const int old_batch_idx = block_src_batch[new_batch_idx];
        auto&     fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);

        if (fork_count == 1) {
            kv_cache_resource->moveBatchResource(new_batch_idx, std::move(old_resources[old_batch_idx]));
        } else {
            auto& block_ids = kv_cache_resource->mutableBlockIds(new_batch_idx);
            kv_cache_resource->setBatchCacheKeys(new_batch_idx, old_resources[old_batch_idx].cacheKeys());
            fullGroup()->reference(block_ids, old_resources[old_batch_idx].blocks());

            if (copy_last_block && !block_ids.blocks().empty()) {
                const int old_block = block_ids.popBack();
                fullGroup()->free({old_block});

                // allocate exactly one new block via kvCacheGroup
                int seq_len_target = (static_cast<int>(block_ids.blocks().size()) + 1) * fullGroup()->seqSizePerBlock();
                bool ok            = fullGroup()->malloc(block_ids, seq_len_target);
                RTP_LLM_CHECK_WITH_INFO(ok, "malloc one block via kvCacheGroup failed during kv cache update");
                const int new_block = block_ids.blocks().back();
                block_update_mapping.push_back(BlockIdPair{old_block, new_block});
            }
        }
        --fork_count;
    }
    return true;
}

int SingleTypeKVCacheAllocator::seqSizePerBlock() const {
    return fullGroup()->seqSizePerBlock();
}

int SingleTypeKVCacheAllocator::singleBatchNeedBlocks(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                      int                            seq_len,
                                                      int                            reserve_step) const {
    (void)batch_kv_cache_resource;
    const int effective_seq_len =
        (cp_slot_mapper_ && cp_slot_mapper_->isSharded()) ? cp_slot_mapper_->effectiveSeqLenForAlloc(seq_len) : seq_len;
    return fullGroup()->needBlocksNum(effective_seq_len, 0, reserve_step);
}

}  // namespace rtp_llm
