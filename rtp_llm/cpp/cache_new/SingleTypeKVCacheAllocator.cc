#include "rtp_llm/cpp/cache_new/SingleTypeKVCacheAllocator.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache_new/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

SingleTypeKVCacheAllocator::SingleTypeKVCacheAllocator(const CacheConfig&   config,
                                                       rtp_llm::DeviceBase* device,
                                                       AllocationType       allocation_type):
    KVCacheAllocator(config, device, allocation_type) {}

bool SingleTypeKVCacheAllocator::init() {
    auto&           spec        = config_.cache_specs[0];
    BlockPoolConfig pool_config = BlockPoolConfigHelper::createLayerFirstConfig(
        static_cast<uint32_t>(config_.layer_num), static_cast<uint32_t>(config_.block_num), spec);
    block_pool_ = std::make_shared<BlockPool>(pool_config, device_, allocation_type_);
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for SingleTypeKVCacheAllocator");
        return false;
    }

    std::vector<int> layer_ids(config_.layer_ids[0]);
    if (config_.cache_specs.empty()) {
        RTP_LLM_LOG_ERROR("no cache_specs found in CacheConfig");
        return false;
    }

    full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(layer_ids, spec, block_pool_, 0);

    if (!full_kv_cache_group_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize FullKVCacheGroup");
        return false;
    }

    RTP_LLM_LOG_INFO("SingleTypeKVCacheAllocator initialized successfully");
    return true;
}

MallocResult SingleTypeKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto& kv_resource    = malloc_info.batch_kv_cache_resource;
    int   reuse_len      = 0;
    int   common_seq_len = malloc_info.common_seq_len >= 0 ? malloc_info.common_seq_len : malloc_info.total_seq_len;
    auto& cache_keys     = kv_resource->cacheKeys(0);
    auto& blocks_0       = kv_resource->blocks(0);
    if (kv_resource->enable_reuse_cache) {
        auto match_result = full_kv_cache_group_->match(cache_keys);
        reuse_len         = static_cast<int>(match_result.reuse_length);
        full_kv_cache_group_->reference(blocks_0, match_result.block_indices);
    }

    if (!full_kv_cache_group_->malloc(cache_keys, blocks_0, common_seq_len)) {
        return {false, 0};
    }

    // other batches reference batch 0's blocks
    for (int batch_id = 1; batch_id < kv_resource->batchSize(); ++batch_id) {
        full_kv_cache_group_->reference(kv_resource->blocks(batch_id), blocks_0);
    }

    return {true, reuse_len};
}

MallocResult SingleTypeKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto& kv_resource    = malloc_info.batch_kv_cache_resource;
    int   batch_size     = kv_resource->batchSize();
    int   current_blocks = kv_resource->maxBlocksNum();
    int   seq_len =
        (malloc_info.total_seq_len >= 0) ? malloc_info.total_seq_len : malloc_info.complete_token_ids->seqLength();

    auto need_blocks = full_kv_cache_group_->needBlocksNum(seq_len, current_blocks);
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
        auto& cache_keys = kv_resource->cacheKeys(current_batch);
        auto& blocks     = kv_resource->blocks(current_batch);
        if (!full_kv_cache_group_->malloc(cache_keys, blocks, seq_len)) {
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
        auto& blocks       = kv_resource->blocks(batch_id);
        auto  original_num = original_blocks_num[batch_id];
        if (blocks.size() > original_num) {
            blocks_to_free.insert(blocks_to_free.end(), blocks.begin() + original_num, blocks.end());
            blocks.resize(original_num);
        }
    }
    if (!blocks_to_free.empty()) {
        full_kv_cache_group_->free(blocks_to_free);
    }
    return {false, 0};
}

void SingleTypeKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;

    if (kv_cache_resource->maxBlocksNum() == 0) {
        return;
    }

    for (auto& resource : kv_cache_resource->batch_resource) {
        full_kv_cache_group_->free(resource.blocks());
    }
    kv_cache_resource->clearBlocks();
}

void SingleTypeKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_resource = insert_info.batch_kv_cache_resource;
    int   batch_size  = kv_resource->batchSize();

    // TODO(chanyin): set batch_size to 1 for now
    batch_size = 1;

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto& cache_keys = kv_resource->cacheKeys(batch_id);
        auto& blocks     = kv_resource->blocks(batch_id);

        size_t block_num = std::min(size_t(cache_keys.size()), size_t(blocks.size()));
        if (block_num == 0) {
            continue;
        }

        CacheKeysType    put_cache_keys(cache_keys.begin(), cache_keys.begin() + block_num);
        BlockIndicesType put_block_ids(blocks.begin(), blocks.begin() + block_num);

        full_kv_cache_group_->insertIntoCache(put_cache_keys, put_block_ids, insert_info.is_resident);
    }
}

CacheLayerLayout SingleTypeKVCacheAllocator::layerCacheBase() const {
    CacheLayerLayout layout;
    auto             layer_tensors = full_kv_cache_group_->layerCacheBase();
    layout.layers_to_buffer_ptrs.clear();
    layout.layers_to_buffer_ptrs.resize(config_.layer_num);
    for (const auto& kv : layer_tensors) {
        int layer_id = kv.first;
        if (layer_id >= 0 && layer_id < config_.layer_num) {
            layout.layers_to_buffer_ptrs[layer_id] = kv.second;
        }
    }
    return layout;
}

BlockAddrInfo SingleTypeKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToAddr(layer_id, block_id);
}

BlockBufferPtrInfo SingleTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToBuffer(layer_id, block_id);
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
        full_kv_cache_group_->free(disused_kv_blocks);
    }

    // ensure there are enough free blocks for last-block copies
    if (new_blocks_num > 0) {
        if (!full_kv_cache_group_->ensureFreeBlocks(static_cast<int>(new_blocks_num))) {
            RTP_LLM_LOG_WARNING("ensure free blocks failed for kv cache update, need %u", new_blocks_num);
            return false;
        }
    }

    // rebuild batch_kv_cache_resource and generate mapping
    // TODO, 这里的move可以吗？这里再优化下。
    std::vector<KVCacheResourceV1> old_resources = std::move(kv_cache_resource->batch_resource);
    kv_cache_resource->batch_resource.clear();
    kv_cache_resource->batch_resource.resize(new_batch_size);

    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        const int old_batch_idx = block_src_batch[new_batch_idx];
        auto&     fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);

        kv_cache_resource->initGroups(1, config_.layer_num);

        if (fork_count == 1) {
            auto& br       = kv_cache_resource->batch_resource[new_batch_idx];
            br.blocks()    = std::move(old_resources[old_batch_idx].blocks());
            br.cacheKeys() = std::move(old_resources[old_batch_idx].cacheKeys());
        } else {
            // create new batch by referencing from source blocks
            auto& br         = kv_cache_resource->batch_resource[new_batch_idx];
            auto& blocks     = br.blocks();
            auto& cache_keys = br.cacheKeys();
            cache_keys       = old_resources[old_batch_idx].cacheKeys();
            full_kv_cache_group_->reference(blocks, old_resources[old_batch_idx].blocks());
            if (copy_last_block && !blocks.empty()) {
                const int old_block = blocks.back();
                full_kv_cache_group_->free({old_block});
                blocks.pop_back();

                // allocate exactly one new block via kvCacheGroup
                int  seq_len_target = (static_cast<int>(blocks.size()) + 1) * full_kv_cache_group_->seqSizePerBlock();
                bool ok             = full_kv_cache_group_->malloc(cache_keys, blocks, seq_len_target);
                RTP_LLM_CHECK_WITH_INFO(ok, "malloc one block via kvCacheGroup failed during kv cache update");
                const int new_block = blocks.back();

                block_update_mapping.push_back(BlockIdPair{old_block, new_block});
            }
        }
        --fork_count;
    }
    return true;
}

int SingleTypeKVCacheAllocator::seqSizePerBlock() const {
    return full_kv_cache_group_->seqSizePerBlock();
}

void SingleTypeKVCacheAllocator::clearCache() {
    if (block_pool_) {
        block_pool_->clearCache();
    }
}

}  // namespace rtp_llm
