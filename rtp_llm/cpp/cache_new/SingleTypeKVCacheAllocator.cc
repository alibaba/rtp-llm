#include "rtp_llm/cpp/cache_new/SingleTypeKVCacheAllocator.h"
#include <algorithm>
#include "rtp_llm/cpp/cache_new/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

SingleTypeKVCacheAllocator::SingleTypeKVCacheAllocator(const CacheConfig&   config,
                                                       rtp_llm::DeviceBase* device,
                                                       AllocationType       atype):
    KVCacheAllocator(config, device, atype) {}

bool SingleTypeKVCacheAllocator::init() {
    auto&           spec        = config_.cache_specs[0];
    BlockPoolConfig pool_config = BlockPoolConfigHelper::createKVFirstConfig(
        static_cast<uint32_t>(config_.layer_num), static_cast<uint32_t>(config_.block_num), spec);

    block_pool_ = std::make_shared<BlockPool>(pool_config, device_, atype_);
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
    // group id is set via constructor

    // kv_cache_groups_.push_back(full_kv_cache_group_);

    RTP_LLM_LOG_INFO("SingleTypeKVCacheAllocator initialized successfully with KV_FIRST layout");
    return true;
}

MallocResult SingleTypeKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    int batch_size = malloc_info.batch_kv_cache_resource->batchSize();
    int reuse_len  = 0;

    int  seq_len        = malloc_info.total_seq_len;
    bool has_common_len = (malloc_info.common_seq_len >= 0) && (malloc_info.common_seq_len <= seq_len);
    int  common_seq_len = has_common_len ? malloc_info.common_seq_len : seq_len;
    // ensure group 0 exists
    auto& br0 = malloc_info.batch_kv_cache_resource->batch_resource[0];

    auto& cache_keys_0    = br0.cache_keys;
    auto& block_indices_0 = br0.group_block_ids[0]->block_indices;
    if (malloc_info.batch_kv_cache_resource->enable_reuse_cache && block_indices_0.size() < cache_keys_0.size()) {
        auto match_result = full_kv_cache_group_->match(cache_keys_0);
        reuse_len         = static_cast<int>(match_result.reuse_length);
        full_kv_cache_group_->reference(block_indices_0, match_result.block_indices);
    }

    if (!full_kv_cache_group_->malloc(cache_keys_0, block_indices_0, common_seq_len)) {
        return {false, 0};
    }

    // reference other batches to group 0
    for (int batch_id = 1; batch_id < batch_size; ++batch_id) {
        auto& br                  = malloc_info.batch_kv_cache_resource->batch_resource[batch_id];
        auto& block_indices_other = br.group_block_ids[0]->block_indices;
        full_kv_cache_group_->reference(block_indices_other, block_indices_0);
    }

    return {true, reuse_len};
}

MallocResult SingleTypeKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    int seq_len =
        (malloc_info.total_seq_len >= 0) ? malloc_info.total_seq_len : malloc_info.complete_token_ids->seqLength();

    int  batch_size     = malloc_info.batch_kv_cache_resource->batchSize();
    int  current_blocks = malloc_info.batch_kv_cache_resource->maxBlockSize();
    auto need_blocks    = full_kv_cache_group_->needBlocksNum(seq_len, current_blocks);
    if (need_blocks == 0) {
        return {true, 0};
    }
    // Record original sizes for rollback in case any subsequent allocation fails
    std::vector<size_t> original_sizes(static_cast<size_t>(batch_size), 0);
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto& br                 = malloc_info.batch_kv_cache_resource->batch_resource[batch_id];
        original_sizes[batch_id] = br.group_block_ids[0]->block_indices.size();
    }

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto& br = malloc_info.batch_kv_cache_resource->batch_resource[batch_id];

        auto& cache_keys    = br.cache_keys;
        auto& block_indices = br.group_block_ids[0]->block_indices;
        if (!full_kv_cache_group_->malloc(cache_keys, block_indices, seq_len)) {
            BlockIndicesType blocks_to_free;
            for (int rb = 0; rb <= batch_id; ++rb) {
                auto&        rb_br      = malloc_info.batch_kv_cache_resource->batch_resource[rb];
                auto&        rb_indices = rb_br.group_block_ids[0]->block_indices;
                const size_t ori_n      = original_sizes[rb];
                if (rb_indices.size() > ori_n) {
                    blocks_to_free.insert(
                        blocks_to_free.end(), rb_indices.begin() + static_cast<long>(ori_n), rb_indices.end());
                    rb_indices.resize(ori_n);
                }
            }
            if (!blocks_to_free.empty()) {
                full_kv_cache_group_->free(blocks_to_free);
            }
            return {false, 0};
        }
    }
    return {true, 0};
}

FreeResult SingleTypeKVCacheAllocator::free(const FreeInfo& free_info) {
    if (!free_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false};
    }

    if (!free_info.complete_token_ids) {
        RTP_LLM_LOG_ERROR("CompleteTokenIds is null");
        return {false};
    }
    // Free all blocks in batch_kv_cache_resource
    for (int batch_id = 0; batch_id < free_info.batch_kv_cache_resource->batchSize(); ++batch_id) {
        auto& br = free_info.batch_kv_cache_resource->batch_resource[batch_id];
        if (br.group_block_ids.empty()) {
            continue;
        }
        auto& batch_blocks = br.group_block_ids[0]->block_indices;

        if (!batch_blocks.empty()) {
            full_kv_cache_group_->free(batch_blocks);
            // batch_blocks.clear();
        }
    }

    return {true};
}

InsertResult SingleTypeKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    int batch_size = insert_info.batch_kv_cache_resource->batchSize();

    // TODO(chanyin): set batch_size to 1 for now
    batch_size = 1;

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto& br = insert_info.batch_kv_cache_resource->batch_resource[batch_id];

        auto& cache_keys = br.cache_keys;
        auto& block_ids  = br.group_block_ids[0]->block_indices;

        size_t block_len = std::min(size_t(cache_keys.size()), size_t(block_ids.size()));
        if (block_len == 0) {
            continue;
        }

        std::vector<CacheKeyType> put_cache_keys(cache_keys.begin(), cache_keys.begin() + block_len);
        std::vector<BlockIdxType> put_block_ids(block_ids.begin(), block_ids.begin() + block_len);

        full_kv_cache_group_->insertIntoCache(put_cache_keys, put_block_ids, insert_info.is_resident);
    }

    return {true};
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

size_t SingleTypeKVCacheAllocator::freeBlocksNum() const {
    return block_pool_->freeBlocksNum();
}

size_t SingleTypeKVCacheAllocator::availableBlocksNum() const {
    // TODO: free blocks nums not equal to available blocks nums when block cache holds blocks reference
    return block_pool_->freeBlocksNum();
}

size_t SingleTypeKVCacheAllocator::totalBlocksNum() const {
    return block_pool_->totalBlocksNum();
}

size_t SingleTypeKVCacheAllocator::maxSeqLen() const {
    return block_pool_->totalBlocksNum() * full_kv_cache_group_->seqSizePerBlock();
}

KVCacheBuffer SingleTypeKVCacheAllocator::kvCacheBuffer() const {
    if (!block_pool_) {
        return KVCacheBuffer{nullptr, nullptr, nullptr, nullptr};
    }
    return block_pool_->kvCacheBuffer();
}

void SingleTypeKVCacheAllocator::regUserMr(size_t model_id) {
    if (block_pool_) {
        block_pool_->regUserMr(model_id);
    }
}

// Update kv blocks for beam search or multi-return sequences.
// - batch_kv_cache_resource: in/out, batch blocks and cache_keys will be rearranged based on block_src_batch
// - block_src_batch: new batch i forks from old batch block_src_batch[i]
// - copy_last_block: whether to copy the last block for each forked batch (instead of sharing)
// - block_update_mapping: out, mapping from old block to new block for batch copy
bool SingleTypeKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                               const std::vector<int>&        block_src_batch,
                                               bool                           copy_last_block,
                                               std::vector<BlockIdPair>&      block_update_mapping) {
    block_update_mapping.clear();

    if (!batch_kv_cache_resource || block_src_batch.empty()) {
        return true;
    }

    const int        old_batch_size = batch_kv_cache_resource->batchSize();
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
    uint32_t         num_new_blocks = 0;
    for (int old_batch_idx = 0; old_batch_idx < old_batch_size; ++old_batch_idx) {
        const int fork_count = batch_fork_count[old_batch_idx];
        if (fork_count == 0) {
            const auto& br_old = batch_kv_cache_resource->batch_resource[old_batch_idx];
            const auto& blocks = br_old.group_block_ids[0]->block_indices;
            disused_kv_blocks.insert(disused_kv_blocks.end(), blocks.begin(), blocks.end());
        } else if (fork_count > 1 && copy_last_block) {
            num_new_blocks += static_cast<uint32_t>(fork_count - 1);
        }
    }

    // free disused first to reclaim capacity
    if (!disused_kv_blocks.empty()) {
        full_kv_cache_group_->free(disused_kv_blocks);
    }
    // ensure there are enough free blocks for last-block copies
    if (num_new_blocks > 0) {
        if (!full_kv_cache_group_->ensureFreeBlocks(static_cast<int>(num_new_blocks))) {
            RTP_LLM_LOG_WARNING("ensure free blocks failed for kv cache update, need %u", num_new_blocks);
            return false;
        }
    }

    // rebuild batch_kv_cache_resource and generate mapping
    std::vector<KVCacheResourceV1> old_resources = std::move(batch_kv_cache_resource->batch_resource);
    batch_kv_cache_resource->batch_resource.reserve(new_batch_size);

    for (int new_batch_idx = 0; new_batch_idx < new_batch_size; ++new_batch_idx) {
        const int old_batch_idx = block_src_batch[new_batch_idx];
        auto&     fork_count    = batch_fork_count[old_batch_idx];
        RTP_LLM_CHECK_WITH_INFO(fork_count > 0, "old batch %d has been forked too many times", old_batch_idx);

        if (fork_count == 1) {
            auto& br = batch_kv_cache_resource->batch_resource[new_batch_idx];
            br.initGroups(1);
            br.group_block_ids[0]->block_indices =
                std::move(old_resources[old_batch_idx].group_block_ids[0]->block_indices);
            br.cache_keys = std::move(old_resources[old_batch_idx].cache_keys);
        } else {
            // create new batch by referencing from source blocks
            auto& br = batch_kv_cache_resource->batch_resource[new_batch_idx];
            br.initGroups(1);
            auto& blocks     = br.group_block_ids[0]->block_indices;
            auto& cache_keys = br.cache_keys;
            cache_keys       = old_resources[old_batch_idx].cache_keys;
            full_kv_cache_group_->reference(blocks, old_resources[old_batch_idx].group_block_ids[0]->block_indices);
            if (copy_last_block && !blocks.empty()) {
                const int old_block = blocks.back();
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

}  // namespace rtp_llm
