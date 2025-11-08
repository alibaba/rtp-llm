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
    auto&           spec       = config_.layer_type_params[0];
    auto            block_size = spec->block_size();
    BlockPoolConfig pool_config =
        BlockPoolConfigHelper::createKVFirstConfig(config_.layer_num, config_.block_num, block_size);

    // Set dtype from spec
    pool_config.dtype = spec->dtype;

    block_pool_ = std::make_shared<BlockPool>(pool_config, device_, atype_);
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for SingleTypeKVCacheAllocator");
        return false;
    }

    std::vector<int> layer_ids(config_.layer_ids[0]);
    if (config_.layer_type_params.empty()) {
        RTP_LLM_LOG_ERROR("no layer_type_params found in CacheConfig");
        return false;
    }

    full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(layer_ids, spec, block_pool_);

    // if (cache_type == KVCacheType::MultiHeadLatentAttention) {
    //     // Use MLA specialization (fields remain default for now)
    //     MLAKVCacheSpec mla_spec;
    //     mla_spec.layer_ids_ = layer_ids;
    //     mla_spec.seq_size_per_block = static_cast<uint>(config_.seq_size_per_block);
    //     // Construct group with MLA spec
    //     auto block_cache = std::make_shared<BlockCacheV1>(static_cast<size_t>(config_.seq_size_per_block));
    //     full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(
    //         layer_ids, mla_spec, block_cache, block_pool_
    //     );
    // } else if (cache_type == KVCacheType::MultiHeadAttention) {
    //     MHAKVCacheSpec mha_spec;
    //     mha_spec.layer_ids_ = layer_ids;
    //     mha_spec.seq_size_per_block = static_cast<uint>(config_.seq_size_per_block);
    //     auto block_cache = std::make_shared<BlockCacheV1>(static_cast<size_t>(config_.seq_size_per_block));
    //     full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(
    //         layer_ids, mha_spec, block_cache, block_pool_
    //     );
    // } else {
    //     RTP_LLM_LOG_ERROR("SingleTypeKVCacheAllocator only supports MHA/MLA full attention");
    //     return false;
    // }

    if (!full_kv_cache_group_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize FullKVCacheGroup");
        return false;
    }

    // kv_cache_groups_.push_back(full_kv_cache_group_);

    RTP_LLM_LOG_INFO("SingleTypeKVCacheAllocator initialized successfully with KV_FIRST layout");
    return true;
}

MallocResult SingleTypeKVCacheAllocator::malloc(const MallocInfo& malloc_info) {
    if (!malloc_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false, 0};
    }

    if (!malloc_info.complete_token_ids) {
        RTP_LLM_LOG_ERROR("CompleteTokenIds is null");
        return {false, 0};
    }

    int batch_size      = malloc_info.batch_kv_cache_resource->batchSize();
    int total_reuse_len = 0;

    // Include reserved steps
    int seq_len =
        (malloc_info.total_seq_len >= 0) ? malloc_info.total_seq_len : malloc_info.complete_token_ids->seqLength();

    // int reuse_blocks = 0;
    // // TODO, match is only in prefill
    // if (malloc_info.batch_kv_cache_resource->enable_reuse_cache && block_indices.size() < cache_keys.size()) {
    //     auto match_result = full_kv_cache_group_->match(cache_keys);
    //     // TODO, modify blocks
    //     reuse_blocks  = static_cast<int>(match_result.reuse_blocks);
    //     block_indices = match_result.block_indices;

    // Determine if we need to handle common/extra blocks separately
    bool has_common_extra_split = (malloc_info.common_seq_len >= 0) && (malloc_info.common_seq_len < seq_len);
    int  common_seq_len         = has_common_extra_split ? malloc_info.common_seq_len : seq_len;

    int current_blocks     = malloc_info.batch_kv_cache_resource->maxBlockSize();
    int common_blocks_need = singleBatchNeedBlocks(common_seq_len, current_blocks);

    // 1. Allocate and clone common blocks (shared across all batches)
    if (common_blocks_need > 0) {
        auto& cache_keys_0    = malloc_info.batch_kv_cache_resource->cache_keys[0];
        auto& block_indices_0 = malloc_info.batch_kv_cache_resource->batch_block_id[0];

        int reuse_len = 0;
        if (malloc_info.batch_kv_cache_resource->enable_reuse_cache && block_indices_0.size() < cache_keys_0.size()) {
            auto match_result = full_kv_cache_group_->match(cache_keys_0);
            reuse_len         = static_cast<int>(match_result.reuse_length);

            // Insert matched blocks into block_indices_0
            // Note: skip blocks that are already in block_indices_0
            for (int i = block_indices_0.size(); i < match_result.block_indices.size(); i++) {
                block_indices_0.push_back(match_result.block_indices[i]);
            }
        }

        int  need_blocks_num = common_blocks_need - static_cast<int>(block_indices_0.size());
        auto free_blocks_num = full_kv_cache_group_->freeBlockNums();
        if (free_blocks_num < need_blocks_num) {
            if (!full_kv_cache_group_->ensureFreeBlocks(need_blocks_num - free_blocks_num)) {
                return {false, 0};
            }
        }

        if (free_blocks_num < static_cast<size_t>(common_blocks_need)) {
            RTP_LLM_LOG_WARNING(
                "Insufficient free blocks for common part: need %d, have %zu", common_blocks_need, free_blocks_num);
        }

        // Allocate common blocks
        if (!full_kv_cache_group_->malloc(cache_keys_0, block_indices_0, common_seq_len)) {
            // TODO，回滚已经成功的batch的资源。
            return {false, 0};
        }

        // Clone common blocks to other batches (increase reference count)
        int newly_allocated_blocks = block_indices_0.size() - current_blocks;
        if (newly_allocated_blocks > 0 && batch_size > 1) {
            std::vector<int> new_common_blocks(block_indices_0.end() - newly_allocated_blocks, block_indices_0.end());

            for (int batch_id = 1; batch_id < batch_size; ++batch_id) {
                // Increase reference count for shared blocks
                full_kv_cache_group_->reference(new_common_blocks);

                // Append to batch_block_id
                auto& batch_blocks = malloc_info.batch_kv_cache_resource->batch_block_id[batch_id];
                batch_blocks.insert(batch_blocks.end(), new_common_blocks.begin(), new_common_blocks.end());
            }
        }

        total_reuse_len = reuse_len;
    }

    // 2. Allocate extra blocks (per-batch independent blocks)
    if (has_common_extra_split) {
        int total_blocks_for_seq        = singleBatchNeedBlocks(seq_len, 0);
        int common_blocks               = singleBatchNeedBlocks(common_seq_len, 0);
        int extra_blocks_need_per_batch = total_blocks_for_seq - common_blocks;

        if (extra_blocks_need_per_batch > 0) {
            int  total_extra_blocks = extra_blocks_need_per_batch * batch_size;
            auto free_blocks_num    = full_kv_cache_group_->freeBlockNums();
            if (free_blocks_num < static_cast<size_t>(total_extra_blocks)) {
                RTP_LLM_LOG_WARNING(
                    "Insufficient free blocks for extra part: need %d, have %zu", total_extra_blocks, free_blocks_num);
                return {false, total_reuse_len};
            }

            // Allocate extra blocks for each batch
            for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
                auto& cache_keys    = malloc_info.batch_kv_cache_resource->cache_keys[batch_id];
                auto& block_indices = malloc_info.batch_kv_cache_resource->batch_block_id[batch_id];
                full_kv_cache_group_->malloc(cache_keys, block_indices, seq_len);
            }
        }
    }

    return {true, total_reuse_len};
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
    std::unordered_set<int> all_blocks;
    for (int batch_id = 0; batch_id < free_info.batch_kv_cache_resource->batchSize(); ++batch_id) {
        auto& batch_blocks = free_info.batch_kv_cache_resource->batch_block_id[batch_id];

        for (int block_id : batch_blocks) {
            all_blocks.insert(block_id);
        }

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
        auto& cache_keys = insert_info.batch_kv_cache_resource->cache_keys[batch_id];
        auto& block_ids  = insert_info.batch_kv_cache_resource->batch_block_id[batch_id];

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

BlockBufferInfo SingleTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToBuffer(layer_id, block_id);
}

size_t SingleTypeKVCacheAllocator::freeBlocksNums() const {
    return block_pool_->freeBlockNums();
}

size_t SingleTypeKVCacheAllocator::availableBlocksNums() const {
    // TODO: free blocks nums not equal to available blocks nums when block cache holds blocks reference
    return block_pool_->freeBlockNums();
}

size_t SingleTypeKVCacheAllocator::totalBlocksNums() const {
    return block_pool_->totalBlockNums();
}

size_t SingleTypeKVCacheAllocator::maxSeqLen() const {
    return block_pool_->totalBlockNums() * full_kv_cache_group_->seqSizePerBlock();
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

}  // namespace rtp_llm
