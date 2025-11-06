#include "rtp_llm/cpp/cache_new/SingleTypeKVCacheAllocator.h"
#include <algorithm>
#include "rtp_llm/cpp/cache_new/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

SingleTypeKVCacheAllocator::SingleTypeKVCacheAllocator(const CacheConfig&   config,
                                                       rtp_llm::DeviceBase* device,
                                                       AllocationType       atype):
    KVCacheAllocator(config, device, atype) {}

bool SingleTypeKVCacheAllocator::init() {
    auto            block_size = config_.layer_type_params[0]->block_size();
    BlockPoolConfig pool_config =
        BlockPoolConfigHelper::createKVFirstConfig(config_.layer_num, config_.block_num, block_size);

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

    auto& spec = config_.layer_type_params[0];

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

    int seq_len = malloc_info.complete_token_ids->seqLength();

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto& cache_keys    = malloc_info.batch_kv_cache_resource->cache_keys[batch_id];
        auto& block_indices = malloc_info.batch_kv_cache_resource->batch_block_id[batch_id];

        int reuse_len = 0;
        if (malloc_info.batch_kv_cache_resource->enable_reuse_cache && block_indices.size() < cache_keys.size()) {
            auto match_result = full_kv_cache_group_->match(cache_keys);
            reuse_len         = static_cast<int>(match_result.reuse_length);
        }

        auto need_blocks_num = full_kv_cache_group_->needBlocksNum(seq_len, block_indices.size());
        auto free_blocks_num = full_kv_cache_group_->freeBlockNums();
        if (free_blocks_num < need_blocks_num) {
            if (!full_kv_cache_group_->ensureFreeBlocks(need_blocks_num - free_blocks_num)) {
                return {false, 0};
            }
        }

        if (!full_kv_cache_group_->malloc(cache_keys, block_indices, seq_len)) {
            // TODO，回滚已经成功的batch的资源。
            return {false, 0};
        }

        // TODO: fix this : use batch[0] reuse_len as total_reuse_len now
        if (batch_id == 0) {
            total_reuse_len = reuse_len;
        }
    }

    return {true, total_reuse_len};
}

FreeResult SingleTypeKVCacheAllocator::free(const FreeInfo& free_info) {
    if (!free_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false};
    }

    for (int batch_id = 0; batch_id < free_info.batch_kv_cache_resource->batchSize(); ++batch_id) {
        auto& blocks = free_info.batch_kv_cache_resource->blocks(batch_id);
        full_kv_cache_group_->free(blocks);
    }

    return {true};
}

InsertResult SingleTypeKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    if (!insert_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false};
    }

    int batch_size         = insert_info.batch_kv_cache_resource->batchSize();
    int seq_size_per_block = full_kv_cache_group_->seqSizePerBlock();

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto& cache_keys = insert_info.batch_kv_cache_resource->cache_keys[batch_id];
        auto& block_ids  = insert_info.batch_kv_cache_resource->batch_block_id[batch_id];

        auto token_ids = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1) {
            continue;
        }
        size_t token_len     = token_ids.size() - 1;
        size_t max_by_tokens = token_len / static_cast<size_t>(seq_size_per_block);
        size_t block_len     = std::min({cache_keys.size(), block_ids.size(), max_by_tokens});
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


void SingleTypeKVCacheAllocator::blockCopy(int src_block_index, int dest_block_index) {
    BlockIdPair copy_mapping{src_block_index, dest_block_index};
    blockBatchCopy(&copy_mapping, &copy_mapping + 1);
}

void SingleTypeKVCacheAllocator::blockBatchCopy(const std::vector<BlockIdPair>& copy_mapping) {
    blockBatchCopy(copy_mapping.data(), copy_mapping.data() + copy_mapping.size());
}

void SingleTypeKVCacheAllocator::blockBatchCopy(const Buffer& copy_mapping) {
    RTP_LLM_CHECK(copy_mapping.dim() == 2 && copy_mapping.shape()[1] == 2);
    const auto* begin_ptr = (const BlockIdPair*)copy_mapping.data();
    size_t      copy_num  = copy_mapping.shape()[0];
    blockBatchCopy(begin_ptr, begin_ptr + copy_num);
}


void SingleTypeKVCacheAllocator::blockBatchCopy(const BlockIdPair* begin_ptr, const BlockIdPair* end_ptr) {
    using CopyType = BatchCopyParams::CopyType;

    if (end_ptr == begin_ptr) {
        return;
    }

    if (!full_kv_cache_group_) {
        RTP_LLM_LOG_ERROR("KV cache group is not initialized");
        return;
    }

    BatchCopyParams copy_params;

    const size_t copy_num = (end_ptr - begin_ptr) * config_.layer_num;

    size_t copy_nums[CopyType::TYPE_SIZE] = {};
    auto copy_type = BatchCopyParams::get_copy_type(atype_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU,
                                                     atype_ == AllocationType::DEVICE ? rtp_llm::MEMORY_GPU : rtp_llm::MEMORY_CPU);
    copy_nums[copy_type] += copy_num * 2; // for k and v 

    for (size_t i = 0; i < CopyType::TYPE_SIZE; ++i) {
        copy_params.reserve(static_cast<CopyType>(i), copy_nums[i]);
    }

    auto& spec = config_.layer_type_params[0];
    size_t k_block_size = spec->k_block_size();
    size_t v_block_size = spec->v_block_size();

    for (auto it = begin_ptr; it != end_ptr; ++it) {
        auto [src_block_index, dest_block_index] = *it;
        
        for (int layer_id = 0; layer_id < config_.layer_num; layer_id++) {
            auto src_addr_info = full_kv_cache_group_->convertIndexToAddr(layer_id, src_block_index);
            auto dst_addr_info = full_kv_cache_group_->convertIndexToAddr(layer_id, dest_block_index);

            if (!src_addr_info.k_addr || !dst_addr_info.k_addr || 
                !src_addr_info.v_addr || !dst_addr_info.v_addr) {
                RTP_LLM_LOG_ERROR("Failed to get block address for layer %d, src_block %d, dst_block %d",
                                 layer_id, src_block_index, dest_block_index);
                continue;
            }

            copy_params.add(dst_addr_info.k_addr, src_addr_info.k_addr, k_block_size, copy_type);
            copy_params.add(dst_addr_info.v_addr, src_addr_info.v_addr, v_block_size, copy_type);
        }
    }

    device_->batchCopy(copy_params);
}


}  // namespace rtp_llm
