#include "rtp_llm/cpp/cache_new/HybridLayerKVCacheAllocator.h"
#include <algorithm>
#include "rtp_llm/cpp/cache_new/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

HybridLayerKVCacheAllocator::HybridLayerKVCacheAllocator(const CacheConfig&   config,
                                                         rtp_llm::DeviceBase* device,
                                                         AllocationType       atype):
    KVCacheAllocator(config, device, atype) {}

bool HybridLayerKVCacheAllocator::init() {
    auto            block_size = config_.layer_type_params[0]->block_size();
    BlockPoolConfig pool_config =
        BlockPoolConfigHelper::createLayerFirstConfig(config_.layer_num, config_.block_num, block_size);

    block_pool_ = std::make_shared<BlockPool>(pool_config, device_, atype_);
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for HybridLayerKVCacheAllocator");
        return false;
    }

    std::vector<int> layer_ids(config_.layer_ids[0]);
    if (config_.layer_type_params.empty()) {
        RTP_LLM_LOG_ERROR("no layer_type_params found in CacheConfig");
        return false;
    }

    auto& spec = config_.layer_type_params[0];

    full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(layer_ids, spec, block_pool_);

    if (!full_kv_cache_group_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize FullKVCacheGroup");
        return false;
    }

    // TODO, user other layer_id
    auto linear_kv_cache_group_1 = std::make_shared<LinearKVCacheGroup>(layer_ids, spec, block_pool_);
    auto linear_kv_cache_group_2 = std::make_shared<LinearKVCacheGroup>(layer_ids, spec, block_pool_);
    linear_kv_cache_groups_.push_back(linear_kv_cache_group_1);
    linear_kv_cache_groups_.push_back(linear_kv_cache_group_2);

    all_kv_cache_groups_.push_back(full_kv_cache_group_);
    all_kv_cache_groups_.insert(
        all_kv_cache_groups_.end(), linear_kv_cache_groups_.begin(), linear_kv_cache_groups_.end());
    for (int i = 0; i < all_kv_cache_groups_.size(); i++) {
        all_kv_cache_groups_[i]->setGroupId(i);
    }

    RTP_LLM_LOG_INFO("HybridLayerKVCacheAllocator initialized successfully with KV_FIRST layout");
    return true;
}

int HybridLayerKVCacheAllocator::reuseCache(const CacheKeysType& cache_keys, GroupBlockIds& group_block_ids) {
    auto                      full_match_result = full_kv_cache_group_->match(cache_keys);
    int                       pos               = full_match_result.reuse_blocks - 1;
    std::vector<BlockIdxType> linear_match_bocks;
    for (; pos >= 0; pos--) {
        linear_match_bocks.clear();
        bool all_linear_matched = true;
        for (auto& linear_group : linear_kv_cache_groups_) {
            auto result = linear_group->matchSingleKey(cache_keys[pos]);
            if (result.reuse_blocks == 0) {
                all_linear_matched = false;
                break;
            }
            linear_match_bocks.push_back(result.block_indices[0]);
        }
        if (all_linear_matched == true) {
            break;
        }
    }

    auto reuse_blocks = static_cast<int>(pos + 1);
    for (auto& group : group_block_ids) {
        group->block_indices.resize(reuse_blocks);
    }

    if (reuse_blocks > 0) {
        group_block_ids[0]->block_indices = full_match_result.block_indices;
        for (int linear_group_id = 0; linear_group_id < linear_kv_cache_groups_.size(); linear_group_id++) {
            group_block_ids[linear_group_id + 1]->block_indices[reuse_blocks - 1] = linear_match_bocks[linear_group_id];
        }
    }
    return reuse_blocks;
}

MallocResult HybridLayerKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    int seq_len =
        (malloc_info.total_seq_len >= 0) ? malloc_info.total_seq_len : malloc_info.complete_token_ids->seqLength();

    int  batch_size     = malloc_info.batch_kv_cache_resource->batchSize();
    int  current_blocks = malloc_info.batch_kv_cache_resource->maxBlockSize();
    auto need_blocks    = full_kv_cache_group_->needBlocksNum(seq_len, current_blocks);
    if (need_blocks == 0) {
        return {true, 0};
    }
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto& cache_keys      = malloc_info.batch_kv_cache_resource->batch_resource[batch_id].cache_keys;
        auto& group_block_ids = malloc_info.batch_kv_cache_resource->batch_resource[batch_id].group_block_ids;
        for (int i = 0; i < group_block_ids.size(); i++) {
            if (!all_kv_cache_groups_[i]->malloc(cache_keys, group_block_ids[i]->block_indices, seq_len)) {
                // TODO，回滚已经malloc的资源。
                return {false, 0};
            }
        }
    }
    return {true, 0};
}

MallocResult HybridLayerKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    int batch_size   = malloc_info.batch_kv_cache_resource->batchSize();
    int reuse_blocks = 0;

    // TODO, 这里是啥含义呢？在外围做掉？
    int seq_len =
        (malloc_info.total_seq_len >= 0) ? malloc_info.total_seq_len : malloc_info.complete_token_ids->seqLength();
    bool has_common_len = (malloc_info.common_seq_len >= 0) && (malloc_info.common_seq_len <= seq_len);
    int  common_seq_len = has_common_len ? malloc_info.common_seq_len : seq_len;

    auto& cache_keys_0      = malloc_info.batch_kv_cache_resource->batch_resource[0].cache_keys;
    auto& group_block_ids_0 = malloc_info.batch_kv_cache_resource->batch_resource[0].group_block_ids;

    if (malloc_info.batch_kv_cache_resource->enable_reuse_cache
        && group_block_ids_0[0]->block_indices.size() < cache_keys_0.size()) {
        reuse_blocks = reuseCache(cache_keys_0, group_block_ids_0);
        for (int i = 0; i < all_kv_cache_groups_.size(); i++) {
            all_kv_cache_groups_[i]->reference(group_block_ids_0[i]->block_indices);
        }
    }

    for (int i = 0; i < all_kv_cache_groups_.size(); i++) {
        if (!all_kv_cache_groups_[i]->malloc(cache_keys_0, group_block_ids_0[i]->block_indices, common_seq_len)) {
            return {false, 0};
        }
    }

    for (int batch_id = 1; batch_id < batch_size; ++batch_id) {
        auto& group_block_ids = malloc_info.batch_kv_cache_resource->batch_resource[batch_id].group_block_ids;
        for (int i = 0; i < all_kv_cache_groups_.size(); i++) {
            all_kv_cache_groups_[i]->reference(group_block_ids[i]->block_indices, group_block_ids_0[i]->block_indices);
        }
    }

    return {true, reuse_blocks * all_kv_cache_groups_[0]->seqSizePerBlock()};
}

FreeResult HybridLayerKVCacheAllocator::free(const FreeInfo& free_info) {
    if (!free_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false};
    }

    for (int batch_id = 0; batch_id < free_info.batch_kv_cache_resource->batchSize(); ++batch_id) {
        auto& group_block_ids = free_info.batch_kv_cache_resource->batch_resource[batch_id].group_block_ids;
        for (int group_id = 0; group_id < group_block_ids.size(); group_id++) {
            all_kv_cache_groups_[group_id]->free(group_block_ids[group_id]->block_indices);
        }
    }

    return {true};
}

InsertResult HybridLayerKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    if (!insert_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false};
    }

    int batch_size         = insert_info.batch_kv_cache_resource->batchSize();
    int seq_size_per_block = full_kv_cache_group_->seqSizePerBlock();

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto& cache_keys      = insert_info.batch_kv_cache_resource->batch_resource[batch_id].cache_keys;
        auto& group_block_ids = insert_info.batch_kv_cache_resource->batch_resource[batch_id].group_block_ids;
        auto  blocks_num      = group_block_ids[0]->block_indices.size();

        auto token_ids = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1) {
            continue;
        }
        size_t token_len     = token_ids.size() - 1;
        size_t max_by_tokens = token_len / static_cast<size_t>(seq_size_per_block);
        blocks_num           = std::min({cache_keys.size(), blocks_num, max_by_tokens});
        if (blocks_num == 0) {
            continue;
        }

        std::vector<CacheKeyType> put_cache_keys(cache_keys.begin(), cache_keys.begin() + blocks_num);
        for (int group_id = 0; group_id < all_kv_cache_groups_.size(); group_id++) {
            const auto&               block_ids = group_block_ids[group_id]->block_indices;
            std::vector<BlockIdxType> put_block_ids(block_ids.begin(), block_ids.begin() + blocks_num);
            all_kv_cache_groups_[group_id]->insertIntoCache(put_cache_keys, put_block_ids, insert_info.is_resident);
        }
    }

    return {true};
}

CacheLayerLayout HybridLayerKVCacheAllocator::layerCacheBase() const {
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

void HybridLayerKVCacheAllocator::regUserMr(size_t model_id) {
    if (block_pool_) {
        block_pool_->regUserMr(model_id);
    }
}

BlockAddrInfo HybridLayerKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToAddr(layer_id, block_id);
}

BlockBufferInfo HybridLayerKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToBuffer(layer_id, block_id);
}

size_t HybridLayerKVCacheAllocator::freeBlocksNums() const {
    return block_pool_->freeBlockNums();
}

size_t HybridLayerKVCacheAllocator::availableBlocksNums() const {
    // TODO: free blocks nums not equal to available blocks nums when block cache holds blocks reference
    return block_pool_->freeBlockNums();
}

size_t HybridLayerKVCacheAllocator::totalBlocksNums() const {
    return block_pool_->totalBlockNums();
}

size_t HybridLayerKVCacheAllocator::maxSeqLen() const {
    return block_pool_->totalBlockNums() * full_kv_cache_group_->seqSizePerBlock();
}

KVCacheBuffer HybridLayerKVCacheAllocator::kvCacheBuffer() const {
    if (!block_pool_) {
        return KVCacheBuffer{nullptr, nullptr, nullptr, nullptr};
    }
    return block_pool_->kvCacheBuffer();
}

}  // namespace rtp_llm
