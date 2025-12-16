#include "rtp_llm/cpp/cache_new/HybridLayerKVCacheAllocator.h"
#include <algorithm>
#include "rtp_llm/cpp/cache_new/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

HybridLayerKVCacheAllocator::HybridLayerKVCacheAllocator(const CacheConfig&   config,
                                                         rtp_llm::DeviceBase* device,
                                                         AllocationType       allocation_type):
    KVCacheAllocator(config, device, allocation_type) {}

bool HybridLayerKVCacheAllocator::init() {
    auto            block_size = config_.cache_specs[0]->block_size();
    BlockPoolConfig pool_config =
        BlockPoolConfigHelper::createLayerFirstConfig(config_.layer_num, config_.block_num, block_size);

    block_pool_ = std::make_shared<BlockPool>(pool_config, device_, allocation_type_);
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for HybridLayerKVCacheAllocator");
        return false;
    }

    std::vector<int> layer_ids(config_.layer_ids[0]);
    if (config_.cache_specs.empty()) {
        RTP_LLM_LOG_ERROR("no cache_specs found in CacheConfig");
        return false;
    }

    auto& spec = config_.cache_specs[0];

    full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(layer_ids, spec, block_pool_, 0);

    if (!full_kv_cache_group_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize FullKVCacheGroup");
        return false;
    }

    // TODO, user other layer_id
    auto linear_kv_cache_group_1 = std::make_shared<LinearKVCacheGroup>(layer_ids, spec, block_pool_, 1);
    auto linear_kv_cache_group_2 = std::make_shared<LinearKVCacheGroup>(layer_ids, spec, block_pool_, 2);
    linear_kv_cache_groups_.push_back(linear_kv_cache_group_1);
    linear_kv_cache_groups_.push_back(linear_kv_cache_group_2);

    all_kv_cache_groups_.push_back(full_kv_cache_group_);
    all_kv_cache_groups_.insert(
        all_kv_cache_groups_.end(), linear_kv_cache_groups_.begin(), linear_kv_cache_groups_.end());
    // group ids have been set via constructors

    RTP_LLM_LOG_INFO("HybridLayerKVCacheAllocator init success");
    return true;
}

int HybridLayerKVCacheAllocator::reuseCache(const CacheKeysType& cache_keys, KVCacheResourceV1& cache_resource) {
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

    auto  reuse_blocks = static_cast<int>(pos + 1);
    auto& blocks_0     = cache_resource.blocks(0);
    blocks_0           = full_match_result.block_indices;
    blocks_0.resize(reuse_blocks);

    if (reuse_blocks > 0) {
        for (int linear_group_id = 0; linear_group_id < linear_kv_cache_groups_.size(); linear_group_id++) {
            auto& blocks = cache_resource.blocks(linear_group_id + 1);
            blocks.resize(reuse_blocks, NULL_BLOCK_IDX);
            blocks[reuse_blocks - 1] = linear_match_bocks[linear_group_id];
        }
    }
    return reuse_blocks;
}

MallocResult HybridLayerKVCacheAllocator::incrMalloc(const MallocInfo& malloc_info) {
    auto& kv_cache_resource = malloc_info.batch_kv_cache_resource;
    int   seq_len =
        (malloc_info.total_seq_len >= 0) ? malloc_info.total_seq_len : malloc_info.complete_token_ids->seqLength();

    int  batch_size     = kv_cache_resource->batchSize();
    int  current_blocks = kv_cache_resource->maxBlocksNum();
    auto need_blocks    = full_kv_cache_group_->needBlocksNum(seq_len, current_blocks);
    if (need_blocks == 0) {
        return {true, 0};
    }
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto& cache_keys = kv_cache_resource->cacheKeys(batch_id);
        int   group_nums = kv_cache_resource->groupNums();
        for (int group_id = 0; group_id < group_nums; group_id++) {
            if (!all_kv_cache_groups_[group_id]->malloc(
                    cache_keys, kv_cache_resource->blocks(batch_id, group_id), seq_len)) {
                // TODO，回滚已经malloc的资源。
                return {false, 0};
            }
        }
    }
    return {true, 0};
}

MallocResult HybridLayerKVCacheAllocator::initMallocForCommonLen(const MallocInfo& malloc_info) {
    auto& kv_cache_resource = malloc_info.batch_kv_cache_resource;
    int   batch_size        = kv_cache_resource->batchSize();
    int   reuse_blocks      = 0;

    // TODO, 这里是啥含义呢？在外围做掉？
    int seq_len =
        (malloc_info.total_seq_len >= 0) ? malloc_info.total_seq_len : malloc_info.complete_token_ids->seqLength();
    bool has_common_len = (malloc_info.common_seq_len >= 0) && (malloc_info.common_seq_len <= seq_len);
    int  common_seq_len = has_common_len ? malloc_info.common_seq_len : 0;

    auto& cache_keys_0 = kv_cache_resource->cacheKeys(0);

    if (kv_cache_resource->enable_reuse_cache && kv_cache_resource->blocksNum(0, 0) < cache_keys_0.size()) {
        reuse_blocks = reuseCache(cache_keys_0, kv_cache_resource->cacheResource(0));
        for (int group_id = 0; group_id < all_kv_cache_groups_.size(); group_id++) {
            all_kv_cache_groups_[group_id]->reference(kv_cache_resource->blocks(0, group_id));
        }
    }

    for (int group_id = 0; group_id < all_kv_cache_groups_.size(); group_id++) {
        if (!all_kv_cache_groups_[group_id]->malloc(
                cache_keys_0, kv_cache_resource->blocks(0, group_id), common_seq_len)) {
            return {false, 0};
        }
    }

    for (int batch_id = 1; batch_id < batch_size; ++batch_id) {
        for (int group_id = 0; group_id < all_kv_cache_groups_.size(); group_id++) {
            all_kv_cache_groups_[group_id]->reference(kv_cache_resource->blocks(batch_id, group_id),
                                                      kv_cache_resource->blocks(0, group_id));
        }
    }

    return {true, reuse_blocks * all_kv_cache_groups_[0]->seqSizePerBlock()};
}

void HybridLayerKVCacheAllocator::free(const FreeInfo& free_info) {
    auto& kv_cache_resource = free_info.batch_kv_cache_resource;

    if (kv_cache_resource->maxBlocksNum() == 0) {
        return;
    }

    for (auto& resource : kv_cache_resource->batch_resource) {
        int group_nums = resource.groupNums();
        for (int id = 0; id < group_nums; id++) {
            all_kv_cache_groups_[id]->free(resource.blocks(id));
        }
    }
    kv_cache_resource->clearBlocks();
}

void HybridLayerKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    auto& kv_cache_resource = insert_info.batch_kv_cache_resource;
    RTP_LLM_CHECK(kv_cache_resource != nullptr);

    int batch_size         = kv_cache_resource->batchSize();
    int seq_size_per_block = full_kv_cache_group_->seqSizePerBlock();

    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
        auto& cache_keys = kv_cache_resource->cacheKeys(batch_id);
        auto  blocks_num = kv_cache_resource->blocksNum(batch_id);

        auto token_ids = insert_info.complete_token_ids->completeTokenIdsVec(batch_id);
        if (token_ids.size() <= 1) {
            continue;
        }
        size_t token_len     = token_ids.size() - 1;
        size_t max_by_tokens = token_len / static_cast<size_t>(seq_size_per_block);
        blocks_num           = std::min((int)cache_keys.size(), std::min((int)blocks_num, (int)max_by_tokens));
        if (blocks_num == 0) {
            continue;
        }

        std::vector<CacheKeyType> put_cache_keys(cache_keys.begin(), cache_keys.begin() + blocks_num);
        for (int group_id = 0; group_id < all_kv_cache_groups_.size(); group_id++) {
            const auto&               block_ids = kv_cache_resource->blocks(batch_id, group_id);
            std::vector<BlockIdxType> put_block_ids(block_ids.begin(), block_ids.begin() + blocks_num);
            all_kv_cache_groups_[group_id]->insertIntoCache(put_cache_keys, put_block_ids, insert_info.is_resident);
        }
    }
}

CacheLayerLayout HybridLayerKVCacheAllocator::layerCacheBase() const {
    CacheLayerLayout layout;
    auto             layer_tensors = full_kv_cache_group_->layerCacheBase();
    layout.layers_to_buffer_ptrs.clear();
    layout.layers_to_buffer_ptrs.resize(config_.layer_num);
    for (const auto& kv : layer_tensors) {
        int layer_id = kv.first;
        if (layer_id >= 0 && layer_id < config_.layer_num) {
            const auto& tensor = kv.second;
            if (tensor.defined() && tensor.numel() > 0) {
                layout.layers_to_buffer_ptrs[layer_id] = torchTensor2Buffer(tensor);
            } else {
                layout.layers_to_buffer_ptrs[layer_id] = nullptr;
            }
        }
    }
    return layout;
}

// TODO, 修改下。
BlockAddrInfo HybridLayerKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToAddr(layer_id, block_id);
}

// TODO, 修改下。
BlockBufferPtrInfo HybridLayerKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BufferPtr> HybridLayerKVCacheAllocator::convertIndexToBuffer(int layer_id,
                                                                         int block_id,
                                                                         int partition_count,
                                                                         int partition_id) const {
    return {};
}

std::shared_ptr<KVCacheResourceV1>
HybridLayerKVCacheAllocator::incrKVCacheRef(const KVCacheResourceV1& kvcache_resource,
                                            const CacheKeysType&     cache_keys) {
    return nullptr;
}

void HybridLayerKVCacheAllocator::decrKVCacheRef(const KVCacheResourceV1& kvcache_resource) {}

int HybridLayerKVCacheAllocator::seqSizePerBlock() const {
    return full_kv_cache_group_->seqSizePerBlock();
}

bool HybridLayerKVCacheAllocator::updateKVBlock(const BatchKVCacheResourcePtr& batch_kv_cache_resource,
                                                const std::vector<int>&        block_src_batch,
                                                bool                           copy_last_block,
                                                std::vector<BlockIdPair>&      block_update_mapping) {
    // TODO(chanyin): may be implemented in Base class in future
    return true;
}

void HybridLayerKVCacheAllocator::clearCache() {
    if (block_pool_) {
        block_pool_->clearCache();
    }
}

}  // namespace rtp_llm
