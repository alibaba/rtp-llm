#include "rtp_llm/cpp/cache_new/SingleTypeKVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache_new/BlockCache.h"
#include "rtp_llm/cpp/cache_new/KVCacheSpec.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

SingleTypeKVCacheAllocator::SingleTypeKVCacheAllocator(const CacheConfig& config, 
                                                       rtp_llm::DeviceBase* device, 
                                                       AllocationType atype)
    : KVCacheAllocator(config, device, atype) {
}

bool SingleTypeKVCacheAllocator::init() {
    BlockPoolConfig pool_config = BlockPoolConfigHelper::createKVFirstConfig(
        config_.layer_num,
        config_.block_num,
        config_.block_size
    );
    
    block_pool_ = std::make_shared<BlockPool>(pool_config, device_, atype_);
    if (!block_pool_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize block pool for SingleTypeKVCacheAllocator");
        return false;
    }
    
    std::vector<int> layer_ids;
    for (int i = 0; i < config_.layer_num; ++i) {
        layer_ids.push_back(i);
    }
    
    KVCacheSpec group_spec;
    group_spec.layer_ids_ = layer_ids;
    group_spec.type_ = KVCacheGroupType::FULL;
    
    auto block_cache = std::make_shared<BlockCacheV1>(1024);  
    
    full_kv_cache_group_ = std::make_shared<FullKVCacheGroup>(
        layer_ids, group_spec, block_cache, block_pool_
    );
    
    if (!full_kv_cache_group_->init()) {
        RTP_LLM_LOG_ERROR("Failed to initialize FullKVCacheGroup");
        return false;
    }
    
    kv_cache_groups_.push_back(full_kv_cache_group_);
    
    RTP_LLM_LOG_INFO("SingleTypeKVCacheAllocator initialized successfully with KV_FIRST layout");
    return true;
}

MallocResult SingleTypeKVCacheAllocator::malloc(const MallocInfo& malloc_info) {
    if (!malloc_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false, 0};
    }

    auto& cache_keys = malloc_info.batch_kv_cache_resource->cache_keys[0];
    auto& block_ids = malloc_info.batch_kv_cache_resource->batch_block_id[0];

    // all block has been allocated
    // if (block_ids.size() >= cache_keys.size()) {
    //     return {true, block_ids.size()};
    // }
    
    // TODO: add reserve_step in malloc_info

    auto& seq_len = malloc_info.complete_token_ids->seqLength();

    auto seq_size_per_block = full_kv_cache_group_->seqSizePerBlock();
    int need_block_num = std::max((seq_len + seq_size_per_block - 1) / seq_size_per_block - block_ids.size(), 0);

    if (malloc_info.batch_kv_cache_resource->enable_reuse_cache && block_ids.size() < cache_keys.size()) {
        vector<int64_t> match_cache_keys(cache_keys.begin() + block_ids.size(), cache_keys.end());
        auto match_result = full_kv_cache_group_->match(cache_keys);
        need_block_num = need_block_num - match_result.reuse_length;
    }

    if (full_kv_cache_group_->freeBlockNums() < need_block_num) {
        full_kv_cache_group_->evict(need_block_num);
    }

    auto new_block_ids = full_kv_cache_group_->alloc(need_block_num);
    if new_block_ids.empty() {
        RTP_LLM_LOG_ERROR("Failed to allocate %d blocks", need_block_num);
        return {false, 0};
    }

    block_ids.insert(block_ids.end(), new_block_ids.begin(), new_block_ids.end());
    return {true, match_result.reuse_length};
}

FreeResult SingleTypeKVCacheAllocator::free(const FreeInfo& free_info) {
    if (!free_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false};
    }
    
    std::vector<int> blocks_to_free;
    for (int batch_id = 0; batch_id < free_info.batch_kv_cache_resource->batchSize(); ++batch_id) {
        const auto& blocks = free_info.batch_kv_cache_resource->blocks(batch_id);
        blocks_to_free.insert(blocks_to_free.end(), blocks.begin(), blocks.end());
    }
    
    if (!blocks_to_free.empty()) {
        full_kv_cache_group_->free(blocks_to_free);
        RTP_LLM_LOG_DEBUG("Freed %zu blocks", blocks_to_free.size());
    }
    
    return {true};
}

InsertResult SingleTypeKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    if (!malloc_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return {false, 0};
    }

    auto& cache_keys = malloc_info.batch_kv_cache_resource->cache_keys[0];
    auto& block_ids = malloc_info.batch_kv_cache_resource->batch_block_id[0];


    auto put_len = std::min(cache_keys.size(), block_ids.size());

    vector<int64_t> put_cache_keys(cache_keys.begin(), cache_keys.begin() + put_len);
    vector<int> put_block_ids(block_ids.begin(), block_ids.begin() + put_len);
    full_kv_cache_group_->insertIntoCache(put_cache_keys, put_block_ids);

    return {true};
}

CacheLayerLayout SingleTypeKVCacheAllocator::layerCacheBase() const {
    CacheLayerLayout layout;
    auto layer_tensors = full_kv_cache_group_->layerCacheBase();
    for (int i = 0; i < layer_tensors.size(); i++) {
        layout.layers_to_buffer_ptrs.push_back(layer_tensors[i]);
    }
    return layout;
}

BlockAddrInfo SingleTypeKVCacheAllocator::convertIndexToAddr(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToAddr(layer_id, block_id);
}

BlockBufferInfo SingleTypeKVCacheAllocator::convertIndexToBuffer(int layer_id, int block_id) const {
    return full_kv_cache_group_->convertIndexToBuffer(layer_id, block_id);
}

}  // namespace rtp_llm
