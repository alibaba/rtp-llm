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

    auto cache_keys = malloc_info.batch_kv_cache_resource->batch_block
    


    
    

}

// MallocResult SingleTypeKVCacheAllocator::mallocWithCache(const MallocInfo& malloc_info) {
//     MallocResult result = {false, 0};
    
//     auto token_ids = malloc_info.complete_token_ids->commonCompleteTokenIdsVec(0);
//     std::vector<int64_t> cache_keys;
//     for (auto token_id : token_ids) {
//         cache_keys.push_back(static_cast<int64_t>(token_id));
//     }
    
//     auto match_result = full_kv_cache_group_->match(cache_keys);
//     result.reuse_len = match_result.reuse_length;
    
//     // 3. 计算需要分配的新块数量
//     int total_needed_blocks = cache_keys.size();
//     int new_blocks_needed = total_needed_blocks - match_result.reuse_length;
    
//     if (new_blocks_needed > 0) {
//         // 4. 分配新块
//         auto new_block_ids = full_kv_cache_group_->alloc(new_blocks_needed);
//         if (new_block_ids.size() != new_blocks_needed) {
//             RTP_LLM_LOG_ERROR("Failed to allocate %d blocks, only got %zu", 
//                              new_blocks_needed, new_block_ids.size());
//             return result;
//         }
        
//         // 5. 将新分配的块添加到 BatchKVCacheResource
//         // 这里需要根据具体的 BatchKVCacheResource 接口来实现
//         // 暂时简化处理
//         RTP_LLM_LOG_DEBUG("Allocated %zu new blocks, reused %d blocks", 
//                          new_block_ids.size(), result.reuse_len);
//     }
    
//     result.success = true;
//     return result;
// }

// MallocResult SingleTypeKVCacheAllocator::mallocSimple(const MallocInfo& malloc_info) {
//     MallocResult result = {false, 0};
    
//     // 简单分配：直接从 block_pool 分配新块
//     int blocks_needed = 1;  // 这里需要根据实际需求计算
    
//     auto new_block_ids = full_kv_cache_group_->alloc(blocks_needed);
//     if (new_block_ids.size() != blocks_needed) {
//         RTP_LLM_LOG_ERROR("Failed to allocate %d blocks, only got %zu", 
//                          blocks_needed, new_block_ids.size());
//         return result;
//     }
    
//     // 将分配的块添加到 BatchKVCacheResource
//     RTP_LLM_LOG_DEBUG("Simple allocation: allocated %zu blocks", new_block_ids.size());
    
//     result.success = true;
//     result.reuse_len = 0;
//     return result;
// }

FreeResult SingleTypeKVCacheAllocator::free(const FreeInfo& free_info) {
    FreeResult result = {false};
    
    if (!free_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return result;
    }
    
    // 收集所有需要释放的块
    std::vector<int> blocks_to_free;
    for (int batch_id = 0; batch_id < free_info.batch_kv_cache_resource->batchSize(); ++batch_id) {
        const auto& blocks = free_info.batch_kv_cache_resource->blocks(batch_id);
        blocks_to_free.insert(blocks_to_free.end(), blocks.begin(), blocks.end());
    }
    
    if (!blocks_to_free.empty()) {
        full_kv_cache_group_->free(blocks_to_free);
        RTP_LLM_LOG_DEBUG("Freed %zu blocks", blocks_to_free.size());
    }
    
    result.success = true;
    return result;
}

InsertResult SingleTypeKVCacheAllocator::insertIntoCache(const InsertInfo& insert_info) {
    InsertResult result = {false};
    
    if (!insert_info.batch_kv_cache_resource) {
        RTP_LLM_LOG_ERROR("BatchKVCacheResource is null");
        return result;
    }
    
    // 这里需要根据具体的缓存插入逻辑来实现
    // 暂时简化处理
    RTP_LLM_LOG_DEBUG("Insert into cache requested");
    
    result.success = true;
    return result;
}

CacheLayerLayout SingleTypeKVCacheAllocator::layerCacheBase() const {
    CacheLayerLayout layout;
    
    if (!full_kv_cache_group_) {
        RTP_LLM_LOG_ERROR("FullKVCacheGroup is not initialized");
        return layout;
    }
    
    // 设置层到组的映射（所有层都属于同一个组）
    layout.layer_to_groups.resize(config_.layer_num, 0);
    
    // 获取层级缓存张量
    auto layer_tensors = full_kv_cache_group_->layerCacheBase();
    layout.layers_to_buffer_ptrs = layer_tensors;
    
    return layout;
}

}  // namespace rtp_llm
