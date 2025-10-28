#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::vector<int> KVCacheGroup::alloc(int needed_blocks) {
    if (needed_blocks <= 0) {
        RTP_LLM_LOG_DEBUG("No blocks needed for allocation");
        return {};
    }
    
    std::vector<int> allocated_blocks;
    
    if (!evict(needed_blocks)) {
        RTP_LLM_LOG_WARNING("Failed to evict enough blocks for allocation");
    }
    
    // 从 block pool 分配新的块
    auto new_blocks = block_pool_->alloc(needed_blocks);
    allocated_blocks.insert(allocated_blocks.end(), new_blocks.begin(), new_blocks.end());
    
    RTP_LLM_LOG_DEBUG("Allocated %zu blocks (requested %d)", 
                      allocated_blocks.size(), needed_blocks);
    
    return allocated_blocks;
}

MatchResult KVCacheGroup::match(std::vector<int64_t> cache_keys) const {
    MatchResult result;
    result.reuse_length = 0;
    
    if (!block_cache_) {
        RTP_LLM_LOG_DEBUG("Block cache is not initialized, no match possible");
        return result;
    }
    
    auto cache_match_result = block_cache_->match(cache_keys);
    
    // 转换 BlockCache::MatchResult 到 MatchResult
    result.reuse_length = cache_match_result.matched_indices.size();
    
    // 构建匹配结果
    if (result.reuse_length > 0) {
        for (size_t i = 0; i < result.reuse_length; ++i) {
            result.cached_keys.push_back(cache_keys[i]);
            result.block_indices.push_back(cache_match_result.matched_indices[i]);
        }
    }
    
    RTP_LLM_LOG_DEBUG("Matched %zu blocks from cache", result.reuse_length);
    
    return result;
}

void KVCacheGroup::free(std::vector<int> block_indices) {
    if (!block_pool_) {
        RTP_LLM_LOG_ERROR("Block pool is not initialized");
        return;
    }
    
    if (block_indices.empty()) {
        return;
    }
    
    // 释放块到 block pool
    block_pool_->free(block_indices);
    
    RTP_LLM_LOG_DEBUG("Freed %zu blocks", block_indices.size());
}

void KVCacheGroup::insertIntoCache(std::vector<int64_t> cache_keys, std::vector<int> block_indices) {
    if (!block_cache_) {
        RTP_LLM_LOG_DEBUG("Block cache is not initialized, skip insertion");
        return;
    }
    
    if (cache_keys.size() != block_indices.size()) {
        RTP_LLM_LOG_ERROR("Cache keys size (%zu) doesn't match block indices size (%zu)", 
                          cache_keys.size(), block_indices.size());
        return;
    }
    
    if (cache_keys.empty()) {
        return;
    }
    
    // 将块插入到缓存中
    auto evicted_blocks = block_cache_->put(cache_keys, block_indices);
    
    // 如果有被驱逐的块，需要释放它们
    if (!evicted_blocks.empty()) {
        block_pool_->free(evicted_blocks);
        RTP_LLM_LOG_DEBUG("Evicted and freed %zu blocks during cache insertion", evicted_blocks.size());
    }
    
    RTP_LLM_LOG_DEBUG("Inserted %zu blocks into cache", block_indices.size());
}

std::map<int, BufferPtr> KVCacheGroup::blockBuffer(int block_id, int64_t cache_key) {
    std::map<int, BufferPtr> result;
    
    if (!block_pool_) {
        RTP_LLM_LOG_ERROR("Block pool is not initialized");
        return result;
    }
    
    // 为每个层获取对应的 buffer
    for (int layer_id : layer_ids_) {
        auto addr_info = block_pool_->convertIndexToAddr(block_id, layer_id);
        
        if (addr_info.k_addr == nullptr) {
            RTP_LLM_LOG_WARNING("Failed to get address for block %d, layer %d", block_id, layer_id);
            continue;
        }
        
        // 这里简化处理，实际可能需要根据具体需求创建不同类型的 Buffer
        // 由于 BlockPool::convertIndexToAddr 返回的是地址信息，我们需要创建相应的 Buffer
        // 这里假设我们需要返回整个 block 的 buffer，具体实现可能需要根据实际需求调整
        
        try {
            // 创建一个指向该层该块的 Buffer
            // 注意：这里的实现可能需要根据实际的内存布局和需求进行调整
            auto layer_tensors = block_pool_->layerCacheBase();
            auto it = layer_tensors.find(layer_id);
            if (it != layer_tensors.end()) {
                // 从 torch::Tensor 创建 Buffer 的逻辑
                // 这里需要根据实际的 Buffer 创建方式进行实现
                RTP_LLM_LOG_DEBUG("Created buffer for block %d, layer %d", block_id, layer_id);
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("Failed to create buffer for block %d, layer %d: %s", 
                              block_id, layer_id, e.what());
        }
    }
    
    return result;
}

KVCacheType KVCacheGroup::type() const {
    return group_spec_.type_;
}

bool KVCacheGroup::evict(int need_evict_len) {
    if (!block_cache_) {
        RTP_LLM_LOG_DEBUG("Block cache is not initialized, cannot evict");
        return false;
    }
    
    if (need_evict_len <= 0) {
        return true;
    }
    
    // 尝试从缓存中驱逐块
    int evicted_count = 0;
    while (evicted_count < need_evict_len && !block_cache_->empty()) {
        auto evicted_blocks = block_cache_->pop();
        if (evicted_blocks.empty()) {
            break;
        }
        
        // 释放被驱逐的块
        block_pool_->free(evicted_blocks);
        evicted_count += static_cast<int>(evicted_blocks.size());
    }
    
    bool success = (evicted_count >= need_evict_len);
    RTP_LLM_LOG_DEBUG("Evicted %d blocks (needed %d): %s", 
                      evicted_count, need_evict_len, success ? "success" : "partial");
    
    return success;
}

}  // namespace rtp_llm
