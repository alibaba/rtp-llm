#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::vector<int> KVCacheGroup::alloc(int needed_blocks) {
    if (needed_blocks <= 0) {
        RTP_LLM_LOG_DEBUG("No blocks needed for allocation");
        return {};
    }

    auto new_blocks = block_pool_->alloc(needed_blocks);

    RTP_LLM_LOG_DEBUG("Allocated %zu blocks (requested %d)", 
                      allocated_blocks.size(), needed_blocks);
    return new_blocks;
}

MatchResult KVCacheGroup::match(std::vector<int64_t> cache_keys) const {
    MatchResult result;
    result.reuse_length = 0;

    if (group_spec_.type_ != KVCacheType::LINEAR) {
        result = block_cache_->prefixMatch(cache_keys);
    } else {
        result = block_cache_->match(cache_keys);
    }
    
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
    
    block_cache_->put(cache_keys, block_indices);
    RTP_LLM_LOG_DEBUG("Inserted %zu blocks into cache", block_indices.size());
}


KVCacheType KVCacheGroup::type() const {
    return group_spec_.type_;
}

bool KVCacheGroup::evict(int need_evict_len) {
    if (need_evict_len <= 0) {
        return true;
    }
    
    vector<int> evicted_blocks;
    while (evicted_blocks.size() < need_evict_len && !block_cache_->empty()) {
        auto evicted_block = block_cache_->pop();
        evicted_blocks.push_back(evicted_block);
    }
    block_pool_->free(evicted_blocks);
    
    bool success = (evicted_count >= need_evict_len);
    RTP_LLM_LOG_DEBUG("Evicted %d blocks (needed %d): %s", 
                      evicted_count, need_evict_len, success ? "success" : "partial");
    
    return success;
}

}  // namespace rtp_llm
