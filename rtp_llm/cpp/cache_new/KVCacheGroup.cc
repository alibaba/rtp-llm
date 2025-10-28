#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool KVCacheGroup::init() {
    auto layer_tensors = block_pool_->layerCacheBase();

    for(int i = 0; i < layer_ids_.size(); ++i) {
        gloabl_layer_to_kv_tensors[layer_ids_[i]] = layer_tensors[i];
        gloabl_layer_to_local_layer[layer_ids_[i]] = i;
    }

    return true;
}


std::vector<int> KVCacheGroup::alloc(int needed_blocks) {
    if (needed_blocks <= 0) {
        RTP_LLM_LOG_DEBUG("No blocks needed for allocation");
        return {};
    }

    auto new_blocks = block_pool_->alloc(needed_blocks);

    RTP_LLM_LOG_DEBUG("Allocated %zu blocks (requested %d)", 
                      new_blocks.size(), needed_blocks);
    return new_blocks;
}

MatchResult KVCacheGroup::match(std::vector<int64_t> cache_keys) const {
    MatchResult result;
    result.reuse_length = 0;

    if (group_spec_.type_ != rtp_llm::KVCacheGroupType::LINEAR) {
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


KVCacheGroupType KVCacheGroup::type() const {
    return group_spec_.type_;
}

bool KVCacheGroup::evict(int need_evict_len) {
    if (need_evict_len <= 0) {
        return true;
    }
    
    vector<int> evicted_blocks;
    while (static_cast<int>(evicted_blocks.size()) < need_evict_len && !block_cache_->empty()) {
        auto evicted_block = block_cache_->pop();
        evicted_blocks.push_back(evicted_block);
    }
    block_pool_->free(evicted_blocks);
    
    int evicted_count = static_cast<int>(evicted_blocks.size());
    bool success = (evicted_count >= need_evict_len);
    RTP_LLM_LOG_DEBUG("Evicted %d blocks (needed %d): %s", 
                      evicted_count, need_evict_len, success ? "success" : "partial");
    
    return success;
}


std::unordered_map<int, torch::Tensor> KVCacheGroup::layerCacheBase() const {
    return gloabl_layer_to_kv_tensors;
}


BufferPtr KVCacheGroup::convertIndexToAddr(int layer_id, int block_id) const {
    local_layer_id = gloabl_layer_to_local_layer[layer_id];
    return block_pool_->convertIndexToAddr(local_layer_id, block_id);
}

}  // namespace rtp_llm
