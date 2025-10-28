#include "rtp_llm/cpp/cache_new/FullKVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool FullKVCacheGroup::init() {
    auto layer_tensors = block_pool_->layerCacheBase();

    for(int i = 0; i < layer_ids_.size(); ++i) {
        gloabl_layer_to_kv_tensors[layer_ids_[i]] = layer_tensors[i];
        gloabl_layer_to_local_layer[layer_ids_[i]] = i;
    }

    return true;
}

std::vector<int> FullKVCacheGroup::alloc(int needed_blocks) {
    if (needed_blocks <= 0) {
        RTP_LLM_LOG_DEBUG("No blocks needed for allocation");
        return {};
    }

    auto new_blocks = block_pool_->alloc(needed_blocks);
    RTP_LLM_LOG_DEBUG("Allocated %zu blocks (requested %d)", 
                      new_blocks.size(), needed_blocks);
    return new_blocks;
}

MatchResult FullKVCacheGroup::match(std::vector<int64_t> cache_keys) const {
    MatchResult result;
    result.reuse_length = 0;

    for(auto& cache_key : cache_keys) {
        auto match_result = block_cache_->match(cache_key);
        if(match_result.matched_index != NULL_BLOCK_IDX) {
            result.reuse_length++;
            result.cached_keys.push_back(cache_key);
            result.block_indices.push_back(match_result.matched_index);
        }
    }
    
    return result;
}

void FullKVCacheGroup::free(std::vector<int> block_indices) {
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

void FullKVCacheGroup::insertIntoCache(std::vector<int64_t> cache_keys, std::vector<int> block_indices) {
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

    for(int i = 0; i < cache_keys.size(); ++i) {
        block_cache_->put({cache_keys[i], block_indices[i], {}, false});
    }
    
    RTP_LLM_LOG_DEBUG("Inserted %zu blocks into cache", block_indices.size());
}

KVCacheGroupType FullKVCacheGroup::type() const {
    return KVCacheGroupType::FULL;
}

bool FullKVCacheGroup::evict(int need_evict_len) {
    if (need_evict_len <= 0) {
        return true;
    }
    
    std::vector<int> evicted_blocks;
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

std::unordered_map<int, torch::Tensor> FullKVCacheGroup::layerCacheBase() const {
    return gloabl_layer_to_kv_tensors;
}

BufferPtr FullKVCacheGroup::convertIndexToAddr(int layer_id, int block_id) const {
    int local_layer_id = gloabl_layer_to_local_layer.at(layer_id);
    return block_pool_->convertIndexToAddr(local_layer_id, block_id);
}

}  // namespace rtp_llm
