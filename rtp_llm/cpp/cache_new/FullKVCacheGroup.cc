#include "rtp_llm/cpp/cache_new/FullKVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool FullKVCacheGroup::init() {
    auto layer_tensors = block_pool_->layerCacheBase();

    // TODO(chanyin): layer_ids might not be set in sequence
    for(int i = 0; i < layer_ids_.size(); ++i) {
        gloabl_layer_to_kv_tensors[layer_ids_[i]] = layer_tensors[i];
        gloabl_layer_to_local_layer[layer_ids_[i]] = i;
    }

    return true;
}

std::vector<BlockIdxType> FullKVCacheGroup::alloc(int needed_blocks) {
    if (needed_blocks <= 0) {
        RTP_LLM_LOG_DEBUG("No blocks needed for allocation");
        return {};
    }

    auto new_blocks = block_pool_->alloc(needed_blocks);
    RTP_LLM_LOG_DEBUG("Allocated %zu blocks (requested %d)", 
                      new_blocks.size(), needed_blocks);
    return new_blocks;
}

MatchResult FullKVCacheGroup::match(std::vector<CacheKeyType> cache_keys) const {
    MatchResult result;
    result.reuse_length = 0;

    for (auto& cache_key : cache_keys) {
        auto mr = block_cache_->match(cache_key);
        if (mr.matched_index != NULL_BLOCK_IDX) {
            result.reuse_length++;
            result.cached_keys.push_back(cache_key);
            result.block_indices.push_back(mr.matched_index);
        }
    }

    return result;
}

void FullKVCacheGroup::free(std::vector<BlockIdxType> block_indices) {
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

void FullKVCacheGroup::insertIntoCache(std::vector<CacheKeyType> cache_keys,
                                       std::vector<BlockIdxType> block_indices,
                                       std::vector<std::vector<float>> loss) {
    if (!block_cache_) {
        RTP_LLM_LOG_DEBUG("Block cache is not initialized, skip insertion");
        return;
    }
    
    if (cache_keys.empty()) {
        return;
    }

    if (cache_keys.size() != block_indices.size() || cache_keys.size() != loss.size()) {
        RTP_LLM_LOG_ERROR("Cache keys size (%zu) doesn't match block indices size (%zu)", 
                          cache_keys.size(), block_indices.size());
        return;
    }

    for (size_t i = 0; i < cache_keys.size(); ++i) {
        BlockCacheV1::CacheItem item;
        item.cache_key = cache_keys[i];
        item.block_index = block_indices[i];
        item.loss = loss[i];
        item.is_resident = false;
        block_cache_->put(item);
    }
    
    RTP_LLM_LOG_DEBUG("Inserted %zu blocks into cache", block_indices.size());
}


size_t FullKVCacheGroup::freeBlockNums() const {
    return block_pool_->freeBlockNums();
}

int FullKVCacheGroup::seqSizePerBlock() const { return static_cast<int>(group_spec_->seq_size_per_block); }

bool FullKVCacheGroup::evict(int need_evict_len) {
    if (need_evict_len <= 0) {
        return true;
    }
    
    // blocks popped by block_cache_ might be occupied by other query
    // it's necessary to checkout whether free blocks are enough
    while(block_pool_->freeBlockNums() < need_evict_len) {
        int need_evict_len_cur = need_evict_len - block_pool_->freeBlockNums();
        auto evicted_blocks = block_cache_->pop(need_evict_len_cur);
        block_pool_->free(evicted_blocks);
    }
    
    return true;
}

std::unordered_map<int, torch::Tensor> FullKVCacheGroup::layerCacheBase() const {
    return gloabl_layer_to_kv_tensors;
}

BlockAddrInfo FullKVCacheGroup::convertIndexToAddr(int layer_id, int block_id) const {
    int local_layer_id = gloabl_layer_to_local_layer.at(layer_id);
    return block_pool_->convertIndexToAddr(local_layer_id, block_id);
}

BlockBufferInfo FullKVCacheGroup::convertIndexToBuffer(int layer_id, int block_id) const {
    int local_layer_id = gloabl_layer_to_local_layer.at(layer_id);
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id);
}

}  // namespace rtp_llm
