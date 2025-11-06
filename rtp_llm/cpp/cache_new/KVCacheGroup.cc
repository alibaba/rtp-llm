#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool KVCacheGroup::init() {
    auto layer_tensors = block_pool_->layerCacheBase();

    // TODO(chanyin): layer_ids might not be set in sequence, move to basic class
    for (int i = 0; i < layer_ids_.size(); ++i) {
        gloabl_layer_to_kv_tensors[layer_ids_[i]]  = layer_tensors[i];
        gloabl_layer_to_local_layer[layer_ids_[i]] = i;
    }

    return true;
}

bool KVCacheGroup::ensureFreeBlocks(int required_blocks) {
    if (required_blocks <= 0) {
        return true;
    }

    // blocks popped by block_cache_ might be occupied by other query
    // it's necessary to checkout whether free blocks are enough
    while (block_pool_->freeBlockNums() < required_blocks) {
        int  need_evict     = required_blocks - block_pool_->freeBlockNums();
        auto evicted_blocks = block_cache_->pop(need_evict);
        if (evicted_blocks.empty()) {
            RTP_LLM_LOG_WARNING("ensure free blocks failed, free blocks : %d, need evict blocks : %d",
                                block_pool_->freeBlockNums(),
                                need_evict);
            return false;
        }
        block_pool_->free(evicted_blocks);
    }

    return true;
}

size_t KVCacheGroup::freeBlockNums() const {
    return block_pool_->freeBlockNums();
}

int KVCacheGroup::seqSizePerBlock() const {
    return seq_size_per_block_;
}

std::unordered_map<int, torch::Tensor> KVCacheGroup::layerCacheBase() const {
    return gloabl_layer_to_kv_tensors;
}

BlockAddrInfo KVCacheGroup::convertIndexToAddr(int layer_id, int block_id) const {
    auto it = gloabl_layer_to_local_layer.find(layer_id);
    if (it == gloabl_layer_to_local_layer.end()) {
        RTP_LLM_LOG_ERROR("Invalid layer_id: %d", layer_id);
        return {nullptr, nullptr, nullptr, nullptr};
    }
    int local_layer_id = it->second;
    return block_pool_->convertIndexToAddr(local_layer_id, block_id);
}

BlockBufferInfo KVCacheGroup::convertIndexToBuffer(int layer_id, int block_id) const {
    auto it = gloabl_layer_to_local_layer.find(layer_id);
    if (it == gloabl_layer_to_local_layer.end()) {
        RTP_LLM_LOG_ERROR("Invalid layer_id: %d", layer_id);
        return {nullptr, nullptr};
    }
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id);
}

MatchResult KVCacheGroup::matchSingleKey(CacheKeyType cache_key) {
    CacheKeysType cache_keys = {cache_key};
    return match(cache_keys);
}

}  // namespace rtp_llm
