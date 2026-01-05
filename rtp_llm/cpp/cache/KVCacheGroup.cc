#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool KVCacheGroup::init() {
    auto layer_tensors = block_pool_->layerCacheBase();
    auto scale_tensors = block_pool_->layerScaleCacheBase();

    for (int i = 0; i < static_cast<int>(layer_ids_.size()); ++i) {
        const int global_layer_id = layer_ids_[i];
        RTP_LLM_CHECK_WITH_INFO(global_layer_id >= 0 && static_cast<size_t>(global_layer_id) < layer_tensors.size(),
                                "global_layer_id out of range in KVCacheGroup::init: id=%d tensors_size=%zu",
                                global_layer_id,
                                layer_tensors.size());
        global_layer_to_kv_tensors[global_layer_id] = layer_tensors[static_cast<size_t>(global_layer_id)];

        if (!scale_tensors.empty()) {
            RTP_LLM_CHECK_WITH_INFO(static_cast<size_t>(global_layer_id) < scale_tensors.size(),
                                    "global_layer_id out of range in scale_tensors: id=%d tensors_size=%zu",
                                    global_layer_id,
                                    scale_tensors.size());
            global_layer_to_kv_scale_tensors[global_layer_id] = scale_tensors[static_cast<size_t>(global_layer_id)];
        }
        global_layer_to_local_layer[layer_ids_[i]] = i;
    }

    return true;
}

bool KVCacheGroup::ensureFreeBlocks(int required_blocks) {
    if (required_blocks <= 0) {
        return true;
    }

    // blocks popped by block cache might be occupied by request
    // it's necessary to checkout whether free blocks are enough
    while (true) {
        const auto free_blocks = block_pool_->freeBlocksNum();
        if (free_blocks >= static_cast<size_t>(required_blocks)) {
            break;
        }

        int  need_evict     = required_blocks - static_cast<int>(free_blocks);
        auto evicted_blocks = block_cache_->pop(need_evict);
        if (evicted_blocks.empty()) {
            RTP_LLM_LOG_WARNING("ensure free blocks failed, free blocks : %d, need evict blocks : %d",
                                block_pool_->freeBlocksNum(),
                                need_evict);
            return false;
        }
        block_pool_->blockCacheFree(evicted_blocks);
    }

    return true;
}

size_t KVCacheGroup::freeBlocksNum() const {
    return block_pool_->freeBlocksNum();
}

int KVCacheGroup::seqSizePerBlock() const {
    return seq_size_per_block_;
}

std::unordered_map<int, torch::Tensor> KVCacheGroup::layerCacheBase() const {
    return global_layer_to_kv_tensors;
}

std::unordered_map<int, torch::Tensor> KVCacheGroup::layerScaleCacheBase() const {
    return global_layer_to_kv_scale_tensors;
}

BlockAddrInfo KVCacheGroup::convertIndexToAddr(int layer_id, int block_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToAddr(local_layer_id, block_id);
}

BlockBufferPtrInfo KVCacheGroup::convertIndexToBuffer(int layer_id, int block_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id);
}

std::vector<BufferPtr>
KVCacheGroup::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id, partition_count, partition_id);
}

MatchResult KVCacheGroup::matchSingleKey(CacheKeyType cache_key) {
    CacheKeysType cache_keys = {cache_key};
    return match(cache_keys);
}

void KVCacheGroup::reference(const BlockIndicesType& new_block_indices) {
    block_pool_->requestReference(new_block_indices);
}

}  // namespace rtp_llm
