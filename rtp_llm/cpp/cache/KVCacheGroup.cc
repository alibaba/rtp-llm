#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool KVCacheGroup::init() {
    auto layer_tensors = block_pool_->allLayerCacheBase();
    auto scale_tensors = block_pool_->allLayerScaleCacheBase();

    // 检查layer_tensors的大小是否足够
    RTP_LLM_CHECK_WITH_INFO(layer_tensors.size() >= layer_ids_.size(),
                            "layer_tensors size (%zu) is less than layer_ids size (%zu)",
                            layer_tensors.size(),
                            layer_ids_.size());
    RTP_LLM_CHECK_WITH_INFO(scale_tensors.size() >= layer_ids_.size(),
                            "scale_tensors size (%zu) is less than layer_ids size (%zu)",
                            scale_tensors.size(),
                            layer_ids_.size());

    for (int i = 0; i < static_cast<int>(layer_ids_.size()); ++i) {
        const int global_layer_id = layer_ids_[i];
        // - For non-hybrid (single-model) layout, BlockPool exposes per-layer tensors indexed by global layer id,
        //   and typically global_layer_id == i.
        // - For hybrid layout, BlockPool exposes per-group "physical layer slot" tensors sized by
        //   CacheConfig.group_layer_num, while layer_ids_ still stores global model layer ids.
        //   In that case, we must bind global_layer_id -> layer_tensors[local_slot=i].

        global_layer_to_kv_tensors[global_layer_id] = layer_tensors[static_cast<size_t>(i)];

        if (!scale_tensors.empty()) {
            global_layer_to_kv_scale_tensors[global_layer_id] = scale_tensors[static_cast<size_t>(i)];
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

        const int need_evict     = required_blocks - static_cast<int>(free_blocks);
        auto      evicted_blocks = block_cache_->pop(need_evict);
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

std::unordered_map<int, torch::Tensor> KVCacheGroup::allLayerCacheBase() const {
    return global_layer_to_kv_tensors;
}

std::unordered_map<int, torch::Tensor> KVCacheGroup::allLayerScaleCacheBase() const {
    return global_layer_to_kv_scale_tensors;
}

BlockAddrInfo KVCacheGroup::convertIndexToAddr(int layer_id, int block_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToAddr(local_layer_id, block_id);
}

std::vector<BlockInfo> KVCacheGroup::convertIndexToBuffer(int layer_id, int block_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id);
}

std::vector<BlockInfo>
KVCacheGroup::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    auto it = global_layer_to_local_layer.find(layer_id);
    RTP_LLM_CHECK_WITH_INFO(it != global_layer_to_local_layer.end(), "invalid layer_id: " + std::to_string(layer_id));
    int local_layer_id = it->second;
    return block_pool_->convertIndexToBuffer(local_layer_id, block_id, partition_count, partition_id);
}

void KVCacheGroup::reference(const BlockIndicesType& new_block_indices) {
    block_pool_->requestReference(new_block_indices);
}

}  // namespace rtp_llm
