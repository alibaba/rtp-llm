#include "rtp_llm/cpp/cache/KVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

bool KVCacheGroup::init() {
    return true;
}

bool KVCacheGroup::ensureFreeBlocks(int required_blocks) {
    if (required_blocks <= 0) {
        return true;
    }

    const bool use_pop_and_free = block_cache_->registeredModelNum() > 0;

    // blocks popped by block cache might be occupied by request
    // it's necessary to checkout whether free blocks are enough
    while (true) {
        const auto free_blocks = block_pool_->freeBlocksNum();
        if (free_blocks >= static_cast<size_t>(required_blocks)) {
            break;
        }

        const size_t deficit = static_cast<size_t>(required_blocks) - free_blocks;
        if (use_pop_and_free) {
            size_t freed = block_cache_->popAndFree(deficit, model_id_);
            if (freed == 0) {
                RTP_LLM_LOG_WARNING("ensure free blocks failed (popAndFree), free blocks : %zu, deficit : %zu",
                                    block_pool_->freeBlocksNum(),
                                    deficit);
                return false;
            }
        } else {
            auto evicted_blocks = block_cache_->pop(static_cast<int>(deficit));
            if (evicted_blocks.empty()) {
                RTP_LLM_LOG_WARNING("ensure free blocks failed, free blocks : %zu, need evict blocks : %zu",
                                    block_pool_->freeBlocksNum(),
                                    deficit);
                return false;
            }
            block_pool_->blockCacheFree(evicted_blocks);
        }
    }

    return true;
}

size_t KVCacheGroup::freeBlocksNum() const {
    return block_pool_->freeBlocksNum();
}

int KVCacheGroup::seqSizePerBlock() const {
    return seq_size_per_block_;
}

int KVCacheGroup::group_id() const {
    return group_id_;
}

BlockAddrInfo KVCacheGroup::convertIndexToAddr(int layer_id, int block_id) const {
    return block_pool_->convertIndexToAddr(layer_id, block_id);
}

std::vector<BlockInfo> KVCacheGroup::convertIndexToBuffer(int layer_id, int block_id) const {
    return block_pool_->convertIndexToBuffer(layer_id, block_id);
}

std::vector<BlockInfo>
KVCacheGroup::convertIndexToBuffer(int layer_id, int block_id, int partition_count, int partition_id) const {
    return block_pool_->convertIndexToBuffer(layer_id, block_id, partition_count, partition_id);
}

void KVCacheGroup::reference(const BlockIndicesType& new_block_indices) {
    block_pool_->requestReference(new_block_indices);
}

}  // namespace rtp_llm
