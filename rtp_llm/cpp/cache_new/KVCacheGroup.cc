#include "rtp_llm/cpp/cache_new/KVCacheGroup.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

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

}  // namespace rtp_llm
