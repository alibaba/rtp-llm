#include "rtp_llm/cpp/disaggregate/cache_store/LockedBlockBufferManager.h"

#include <mutex>
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

bool LockedBlockBufferManager::lock(const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    std::unique_lock<std::mutex> lock(block_map_mutex_);

    // check available block
    bool no_confict = true;
    for (auto& block : blocks) {
        auto iter = block_map_.find(block->key);
        if (iter == block_map_.end()) {
            continue;
        }
        // 先清理过期的locked block
        for (auto lock_iter = iter->second.begin(); lock_iter != iter->second.end();) {
            if (currentTimeUs() - lock_iter->locked_deadline_us > 0) {
                RTP_LLM_LOG_INFO("locked block buffer manager erase expired locked block %s", block->key.c_str());
                lock_iter = iter->second.erase(lock_iter);
            } else {
                ++lock_iter;
            }
        }

        // 查看有没有交集的block
        for (auto& locked_block : iter->second) {
            int64_t start_addr = (int64_t)locked_block.block->addr.get();
            int64_t end_addr   = start_addr + locked_block.block->len;

            int64_t check_start_addr = (int64_t)block->addr.get();
            int64_t check_end_addr   = check_start_addr + block->len;

            if (check_end_addr <= start_addr || end_addr <= check_start_addr) {
                continue;
            }
            no_confict = false;

            RTP_LLM_LOG_WARNING("locked block %s %p len %d conflict with %p, len %d",
                                locked_block.block->key.c_str(),
                                block->addr.get(),
                                block->len,
                                locked_block.block->addr.get(),
                                locked_block.block->len);
            break;
        }
        if (!no_confict) {
            break;
        }
    }

    if (!no_confict) {
        return false;
    }

    // do lock
    for (auto& block : blocks) {
        block_map_[block->key].emplace_back(LockBlockBuffer{block, currentTimeUs() + expired_time_us_});
    }
    return true;
}

void LockedBlockBufferManager::unlock(const std::vector<std::shared_ptr<BlockBuffer>>& blocks) {
    std::unique_lock<std::mutex> lock(block_map_mutex_);
    for (auto& block : blocks) {
        auto iter = block_map_.find(block->key);
        if (iter == block_map_.end()) {
            continue;
        }

        for (auto lock_iter = iter->second.begin(); lock_iter != iter->second.end();) {
            if (lock_iter->block->addr == block->addr && lock_iter->block->len == block->len) {
                lock_iter = iter->second.erase(lock_iter);
            } else {
                ++lock_iter;
            }
        }

        if (iter->second.size() == 0) {
            block_map_.erase(iter);
        }
    }
}

}  // namespace rtp_llm
