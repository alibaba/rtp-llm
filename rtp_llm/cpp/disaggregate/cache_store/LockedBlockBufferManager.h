#pragma once

#include "rtp_llm/cpp/disaggregate/cache_store/RequestBlockBuffer.h"

namespace rtp_llm {

class LockedBlockBufferManager {
public:
    LockedBlockBufferManager(int64_t expired_time_us = 600 * 1000 * 1000): expired_time_us_(expired_time_us) {}

public:
    bool lock(const std::vector<std::shared_ptr<BlockBuffer>>& blocks);
    void unlock(const std::vector<std::shared_ptr<BlockBuffer>>& blocks);

private:
    int64_t expired_time_us_ = 60 * 1000 * 1000;

    // optimize with lock on block key
    struct LockBlockBuffer {
        std::shared_ptr<BlockBuffer> block;
        int64_t                      locked_deadline_us = 0;
        // maybe associated with connection to cancel
        LockBlockBuffer(const std::shared_ptr<BlockBuffer>& block, int64_t locked_deadline_us):
            block(block), locked_deadline_us(locked_deadline_us) {}
    };

    std::mutex                                          block_map_mutex_;
    std::map<std::string, std::vector<LockBlockBuffer>> block_map_;
};

}  // namespace rtp_llm