#pragma once

#include <unistd.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace multi_call {

class SharedPtrUtil
{
public:
    template <class T>
    static bool waitUseCount(const std::shared_ptr<T> &ptr, uint32_t expectUseCount = 1,
                             int64_t timeout = -1) {
        const int64_t interval = 10 * 1000;
        while (ptr.use_count() > expectUseCount) {
            if (timeout != -1) {
                if (timeout <= interval) {
                    return false;
                }
                timeout -= interval;
            }
            usleep(interval);
        }
        return true;
    }
};

} // namespace multi_call

