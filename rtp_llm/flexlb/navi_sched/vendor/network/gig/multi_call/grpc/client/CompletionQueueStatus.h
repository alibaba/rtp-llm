#pragma once

#include <grpc++/completion_queue.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "autil/Lock.h"

namespace multi_call {

typedef std::shared_ptr<grpc::CompletionQueue> CompletionQueuePtr;

struct CompletionQueueStatus {
    CompletionQueueStatus(const CompletionQueuePtr &cq_) : stopped(false), cq(cq_) {
    }
    volatile bool stopped;
    autil::ReadWriteLock enqueueLock;
    CompletionQueuePtr cq;
};

MULTI_CALL_TYPEDEF_PTR(CompletionQueueStatus);

} // namespace multi_call

