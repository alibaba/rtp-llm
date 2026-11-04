#pragma once

#include <grpc++/alarm.h>
#include <grpc++/grpc++.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/CompletionQueueStatus.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/RandomGenerator.h"
#include "autil/Thread.h"

namespace multi_call {

class GrpcClientWorker
{
public:
    GrpcClientWorker(size_t threadNum);
    ~GrpcClientWorker();

private:
    GrpcClientWorker(const GrpcClientWorker &);
    GrpcClientWorker &operator=(const GrpcClientWorker &);

public:
    void stop();
    const CompletionQueueStatusPtr &getCompletionQueue(size_t allocId);

private:
    void workLoop(size_t idx, CompletionQueueStatusPtr cqsPtr);

private:
    std::vector<::grpc::Alarm *> _shutdownAlarms;
    std::vector<autil::ThreadPtr> _workThreads;
    std::vector<CompletionQueueStatusPtr> _completionQueues;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GrpcClientWorker);

} // namespace multi_call

