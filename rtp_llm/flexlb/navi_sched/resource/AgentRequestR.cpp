#include "aios/navi_sched/resource/AgentRequestR.h"

namespace navi_sched {

const std::string AgentRequestR::RESOURCE_ID = "navi_sched.agent_request_r";

AgentRequestR::AgentRequestR() : navi::ExternalResource(RESOURCE_ID, navi::RS_EXTERNAL) {}

void AgentRequestR::setCompletionHandler(const CompletionHandler &handler)
{
    autil::ScopedWriteLock lock(_handlerLock);
    _completionHandler = handler;
}

void AgentRequestR::setFinishHandler(const std::function<bool()> &handler)
{
    autil::ScopedWriteLock lock(_handlerLock);
    _finishHandler = handler;
}

bool AgentRequestR::submit(const NaviSchedForwardRequest &request,
                           const navi::AsyncPipePtr &completionPipe) const
{
    return SubmitResult::ACCEPTED == submitWithResult(request, completionPipe);
}

AgentRequestR::SubmitResult AgentRequestR::submitWithResult(
    const NaviSchedForwardRequest &request,
    const navi::AsyncPipePtr &completionPipe) const
{
    CompletionHandler completionHandler;
    {
        autil::ScopedReadLock lock(_handlerLock);
        completionHandler = _completionHandler;
    }
    if (!completionHandler) {
        _rejectedRequestCount.fetch_add(1, std::memory_order_relaxed);
        return SubmitResult::COMPLETION_HANDLER_NOT_SET;
    }
    if (!completionHandler(request, completionPipe)) {
        _rejectedRequestCount.fetch_add(1, std::memory_order_relaxed);
        return SubmitResult::COMPLETION_HANDLER_REJECTED;
    }
    _requestCount.fetch_add(1, std::memory_order_relaxed);
    return SubmitResult::ACCEPTED;
}

bool AgentRequestR::finish() const
{
    std::function<bool()> handler;
    {
        autil::ScopedReadLock lock(_handlerLock);
        handler = _finishHandler;
    }
    return !handler || handler();
}

navi::SummaryErrorCode AgentRequestR::getSummary(const std::string &,
                                                 navi::SummaryBuilder &builder) const
{
    bool completionHandlerSet = false;
    {
        autil::ScopedReadLock lock(_handlerLock);
        completionHandlerSet = bool(_completionHandler);
    }
    auto table = builder.newTable("NaviSchedAgentRequestOverview");
    TABLE_ADD_COLUMN_VALUE(table, "completion_handler_set", completionHandlerSet);
    TABLE_ADD_COLUMN_VALUE(table, "request_count",
                           _requestCount.load(std::memory_order_relaxed));
    TABLE_ADD_COLUMN_VALUE(table, "rejected_request_count",
                           _rejectedRequestCount.load(std::memory_order_relaxed));
    return navi::SEC_OK;
}

REGISTER_RESOURCE(AgentRequestR);

}
