#pragma once

#include "rtp_llm/flexlb/navi_sched/common/NaviSchedScheduleData.h"
#include "autil/Lock.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include <atomic>

namespace navi_sched {

class AgentRequestR : public navi::ExternalResource {
public:
    enum class SubmitResult {
        ACCEPTED,
        COMPLETION_HANDLER_NOT_SET,
        COMPLETION_HANDLER_REJECTED,
    };

    using CompletionHandler =
        std::function<bool(const NaviSchedForwardRequest &, const navi::AsyncPipePtr &)>;

    AgentRequestR();

public:
    void setCompletionHandler(const CompletionHandler &handler);
    void setFinishHandler(const std::function<bool()> &handler);
    bool submit(const NaviSchedForwardRequest &request,
                const navi::AsyncPipePtr &completionPipe = {}) const;
    SubmitResult submitWithResult(
        const NaviSchedForwardRequest &request,
        const navi::AsyncPipePtr &completionPipe = {}) const;
    bool finish() const;
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    mutable autil::ReadWriteLock _handlerLock;
    CompletionHandler _completionHandler;
    std::function<bool()> _finishHandler;
    mutable std::atomic<uint64_t> _requestCount{0};
    mutable std::atomic<uint64_t> _rejectedRequestCount{0};
};

NAVI_TYPEDEF_PTR(AgentRequestR);

} // namespace navi_sched
