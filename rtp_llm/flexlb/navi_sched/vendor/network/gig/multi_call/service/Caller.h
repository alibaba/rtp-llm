#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Closure.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/QuerySessionContext.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceResource.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/TerminateNotifier.h"
#include "autil/Lock.h"

namespace multi_call {

class ChildNodeReply;
typedef std::shared_ptr<ChildNodeReply> ChildNodeReplyPtr;

class Caller : public std::enable_shared_from_this<Caller>
{
public:
    /**
     * @func: called whenever a response arrived
     */
    Caller(const ChildNodeReplyPtr &reply,
           const std::shared_ptr<QuerySessionContext> &sessionContext);
    virtual ~Caller();

private:
    Caller(const Caller &);
    Caller &operator=(const Caller &);

public:
    void call(const SearchServiceResourcePtr &resource, bool isRetry);
    void earlyTerminate();
    bool canTerminate() const {
        if (_hasClosure) {
            return _notifier.canTerminate();
        } else {
            return _stopped;
        }
    }
    void afterCall(Closure *closure) {
        _hasClosure = true;
        _notifier.setClosure(closure);
    }
    bool hasClosure() const {
        return _hasClosure;
    }
    void stop() {
        _stopped = true;
    }
    Closure *stealClosure() {
        return _notifier.stealClosure();
    }
    void decNotifier(PartIdTy partIdIndex) {
        _notifier.dec(partIdIndex);
    }
    const ChildNodeReplyPtr &getReply() const {
        return _reply;
    }

protected:
    bool _hasClosure : 1;
    bool _stopped    : 1;
    TerminateNotifier _notifier;
    ChildNodeReplyPtr _reply;
    std::shared_ptr<QuerySessionContext> _sessionContext;
    std::map<PartIdTy, PartIdTy> _partIdToIndexMap;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(Caller);

} // namespace multi_call

