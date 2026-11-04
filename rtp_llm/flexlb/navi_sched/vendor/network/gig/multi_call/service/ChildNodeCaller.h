#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/QuerySession.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Reply.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/SearchService.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CallDelegationThread.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ChildNodeReply.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/FlowConfigSnapshot.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/LatencyTimeSnapshot.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/RetryLimitChecker.h"

namespace multi_call {

class ChildNodeCaller
{
public:
    ChildNodeCaller(const SearchServiceSnapshotPtr &snapshot,
                    const FlowConfigSnapshotPtr &flowConfigSnapshot,
                    const CallDelegationThreadPtr &callDelegationThread,
                    const RequestGeneratorVec &generatorVec,
                    const RetryLimitCheckerPtr &retryLimitChecker,
                    const LatencyTimeSnapshotPtr &latencyTimeSnapshot,
                    const QuerySessionPtr &querySession);
    ~ChildNodeCaller();

private:
    ChildNodeCaller(const ChildNodeCaller &);
    ChildNodeCaller &operator=(const ChildNodeCaller &);

public:
    ChildNodeReplyPtr call();
    ChildNodeReplyPtr call(SearchServiceResourceVector &resourceVec);
    void afterCall(Closure *closure);
    bool isDetectionOn() const;
    bool isRetryOn() const;
    const CallerPtr &getCaller() const;
    void setDisableDetection(bool value) {
        _disableDetection = value;
    }

public:
    // for java client
    bool initResource(SearchServiceResourceVector &resourceVec);
    const ReplyInfoCollectorPtr &getReplyInfoCollector() const {
        return _replyInfoCollector;
    }

private:
    void initCaller();
    void prepareSearchResource(SearchServiceResourceVector &resourceVec);
    void doCall(const SearchServiceResourcePtr &resource);
    void fillSourceId();

private:
    static uint64_t getMaxTimeout(const SearchServiceResourceVector &resourceVec);

private:
    int64_t _callBeginTime;
    SearchServiceSnapshotPtr _snapshot;
    FlowConfigSnapshotPtr _flowConfigSnapshot;
    CallDelegationThreadPtr _callDelegationThread;
    RequestGeneratorVec _generatorVec;
    RetryLimitCheckerPtr _retryLimitChecker;
    LatencyTimeSnapshotPtr _latencyTimeSnapshot;
    QuerySessionPtr _querySession;
    ChildNodeReplyPtr _reply;
    ReplyInfoCollectorPtr _replyInfoCollector;
    CallerPtr _caller;
    bool _earlyTerminationEnabled;
    bool _retryEnabled;
    bool _singleRetryEnable;
    bool _canRetry;
    bool _disableDetection : 1;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ChildNodeCaller);

} // namespace multi_call
