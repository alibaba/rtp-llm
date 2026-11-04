#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/WorkerMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CallDelegationWorkItem.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceManager.h"
#include "autil/Lock.h"
#include "autil/LoopThread.h"

namespace multi_call {

class CallDelegationThread
{
public:
    CallDelegationThread(const SearchServiceManagerPtr &searchServiceManagerPtr,
                         const WorkerMetricReporterPtr &metricReporter,
                         int64_t processInterval = PROCESS_INTERVAL,
                         uint64_t queueSize = std::numeric_limits<uint64_t>::max());
    ~CallDelegationThread();

public:
    typedef std::deque<CallDelegationWorkItem *> CallDelegationQueue;

private:
    CallDelegationThread(const CallDelegationThread &);
    CallDelegationThread &operator=(const CallDelegationThread &);

public:
    bool start();
    void stop();
    bool isStarted() const {
        return _thread != NULL;
    }
    void pushWorkItem(CallDelegationWorkItem *workItem);
    int64_t getProcessInterval() const {
        return _processInterval;
    }

private:
    void workLoop();
    void mergeQueue();
    bool isAllQueueEmpty() const;

private:
    CallDelegationQueue _queue;
    CallDelegationQueue _pushQueue;
    autil::LoopThreadPtr _thread;
    mutable autil::ThreadMutex _mutex;
    SearchServiceManagerPtr _searchServiceManager;
    WorkerMetricReporterPtr _metricReporter;
    int64_t _processInterval;
    uint64_t _queueMaxSize;
    uint64_t _dropRequestCount;
    uint64_t _totalRequestCount;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(CallDelegationThread);
} // namespace multi_call

