#pragma once

#include "navi/engine/ScheduleInfo.h"
#include "navi/perf/NaviPerfResult.h"
#include "navi/memleak/MemLeakCollector.h"
#include "navi/util/CommonUtil.h"
#include <autil/DataBuffer.h>
#include <list>
#include <sys/resource.h>
#include <sys/time.h>

namespace navi {

class KernelMetricDef;
class KernelComputeEventDef;

struct KernelComputeEvent {
public:
    void fillProto(KernelComputeEventDef *event, StackTable &stackTable) const;

public:
    int32_t type;
    NaviPartId partId;
    int64_t scheduleId;
    ScheduleInfo scheduleInfo;
    int64_t beginTime;
    int64_t endTime;
    int64_t utimeInUs;
    int64_t stimeInUs;
    int64_t nvcsw;
    int64_t nivcsw;
    int64_t minflt;
    int64_t majflt;
    NaviPerfResultPtr perfResult;
    ThreadMemLeakResultPtr threadMemLeakResult;
};

class KernelMetric;
NAVI_TYPEDEF_PTR(KernelMetric);

class KernelMetric
{
public:
    KernelMetric(GraphId graphId,
                 const std::string &node,
                 const std::string &kernel);
    KernelMetric();
    ~KernelMetric();
private:
    KernelMetric(const KernelMetric &);
    KernelMetric &operator=(const KernelMetric &);
public:
    const std::string &getNode() const {
        return _node;
    }
    void setGraphId(GraphId graphId) {
        _graphId = graphId;
    }
    void setKernel(const std::string &kernel) {
        autil::ScopedLock lock(_mutex);
        _kernel = kernel;
    }
    void setIsGraphInit(bool value) {
        _isGraphInit = value;
    }
    bool isGraphInit() const {
        return _isGraphInit;
    }
    int64_t scheduleCount() const;
    int64_t tryScheduleCount() const;
    int64_t queueLatencyUs() const;
    int64_t totalLatencyUs() const;
    inline int64_t getMicroSecond(const timeval &val) const {
        return val.tv_sec * 1000000 + val.tv_usec;
    }
    void setKernelStates(KernelStatesDef *states);
    std::string toString() const;
    size_t eventCount() const;
private:
    void incScheduleCount();
    void incTryScheduleCount();
    void reportTime(ErrorCode ec, int32_t type, NaviPartId partId,
                    int64_t beginTime, int64_t endTime,
                    const rusage &usageBegin, const rusage &usageEnd,
                    const ScheduleInfo &schedInfo,
                    const NaviPerfResultPtr &perfResult,
                    const ThreadMemLeakResultPtr &threadMemLeakResult);
    void reportTime(ErrorCode ec, int64_t beginTime, int64_t endTime,
                    const ScheduleInfo &schedInfo);
    void doReportTime(ErrorCode ec, int64_t beginTime, int64_t endTime,
                      const ScheduleInfo &schedInfo)
    {
        if (EC_NONE == _ec) {
            _ec = ec;
        }
        _queueLatencyNs += beginTime - schedInfo.enqueueTime;
        _computeLatencyNs += endTime - beginTime;
    }
    void setFinish() {
        _finishTime = CommonUtil::getTimelineTimeNs();
    }
    void fillProto(KernelMetricDef *metric, StackTable &stackTable) const;

private:
    friend class Node;
    friend class ComputeScope;
    friend class GraphMetric;
    friend class NaviResult;
    friend class StreamMetric;
    friend class NaviWorkerBase;
private:
    GraphId _graphId;
    std::string _node;
    std::string _kernel;
    mutable bool _collected = false;
    bool _isGraphInit = false;

    ErrorCode _ec; // need_restore
    int64_t _scheduleCount;
    int64_t _tryScheduleCount;
    int64_t _queueLatencyNs;
    int64_t _computeLatencyNs;
    int64_t _finishTime = 0;
    mutable autil::ThreadMutex _mutex;
    std::shared_ptr<std::list<KernelComputeEvent>> _eventList; // need_restore
    KernelStatesDef *_states = nullptr;
};

}
