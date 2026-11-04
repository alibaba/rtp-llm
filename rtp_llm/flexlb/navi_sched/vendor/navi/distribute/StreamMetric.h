#pragma once

#include <multi_call/stream/GigStreamBase.h>
#include "navi/engine/KernelMetric.h"
#include "navi/engine/ComputeScope.h"

namespace navi {

class NaviPerf;
class PerfEnableScope;
class GraphMetric;

struct StreamSchedInfo {
public:
    StreamSchedInfo(int64_t enqueueTime);
public:
    ScheduleInfo schedInfo;
};

class StreamMetricScope {
public:
    StreamMetricScope(NaviPerf *naviPerf,
                      MemLeakCollector *memLeakCollector,
                      KernelMetric *metric,
                      NaviPartId partId,
                      const CollectFlags &collectFlags,
                      const MallocTraceParam *mallocParam,
                      GraphEventType type,
                      int64_t enqueueTime = -1);
    ~StreamMetricScope();
public:
    void setFakeEvent(bool fake) {
        _scope.setFakeEvent(fake);
    }
private:
    std::shared_ptr<PerfEnableScope> _perfEnableScope;
    StreamSchedInfo _schedInfo;
    ComputeScope _scope;
};

class StreamMetricScopeWithSchedInfo {
public:
    StreamMetricScopeWithSchedInfo(NaviPerf *naviPerf,
                                   MemLeakCollector *memLeakCollector,
                                   KernelMetric *metric,
                                   NaviPartId partId,
                                   const CollectFlags &collectFlags,
                                   const MallocTraceParam *mallocParam,
                                   GraphEventType type,
                                   const ScheduleInfo &schedInfo);
    ~StreamMetricScopeWithSchedInfo();
    void report() {
        _scope.report();
    }
private:
    std::shared_ptr<PerfEnableScope> _perfEnableScope;
    ComputeScope _scope;
    StreamMetricScopeWithSchedInfo *_prevScope = nullptr;
};

struct StreamPartMetric {
public:
    StreamPartMetric(NaviPartId partId, bool isClient);
public:
    KernelMetricPtr taskQueueMetric;
    KernelMetricPtr receiveQueueMetric;
    KernelMetricPtr sendMetric;
    KernelMetricPtr receiveMetric;
};

class StreamMetric
{
public:
    StreamMetric(NaviPartId partCount,
                 NaviPerf *naviPerf,
                 MemLeakCollectorPtr memLeakCollector,
                 bool isClient,
                 CollectFlags collectFlags,
                 const MallocTraceParam *mallocParam);
    ~StreamMetric();
    StreamMetric(const StreamMetric &) = delete;
    StreamMetric &operator=(const StreamMetric &) = delete;
public:
    void *beginTaskQueue(multi_call::PartIdTy partId, const ScheduleInfo &schedInfo);
    void *beginReceiveQueue(multi_call::PartIdTy partId, const ScheduleInfo &schedInfo);
    void endQueueHandle(void *handle);
    void *beginReceive(multi_call::PartIdTy partId);
    void *beginSend(multi_call::PartIdTy partId);
    void endHandle(void *handle, bool fake);
    void collectMetric(const std::string &inputBizName, GraphId graphId, GraphMetric *graphMetric);
    const MemLeakCollectorPtr &getMemLeakCollector() const {
        return _memLeakCollector;
    }
private:
    void collectOneMetric(const KernelMetricPtr &metric,
                          const std::string &bizName,
                          GraphId graphId,
                          GraphMetric *graphMetric);

private:
    NaviPartId _partCount = 0;
    NaviPerf *_naviPerf = nullptr;
    MemLeakCollectorPtr _memLeakCollector;
    CollectFlags _collectFlags;
    MallocTraceParam _mallocParam;
    std::vector<StreamPartMetric> _partMetrics;
};

NAVI_TYPEDEF_PTR(StreamMetric);

}

