#include "navi/distribute/StreamMetric.h"
#include "navi/engine/GraphMetric.h"
#include "navi/engine/NaviThreadPool.h"

namespace navi {

thread_local extern size_t current_thread_counter;
thread_local extern size_t current_thread_wait_counter;
thread_local StreamMetricScopeWithSchedInfo *current_compute_scope = nullptr;

StreamSchedInfo::StreamSchedInfo(int64_t enqueueTime) {
    auto time = CommonUtil::getTimelineTimeNs();
    if (unlikely(0 == current_thread_id)) {
        current_thread_id = (long)syscall(SYS_gettid);
    }
    auto threadId = current_thread_id;
    schedInfo.enqueueTime = enqueueTime > 0 ? enqueueTime : time;
    schedInfo.dequeueTime = time;
    schedInfo.schedTid = threadId;
    schedInfo.signalTid = threadId;
    schedInfo.processTid = threadId;
    schedInfo.runningThread = 0;
    schedInfo.threadCounter = current_thread_counter;
    schedInfo.threadWaitCounter = current_thread_wait_counter;
    schedInfo.queueSize = 0;
}

StreamMetricScope::StreamMetricScope(NaviPerf *naviPerf,
                                     MemLeakCollector *memLeakCollector,
                                     KernelMetric *metric,
                                     NaviPartId partId,
                                     const CollectFlags &collectFlags,
                                     const MallocTraceParam *mallocParam,
                                     GraphEventType type,
                                     int64_t enqueueTime)
    : _perfEnableScope(
          collectFlags.needCollect()
              ? std::make_shared<PerfEnableScope>(naviPerf, collectFlags)
              : nullptr)
    , _schedInfo(enqueueTime)
    , _scope(metric,
             nullptr,
             partId,
             _perfEnableScope ? _perfEnableScope->getPerfThread() : nullptr,
             memLeakCollector,
             _schedInfo.schedInfo,
             collectFlags,
             mallocParam)
{
    _scope.setType(type);
}

StreamMetricScope::~StreamMetricScope() {
}

StreamMetricScopeWithSchedInfo::StreamMetricScopeWithSchedInfo(NaviPerf *naviPerf,
                                                               MemLeakCollector *memLeakCollector,
                                                               KernelMetric *metric,
                                                               NaviPartId partId,
                                                               const CollectFlags &collectFlags,
                                                               const MallocTraceParam *mallocParam,
                                                               GraphEventType type,
                                                               const ScheduleInfo &schedInfo)
    : _perfEnableScope(collectFlags.needCollect() ? std::make_shared<PerfEnableScope>(naviPerf, collectFlags) : nullptr)
    , _scope(metric,
             nullptr,
             partId,
             _perfEnableScope ? _perfEnableScope->getPerfThread() : nullptr,
             memLeakCollector,
             schedInfo,
             collectFlags,
             mallocParam)
{
    _scope.setType(type);
    _prevScope = current_compute_scope;
    current_compute_scope = this;
}

StreamMetricScopeWithSchedInfo::~StreamMetricScopeWithSchedInfo() {
    current_compute_scope = _prevScope;
}

StreamPartMetric::StreamPartMetric(NaviPartId partId, bool isClient) {
    const auto &partIdStr = std::to_string(partId);
    taskQueueMetric = std::make_shared<KernelMetric>(INVALID_GRAPH_ID,
                                                     isClient ? "navi.client.taskQueue_" + partIdStr
                                                              : "navi.server.taskQueue_" + partIdStr,
                                                     "");
    receiveQueueMetric = std::make_shared<KernelMetric>(INVALID_GRAPH_ID,
                                                        isClient ? "navi.client.receiveQueue_" + partIdStr
                                                                 : "navi.server.receiveQueue_" + partIdStr,
                                                        "");
    sendMetric = std::make_shared<KernelMetric>(
        INVALID_GRAPH_ID, isClient ? "navi.client.send_" + partIdStr : "navi.server.send_" + partIdStr, "");
    receiveMetric = std::make_shared<KernelMetric>(INVALID_GRAPH_ID,
                                                   isClient ? "navi.client.receive_" + partIdStr
                                                            : "navi.server.receive_" + partIdStr,
                                                   "");
}

StreamMetric::StreamMetric(
    NaviPartId partCount, NaviPerf *naviPerf,
    MemLeakCollectorPtr memLeakCollector,
    bool isClient, CollectFlags collectFlags,
    const MallocTraceParam *mallocParam)
    : _partCount(partCount)
    , _naviPerf(naviPerf)
    , _memLeakCollector(memLeakCollector)
    , _collectFlags(collectFlags)
{
    if (mallocParam) {
        _mallocParam = *mallocParam;
    }
    for (NaviPartId partId = 0; partId < _partCount; partId++) {
        _partMetrics.emplace_back(StreamPartMetric(partId, isClient));
    }
}

StreamMetric::~StreamMetric() {
}

void *StreamMetric::beginTaskQueue(multi_call::PartIdTy partId, const ScheduleInfo &schedInfo) {
    if (partId >= _partCount) {
        return nullptr;
    }
    return new StreamMetricScopeWithSchedInfo(
        _naviPerf,
        _memLeakCollector.get(),
        _partMetrics[partId].taskQueueMetric.get(),
        partId,
        _collectFlags,
        &_mallocParam,
        GET_RPC_TASK_QUEUE,
        schedInfo);
}

void *StreamMetric::beginReceiveQueue(multi_call::PartIdTy partId, const ScheduleInfo &schedInfo) {
    if (partId >= _partCount) {
        return nullptr;
    }
    return new StreamMetricScopeWithSchedInfo(
        _naviPerf,
        _memLeakCollector.get(),
        _partMetrics[partId].receiveQueueMetric.get(),
        partId,
        _collectFlags,
        &_mallocParam,
        GET_RPC_RECEIVE_QUEUE,
        schedInfo);
}

void *StreamMetric::beginReceive(multi_call::PartIdTy partId) {
    if (partId >= _partCount) {
        return nullptr;
    }
    return new StreamMetricScope(_naviPerf,
                                 _memLeakCollector.get(),
                                 _partMetrics[partId].receiveMetric.get(),
                                 partId,
                                 _collectFlags,
                                 &_mallocParam,
                                 GET_RPC_RECEIVE);
}

void *StreamMetric::beginSend(multi_call::PartIdTy partId) {
    if (partId >= _partCount) {
        return nullptr;
    }
    return new StreamMetricScope(_naviPerf,
                                 _memLeakCollector.get(),
                                 _partMetrics[partId].sendMetric.get(),
                                 partId,
                                 _collectFlags,
                                 &_mallocParam,
                                 GET_RPC_SEND);
}

void StreamMetric::endQueueHandle(void *handle) {
    auto scope = (StreamMetricScopeWithSchedInfo *)handle;
    delete scope;
}

void StreamMetric::endHandle(void *handle, bool fake) {
    auto scope = (StreamMetricScope *)handle;
    if (fake) {
        scope->setFakeEvent(fake);
    }
    delete scope;
}

void StreamMetric::collectMetric(const std::string &inputBizName, GraphId graphId, GraphMetric *graphMetric) {
    if (current_compute_scope) {
        current_compute_scope->report();
    }
    std::string bizName;
    {

        bizName = std::string(inputBizName.c_str(), inputBizName.size());
    }
    for (const auto &partMetric : _partMetrics) {
        collectOneMetric(partMetric.taskQueueMetric, bizName, graphId, graphMetric);
        collectOneMetric(partMetric.receiveQueueMetric, bizName, graphId, graphMetric);
        collectOneMetric(partMetric.sendMetric, bizName, graphId, graphMetric);
        collectOneMetric(partMetric.receiveMetric, bizName, graphId, graphMetric);
    }
}

void StreamMetric::collectOneMetric(const KernelMetricPtr &metric,
                                    const std::string &bizName,
                                    GraphId graphId,
                                    GraphMetric *graphMetric)
{
    if (0 == metric->eventCount()) {
        return;
    }
    metric->setFinish();
    metric->setGraphId(graphId);
    metric->setKernel(bizName);
    graphMetric->addKernelMetricPtr(metric);
}

}
