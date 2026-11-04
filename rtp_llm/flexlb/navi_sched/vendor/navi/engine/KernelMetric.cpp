#include "navi/engine/KernelMetric.h"
#include "autil/StringUtil.h"
#include "navi/proto/GraphVis.pb.h"

namespace navi {

void KernelComputeEvent::fillProto(KernelComputeEventDef *event, StackTable &stackTable) const {
    event->set_type((GraphEventType)type);
    event->set_part_id(partId);
    event->set_schedule_id(scheduleId);
    scheduleInfo.fillProto(event->mutable_schedinfo());
    event->set_begin_time_us(beginTime);
    event->set_end_time_us(endTime);
    event->set_utime_in_us(utimeInUs);
    event->set_stime_in_us(stimeInUs);
    event->set_nvcsw(nvcsw);
    event->set_nivcsw(nivcsw);
    event->set_minflt(minflt);
    event->set_majflt(majflt);
    if (perfResult) {
        perfResult->fillProto(event->mutable_perf(), stackTable);
    }
    if (threadMemLeakResult) {
        threadMemLeakResult->fillProto(event->mutable_malloc(), stackTable);
    }
}

KernelMetric::KernelMetric(GraphId graphId,
                           const std::string &node,
                           const std::string &kernel)
    : _graphId(graphId)
    , _node(node)
    , _kernel(kernel)
    , _scheduleCount(0)
    , _tryScheduleCount(0)
    , _queueLatencyNs(0)
    , _computeLatencyNs(0)
{
    _ec = EC_NONE;
}

size_t KernelMetric::eventCount() const {
    autil::ScopedLock lock(_mutex);
    return _eventList ? _eventList->size() : 0;
}

KernelMetric::~KernelMetric() {
    DELETE_AND_SET_NULL(_states);

    _eventList.reset();
}

void KernelMetric::incScheduleCount() {
    _scheduleCount++;
}

void KernelMetric::incTryScheduleCount() {
    _tryScheduleCount++;
}

int64_t KernelMetric::scheduleCount() const {
    return _scheduleCount;
}

int64_t KernelMetric::tryScheduleCount() const {
    return _tryScheduleCount;
}

void KernelMetric::reportTime(ErrorCode ec,
                              int32_t type,
                              NaviPartId partId,
                              int64_t beginTime,
                              int64_t endTime,
                              const rusage &usageBegin,
                              const rusage &usageEnd,
                              const ScheduleInfo &schedInfo,
                              const NaviPerfResultPtr &perfResult,
                              const ThreadMemLeakResultPtr &threadMemLeakResult)
{
    if (_collected) {
        return;
    }
    KernelComputeEvent event;
    event.type = type;
    event.partId = partId;
    event.scheduleId = scheduleCount();
    event.scheduleInfo = schedInfo;
    event.beginTime = beginTime;
    event.endTime = endTime;
    event.utimeInUs =
        getMicroSecond(usageEnd.ru_utime) - getMicroSecond(usageBegin.ru_utime);
    event.stimeInUs =
        getMicroSecond(usageEnd.ru_stime) - getMicroSecond(usageBegin.ru_stime);
    event.nvcsw = usageEnd.ru_nvcsw - usageBegin.ru_nvcsw;
    event.nivcsw = usageEnd.ru_nivcsw - usageBegin.ru_nivcsw;
    event.minflt = usageEnd.ru_minflt - usageBegin.ru_minflt;
    event.majflt = usageEnd.ru_majflt - usageBegin.ru_majflt;
    event.perfResult = perfResult;
    event.threadMemLeakResult = threadMemLeakResult;
    autil::ScopedLock lock(_mutex);
    doReportTime(ec, beginTime, endTime, schedInfo);

    if (!_eventList) {
        _eventList = std::make_shared<std::list<KernelComputeEvent>>();
        _eventList->emplace_back(event);
    } else {
        _eventList->emplace_back(event);
    }
}

void KernelMetric::reportTime(ErrorCode ec, int64_t beginTime, int64_t endTime, const ScheduleInfo &schedInfo)
{
    autil::ScopedLock lock(_mutex);
    doReportTime(ec, beginTime, endTime, schedInfo);
}

void KernelMetric::setKernelStates(KernelStatesDef *states) {
    autil::ScopedLock lock(_mutex);
    if (_states) {
        DELETE_AND_SET_NULL(_states);
    }
    _states = states;
}

int64_t KernelMetric::queueLatencyUs() const {
    return _queueLatencyNs / 1000;
}

int64_t KernelMetric::totalLatencyUs() const {
    return _computeLatencyNs / 1000;
}

void KernelMetric::fillProto(KernelMetricDef *metric, StackTable &stackTable) const {
    metric->set_graph_id(_graphId);
    metric->set_node(_node);
    metric->set_ec(_ec);
    metric->set_ec_str(CommonUtil::getErrorString(_ec));
    metric->set_schedule_count(_scheduleCount);
    metric->set_try_schedule_count(_tryScheduleCount);
    metric->set_queue_latency_us(queueLatencyUs());
    metric->set_compute_latency_us(totalLatencyUs());
    metric->set_finish_time(_finishTime);

    if (_states) {
        metric->mutable_states()->Swap(_states);
    }
    std::shared_ptr<std::list<KernelComputeEvent>> eventListPtr;
    {
        autil::ScopedLock lock(_mutex);
        eventListPtr = std::move(_eventList);
        metric->set_kernel(_kernel);
    }
    if (eventListPtr) {
        const auto &eventList = *eventListPtr;
        for (const auto &event : eventList) {
            event.fillProto(metric->add_events(), stackTable);
        }
    }
    _collected = true;
}

std::string KernelMetric::toString() const {
    autil::ScopedLock lock(_mutex);
    std::string ret;
    ret += "graphId: " + autil::StringUtil::toString(_graphId)
           + ", node: " + autil::StringUtil::toString(_node)
           + ", kernel: " + autil::StringUtil::toString(_kernel)
           + ", scheduleCount: " + autil::StringUtil::toString(_scheduleCount)
           + ", tryScheduleCount: " + autil::StringUtil::toString(_tryScheduleCount)
           + ", queueLatency: " + autil::StringUtil::toString(queueLatencyUs())
           + ", totalLatencyUs: " + autil::StringUtil::toString(totalLatencyUs());
    ret += "\ntimeline: \n";
    if (_eventList) {
        const auto &eventList = *_eventList;
        for (const auto &event : eventList) {
            ret += "enqueue: " +
                   autil::StringUtil::toString(event.scheduleInfo.enqueueTime) +
                   ", begin: " + autil::StringUtil::toString(event.beginTime) +
                   ", end: " + autil::StringUtil::toString(event.endTime);
            ret += "\n";
        }
    }
    return ret;
}

}
