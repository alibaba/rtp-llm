#pragma once

#include "navi/engine/CreatorStats.h"
#include "navi/engine/RunGraphParams.h"
#include "navi/engine/KernelMetric.h"
#include "navi/engine/ParamTopoTreeNode.h"
#include "navi/perf/NaviPerf.h"
#include "navi/perf/NaviPerfThread.h"
#include "navi/memleak/MemLeakCollector.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/util/CommonUtil.h"

namespace navi {

thread_local extern size_t current_thread_id;
thread_local extern CollectFlags current_collect_flags;

class PerfEnableScope {
public:
    PerfEnableScope(NaviPerf *perf, const CollectFlags &flags)
        : _collectFlags(flags)
    {
        _saveFlags = current_collect_flags;
        if (unlikely(!perf)) {
            current_collect_flags = _collectFlags;
            return;
        }
        auto thisFlag = _collectFlags.getPerfFlags();
        auto saveFlag = _saveFlags.getPerfFlags();
        if ((thisFlag != 0) || (thisFlag != saveFlag)) {
            _perfThread = perf->getNaviPerfThread(current_thread_id);
        }
        auto realFlags = _collectFlags;
        if ((thisFlag != saveFlag) && _perfThread) {
            auto ret = _perfThread->setCollectState(_collectFlags);
            realFlags.setPerfFlags(ret.getPerfFlags());
            _needRecover = true;
        }
        current_collect_flags = realFlags;
    }
    ~PerfEnableScope() {
        current_collect_flags = _saveFlags;
        if (_needRecover && _perfThread) {
            _perfThread->setCollectState(_saveFlags);
        }
    }
    NaviPerfThread *getPerfThread() const {
        return _perfThread;
    }
private:
    NaviPerfThread *_perfThread = nullptr;
    CollectFlags _collectFlags;
    CollectFlags _saveFlags;
    bool _needRecover = false;
};

class ComputeScope {
public:
    ComputeScope(KernelMetric *metric,
                 CreatorStats *creatorStats,
                 NaviPartId partId,
                 NaviPerfThread *perfThread,
                 MemLeakCollector *memLeakCollector,
                 const ScheduleInfo &schedInfo,
                 const CollectFlags &collectFlags,
                 const MallocTraceParam *mallocTraceParam)
        : _metric(metric)
        , _type(GET_KERNEL_UNKNOWN)
        , _partId(partId)
        , _perfThread(perfThread)
        , _schedInfo(schedInfo)
        , _beginTime(CommonUtil::getTimelineTimeNs())
        , _endTime(0)
        , _ec(EC_NONE)
        , _collectFlags(collectFlags)
        , _creatorStats(creatorStats)
    {
        if (_collectFlags.needCollect()) {
            getrusage(RUSAGE_THREAD, &_usageBegin);
        }
        if (_collectFlags.needPerfCollect() && _perfThread) {
            _perfThread->beginSample();
        }
        if (_metric) {
            _metric->incScheduleCount();
        }
        if (unlikely(nullptr != memLeakCollector)) {
            int64_t mallocTraceId = -1;
            if (_metric && mallocTraceParam) {
                mallocTraceId = mallocTraceParam->getMallocId(
                    _metric->getNode(), _metric->scheduleCount());
            }
            _threadMemLeakCollector =
                memLeakCollector->getThreadCollector(mallocTraceId);
        }
    }
    void setType(GraphEventType type) {
        _type = type;
    }
    void setErrorCode(ErrorCode ec) {
        _ec = ec;
    }
    void setFakeEvent(bool fake) {
        _fakeEvent = fake;
    }
    const ThreadMemLeakCollectorPtr &getThreadMemLeakCollector() const {
        return _threadMemLeakCollector;
    }
    void stop() {
        if (_endTime == 0) {
            if (_collectFlags.needCollect()) {
                getrusage(RUSAGE_THREAD, &_usageEnd);
            }
            if (_collectFlags.needPerfCollect() && _perfThread) {
                _result = _perfThread->endSample();
            }
            _endTime = CommonUtil::getTimelineTimeNs();
            if (_creatorStats && EC_NONE == _ec) {
                auto latency = (_endTime - _beginTime) / 1000.0f;
                if (GET_KERNEL_CREATE == _type) {
                    _creatorStats->updateCreateLatency(latency);
                } else if (GET_KERNEL_INIT == _type) {
                    _creatorStats->updateInitLatency(latency);
                }
            }
        }
        if (_threadMemLeakCollector) {
            _threadMemLeakCollector->stop();
        }
    }
    void report() {
        stop();
        if (!_fakeEvent && _metric) {
            if ((GET_KERNEL_UNKNOWN != _type) && _collectFlags.needCollect()) {
                _metric->reportTime(_ec,
                                    (int32_t)_type,
                                    _partId,
                                    _beginTime,
                                    _endTime,
                                    _usageBegin,
                                    _usageEnd,
                                    _schedInfo,
                                    _result,
                                    _threadMemLeakCollector ? _threadMemLeakCollector->getResult() : nullptr);
            } else {
                _metric->reportTime(_ec, _beginTime, _endTime, _schedInfo);
            }
            _threadMemLeakCollector.reset();
            _result.reset();
            _metric = nullptr;
        }
    }
public:
    ~ComputeScope() {
        report();
    }
private:
    KernelMetric *_metric;
    GraphEventType _type;
    NaviPartId _partId;
    NaviPerfThread *_perfThread;
    ThreadMemLeakCollectorPtr _threadMemLeakCollector;
    ScheduleInfo _schedInfo;
    int64_t _beginTime;
    int64_t _endTime;
    ErrorCode _ec;
    CollectFlags _collectFlags;
    struct rusage _usageBegin;
    struct rusage _usageEnd;
    NaviPerfResultPtr _result;
    CreatorStats *_creatorStats;
    bool _fakeEvent = false;
};

NAVI_TYPEDEF_PTR(ComputeScope);

}

