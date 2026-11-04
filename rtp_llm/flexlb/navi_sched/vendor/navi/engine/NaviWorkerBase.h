#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/common/LockFreeQueue.h"
#include "autil/ObjectTracer.h"
#include "autil/TimeUtility.h"
#include "navi/common.h"
#include "navi/engine/EvTimer.h"
#include "navi/engine/GraphParam.h"
#include "navi/engine/NaviMetricsGroup.h"
#include "navi/engine/NaviThreadPool.h"
#include "navi/engine/NaviUserResult.h"
#include "navi/engine/TaskQueue.h"
#include "navi/log/NaviLogger.h"
#include "navi/perf/NaviPerfThread.h"
#include "navi/memleak/MemLeakCollector.h"
#include "navi/util/CommonUtil.h"

namespace multi_call {
class QueryInfo;
} // namespace multi_call

namespace navi {

class NaviWorkerBase;
class RunGraphParams;
class NaviClosure;
class TaskQueue;
class NaviPerf;
class NaviHostInfo;

class NaviWorkerItem : public NaviThreadPoolItemBase
{
public:
    NaviWorkerItem(NaviWorkerBase *worker, const CollectFlags &collectFlags)
        : _worker(worker)
        , _collectFlags(collectFlags)
    {
        _syncMode = false;
        _delayDestruct = false;
    }
    virtual ~NaviWorkerItem() {
    }
public:
    virtual void doProcess(NaviPerfThread *perfThread) = 0;
public:
    void process() override;
    void destroy() override;
    bool syncMode() const override final;
    virtual void drop();
public:
    void setSyncMode(bool syncMode);
    void setDelayDestruct(bool delay) {
        _delayDestruct = delay;
    }
    bool delayDestruct() const {
        return _delayDestruct;
    }
protected:
    NaviWorkerBase *_worker = nullptr;
    CollectFlags _collectFlags;
    bool _syncMode : 1;
    bool _delayDestruct: 1;
};

class NaviNoDropWorkerItem : public NaviWorkerItem
{
public:
    NaviNoDropWorkerItem(NaviWorkerBase *worker, const CollectFlags &collectFlags)
        : NaviWorkerItem(worker, collectFlags)
        , _done(false)
    {
    }
    virtual ~NaviNoDropWorkerItem() {
    }
public:
    void process() override final;
    void destroy() override final;
    void drop() override final;
    void doWork();
private:
    bool _done;
};

class BizManager;

class NaviWorkerBase : public autil::ObjectTracer<NaviWorkerBase, true> {
private:
    class WorkerInitItem : public NaviWorkerItem {
      public:
        WorkerInitItem(NaviWorkerBase *worker, const CollectFlags &collectFlags)
            : NaviWorkerItem(worker, collectFlags)
        {
        }
      public:
        void doProcess(NaviPerfThread *perfThread) override;
    };
public:
    NaviWorkerBase(TaskQueue *taskQueue,
                   BizManager *bizManager,
                   NaviUserResultClosure *closure = nullptr);
    virtual ~NaviWorkerBase();
public:
    bool init(const RunGraphParams &params,
              const std::shared_ptr<multi_call::QueryInfo> &queryInfo,
              const std::shared_ptr<kmonitor::MetricsReporter> &metricsReporter);
    void initSchedule();
    bool schedule(NaviWorkerItem *item, bool force = false);
    void dec();
    const NaviObjectLogger &getLogger() const {
        return _logger;
    }
    BizManager *getBizManager() const {
        return _bizManager;
    }
    void setFinish();
    const NaviUserResultPtr &getUserResult() const;
    void notifyFinish();
    void setNewSessionLatency(int64_t newSessionLatency);
    std::shared_ptr<multi_call::QuerySession> tryCreateQuerySession(
            const std::shared_ptr<multi_call::QuerySession> &originQuerySession,
            const std::shared_ptr<multi_call::QueryInfo> &queryInfo);
    void initMetricsReporter(kmonitor::MetricsReporter &metricsReporter,
                             const std::shared_ptr<QuerySessionR> &querySessionR);
    void setDefaultThreadPool(const NaviThreadPoolPtr &defaultThreadPool);
    void setNaviPerf(NaviPerf *naviPerf);
    NaviPerf *getNaviPerf() const {
        return _naviPerf;
    }
    const MemLeakCollectorPtr &getMemLeakCollector() const {
        return _memLeakCollector;
    }
    void setMemLeakCollector(const MemLeakCollectorPtr &memLeakCollector) {
        _memLeakCollector = memLeakCollector;
    }
    void setNaviSymbolTable(const NaviSymbolTablePtr &naviSymbolTable);
    void setEvTimer(EvTimer *evTimer);
    EvTimer *getEvTimer() const;
    void setHostInfo(const std::shared_ptr<NaviHostInfo> &hostInfo) {
        _hostInfo = hostInfo;
        _userResult->setWorkerInfo(hostInfo, this);
    }
    void setDestructClosure(NaviUserResultClosure *closure) {
        _destructClosure = closure;
    }
    NaviUserResultClosure *stealDestructClosure() {
        auto temp = _destructClosure;
        _destructClosure = nullptr;
        return temp;
    }
    const NaviSymbolTablePtr &getNaviSymbolTable() const;
    void clear();
    void deleteThis();
    void incItemCount();
    void decItemCount();
    int64_t itemCount();
    bool canInline() const;
    int64_t getIdleThreadCount() const;
    int64_t getRunningThreadCount() const;
    size_t getQueueSize() const;
    void inlineBegin() const;
    void inlineEnd() const;
    void finalize();
    ScheduleInfoPtr makeInlineSchedInfo();
    void setTaskQueueHolder(TaskQueueHolder token);
    int64_t getSessionBeginTime() const {
        return _metricsCollector.sessionBeginTime;
    }
    int64_t getInitScheduleTime() const {
        return _metricsCollector.initScheduleTime;
    }
    int64_t getGlobalStartTimeUs() const {
        return _globalStartTimeUs;
    }
    void updateThreadLimit(uint32_t threadLimit) {
        _threadLimit = threadLimit;
    }
    const std::string &getPeerIp() const {
        return _peerIp;
    }
public:
    void setTestMode(TestMode testMode) {
        _testMode = testMode;
    }
    TestMode getTestMode() const {
        return _testMode;
    }
protected:
public:
    NaviThreadPool *getThreadPool() const {
        return _threadPool;
    }
    const NaviThreadPoolPtr &getDefaultThreadPool() const;
public:
    virtual bool doInit(GraphParam *param) = 0;
    virtual void run(const std::shared_ptr<ComputeScope> &scope) = 0;
    virtual void drop() = 0;
public:
    bool isFinished() const;
private:
    void initCounters();
    void doSchedule();
    bool isStopped() const;
    virtual bool isTimeout() const = 0;
    bool reportRpcLackByBiz(const multi_call::GigStreamRpcInfoMap &rpcInfoMap);
    void postFillResultMetrics();
    friend class NaviThreadPool;
    friend class NaviWorkerItem;
    friend class KernelWorkItem;
    friend class NaviNoDropWorkerItem;
    friend class WorkerInitItem;
    friend class WorkerDestructItem;
protected:
    DECLARE_LOGGER();
    int64_t _beginTimeNs = 0;
    RunGraphMetricsCollector _metricsCollector;
private:
    BizManager *_bizManager;
    volatile bool _finish;
    NaviUserResultPtr _userResult;
    NaviSymbolTablePtr _naviSymbolTable;
    TestMode _testMode;
    bool _initSuccess;
    NaviThreadPool *_threadPool;
    TaskQueue *_taskQueue;
    NaviThreadPoolPtr _defaultThreadPool;
    NaviPerf *_naviPerf;
    EvTimer *_evTimer;
    std::shared_ptr<NaviHostInfo> _hostInfo;
    uint32_t _threadLimit;
    int32_t _maxInline;
    int64_t _globalStartTimeUs;
    atomic64_t _scheduleId;
    arpc::common::LockFreeQueue<NaviWorkerItem *> _scheduleQueue;
    atomic64_t _itemCount;
    atomic64_t _processingCount;
    // reuse kernel metric to collect init metric
    KernelMetricPtr _metric;
    TaskQueueHolder _taskQueueHolder;
    MemLeakCollectorPtr _memLeakCollector;
    NaviUserResultClosure *_destructClosure = nullptr;
    kmonitor::MetricsReporterPtr _metricsReporter;
protected:
    CollectFlags _collectFlags;
    std::shared_ptr<TopoNodeParam> _rootTopoParam;
    // for gdb
    std::string _peerIp;
};

}
