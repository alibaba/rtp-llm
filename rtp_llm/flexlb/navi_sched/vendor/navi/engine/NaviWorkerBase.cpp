#include "navi/engine/NaviWorkerBase.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/QueryInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/QuerySession.h"
#include "navi/engine/BizManager.h"
#include "navi/engine/ComputeScope.h"
#include "navi/engine/TaskQueue.h"
#include "navi/log/TraceAppender.h"
#include "navi/resource/GigClientR.h"
#include "navi/resource/QuerySessionR.h"
#include "navi/util/CommonUtil.h"
#include "navi/util/NaviClosure.h"

namespace navi {

thread_local extern size_t current_thread_counter;
thread_local extern size_t current_thread_wait_counter;

void NaviWorkerItem::process() {
    if (!_worker->isStopped() || _collectFlags.collectMalloc()) {
        PerfEnableScope perfEnableScope(_worker->getNaviPerf(), _collectFlags);
        {
            NaviLoggerScope scope(_worker->getLogger());
            doProcess(perfEnableScope.getPerfThread());
        }
    } else if (!_worker->isFinished()) {
        _worker->drop();
    }
}

void NaviWorkerItem::destroy() {
    auto worker = _worker;
    delete this;
    worker->dec();
}

void NaviWorkerItem::drop() {
    delete this;
}

bool NaviWorkerItem::syncMode() const {
    return _syncMode;
}

void NaviWorkerItem::setSyncMode(bool syncMode) {
    _syncMode = syncMode;
}

void NaviNoDropWorkerItem::process() {
    doWork();
    if (_worker->isStopped() && !_worker->isFinished()) {
        _worker->drop();
    }
}

void NaviNoDropWorkerItem::destroy() {
    doWork();
    auto worker = _worker;
    delete this;
    worker->dec();
}

void NaviNoDropWorkerItem::drop() {
    setScheduleInfo(CommonUtil::getTimelineTimeNs(),
                    _worker->getRunningThreadCount(),
                    current_thread_id,
                    current_thread_counter,
                    current_thread_wait_counter,
                    ICT_DROP);
    doWork();
    delete this;
}

void NaviNoDropWorkerItem::doWork() {
    if (!_done) {
        PerfEnableScope perfEnableScope(_worker->getNaviPerf(), _collectFlags);
        {
            NaviLoggerScope scope(_worker->getLogger());
            doProcess(perfEnableScope.getPerfThread());
            _done = true;
        }
    }
}

class WorkerDestructItem : public NaviThreadPoolItemBase
{
public:
    WorkerDestructItem(NaviWorkerBase *worker)
        : _worker(worker)
    {
    }
public:
    void process() override {
        if (_worker) {
            auto metric = _worker->_metric;
            auto userResult = _worker->_userResult;
            CondWaiterPtr destructWaiter;
            if (userResult) {
                destructWaiter = userResult->getDestructWaiter();
            }
            auto destructClosure = _worker->stealDestructClosure();
            if (unlikely(nullptr != metric)) {
                auto collectFlags = _worker->_collectFlags;
                auto schedInfo = _worker->makeInlineSchedInfo();
                PerfEnableScope perfEnableScope(_worker->getNaviPerf(),
                                                collectFlags);
                auto computeScope = std::make_shared<ComputeScope>(
                    metric.get(),
                    nullptr,
                    0,
                    perfEnableScope.getPerfThread(),
                    _worker->_memLeakCollector.get(),
                    *schedInfo,
                    collectFlags,
                    _worker->_rootTopoParam
                        ? &_worker->_rootTopoParam->getMallocTraceParam()
                        : nullptr);
                computeScope->setType(GET_GRAPH_DESTRUCT);
                _worker->finalize();
                if (unlikely(nullptr != destructClosure)) {
                    destructClosure->run(std::move(userResult), computeScope);
                } else {
                    userResult.reset();
                }
            } else {
                _worker->finalize();
                if (unlikely(nullptr != destructClosure)) {
                    destructClosure->run(std::move(userResult), nullptr);
                }
            }
            _worker = nullptr;
            if (destructWaiter) {
                destructWaiter->notify();
            }
        }
    }
    void destroy() override {
        process();
        delete this;
    }
private:
    NaviWorkerBase *_worker;
};

void NaviWorkerBase::WorkerInitItem::doProcess(NaviPerfThread *perfThread) {
    ComputeScopePtr scope;
    if (_collectFlags.needCollect()) {
        scope = std::make_shared<ComputeScope>(
            _worker->_metric.get(),
            nullptr,
            0,
            perfThread,
            _worker->getMemLeakCollector().get(),
            _schedInfo,
            _collectFlags,
            _worker->_rootTopoParam
                ? &_worker->_rootTopoParam->getMallocTraceParam()
                : nullptr);
        scope->setType(GET_GRAPH_INIT);
    }
    _worker->run(scope);
}

NaviWorkerBase::NaviWorkerBase(
        TaskQueue *taskQueue,
        BizManager *bizManager,
        NaviUserResultClosure *closure)
    : _logger(this, "session")
    , _bizManager(bizManager)
    , _finish(false)
    , _testMode(TM_NONE)
    , _initSuccess(false)
    , _threadPool(taskQueue->getThreadPool())
    , _taskQueue(taskQueue)
    , _naviPerf(nullptr)
    , _evTimer(nullptr)
    , _threadLimit(DEFAULT_THREAD_LIMIT)
    , _maxInline(DEFAULT_MAX_INLINE)
    , _globalStartTimeUs(0)
    , _metricsReporter(nullptr)
{
    atomic_set(&_scheduleId, 1);
    _metricsCollector.sessionBeginTime = autil::TimeUtility::currentTime();
    _beginTimeNs = CommonUtil::getTimelineTimeNs();

    if (closure) {
        if (closure->waitDestruct) {
            setDestructClosure(closure);
            _userResult = std::make_shared<NaviAsyncUserResult>(nullptr);
        } else {
            _userResult = std::make_shared<NaviAsyncUserResult>(closure);
        }
    } else {
        _userResult = std::make_shared<NaviSyncUserResult>();
    }
    initCounters();
    NAVI_LOG(SCHEDULE1, "construct");
}

NaviWorkerBase::~NaviWorkerBase() {
    if (_metricsReporter) {
        _metricsReporter->report<RunGraphMetrics>(nullptr, &_metricsCollector);
    }
    assert(_scheduleQueue.Empty());
    NAVI_LOG(SCHEDULE1, "destruct");
    _logger.logger.reset();
    auto destructClosure = stealDestructClosure();
    if (unlikely(nullptr != destructClosure)) {
        destructClosure->run(nullptr, nullptr);
        destructClosure = nullptr;
    }
}

void NaviWorkerBase::setNewSessionLatency(int64_t newSessionLatency) {
    _metricsCollector.newSessionLatency = newSessionLatency;
}

void NaviWorkerBase::setDefaultThreadPool(const NaviThreadPoolPtr &defaultThreadPool) {
    _defaultThreadPool = defaultThreadPool;
}

void NaviWorkerBase::setNaviPerf(NaviPerf *naviPerf) {
    _naviPerf = naviPerf;
}

void NaviWorkerBase::setNaviSymbolTable(
    const NaviSymbolTablePtr &naviSymbolTable)
{
    _naviSymbolTable = naviSymbolTable;
}

const NaviSymbolTablePtr &NaviWorkerBase::getNaviSymbolTable() const {
    return _naviSymbolTable;
}

void NaviWorkerBase::setEvTimer(EvTimer *evTimer) {
    _evTimer = evTimer;
}

EvTimer *NaviWorkerBase::getEvTimer() const {
    return _evTimer;
}

ScheduleInfoPtr NaviWorkerBase::makeInlineSchedInfo() {
    auto schedInfo = ScheduleInfo::makeInlineSchedInfo();
    schedInfo->runningThread = getRunningThreadCount();
    schedInfo->queueSize = getQueueSize();
    return schedInfo;
}

void NaviWorkerBase::setTaskQueueHolder(TaskQueueHolder token) {
    _taskQueueHolder = std::move(token);
}

bool NaviWorkerBase::init(
    const RunGraphParams &runParams,
    const std::shared_ptr<multi_call::QueryInfo> &queryInfo,
    const std::shared_ptr<kmonitor::MetricsReporter> &metricsReporter)
{

    _collectFlags = runParams.getCollectFlags();
    _rootTopoParam = runParams.getTopoParam();
    PerfEnableScope perfEnableScope(_naviPerf, _collectFlags);
    ScheduleInfoPtr schedInfo;
    ComputeScopePtr computeScope;
    if (_collectFlags.collectMalloc()) {
        if (!_memLeakCollector) {

            _memLeakCollector = std::make_shared<MemLeakCollector>();
        }
        _userResult->initDestructWaiter();
    }
    if (_collectFlags.needCollect()) {
        {

            _metric = std::make_shared<KernelMetric>(INVALID_GRAPH_ID, "navi.graph_init", "");
        }
        _metric->setIsGraphInit(true);
        schedInfo = makeInlineSchedInfo();
        schedInfo->enqueueTime = _beginTimeNs;
        const auto &topoParam = runParams.getTopoParam();
        computeScope = std::make_shared<ComputeScope>(
            _metric.get(),
            nullptr,
            0,
            perfEnableScope.getPerfThread(),
            _memLeakCollector.get(),
            *schedInfo,
            _collectFlags,
            topoParam ? &topoParam->getMallocTraceParam() : nullptr);
        computeScope->setType(GET_GRAPH_INIT_PREPARE);
    }
    if (!_logger.logger->hasTraceAppender()) {
        const auto &traceLevel = runParams.getTraceLevel();
        std::shared_ptr<TraceAppender> traceAppender;
        {

            traceAppender = std::make_shared<TraceAppender>(traceLevel);
        }
        runParams.fillFilterParam(traceAppender.get());
        _logger.logger->setTraceAppender(traceAppender);
    }
    _threadLimit = runParams.getThreadLimit();
    _maxInline = runParams.getMaxInline();
    auto param = new GraphParam();
    auto sessionId = _logger.logger->getLoggerId();
    if (!param->init(sessionId, this, _logger.logger, runParams,
                     _naviSymbolTable, _hostInfo.get(), _memLeakCollector))
    {
        delete param;
        return false;
    }
    auto graphMetric = param->graphResult->getGraphMetric();
    graphMetric->updateBeginTime(getSessionBeginTime(), _beginTimeNs);
    if (_collectFlags.needCollect()) {
        graphMetric->addKernelMetricPtr(_metric);
    }
    const auto &querySession = tryCreateQuerySession(runParams.getQuerySession(), queryInfo);
    if (querySession) {
        param->querySessionR = std::make_shared<QuerySessionR>(querySession);
    }
    if (!runParams.isLoadGraph()) {
        if (metricsReporter) {
            initMetricsReporter(*metricsReporter, param->querySessionR);
        }
    }
    _userResult->setGraphResult(param->graphResult);
    _globalStartTimeUs = param->runParams.getGlobalStartTimeUs();
    return doInit(param);
}

std::shared_ptr<multi_call::QuerySession> NaviWorkerBase::tryCreateQuerySession(
        const std::shared_ptr<multi_call::QuerySession> &originQuerySession,
        const std::shared_ptr<multi_call::QueryInfo> &queryInfo)
{
    auto resource = _bizManager->getSnapshotResource(GIG_CLIENT_RESOURCE_ID);
    if (!resource) {
        NAVI_LOG(SCHEDULE1, "get gig client resource failed, skip");
        return std::make_shared<multi_call::QuerySession>(multi_call::SearchServicePtr(),
                std::shared_ptr<multi_call::QuerySessionContext>());
    }
    auto gigResource = dynamic_cast<GigClientR *>(resource);
    assert(gigResource != nullptr && "gig resource type error");
    auto searchService = gigResource->getSearchService();
    if (!searchService) {
        NAVI_LOG(SCHEDULE1, "get search service failed, skip");
        return std::make_shared<multi_call::QuerySession>(multi_call::SearchServicePtr(),
                std::shared_ptr<multi_call::QuerySessionContext>());
    }
    if (originQuerySession) {
        auto querySession = multi_call::QuerySession::createWithOrigin(searchService, originQuerySession);
            return querySession;
    } else {
        auto querySession = std::make_shared<multi_call::QuerySession>(searchService, queryInfo);
            return querySession;
    }
}

void NaviWorkerBase::initMetricsReporter(kmonitor::MetricsReporter &metricsReporter,
                                         const std::shared_ptr<QuerySessionR> &querySessionR)
{
    if (querySessionR) {
        _metricsReporter = metricsReporter.getSubReporter("kmon.sql", querySessionR->getTagsMap());
    } else {
        _metricsReporter = metricsReporter.getSubReporter("kmon.sql", {});
    }
}

void NaviWorkerBase::initCounters() {
    atomic_set(&_itemCount, 0);
    atomic_set(&_processingCount, 0);
}

void NaviWorkerBase::initSchedule() {
    NAVI_LOG(SCHEDULE1, "init schedule with sync mode");
    _metricsCollector.initScheduleTime = autil::TimeUtility::currentTime();

    if (_threadPool->incWorkerCount()) {
        _initSuccess = true;
        auto item = new WorkerInitItem(this, _collectFlags);
        item->setSyncMode(true);
        schedule(item);
    } else {
        drop();
    }
}

bool NaviWorkerBase::schedule(NaviWorkerItem *item, bool force) {
    if (!force && isStopped()) {
        incItemCount();
        drop();
        item->drop();
        clear();
        decItemCount();
        return false;
    }
    incItemCount();
    if (force || atomic_read(&_processingCount) < _threadLimit) {
        if (DEFAULT_THREAD_LIMIT != _threadLimit) {
            atomic_inc(&_processingCount);
        }
        _threadPool->push(_globalStartTimeUs,
                          atomic_inc_return(&_scheduleId),
                          item,
                          _collectFlags.collectMalloc());
        return true;
    }
    _scheduleQueue.Push(item);
    doSchedule();
    return true;
}

void NaviWorkerBase::doSchedule() {
    NaviWorkerItem *item = nullptr;
    if (atomic_read(&_processingCount) < _threadLimit &&
        _scheduleQueue.Pop(&item))
    {
        if (DEFAULT_THREAD_LIMIT != _threadLimit) {
            atomic_inc(&_processingCount);
        }
        auto priorityH = _globalStartTimeUs;
        if (item->delayDestruct()) {
            priorityH += DESTRUCT_DELAY_US;
        }
        _threadPool->push(priorityH,
                          atomic_inc_return(&_scheduleId),
                          item,
                          _collectFlags.collectMalloc());
    }
}

void NaviWorkerBase::clear() {
    NaviWorkerItem *item = nullptr;
    while (_scheduleQueue.Pop(&item)) {
        // will not destruct this if called from another NaviWorkerItem
        item->destroy();
    }
}

const NaviThreadPoolPtr &NaviWorkerBase::getDefaultThreadPool() const {
    return _defaultThreadPool;
}

void NaviWorkerBase::setFinish() {
    NAVI_LOG(SCHEDULE1, "finish");
    // _logger.logger->stop();
    _finish = true;
    if (_metric) {
        _metric->setFinish();
    }
}

bool NaviWorkerBase::isStopped() const {
    return isFinished() || isTimeout();
}

bool NaviWorkerBase::isFinished() const {
    return _finish;
}

bool NaviWorkerBase::reportRpcLackByBiz(const multi_call::GigStreamRpcInfoMap &rpcInfoMap) {
    bool hasLack = false;
    for (const auto &pair : rpcInfoMap) {
        size_t bizUsedRpcCount, bizLackRpcCount;
        calcStreamRpcConverage(pair.second, bizUsedRpcCount, bizLackRpcCount);
        if (bizLackRpcCount > 0) {
            static const std::string metricName = "user.ops.ExchangeKernel.LackRatio";
            double value = 1.0 * bizLackRpcCount / bizUsedRpcCount;
            kmonitor::MetricsTags tags{"biz", pair.first.second};
            REPORT_USER_MUTABLE_METRIC_TAGS(_metricsReporter, metricName, value, &tags);
            hasLack = true;
        }
    }
    return hasLack;
}

void NaviWorkerBase::postFillResultMetrics() {
    const auto &naviResult = _userResult->getNaviResultWithoutCheck();
    if (naviResult) {
        _metricsCollector.totalQueueLatency = naviResult->queueUs / 1000;
        _metricsCollector.totalComputeLatency = naviResult->computeUs / 1000;
        _metricsCollector.hasError = (nullptr != naviResult->error);
        auto rpcInfoMap = naviResult->getRpcInfoMap();
        if (rpcInfoMap) {
            _metricsCollector.hasLack = reportRpcLackByBiz(*rpcInfoMap);
        }
    }
}

const NaviUserResultPtr &NaviWorkerBase::getUserResult() const {
    return _userResult;
}

void NaviWorkerBase::notifyFinish() {
    NAVI_LOG(SCHEDULE1, "worker finish");
    bool isFirst = (_metricsCollector.fillResultTime == 0); // for: may enter multiple times
    if (isFirst) {
        _metricsCollector.fillResultTime = autil::TimeUtility::currentTime();
    }
    _userResult->notify(true);
    if (isFirst) {
        postFillResultMetrics();
    }
}

void NaviWorkerBase::dec() {
    if (DEFAULT_THREAD_LIMIT != _threadLimit) {
        atomic_dec(&_processingCount);
    }
    decItemCount();
}

void NaviWorkerBase::finalize() {
    NAVI_LOG(SCHEDULE1, "finalize")
    if (_initSuccess) {
        _threadPool->decWorkerCount();
    }
    deleteThis();
}

void NaviWorkerBase::deleteThis() {
    delete this;
}

void NaviWorkerBase::incItemCount() {
    NAVI_LOG(SCHEDULE2, "inc item count from [%lld]", atomic_read(&_itemCount));
    atomic_inc(&_itemCount);
}

void NaviWorkerBase::decItemCount() {
    NAVI_LOG(SCHEDULE2, "dec item count from [%lld]", atomic_read(&_itemCount));
    doSchedule();
    if (unlikely(atomic_dec_return(&_itemCount) == 0)) {
        NAVI_LOG(SCHEDULE1, "item count clear");
        _taskQueueHolder.release();
        auto item = new WorkerDestructItem(this);
        _defaultThreadPool->push(_globalStartTimeUs + DESTRUCT_DELAY_US,
                                 atomic_inc_return(&_scheduleId),
                                 item,
                                 _collectFlags.collectMalloc());
    }
}

int64_t NaviWorkerBase::itemCount() {
    return atomic_read(&_itemCount);
}

bool NaviWorkerBase::canInline() const {
    if (INVALID_INLINE_DEPTH != INLINE_DEPTH_TLS &&
        INLINE_DEPTH_TLS < _maxInline)
    {
        return true;
    }
    return false;
}

int64_t NaviWorkerBase::getIdleThreadCount() const {
    return _threadPool->getIdleThreadCount();
}

int64_t NaviWorkerBase::getRunningThreadCount() const {
    return _threadPool->getRunningThreadCount();
}

size_t NaviWorkerBase::getQueueSize() const {
    return _threadPool->getQueueSize();
}

void NaviWorkerBase::inlineBegin() const {
    if (INVALID_INLINE_DEPTH != INLINE_DEPTH_TLS) {
        INLINE_DEPTH_TLS++;
    }
}

void NaviWorkerBase::inlineEnd() const {
    if (INLINE_DEPTH_TLS >= 1) {
        INLINE_DEPTH_TLS--;
    }
}

}
