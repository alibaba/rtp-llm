#include "navi/engine/NaviSnapshot.h"

#include "autil/StringUtil.h"
#include "multi_call/interface/QuerySession.h"
#include "navi/engine/EvTimer.h"
#include "navi/engine/NaviSession.h"
#include "navi/engine/NaviSnapshotSummary.h"
#include "navi/engine/NaviStreamSession.h"
#include "navi/engine/NaviThreadPool.h"
#include "navi/engine/TaskQueue.h"
#include "navi/engine/NaviHostInfo.h"
#include "navi/engine/ParamTopoTreeNode.h"
#include "navi/perf/NaviSymbolTable.h"
#include "navi/resource/ResourceErrorInfoCollectorR.h"

namespace navi {

template <typename T, typename... U>
static int64_t timerAllocAndNewObject(T *&obj, U &&...args) {
    autil::ScopedTime2 timer;
    obj = (T *)operator new (sizeof(T), std::align_val_t{alignof(T)});
    int64_t latency = timer.done_us();
    new (obj) T(std::forward<U>(args)...);
    return latency;
}

class NaviSessionScheduleItem : public TaskQueueScheduleItemBase {
public:
    NaviSessionScheduleItem(NaviSession *session) : _session(session) {}
    ~NaviSessionScheduleItem() {}

public:
    void process() override {
        if (_session) {
            _session->setTaskQueueHolder(stealTaskQueueHolder());
            auto logger = _session->getLogger();
            NaviLoggerScope scope(logger);
            _session->initSchedule();
        }
        _session = nullptr;
    }
    void destroy() override {
        DELETE_AND_SET_NULL(_session);
        delete this;
    }

private:
    NaviSession *_session = nullptr;
};

NaviSnapshot::NaviSnapshot(
        const NaviHostInfoPtr &hostInfo,
        const EvTimerPtr &evTimer)
    : _hostInfo(hostInfo)
    , _errorInfoCollectorR(std::make_shared<ResourceErrorInfoCollectorR>())
    , _evTimer(evTimer)
    , _testMode(TM_NONE)
    , _gigRpcServer(nullptr)
{
}

NaviSnapshot::~NaviSnapshot() {
    stop();
    if (_logManager) {
        _logManager->setNaviSymbolTable(nullptr);
        _logManager->flush(true);
    }
    _logger.logger.reset();
}

bool NaviSnapshot::init(bool isRealNavi,
                        const NaviLoggerPtr &defaultLogger,
                        const std::shared_ptr<NaviSymbolTable> &naviSymbolTable,
                        const std::string &initConfigPath,
                        InstanceId instanceId,
                        multi_call::GigRpcServer *gigRpcServer,
                        const ModuleManagerPtr &moduleManager,
                        const NaviSnapshotPtr &oldSnapshot,
                        const NaviConfigPtr &config,
                        const ResourceMap &rootResourceMap)
{
    _config = config;
    _gigRpcServer = gigRpcServer;
    if (!initLogger(instanceId, defaultLogger)) {
        return false;
    }
    NAVI_LOG(INFO, "config [%s]", _config->configStr.c_str());
    if (!initTaskQueue()) {
        return false;
    }
    if (!initNaviPerf()) {
        NAVI_LOG(WARN, "navi perf init failed, disabled");
    }
    if (!initSymbolTable(naviSymbolTable, oldSnapshot)) {
        return false;
    }
    if (!_evTimer) {
        NAVI_LOG(ERROR, "timer is null");
        return false;
    }
    if (!initBizManager(isRealNavi, moduleManager, gigRpcServer, oldSnapshot, rootResourceMap)) {
        NAVI_LOG(ERROR, "init biz manager failed");
        return false;
    }
    NAVI_LOG(INFO, "snapshot start success");
    return true;
}

bool NaviSnapshot::start(const NaviSnapshotPtr &oldSnapshot) {
    BizManagerPtr oldBizManager;
    if (oldSnapshot) {
        oldBizManager = oldSnapshot->_bizManager;
    }
    if (!_bizManager->start(_gigRpcServer, oldBizManager.get())) {
        return false;
    }
    if (_naviRpcServer && !_naviRpcServer->flushRegistry()) {
        return false;
    }
    return true;
}

void NaviSnapshot::initMetrics(kmonitor::MetricsReporter &reporter) {
    _metricsReporter = reporter.getSubReporter("", {});
}

bool NaviSnapshot::initLogger(InstanceId instanceId, const NaviLoggerPtr &defaultLogger) {
    if (!defaultLogger) {
        NAVI_KERNEL_LOG(ERROR, "default logger is null");
        return false;
    }
    _logManager = defaultLogger->getLogManager();
    _logger = NaviObjectLogger(this, "snapshot", defaultLogger);
    return true;
}

bool NaviSnapshot::initSingleTaskQueue(const std::string &name,
                                       const ConcurrencyConfig &config,
                                       std::unique_ptr<TaskQueue> &taskQueue)
{
    taskQueue = std::make_unique<TaskQueue>();
    if (!taskQueue->init(name, config, getTestMode())) {
        return false;
    }
    return true;
}

bool NaviSnapshot::initTaskQueue() {
    if (!initSingleTaskQueue("", _config->engineConfig.builtinTaskQueue,
                             _defaultTaskQueue))
    {
        NAVI_LOG(ERROR, "init default task queue failed");
        return false;
    }
    for (auto &pair : _config->engineConfig.extraTaskQueueMap) {
        if (pair.first.empty()) {
            NAVI_LOG(ERROR, "config error, empty extra queue name is not allowed");
            return false;
        }
        std::unique_ptr<TaskQueue> entry;
        if (!initSingleTaskQueue(pair.first, pair.second, entry)) {
            NAVI_LOG(ERROR, "init task queue[%s] failed", pair.first.c_str());
            return false;
        }
        _extraTaskQueueMap.emplace(pair.first, std::move(entry));
    }
    NAVI_LOG(INFO, "extra task queue map after init [%s]", autil::StringUtil::toString(_extraTaskQueueMap).c_str());
    return true;
}

bool NaviSnapshot::initNaviPerf() {
    if (_testMode != TM_NONE) {
        return true;
    }
    if (_config->engineConfig.disablePerf) {
        return true;
    }
    const auto &pidVec = _defaultTaskQueue->getThreadPool()->getPidVec();
    for (auto pid : pidVec) {
        _naviPerf.addPid(pid);
    }
    for (const auto &pair : _extraTaskQueueMap) {
        const auto &extraPidVec = pair.second->getThreadPool()->getPidVec();
        for (auto pid : extraPidVec) {
            _naviPerf.addPid(pid);
        }
    }
    if (_gigRpcServer) {
        auto grpcThreads = _gigRpcServer->getGrpcIoPids();
        for (auto pid : grpcThreads) {
            _naviPerf.addPid(pid);
        }
    }
    if (!_naviPerf.init(_logger.logger)) {
        return false;
    }
    return true;
}

bool NaviSnapshot::initSymbolTable(const std::shared_ptr<NaviSymbolTable> &naviSymbolTable,
                                   const NaviSnapshotPtr &oldSnapshot)
{
    if (_testMode != TM_NONE) {
        return true;
    }
    if (_config->engineConfig.disableSymbolTable) {
        return true;
    }
    _naviSymbolTable = std::make_shared<NaviSymbolTable>();
    std::shared_ptr<NaviSymbolTable> oldSymbolTable = oldSnapshot ? oldSnapshot->getNaviSymbolTable() : nullptr;
    if (!oldSymbolTable) {
        oldSymbolTable = naviSymbolTable;
    }
    if (!_naviSymbolTable->init(_logger.logger, oldSymbolTable.get())) {
        _naviSymbolTable.reset();
    }
    _logManager->setNaviSymbolTable(_naviSymbolTable);
    return true;
}

bool NaviSnapshot::reInitSymbolTable() {
    if (!_naviSymbolTable) {
        return true;
    }
    auto naviSymbolTable = std::make_shared<NaviSymbolTable>();
    if (!naviSymbolTable->init(_logger.logger, _naviSymbolTable.get())) {
        naviSymbolTable.reset();
    }
    _logManager->setNaviSymbolTable(naviSymbolTable);
    _naviSymbolTable = naviSymbolTable;
    return true;
}

bool NaviSnapshot::initBizManager(bool isRealNavi,
                                  const ModuleManagerPtr &moduleManager,
                                  multi_call::GigRpcServer *gigRpcServer,
                                  const NaviSnapshotPtr &oldSnapshot,
                                  const ResourceMap &rootResourceMapIn)
{
    ResourceMap rootResourceMap;
    rootResourceMap.addResource(rootResourceMapIn);

    BizManagerPtr oldBizManager;
    if (oldSnapshot) {
        oldBizManager = oldSnapshot->_bizManager;
    }
    ResourceMap resourceMap(rootResourceMap.getMap());
    addBuildinExternalResource(gigRpcServer, resourceMap);
    _bizManager = std::make_shared<BizManager>(isRealNavi, _hostInfo->getNaviName(), moduleManager);
    _bizManager->setTestMode(_testMode);
    if (!_bizManager->preInit(_logger.logger, oldBizManager.get(), *_config, resourceMap)) {
        return false;
    }
    if (!initBizResource(resourceMap)) {
        return false;
    }
    if (!_bizManager->postInit()) {
        return false;
    }
    return true;
}

void NaviSnapshot::addBuildinExternalResource(
    multi_call::GigRpcServer *gigRpcServer, ResourceMap &resourceMap)
{
    if (gigRpcServer) {
        _naviRpcServer = std::make_shared<NaviRpcServerR>(this, gigRpcServer);
        resourceMap.addResource(_naviRpcServer);
    }
    resourceMap.addResource(_errorInfoCollectorR);
}

bool NaviSnapshot::initBizResource(const ResourceMap &rootResourceMap) {
    auto graphDef = _bizManager->cloneResourceGraph();
    if (!graphDef) {
        NAVI_LOG(WARN, "empty init resource");
        return true;
    }
    NAVI_LOG(INFO, "root resource map [%s]", autil::StringUtil::toString(rootResourceMap).c_str());
    RunGraphParams params;
    params.setResourceStage(RS_SNAPSHOT);
    params.setTimeoutMs(_config->initTimeoutMs);
    params.setCollectMetric(_config->collectInitMetric);
    params.setTraceLevel("INFO");
    params.addParamTopoTreeStr("edge_key=*");
    params.setLoadGraph(true);
    NaviUserResultPtr result;
    {
        NaviLoggerScope scope(nullptr);
        result = runGraph(graphDef, params, &rootResourceMap);
    }
    if (!result) {
        NAVI_LOG(ERROR, "run resource graph failed");
        return false;
    }
    setInitResult(result);
    _collectThread = autil::Thread::createThread(std::bind(&NaviSnapshot::collectInitResource, this), "navi_collect");
    if (!_collectThread) {
        NAVI_LOG(ERROR, "create collect thread failed");
        return false;
    }
    {
        NaviLoggerScope scope(_logger);
        if (!_collectWaiter.wait(WAIT_TIME)) {
            NAVI_LOG(ERROR, "wait collect resource failed");
            return false;
        }
    }
    if (result->_logger.logger) {
        result->_logger.logger->stopTraceAppender();
    }
    return true;
}

void NaviSnapshot::collectInitResource() {
    ResourceMap resourceMap;
    collectResultResource(getInitResult(), resourceMap, &_collectWaiter);
}

void InitWaiter::setExpectCount(size_t expectCount) {
    autil::ScopedLock lock(_collectCond);
    _expectCount = expectCount;
}

bool InitWaiter::wait(int64_t timeoutUs) {
    auto begin = autil::TimeUtility::currentTime();
    while (true) {
        autil::ScopedLock lock(_collectCond);
        auto readyCount = _readyBatchAndMeta.empty() ? 0 : _readyBatchAndMeta.begin()->second.first.size();
        auto result = (_expectCount > 0 && _expectCount <= readyCount);
        if (result) {
            return true;
        }
        if (_stopped) {
            if (!result) {
                NAVI_KERNEL_LOG(
                    ERROR, "wait failed, expectCount: %lu, readyCount: %lu", _expectCount, readyCount);
            }
            return result;
        }
        if (autil::TimeUtility::currentTime() - begin > timeoutUs) {
            NAVI_KERNEL_LOG(ERROR, "wait timeout, timeoutUs: %ld", timeoutUs);
            break;
        }
        _collectCond.wait(timeoutUs);
    }
    return false;
}

void InitWaiter::notify(const NaviUserData &data, const multi_call::TopoSignatureSetPtr &signatureSet) {
    auto index = data.index;
    auto streamId = data.streamId;
    auto eof = data.eof;
    autil::ScopedLock lock(_collectCond);
    if (_readyBatchAndMeta.end() == _readyBatchAndMeta.find(streamId)) {
        std::set<size_t> eofIndexSet;
        for (const auto &pair : _eofIndexMeta) {
            eofIndexSet.insert(pair.first);
        }
        multi_call::TopoSignatureSetPtr eofSignatureSet;
        if (!_finishWaitGigMeta) {
            for (const auto &pair : _eofIndexMeta) {
                ResourceData::mergeGigLocalSubResult(pair.second, eofSignatureSet);
            }
        }
        _readyBatchAndMeta.emplace(std::make_pair(streamId, std::make_pair(eofIndexSet, eofSignatureSet)));
    }
    _readyBatchAndMeta[streamId].first.insert(index);
    if (!_finishWaitGigMeta) {
        ResourceData::mergeGigLocalSubResult(signatureSet, _readyBatchAndMeta[streamId].second);
        if (eof) {
            NAVI_KERNEL_LOG(INFO, "output index [%lu] streamId [%lu] reach eof", index, streamId);
            _eofIndexMeta[index] = signatureSet;
            for (size_t i = streamId + 1; i < _readyBatchAndMeta.rbegin()->first; i++) {
                _readyBatchAndMeta[i].first.insert(index);
                ResourceData::mergeGigLocalSubResult(signatureSet, _readyBatchAndMeta[i].second);
            }
        }
    }
    bool hasReadyBatch = false;
    if (!_readyBatchAndMeta.empty()) {
        auto iter = _readyBatchAndMeta.begin();
        for (; iter != _readyBatchAndMeta.end(); iter++) {
            if (iter->second.first.size() < _expectCount) {
                break;
            }
        }
        if (iter != _readyBatchAndMeta.begin()) {
            hasReadyBatch = true;
            iter--;
            _readyBatchAndMeta.erase(_readyBatchAndMeta.begin(), iter);
        }
    }
    if (hasReadyBatch) {
        _collectCond.broadcast();
    }
}

bool InitWaiter::waitLocalSubComplete(int64_t timeoutUs,
                                      const std::vector<multi_call::SignatureTy> &publishSignatures) {
    auto begin = autil::TimeUtility::currentTime();
    while (true) {
        autil::ScopedLock lock(_collectCond);
        auto readyCount = _readyBatchAndMeta.empty() ? 0 : _readyBatchAndMeta.begin()->second.first.size();
        if (readyCount >= _expectCount) {
            NAVI_KERNEL_LOG(INFO, "ready batch [%lu]", _readyBatchAndMeta.begin()->first);
            const auto &gigMetaLocalSignatures = _readyBatchAndMeta.begin()->second.second;
            if (!gigMetaLocalSignatures || _finishWaitGigMeta) {
                NAVI_KERNEL_LOG(INFO,
                                "gigMetaLocalSignatures [%p], _finishWaitGigMeta [%d], return true",
                                gigMetaLocalSignatures.get(),
                                _finishWaitGigMeta);
                return true;
            }
            bool allFound = true;
            NAVI_KERNEL_LOG(
                INFO, "published local signatures: %s", autil::StringUtil::toString(publishSignatures).c_str());
            for (const auto &sig : publishSignatures) {
                if (gigMetaLocalSignatures->end() == gigMetaLocalSignatures->find(sig)) {
                    NAVI_KERNEL_LOG(INFO, "lack published local signatures: %lu, still wait", sig);
                    allFound = false;
                    break;
                }
            }
            if (allFound) {
                NAVI_KERNEL_LOG(INFO, "finish wait all published local signatures, return true");
                _finishWaitGigMeta = true;
                return true;
            }
        } else {
            NAVI_KERNEL_LOG(INFO, "no ready batch, expectCount: %lu, readyCount: %lu", _expectCount, readyCount);
        }
        if (_stopped) {
            NAVI_KERNEL_LOG(ERROR, "waiter stopped, timeoutUs: %ld", timeoutUs);
            return false;
        }
        if (autil::TimeUtility::currentTime() - begin > timeoutUs) {
            NAVI_KERNEL_LOG(ERROR, "wait timeout, timeoutUs: %ld", timeoutUs);
            return false;
        }
        _collectCond.wait(2 * 1000 * 1000);
    }
}

void InitWaiter::stop() {
    autil::ScopedLock lock(_collectCond);
    _stopped = true;
    _collectCond.broadcast();
}

bool NaviSnapshot::collectResultResource(const NaviUserResultPtr &result,
        ResourceMap &resourceMap, InitWaiter *waiter)
{
    if (!result) {
        NAVI_LOG(ERROR, "run resource graph failed");
        return false;
    }
    while (true) {
        NaviUserData data;
        bool eof = false;
        if (result->nextData(data, eof)) {
            multi_call::TopoSignatureSetPtr gigMetaLocalSignatures;
            if (data.data) {
                auto resourceData =
                    std::dynamic_pointer_cast<ResourceData>(data.data);
                bool success = false;
                if (resourceData) {
                    gigMetaLocalSignatures = resourceData->_gigMetaLocalSignatures;
                    if (resourceData->_resource) {
                        NAVI_LOG(SCHEDULE1, "create resource success [%s] [%p]",
                            data.name.c_str(), resourceData->_resource.get());
                        resourceMap.addResource(resourceData->_resource);
                        success = true;
                    } else if (!resourceData->_require) {
                        success = true;
                    } else if (resourceData->_wait) {
                        NAVI_LOG(DEBUG, "need wait for resource creation [%s]", resourceData->_name.c_str());
                        success = true;
                    }
                }
                if (!success) {
                    NAVI_LOG(ERROR, "create resource failed [%s]",
                            data.name.c_str());
                    result->abort();
                    if (waiter) {
                        waiter->stop();
                    }
                    return false;
                }
            }
            if (waiter) {
                waiter->setExpectCount(result->outputCount());
                waiter->notify(data, gigMetaLocalSignatures);
            }
        }
        if (eof) {
            break;
        }
    }
    if (waiter) {
        waiter->stop();
    }
    auto naviResult = result->getUnfinishedNaviResult();
    if (!naviResult) {
        NAVI_LOG(ERROR, "get navi result failed");
        return false;
    }
    return !naviResult->hasError();
}

bool NaviSnapshot::waitLocalSubComplete() {
    const auto &signatures = _bizManager->getPublishSignatures();
    return _collectWaiter.waitLocalSubComplete(WAIT_TIME, signatures);
}

void NaviSnapshot::logVisData(const NaviResultPtr &naviResult) const {
    if (!naviResult) {
        NAVI_LOG(ERROR, "log init vis data failed, null navi result");
        return;
    }
    auto visProto = naviResult->getVisProto();
    if (!visProto) {
        NAVI_LOG(ERROR, "log init vis data failed, null vis data");
        return;
    }
    const auto &naviName = _hostInfo->getNaviName();
    std::string filePrefix;
    if (!naviName.empty()) {
        filePrefix = "./navi." + naviName + ".snapshot";
    } else {
        filePrefix = "./navi.snapshot";
    }
    NaviSnapshotSummary::writeFile(filePrefix + ".vis",
                                   visProto->SerializeAsString());
}

NaviObjectLogger NaviSnapshot::getObjectLogger(SessionId id) {
    auto logger = getTlsLogger();
    if (!logger) {
        logger = _logManager->createLogger(id);
    }
    return NaviObjectLogger(this, "snapshot", logger);
}

const NaviLoggerPtr &NaviSnapshot::getLogger() const {
    return _logger.logger;
}

NaviLoggerPtr NaviSnapshot::getTlsLogger() {
    if (NAVI_TLS_LOGGER && NAVI_TLS_LOGGER->logger) {
        return NAVI_TLS_LOGGER->logger;
    } else {
        return nullptr;
    }
}

ModuleManagerPtr NaviSnapshot::getModuleManager() const {
    if (_bizManager) {
        return _bizManager->getModuleManager();
    }
    return nullptr;
}

void NaviSnapshot::cleanupModule() {
    if (_bizManager) {
        return _bizManager->cleanupModule();
    }
}

void NaviSnapshot::stop() {
    if (_naviRpcServer) {
        _naviRpcServer->stop();
    }
    auto initResult = getInitResult();
    if (initResult) {
        initResult->abort(EC_NONE);
    }
    _collectThread.reset();
    NaviLoggerScope scope(_logger);
    _naviPerf.stop();
    if (_defaultTaskQueue) {
        _defaultTaskQueue->stop();
    }
    for (auto &pair : _extraTaskQueueMap) {
        pair.second->stop();
    }
    _bizManager.reset();
    NAVI_LOG(INFO, "snapshot stopped");
}

const std::shared_ptr<NaviThreadPool> &NaviSnapshot::getDefaultThreadPool() const {
    return _defaultTaskQueue->getThreadPoolPtr();
}

void NaviSnapshot::setTestMode(TestMode testMode) {
    _testMode = testMode;
}

NaviUserResultPtr NaviSnapshot::runGraph(GraphDef *graphDef,
        const RunGraphParams &params,
        const ResourceMap *resourceMap)
{
    NaviUserResultPtr naviUserResult;
    if (!runGraphImpl(graphDef, params, resourceMap, naviUserResult, nullptr)) {
        NAVI_KERNEL_LOG(WARN, "run graph inner failed");
        return nullptr;
    }
    return naviUserResult;
}


NaviUserResultPtr NaviSnapshot::runGraphAsync(
        GraphDef *graphDef,
        const RunGraphParams &params,
        const ResourceMap *resourceMap,
        NaviUserResultClosure *closure)
{
    NaviUserResultPtr naviUserResult;
    if (!runGraphImpl(graphDef, params, resourceMap, naviUserResult, closure)) {
        NAVI_KERNEL_LOG(WARN, "run graph inner failed");
        closure->run(nullptr, nullptr);
        return nullptr;
    }
    return naviUserResult;
}

// push stream session to schedule queue
void NaviSnapshot::runStreamSession(const std::string &taskQueueName,
                                    const SessionId &sessionId,
                                    int64_t globalStartTimeUs,
                                    TaskQueueScheduleItemBase *item,
                                    std::shared_ptr<NaviLogger> &logger)
{

    auto objectLogger = getObjectLogger(sessionId);
    logger = objectLogger.logger;
    NaviLoggerScope scope(objectLogger);

    auto *taskQueue = getTaskQueue(taskQueueName);
    if (!taskQueue->push(globalStartTimeUs, item)) { // dropped as schedule queue full
        NAVI_KERNEL_LOG(WARN, "session push failed, dropped");
        REPORT_USER_MUTABLE_QPS(_metricsReporter, "dropQps");
    }
}

TaskQueue *NaviSnapshot::getTaskQueue(const std::string &taskQueueName) {
    auto iter = _extraTaskQueueMap.find(taskQueueName);
    if (iter != _extraTaskQueueMap.end()) {
        return iter->second.get();
    }
    return _defaultTaskQueue.get();
}

bool NaviSnapshot::runGraphImpl(GraphDef *graphDef,
                                const RunGraphParams &params,
                                const ResourceMap *resourceMap,
                                NaviUserResultPtr &naviUserResult,
                                NaviUserResultClosure *closure) {
    if (!graphDef) {
        return false;
    }
    auto *taskQueue = getTaskQueue(params.getTaskQueueName());
    auto logger = getObjectLogger(SessionId());
    NaviLoggerScope scope(logger);
    NaviSession *session = nullptr;
    {

        session = new NaviSession(taskQueue, _bizManager.get(), graphDef, resourceMap, closure);
    }
    session->setTestMode(getTestMode());

    naviUserResult = session->getUserResult();
    fillSnapshotResource(params, session);
    if (!session->init(params, nullptr, _metricsReporter)) {
        DELETE_AND_SET_NULL(session);
        NAVI_KERNEL_LOG(WARN, "session init failed");
        REPORT_USER_MUTABLE_QPS(_metricsReporter, "initFailedQps");
        return false;
    }
    auto globalStartTimeUs = session->getGlobalStartTimeUs();
    auto *item = new NaviSessionScheduleItem(session);
    if (!taskQueue->push(globalStartTimeUs, item)) {
        NAVI_KERNEL_LOG(WARN, "session push failed, dropped");
        REPORT_USER_MUTABLE_QPS(_metricsReporter, "dropQps");
        return false;
    }
    if (!closure) { // for sync run mode
        naviUserResult->waitInit();
    }
    return true;
}

// run stream session after popped from schedule queue
bool NaviSnapshot::doRunStreamSession(NaviMessage *request,
                                      const ArenaPtr &arena,
                                      const RunParams &pbParams,
                                      const std::shared_ptr<multi_call::GigStreamBase> &stream,
                                      const std::shared_ptr<MemLeakCollector> &memLeakCollector) {
    auto *taskQueue = getTaskQueue(pbParams.task_queue_name());
    NaviStreamSession *session = nullptr;
    {

        session = new NaviStreamSession(taskQueue, _bizManager.get(), request, arena, stream);
    }
    session->setTestMode(getTestMode());
    RunGraphParams params;
    if (!parseRunParams(pbParams, params)) {
        DELETE_AND_SET_NULL(session);
        return false;
    }
    params.setAsyncIo(stream->isAsyncMode());
    fillSnapshotResource(params, session);
    session->setMemLeakCollector(memLeakCollector);
    if (!session->init(params, stream->getQueryInfo(), _metricsReporter)) {
        DELETE_AND_SET_NULL(session);
        NAVI_KERNEL_LOG(WARN, "stream session init failed, dropped");
        REPORT_USER_MUTABLE_QPS(_metricsReporter, "initStreamFailedQps");
        return false;
    }
    session->initSchedule();
    return true;
}

bool NaviSnapshot::parseRunParams(const RunParams &pbParams,
                                  RunGraphParams &params)
{
    const auto &creatorManager = _bizManager->getBuildinCreatorManager();
    return RunGraphParams::fromProto(pbParams, creatorManager.get(), params);
}

void NaviSnapshot::fillSnapshotResource(const RunGraphParams &params, NaviWorkerBase *session) {
    session->setDefaultThreadPool(getDefaultThreadPool());
    session->setNaviPerf(&_naviPerf);
    session->setNaviSymbolTable(_naviSymbolTable);
    session->setEvTimer(_evTimer.get());
    session->setHostInfo(_hostInfo);
}

bool NaviSnapshot::createResource(const std::string &bizName,
                                  NaviPartId partCount,
                                  NaviPartId partId,
                                  const std::set<std::string> &resources,
                                  ResourceMap &resourceMap)
{
    if (!_bizManager) {
        return false;
    }
    auto resourceSet = resources;
    if (resourceSet.empty()) {
        return true;
    }
    auto graphDef = _bizManager->createResourceGraph(bizName, partCount, partId,
                                                     resourceSet);
    RunGraphParams params;
    params.setLoadGraph(true);
    auto result = runGraph(graphDef, params, nullptr);
    if (!collectResultResource(result, resourceMap, nullptr)) {
        return false;
    }
    return true;
}

void NaviSnapshot::loop() {
    static size_t count = 0;
    if (0 == (count++) % 128) {
        if (_bizManager) {
            NaviLoggerScope scope(_logger);
            _bizManager->reclaimCache();
        }
    }
    if (_metricsReporter) {
        {
            auto stat = _defaultTaskQueue->getStat();
            kmonitor::MetricsTags tag{"name", "builtin"};
            _metricsReporter->report<TaskQueueStatMetrics>(&tag, &stat);
        }
        for (auto &pair : _extraTaskQueueMap) {
            auto stat = pair.second->getStat();
            kmonitor::MetricsTags tag{"name", pair.first};
            _metricsReporter->report<TaskQueueStatMetrics>(&tag, &stat);
        }
    }
}

void NaviSnapshot::logSummary(const std::string &stage) const {
    auto initResult = getInitResult();
    if (initResult) {
        auto naviResult = initResult->getUnfinishedNaviResult();
        if (naviResult && _testMode == TM_NONE) {
            // DO NOT steal trace in test mode!!!
            // otherwise trace check may be unstable
            logVisData(naviResult);
        }
    }
    NaviSnapshotSummary summary;
    summary.naviName = _hostInfo->getNaviName();
    summary.stage = stage;
    summary.config = _config.get();
    if (_bizManager) {
        _bizManager->fillSummary(summary);
    }
    NaviSnapshotSummary::logSummary(summary);
}

std::shared_ptr<NaviUserResult> NaviSnapshot::getInitResult() const {
    autil::ScopedLock lock(_resultLock);
    return _initResult;
}

void NaviSnapshot::setInitResult(const std::shared_ptr<NaviUserResult> &initResult) {
    autil::ScopedLock lock(_resultLock);
    _initResult = initResult;
}

std::shared_ptr<GraphVisDef> NaviSnapshot::getInitVisProto() const {
    auto initResult = getInitResult();
    if (!initResult) {
        return nullptr;
    }
    auto naviResult = initResult->getUnfinishedNaviResult();
    if (!naviResult) {
        return nullptr;
    }
    auto visProto = naviResult->getVisProto();
    if (visProto && visProto->nodes_size() > 0) {
        fillSummaryToVis(visProto, _config ? _config->configStr : "null config");
        updateError(visProto);
    }
    return visProto;
}

void NaviSnapshot::fillSummaryToVis(const std::shared_ptr<GraphVisDef> &visProto,
                                    const std::string &configStr) const
{
    auto nodeDef = visProto->mutable_nodes(0);
    auto naviSummaryDef = nodeDef->mutable_navi_summary();
    naviSummaryDef->mutable_snapshot_config()->set_config_json(configStr);
    fillNaviSummary(*naviSummaryDef);
    auto registryTable = naviSummaryDef->mutable_registry_table();
    nodeDef->mutable_symbol_table()->MergeFrom(registryTable->symbol_table());
    registryTable->clear_symbol_table();
}

void NaviSnapshot::updateProto(NaviSummaryDef &summaryDef) const {
    if (_bizManager) {
        _bizManager->updateProto(summaryDef);
    }
}

void NaviSnapshot::updateError(const std::shared_ptr<GraphVisDef> &visProto) const {
    auto errorInfo = _errorInfoCollectorR->getErrorInfo();
    if (!errorInfo) {
        return;
    }
    auto nodeDef = visProto->mutable_nodes(0);
    auto graphResult = nodeDef->mutable_result();
    if (!graphResult->has_traces()) {
        return;
    }
    auto object = (uint64_t)errorInfo->object;
    auto beginTimeUs = errorInfo->beginTimeUs;
    auto endTimeUs = errorInfo->endTimeUs;
    LoggingEventDef *targetEventDef = nullptr;
    auto &traceDef = *graphResult->mutable_traces();
    int64_t count = traceDef.events_size();
    for (int64_t i = 0; i < count; i++) {
        auto &eventDef = *traceDef.mutable_events(i);
        if (eventDef.level() > LOG_LEVEL_ERROR || object != eventDef.object()) {
            continue;
        }
        auto timeUs = CommonUtil::getTimeUs(eventDef.time_sec(), eventDef.time_usec());
        if (timeUs < beginTimeUs || timeUs > endTimeUs) {
            continue;
        }
        if (!targetEventDef) {
            targetEventDef = &eventDef;
        } else {
            auto targetTimeUs = CommonUtil::getTimeUs(targetEventDef->time_sec(), targetEventDef->time_usec());
            if (targetTimeUs > timeUs) {
                targetEventDef = &eventDef;
            } else if (targetTimeUs == timeUs) {
                if (targetEventDef->tid() > eventDef.tid()) {
                    targetEventDef = &eventDef;
                }
            }
        }
    }
    if (targetEventDef) {
        targetEventDef->set_major_error(true);
        auto errorDef = graphResult->mutable_error();
        auto oldErrorEvent = errorDef->mutable_error_event();
        {
            LoggingEvent event;
            event.fromProto(*targetEventDef, &nodeDef->symbol_table());
            oldErrorEvent->Clear();
            event.toProto(oldErrorEvent, nullptr);
        }
        auto ec = errorInfo->ec;
        errorDef->set_ec(ec);
        errorDef->set_ec_str(CommonUtil::getErrorString(ec));
    }
}

std::vector<std::string> NaviSnapshot::getBizList() const {
    if (_bizManager) {
        return _bizManager->getBizList();
    } else {
        return {};
    }
}

void NaviSnapshot::fillNaviSummary(NaviSummaryDef &summaryDef) const {
    if (_config) {
        _config->fillSnapshotConfig(*summaryDef.mutable_snapshot_config());
    }
    if (_bizManager) {
        _bizManager->fillProto(summaryDef);
    }
    if (_naviRpcServer) {
        _naviRpcServer->fillProto(*summaryDef.mutable_rpc());
    }
}
}
