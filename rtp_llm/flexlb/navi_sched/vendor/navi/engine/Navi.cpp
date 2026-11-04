#include "navi/engine/Navi.h"

#include "autil/FileUtil.h"
#include <multi_call/agent/GigAgent.h>
#include <multi_call/rpc/GigRpcServer.h>

#include "navi/builder/GraphBuilder.h"
#include "navi/config/NaviConfig.h"
#include "navi/config_loader/Config.h"
#include "navi/distribute/NaviServerStreamCreator.h"
#include "navi/engine/EvTimer.h"
#include "navi/engine/NaviMetricsGroup.h"
#include "navi/engine/NaviR.h"
#include "navi/engine/NaviSnapshot.h"
#include "navi/engine/NaviHostInfo.h"
#include "navi/resource/MemoryPoolR.h"
#include "navi/resource/NaviStateRpcR.h"
#include "navi/resource/NaviStateR.h"
#include "navi/util/CommonUtil.h"
#include "navi/util/EnvParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCServer.h"

namespace navi {

Navi::Navi(const std::string &name)
    : _name(name)
    , _gigRpcServer(nullptr)
    , _testMode(TM_NONE)
{
    initInstanceId();
}

Navi::~Navi() {
    stop();
}

bool Navi::init(const std::string &initConfigPath,
                multi_call::GigRpcServer *gigRpcServer)
{
    _initConfigPath = initConfigPath;
    _gigRpcServer = gigRpcServer;
    if (!initEvTimer()) {
        return false;
    }
    if (_isRealNavi) {
        return doInit();
    } else {
        return true;
    }
}

bool Navi::initWithGigRpcServer(
        const std::string &initConfigPath,
        const std::shared_ptr<multi_call::GigRpcServer> &gigRpcServer)
{
    _gigRpcServerPtr = gigRpcServer;
    return init(initConfigPath, gigRpcServer.get());
}

bool Navi::doInit() {
    initDefaultLogger();
    NAVI_LOG(INFO, "start init with gigRpcServer [%p], config [%s]",
             _gigRpcServer,
             _initConfigPath.c_str());
    if (!initHostInfo()) {
        return false;
    }
    if (!initRpc()) {
        return false;
    }
    if (!initBackgroundThread()) {
        return false;
    }
    if (!initMemoryPoolR()) {
        return false;
    }
    return true;
}

void Navi::initMetrics(kmonitor::MetricsReporter &reporter) {
    _metricsReporter = reporter.getSubReporter("", {});
}

bool Navi::initRpc() {
    if (!_gigRpcServer) {
        NAVI_LOG(INFO, "skip init rpc as gigRpcServer is nullptr.");
        return true;
    }
    if (!_isRealNavi) {
        return true;
    }
    _streamCreator = std::make_shared<NaviServerStreamCreator>(_logger.logger, this, _name);
    if (!_gigRpcServer->registerGrpcStreamService(
            std::dynamic_pointer_cast<multi_call::GigServerStreamCreator>(
                _streamCreator)))
    {
        NAVI_LOG(ERROR,
                 "register navi stream rpc failed, check whether the grpc server is initialized");
        return false;
    }
    _gigRpcServer->startAgent(); // start agent to receive query
    NAVI_LOG(INFO,
             "register navi stream rpc success, navi: [%s %p], stream creator [%p], isRealNavi: %d",
             _name.c_str(),
             this,
             _streamCreator.get(),
             _isRealNavi);
    return true;
}

// 1
bool Navi::update(const std::string &configLoader,
                  const std::string &configPath,
                  const std::string &loadParam,
                  const ResourceMap &rootResourceMap)
{
    return update(configLoader, configPath, loadParam, rootResourceMap, nullptr);
}

// 2
bool Navi::update(const NaviConfig &config,
                  const ResourceMap &rootResourceMap)
{
    std::string configStr;
    {
        initDefaultLogger();
        NaviLoggerScope scope(_logger);
        if (!NaviConfig::prettyDumpToStr(config, configStr)) {
            NAVI_LOG(ERROR, "dump navi config to str failed");
            return false;
        }
    }
    return update(configStr, rootResourceMap);
}

// 3
bool Navi::update(const std::string &configStr,
                  const ResourceMap &rootResourceMapIn)
{
    return update(configStr, rootResourceMapIn, nullptr, nullptr);
}

bool Navi::update(const std::string &configLoader,
                  const std::string &configPath,
                  const std::string &loadParam,
                  const ResourceMap &rootResourceMap,
                  const std::shared_ptr<ModuleManager> &moduleManager)
{
    auto configLoadFunc = [this, configLoader, configPath, loadParam](std::string &configStr) {
        initDefaultLogger();
        NaviLoggerScope scope(_logger);
        if (!Config::load(getNaviPythonPath(), configLoader, configPath, loadParam, configStr)) {
            NAVI_LOG(ERROR, "load navi config from python failed");
            return false;
        } else {
            return true;
        }
    };
    return update(std::string(), rootResourceMap, moduleManager, configLoadFunc);
}

bool Navi::update(const std::string &configStr,
                  const ResourceMap &rootResourceMapIn,
                  const std::shared_ptr<ModuleManager> &moduleManager,
                  const std::function<bool(std::string &)> &configLoadFunc)
{
    if (!_isRealNavi) {
        return runUpdateGraph(configStr, rootResourceMapIn, moduleManager, configLoadFunc);
    } else {
        return doUpdate(configStr, nullptr, nullptr, nullptr, rootResourceMapIn, moduleManager);
    }
}

bool Navi::doUpdate(const std::string &configStr,
                    const NaviConfigPtr &naviConfig,
                    const NaviLoggerPtr &logger,
                    const std::shared_ptr<NaviSymbolTable> &naviSymbolTable,
                    const ResourceMap &rootResourceMapIn,
                    const std::shared_ptr<ModuleManager> &moduleManager)
{
    ResourceMap rootResourceMap;
    rootResourceMap.addResource(rootResourceMapIn);
    rootResourceMap.addResource(_memoryPoolR);
    NaviConfigPtr config;
    if (naviConfig) {
        config = naviConfig;
    } else {
        config = std::make_shared<NaviConfig>();
        initDefaultLogger();
        NaviLoggerScope scope(_logger);
        if (!config->loadConfig(configStr)) {
            return false;
        }
    }
    if (config->sleepBeforeUpdateUs > 0) {
        usleep(config->sleepBeforeUpdateUs);
    }
    if (!_hostInfo) {
        NAVI_LOG(ERROR, "no host info, call navi init before update");
        return false;
    }
    auto snapshot = std::make_shared<NaviSnapshot>(_hostInfo, _evTimer);
    snapshot->setTestMode(_testMode);
    if (_isRealNavi && _metricsReporter) {
        snapshot->initMetrics(*_metricsReporter);
    }
    if (!snapshot->init(_isRealNavi,
                        logger ? logger : _logger.logger,
                        naviSymbolTable,
                        _initConfigPath,
                        _instanceId,
                        _gigRpcServer,
                        moduleManager,
                        getSnapshot(),
                        config,
                        rootResourceMap))
    {
        NaviLoggerScope scope(_logger);
        snapshot->logSummary("snapshot");
        setLastVisProto(snapshot->getInitVisProto());
        snapshot->stop();
        return false;
    }
    NaviLoggerScope scope(snapshot->_logger);
    updateSnapshot(snapshot);
    cleanupModule();
    {
        auto newSnapshot = getSnapshot();
        if (newSnapshot) {
            newSnapshot->reInitSymbolTable();
        }
    }
    if (config->sleepAfterUpdateUs > 0) {
        usleep(config->sleepAfterUpdateUs);
    }
    setLastVisProto(snapshot->getInitVisProto());
    if (_isRealNavi) {
        logSnapshotSummary();
    }
    return true;
}

void Navi::updateSnapshot(const NaviSnapshotPtr &newSnapshot) {
    NAVI_KERNEL_LOG(INFO, "begin update snapshot, new [%p]", newSnapshot.get());
    _updatingSnapshotStatus = true;
    auto configSnapshot = newSnapshot;
    if (!configSnapshot) {
        configSnapshot = getSnapshot();
    }
    auto smoothIntervalUs = multi_call::INVALID_SMOOTH_WEIGHT_INTERVAL;
    if (_waitNoQuery && configSnapshot) {
        smoothIntervalUs = configSnapshot->getGigSmoothIntervalUs();
    }
    auto disableGigAllReplicaStopped = false;
    if (configSnapshot) {
        disableGigAllReplicaStopped = configSnapshot->disableGigAllReplicaStopped();
    }
    if (_isRealNavi && _gigRpcServer) {
        NAVI_KERNEL_LOG(INFO, "stop gig agent");
        auto oldSnapshot = getSnapshot();
        _gigRpcServer->stopAgent(oldSnapshot ? smoothIntervalUs : multi_call::INVALID_SMOOTH_WEIGHT_INTERVAL);
        if (oldSnapshot) {
            oldSnapshot->setStopping(true);
            if (_waitNoQuery) {
                waitUntilNoQuery(disableGigAllReplicaStopped);
            }
        }
    }
    if (newSnapshot) {
        newSnapshot->start(getSnapshot());
    }
    NaviSnapshotPtr oldSnapshot;
    {
        autil::ScopedWriteLock scope(_snapshotLock);
        oldSnapshot = _snapshot;
        _snapshot = newSnapshot;
        if (newSnapshot) {
            _logger.logger = newSnapshot->getLogger();
            _logger.prefix.clear();
            _logger.addPrefix("navi");
        }
    }
    configSnapshot.reset();
    if (oldSnapshot) {
        NaviLoggerScope scope(oldSnapshot->_logger);
        CommonUtil::waitUseCount(oldSnapshot, 1);
        oldSnapshot->stop();
    }
    if (_isRealNavi && newSnapshot && _gigRpcServer) {
        NAVI_KERNEL_LOG(INFO, "new snapshot[%p] start wait local sub complete", newSnapshot.get());
        newSnapshot->waitLocalSubComplete();
        NAVI_KERNEL_LOG(INFO, "new snapshot[%p] end wait local sub complete", newSnapshot.get());
        _gigRpcServer->startAgent(smoothIntervalUs);
    }
    _updatingSnapshotStatus = false;
    NAVI_KERNEL_LOG(INFO, "end update snapshot, new [%p], navi: [%p]", _snapshot.get(), this);
}

NaviSnapshotPtr Navi::getSnapshot() const {
    autil::ScopedReadLock scope(_snapshotLock);
    return _snapshot;
}

void Navi::logSnapshotSummary() const {
    auto snapshot = getSnapshot();
    if (!snapshot) {
        return;
    }
    NaviLoggerScope scope(_logger);
    snapshot->logSummary("snapshot");
}

void Navi::initDefaultLogger() {
    if (_logger.logger) {
        if (!_logger.object) {
            _logger.object = this;
            _logger.prefix.clear();
            _logger.addPrefix("navi [%s] snapshot before load config", _name.c_str());
        }
        return;
    }
    auto oldSnapshot = getSnapshot();
    if (oldSnapshot) {
        _logger.logger = oldSnapshot->getLogger();
        _logger.object = this;
        return;
    }
    auto tlsLogger = NaviSnapshot::getTlsLogger();
    if (tlsLogger) {
        _logger.logger = tlsLogger;
    } else {
        _logger.logger = NaviLogManager::createDefaultLogger(_instanceId, _name);
    }
    _logger.object = this;
    _logger.prefix.clear();
    _logger.addPrefix("navi [%s] snapshot before load config", _name.c_str());
}

void Navi::stopSnapshot() {
    if (_isRealNavi) {
        NAVI_LOG(INFO, "navi stop snapshot begin");
        NaviLoggerScope scope(_logger);
        updateSnapshot(nullptr);
    } else {
        auto beginTimeUs = autil::TimeUtility::currentTime();
        auto realNavi = getRealNavi();
        if (realNavi) {
            realNavi->stopSnapshot();
        }
        if (_hostInfo) {
            _hostInfo->getHistory()->addControlEvent(NHET_STOP, beginTimeUs, autil::TimeUtility::currentTime());
        }
    }
}

void Navi::stop() {
    auto realNavi = getRealNavi();
    if (realNavi) {
        realNavi->stop();
    }
    NAVI_LOG(INFO, "navi stop begin");
    NaviLoggerScope scope(_logger);
    NAVI_LOG(INFO, "navi stop snapshot begin");
    updateSnapshot(nullptr);
    _backgroundThread.reset();
    if (_streamCreator) {
        if (_gigRpcServer) {
            _gigRpcServer->unRegisterGrpcStreamService(_streamCreator);
        }
        _streamCreator->stop();
        _streamCreator.reset();
    }
    _evTimer.reset();
    NAVI_LOG(INFO, "instance stopped");
}

std::string Navi::getTopoInfoStr() const {
    if (_isRealNavi) {
        auto realNavi = getRealNavi();
        if (realNavi) {
            return realNavi->getTopoInfoStr();
        } else {
            return std::string();
        }
    }
    if (!_gigRpcServer) {
        return std::string();
    }
    return _gigRpcServer->getTopoInfoStr();
}

NaviUserResultPtr Navi::runLocalGraph(GraphDef *graphDef,
                                      const RunGraphParams &params,
                                      const ResourceMap &resourceMap)
{
    if (_isRealNavi) {
        return doRunGraph(graphDef, params, &resourceMap);
    } else {
        auto realNavi = getRealNavi();
        if (realNavi) {
            return realNavi->doRunGraph(graphDef, params, &resourceMap);
        } else {
            NAVI_LOG(ERROR, "real navi is null");
            return nullptr;
        }
    }
}

NaviUserResultPtr Navi::runLocalGraphAsync(GraphDef *graphDef,
                                           const RunGraphParams &params,
                                           const ResourceMap &resourceMap,
                                           NaviUserResultClosure *closure)
{
    if (_isRealNavi) {
        return doRunGraphAsync(graphDef, params, &resourceMap, closure);
    }
    auto realNavi = getRealNavi();
    if (realNavi) {
        return realNavi->doRunGraphAsync(
            graphDef, params, &resourceMap, closure);
    }
    multi_call::freeProtoMessage(graphDef);
    NAVI_LOG(ERROR, "real navi is null");
    closure->run(nullptr, nullptr);
    return nullptr;
}

NaviUserResultPtr Navi::runGraph(GraphDef *graphDef,
                                 const RunGraphParams &params)
{
    if (_isRealNavi) {
        return doRunGraph(graphDef, params, nullptr);
    } else {
        auto realNavi = getRealNavi();
        if (realNavi) {
            return realNavi->doRunGraph(graphDef, params, nullptr);
        } else {
            NAVI_LOG(ERROR, "real navi is null");
            return nullptr;
        }
    }
}

NaviUserResultPtr Navi::doRunGraph(GraphDef *graphDef,
                                   const RunGraphParams &params,
                                   const ResourceMap *resourceMap)
{
    auto snapshot = getSnapshot();
    if (!snapshot) {
        multi_call::freeProtoMessage(graphDef);
        NAVI_LOG(WARN, "snapshot not exists, run graph skipped");
        return nullptr;
    }
    return snapshot->runGraph(graphDef, params, resourceMap);
}

NaviUserResultPtr Navi::doRunGraphAsync(GraphDef *graphDef,
                                        const RunGraphParams &params,
                                        const ResourceMap *resourceMap,
                                        NaviUserResultClosure *closure)
{
    auto snapshot = getSnapshot();
    if (!snapshot) {
        multi_call::freeProtoMessage(graphDef);
        NAVI_LOG(WARN, "snapshot not exists, run graph skipped");
        closure->run(nullptr, nullptr);
        return nullptr;
    }
    return snapshot->runGraphAsync(graphDef, params, resourceMap, closure);
}

std::shared_ptr<ModuleManager> Navi::getModuleManager() const {
    auto realNavi = getRealNavi();
    if (realNavi) {
        return realNavi->getModuleManager();
    }
    auto snapshot = getSnapshot();
    if (!snapshot) {
        return nullptr;
    }
    return snapshot->getModuleManager();
}

void Navi::cleanupModule() {
    auto snapshot = getSnapshot();
    if (snapshot) {
        snapshot->cleanupModule();
    }
}

bool Navi::initBackgroundThread() {
    if (!_isRealNavi) {
        return true;
    }
    _backgroundThread = autil::LoopThread::createLoopThread(
            std::bind(&Navi::loop, this),
            10 * 1000, "naviBg");
    if (!_backgroundThread) {
        NAVI_KERNEL_LOG(ERROR, "Create metrics report thread fail!");
        return false;
    }
    return true;
}

bool Navi::initMemoryPoolR() {
    if (_memoryPoolR) {
        return true;
    }
    NaviLoggerScope scope(_logger);
    auto resource = std::make_shared<MemoryPoolR>();
    if (!resource->Init(_metricsReporter)) {
        NAVI_KERNEL_LOG(ERROR, "Create worker level memory pool resource failed");
        return false;
    }
    _memoryPoolR = std::move(resource);
    return true;
}

bool Navi::initEvTimer() {
    if (_evTimer) {
        return true;
    }
    _evTimer = std::make_shared<EvTimer>("navi_timeout");
    if (_evTimer->start()) {
        return true;
    }
    _evTimer.reset();
    NAVI_KERNEL_LOG(ERROR, "start timer thread failed");
    return false;
}

void Navi::setIsRealNavi() {
    _isRealNavi = true;
}

bool Navi::initHostInfo() {
    if (_hostInfo) {
        return true;
    }
    _hostInfo = std::make_shared<NaviHostInfo>();
    NaviLoggerScope scope(_logger);
    return _hostInfo->init(_name, _initConfigPath, _gigRpcServer);
}

void Navi::initInstanceId() {
    _instanceId = 0;
    while (0 == _instanceId) {
        _instanceId = CommonUtil::random64();
    }
}

InstanceId Navi::getInstanceId() const {
    return _instanceId;
}

void Navi::setTestMode(TestMode testMode) {
    _testMode = testMode;
}

void Navi::setWaitNoQuery(bool needWait) {
    _waitNoQuery = needWait;
}

bool Navi::createResource(const std::string &bizName,
                          NaviPartId partCount,
                          NaviPartId partId,
                          const std::set<std::string> &resources,
                          ResourceMap &resourceMap) const
{
    auto realNavi = getRealNavi();
    if (realNavi) {
        return realNavi->createResource(bizName, partCount, partId, resources, resourceMap);
    }
    resourceMap.addResource(_memoryPoolR);
    auto snapshot = getSnapshot();
    if (!snapshot) {
        return false;
    }
    return snapshot->createResource(bizName, partCount, partId, resources,
                                    resourceMap);
}

std::string Navi::getNaviPythonPath() {
    return autil::FileUtil::joinPath(_initConfigPath, "usr/local/lib/python/site-packages/navi");
}

void Navi::loop() {
    auto snapshot = getSnapshot();
    if (snapshot) {
        snapshot->loop();
    }
    if (_memoryPoolR) {
        _memoryPoolR->workLoop();
    }
    if (_metricsReporter) {
        NaviStat stat;
        stat.serviceStatus = (!snapshot || snapshot->isStopping()) ? 0 : 1;
        stat.updatingStatus = _updatingSnapshotStatus.load(std::memory_order_relaxed);
        _metricsReporter->report<NaviStatMetrics>(nullptr, &stat);
    }
}

void Navi::waitUntilNoQuery(bool disableGigAllReplicaStopped) {
    int64_t forceWaitNoQueryTime = EnvParam::inst().forceWaitNoQueryTime;
    NAVI_KERNEL_LOG(INFO, "begin wait until no query, force wait [%ld]s",
                    forceWaitNoQueryTime);
    usleep(forceWaitNoQueryTime * 1000 * 1000);
    if (_gigRpcServer) {
        _gigRpcServer->waitUntilNoQuery(10, std::numeric_limits<int64_t>::max(), disableGigAllReplicaStopped);
    }
    NAVI_KERNEL_LOG(INFO, "end wait until no query");
}

bool Navi::runInit(const NaviConfigPtr &inputConfig) {
    if (getSnapshot()) {
        return true;
    }
    if (!doInit()) {
        return false;
    }
    auto config = std::make_shared<NaviConfig>();
    config->engineConfig.builtinTaskQueue.threadNum = 2;
    config->engineConfig.disablePerf = inputConfig ? inputConfig->engineConfig.disablePerf : true;
    config->engineConfig.disableSymbolTable = inputConfig ? inputConfig->engineConfig.disableSymbolTable : true;
    if (_gigRpcServer) {
        config->resources.insert(NaviStateRpcR::RESOURCE_ID);
    }
    ResourceMap rootResourceMap;
    auto naviStateR = std::make_shared<NaviStateR>();
    naviStateR->_navi = this;
    rootResourceMap.addResource(naviStateR);
    {
        NaviLoggerScope scope(_logger);
        if (!doUpdate("", config, nullptr, nullptr, rootResourceMap, nullptr)) {
            return false;
        }
    }
    return true;
}

bool Navi::runUpdateGraph(const std::string &inputConfigStr,
                          const ResourceMap &rootResourceMapIn,
                          const std::shared_ptr<ModuleManager> &moduleManager,
                          const std::function<bool(std::string &)> &configLoadFunc)
{
    initDefaultLogger();
    _logger.logger->clearFirstErrorEvent();
    if (!initHostInfo()) {
        return false;
    }
    auto configStr = inputConfigStr;
    UpdateHistoryScope historyScope(configStr);
    historyScope.setHistory(_hostInfo->getHistory());
    bool valid = false;
    if (configLoadFunc) {
        if (configLoadFunc(configStr)) {
            historyScope.setConfigStr(configStr);
            valid = true;
        } else {
            NAVI_LOG(ERROR, "load python config failed: [%s]", configStr.c_str());
            historyScope.setError(NaviError::makeError(_logger.logger, EC_ABORT));
            valid = false;
        }
    } else {
        valid = true;
    }
    NaviConfigPtr inputConfig;
    if (valid) {
        inputConfig = std::make_shared<NaviConfig>();
        NaviLoggerScope scope(_logger);
        if (!inputConfig->loadConfig(configStr)) {
            NAVI_LOG(ERROR, "load input config failed: [%s]", configStr.c_str());
            historyScope.setError(NaviError::makeError(_logger.logger, EC_ABORT));
            inputConfig.reset();
        }
    }
    if (!runInit(inputConfig)) {
        historyScope.setError(NaviError::makeError(_logger.logger, EC_ABORT));
        return false;
    }
    if (!inputConfig) {
        auto thisProto = getLastVisProto();
        if (thisProto) {
            auto useProto = std::make_shared<GraphVisDef>();
            useProto->CopyFrom(*thisProto);
            if (useProto->nodes_size() > 0) {
                auto node0 = useProto->mutable_nodes(0);
                node0->set_user_result("navi update failed before load");
                auto naviError = historyScope.getError();
                if (naviError) {
                    naviError->toProto(*node0->mutable_result()->mutable_error());
                }
            }
            _hostInfo->setInitVisProto(useProto);
        }
        return false;
    }
    auto graphDef = new GraphDef();
    GraphBuilder builder(graphDef);
    builder.newSubGraph(NAVI_BUILDIN_BIZ);
    builder.inlineMode(true);
    auto n = builder.node("navi").kernel(NAVI_KERNEL_NAME);
    n.out(NAVI_KERNEL_OUTPUT_PORT).asGraphOutput("navi_output");
    RunGraphParams params;
    params.setCollectMetric(true);
    params.setTraceLevel("INFO");
    ResourceMap resourceMap;
    auto naviR = std::make_shared<NaviR>();
    naviR->_gigRpcServerPtr = _gigRpcServerPtr;
    naviR->_gigRpcServer = _gigRpcServer;
    naviR->_name = _name;
    naviR->_initConfigPath = _initConfigPath;
    naviR->_testMode = _testMode;
    naviR->_waitNoQuery = _waitNoQuery;
    naviR->_metricsReporter = _metricsReporter;
    naviR->_memoryPoolR = _memoryPoolR;
    naviR->_hostInfo = _hostInfo;
    naviR->_evTimer = _evTimer;
    naviR->_config = inputConfig;
    naviR->_rootResourceMap.addResource(rootResourceMapIn);
    naviR->_moduleManager = moduleManager;
    naviR->_navi = getRealNavi();
    {
        auto snapshot = getSnapshot();
        naviR->_naviSymbolTable = snapshot ? snapshot->getNaviSymbolTable() : nullptr;
    }
    resourceMap.addResource(naviR);
    auto naviUserResult = doRunGraph(graphDef, params, &resourceMap);
    if (!naviUserResult) {
        NAVI_LOG(ERROR, "user result is null");
        historyScope.setError(NaviError::makeError(_logger.logger, EC_ABORT));
        return false;
    }
    while (true) {
        NaviUserData data;
        bool eof = false;
        naviUserResult->nextData(data, eof);
        if (eof) {
            break;
        }
    }
    auto naviResult = naviUserResult->getNaviResult();
    if (!naviResult) {
        NAVI_LOG(ERROR, "navi result is null");
        historyScope.setError(NaviError::makeError(_logger.logger, EC_ABORT));
        return false;
    }
    bool ret = true;
    // root vis
    auto thisVisProto = naviResult->getVisProto();
    if (!naviResult->hasError()) {
        auto realNavi = naviR->_navi;
        setRealNavi(realNavi);
        if (!realNavi) {
            NAVI_LOG(ERROR, "init navi failed");
            historyScope.setError(NaviError::makeError(_logger.logger, EC_ABORT));
            ret = false;
        } else {
            ret = true;
        }
    } else {
        historyScope.setError(naviResult->error);
        ret = false;
    }
    {
        auto thisSnapshot = getSnapshot();
        if (thisSnapshot) {
            // add summary to root vis, fill real config
            thisSnapshot->fillSummaryToVis(thisVisProto, configStr);
        }
    }
    if (thisVisProto && thisVisProto->nodes_size() > 0 && naviR->_navi) {
        auto thisDef = thisVisProto->mutable_nodes(0);
        // snapshot vis
        auto realProto = naviR->_navi->getLastVisProto();
        if (realProto && realProto->nodes_size() > 0) {
            auto node0 = realProto->mutable_nodes(0);
            if (node0->result().has_error()) {
                const auto &nodeError = node0->result().error();
                auto naviError = std::make_shared<NaviError>();
                naviError->fromProto(nodeError);
                historyScope.setError(naviError);
                auto parentResult = thisDef->mutable_result();
                parentResult->clear_error();
                parentResult->mutable_error()->CopyFrom(nodeError);
            }
            NaviHostInfo::addVisProtoChild(thisDef, _name + "_snapshot", node0);
        } else {
            // failed before snapshot initBizResource
            realProto = getLastVisProto();
            if (realProto && realProto->nodes_size() > 0) {
                auto node0 = realProto->mutable_nodes(0);
                if (node0->result().has_error()) {
                    auto naviError = std::make_shared<NaviError>();
                    naviError->fromProto(node0->result().error());
                    historyScope.setError(naviError);
                }
                NaviHostInfo::addVisProtoChild(thisDef, "navi_registry", node0);
            }
        }
        std::string userResult = "Navi State Summary\n\n";
        if (_gigRpcServer) {
            auto httpArpcServer = _gigRpcServer->getHttpArpcServer();
            if (httpArpcServer) {
                auto rpcNames = httpArpcServer->getRPCNames();
                userResult += "RPC Methods: \n" + autil::StringUtil::toString(rpcNames, "\n");
            }
        }
        auto bizList = naviR->_navi->getBizList();
        userResult += "\n\nBiz List: \n" + autil::StringUtil::toString(bizList, "\n");
        thisDef->set_user_result(userResult);
    }
    if (thisVisProto) {
        _hostInfo->setInitVisProto(thisVisProto);
    }
    if (TM_NONE == _testMode && naviResult->_logger.logger) {
        naviResult->_logger.logger->stopTraceAppender();
    }
    if (ret) {
        _hostInfo->getHistory()->updateConfig(inputConfig->historyConfig);
    }
    return ret;
}

void Navi::setLastVisProto(const std::shared_ptr<GraphVisDef> &lastVisProto) {
    autil::ScopedWriteLock scope(_snapshotLock);
    _lastVisProto = lastVisProto;
}

std::shared_ptr<GraphVisDef> Navi::getLastVisProto() const {
    autil::ScopedReadLock scope(_snapshotLock);
    return _lastVisProto;
}

std::shared_ptr<NaviHostInfo> Navi::getHostInfo() const {
    return _hostInfo;
}

std::vector<std::string> Navi::getBizList() const {
    auto snapshot = getSnapshot();
    if (snapshot) {
        return snapshot->getBizList();
    } else {
        return {};
    }
}

void Navi::setRealNavi(const NaviPtr &navi) {
    autil::ScopedLock lock(_realNaviLock);
    _realNavi = navi;
}

NaviPtr Navi::getRealNavi() const {
    autil::ScopedLock lock(_realNaviLock);
    return _realNavi;
}

}
