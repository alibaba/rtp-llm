#pragma once

#include "autil/LoopThread.h"
#include "autil/ObjectTracer.h"
#include "navi/config/NaviConfig.h"
#include "navi/engine/NaviUserResult.h"
#include "navi/engine/ResourceMap.h"
#include "navi/engine/RunGraphParams.h"
#include "navi/log/NaviLogger.h"

namespace multi_call {
class GigRpcServer;
class GigStreamBase;
}

namespace kmonitor {
class MetricsReporter;
}

namespace navi {

class NaviGraphRunner;
class GraphDef;
class RunGraphParams;
class ModuleManager;
class NaviMessage;
class NaviClosure;
class NaviSnapshot;
class NaviServerStreamCreator;
class NaviHostInfo;
class NaviSymbolTable;
class MemoryPoolR;
class EvTimer;

class Navi;
NAVI_TYPEDEF_PTR(Navi);

class Navi : public autil::ObjectTracer<Navi, true>
{
public:
    Navi(const std::string &name = "");
    ~Navi();
private:
    Navi(const Navi &);
    Navi &operator=(const Navi &);
public:
    void initMetrics(kmonitor::MetricsReporter &reporter);
    // init/stop
    bool init(const std::string &initConfigPath,
              multi_call::GigRpcServer *gigRpcServer);
    bool initWithGigRpcServer(
        const std::string &initConfigPath,
        const std::shared_ptr<multi_call::GigRpcServer> &gigRpcServer);
    bool update(const std::string &configLoader,
                const std::string &configPath,
                const std::string &loadParam,
                const ResourceMap &rootResourceMap);
    bool update(const std::string &configStr,
                const ResourceMap &rootResourceMapIn);
    bool update(const NaviConfig &config,
                const ResourceMap &rootResourceMap);
    void stopSnapshot();
    void stop();
    std::string getTopoInfoStr() const;
public:
    // run
    NaviUserResultPtr runLocalGraph(GraphDef *graphDef,
                                    const RunGraphParams &params,
                                    const ResourceMap &resourceMap);
    NaviUserResultPtr runLocalGraphAsync(GraphDef *graphDef,
                                         const RunGraphParams &params,
                                         const ResourceMap &resourceMap,
                                         NaviUserResultClosure *closure);
    NaviUserResultPtr runGraph(GraphDef *graphDef,
                               const RunGraphParams &params);
private:
    void setIsRealNavi();
    bool runInit(const std::shared_ptr<NaviConfig> &inputConfig);
    bool runUpdateGraph(const std::string &configStr,
                        const ResourceMap &rootResourceMapIn,
                        const std::shared_ptr<ModuleManager> &moduleManager,
                        const std::function<bool(std::string &)> &configLoadFunc);
private:
    bool doInit();
    bool doUpdate(const std::string &configStr,
                  const NaviConfigPtr &naviConfig,
                  const NaviLoggerPtr &logger,
                  const std::shared_ptr<NaviSymbolTable> &naviSymbolTable,
                  const ResourceMap &rootResourceMapIn,
                  const std::shared_ptr<ModuleManager> &moduleManager);
    bool initHostInfo();
    bool initBackgroundThread();
    bool initMemoryPoolR();
    bool initEvTimer();
    void initInstanceId();
    bool initRpc();
    void initDefaultLogger();
    void updateSnapshot(const std::shared_ptr<NaviSnapshot> &newSnapshot);
    void cleanupModule();
    std::shared_ptr<NaviSnapshot> getSnapshot() const;
    NaviUserResultPtr doRunGraph(GraphDef *graphDef,
                                 const RunGraphParams &params,
                                 const ResourceMap *resourceMap);
    NaviUserResultPtr doRunGraphAsync(GraphDef *graphDef,
                                      const RunGraphParams &params,
                                      const ResourceMap *resourceMap,
                                      NaviUserResultClosure *closure);
    bool parseRunParams(const RunParams &pbParams, RunGraphParams &params);
    std::string getNaviPythonPath();
    void loop();
    void logSnapshotSummary() const;
    void waitUntilNoQuery(bool disableGigAllReplicaStopped);
    void setLastVisProto(const std::shared_ptr<GraphVisDef> &lastVisProto);
    std::shared_ptr<GraphVisDef> getLastVisProto() const;
    std::shared_ptr<NaviHostInfo> getHostInfo() const;
    std::vector<std::string> getBizList() const;
    void setRealNavi(const NaviPtr &navi);
    NaviPtr getRealNavi() const;
private:
    bool update(const std::string &configLoader,
                const std::string &configPath,
                const std::string &loadParam,
                const ResourceMap &rootResourceMap,
                const std::shared_ptr<ModuleManager> &moduleManager);
    bool update(const std::string &configStr,
                const ResourceMap &rootResourceMapIn,
                const std::shared_ptr<ModuleManager> &moduleManager,
                const std::function<bool(std::string &)> &configLoadFunc);
    std::shared_ptr<ModuleManager> getModuleManager() const;
public:
    InstanceId getInstanceId() const;
    void setTestMode(TestMode testMode);
    void setWaitNoQuery(bool needWait);
    bool createResource(const std::string &bizName,
                        NaviPartId partCount,
                        NaviPartId partId,
                        const std::set<std::string> &resources,
                        ResourceMap &resourceMap) const;
private:
    friend class NaviServerStreamCreator;
    friend class NaviGraphRunner;
    friend class NaviR;
    friend class NaviStateKernel;
    friend class NaviTestServer;
private:
    DECLARE_LOGGER();
    bool _isRealNavi = false;
    mutable autil::ThreadMutex _realNaviLock;
    NaviPtr _realNavi;
    std::string _name;
    std::string _initConfigPath;
    std::shared_ptr<NaviHostInfo> _hostInfo;
    std::shared_ptr<kmonitor::MetricsReporter> _metricsReporter;
    std::shared_ptr<multi_call::GigRpcServer> _gigRpcServerPtr;
    multi_call::GigRpcServer *_gigRpcServer;
    std::shared_ptr<NaviServerStreamCreator> _streamCreator;
    InstanceId _instanceId;
    autil::LoopThreadPtr _backgroundThread;
    std::shared_ptr<MemoryPoolR> _memoryPoolR;
    std::shared_ptr<EvTimer> _evTimer;
    mutable autil::ReadWriteLock _snapshotLock;
    std::shared_ptr<NaviSnapshot> _snapshot;
    std::atomic<bool> _updatingSnapshotStatus = {false};
    TestMode _testMode;
    bool _waitNoQuery = true;
    std::shared_ptr<GraphVisDef> _lastVisProto;
};

}
