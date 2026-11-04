#pragma once

#include "navi/engine/BizManager.h"
#include "navi/rpc_server/NaviRpcServerR.h"
#include "navi/log/NaviLogManager.h"
#include "navi/log/NaviLogger.h"
#include "navi/ops/ResourceData.h"
#include "navi/perf/NaviPerf.h"
#include "navi/proto/NaviStream.pb.h"
#include "navi/util/NaviClosure.h"
#include "autil/ObjectTracer.h"
#include "autil/LoopThread.h"

namespace kmonitor {
class MetricsReporter;
}

namespace multi_call {
class GigStreamBase;
class GigRpcServer;
}

namespace navi {

class NaviSnapshot;
class NaviGraphRunner;
class NaviThreadPool;
class NaviUserResult;
struct NaviUserData;
class NaviUserResultClosure;
class RunGraphParams;
class NaviSymbolTable;
class TaskQueue;
class NaviThreadPoolItemBase;
class NaviWorkerBase;
class EvTimer;
class NaviResult;
class TaskQueueScheduleItemBase;
class NaviHostInfo;
class MemLeakCollector;
class ResourceErrorInfoCollectorR;

NAVI_TYPEDEF_PTR(NaviSnapshot);

struct InitWaiter {
public:
    InitWaiter() {}
public:
    void setExpectCount(size_t expectCount);
    bool wait(int64_t timeoutUs);
    void notify(const NaviUserData &data, const multi_call::TopoSignatureSetPtr &signatureSet);
    void stop();
    bool waitLocalSubComplete(int64_t timeoutUs, const std::vector<multi_call::SignatureTy> &publishSignatures);

private:
    autil::ThreadCond _collectCond;
    bool _stopped = false;
    bool _finishWaitGigMeta = false;
    size_t _expectCount = 0;
    std::map<size_t, std::pair<std::set<size_t>, multi_call::TopoSignatureSetPtr>> _readyBatchAndMeta;
    std::map<size_t, multi_call::TopoSignatureSetPtr> _eofIndexMeta;
};

class NaviSnapshot : public autil::ObjectTracer<NaviSnapshot, true>
{
private:
    static constexpr int64_t WAIT_TIME = 700 * 1000 * 1000;
public:
    NaviSnapshot(
        const std::shared_ptr<NaviHostInfo> &hostInfo,
        const std::shared_ptr<EvTimer> &evTimer);
    ~NaviSnapshot();
private:
    NaviSnapshot(const NaviSnapshot &);
    NaviSnapshot &operator=(const NaviSnapshot &);
public:
    friend class NaviGraphRunner;
    friend class Navi;
    friend class NaviServerStream;
public:
    bool init(bool isRealNavi,
              const NaviLoggerPtr &defaultLogger,
              const std::shared_ptr<NaviSymbolTable> &naviSymbolTable,
              const std::string &initConfigPath,
              InstanceId instanceId,
              multi_call::GigRpcServer *gigRpcServer,
              const ModuleManagerPtr &moduleManager,
              const NaviSnapshotPtr &oldSnapshot,
              const NaviConfigPtr &config,
              const ResourceMap &rootResourceMap);
    bool start(const NaviSnapshotPtr &oldSnapshot);
    void initMetrics(kmonitor::MetricsReporter &reporter);
    void logSummary(const std::string &stage) const;
    const NaviLoggerPtr &getLogger() const;
    ModuleManagerPtr getModuleManager() const;
    bool reInitSymbolTable();
    void stop();
    void setTestMode(TestMode testMode);
    TestMode getTestMode() const { return _testMode; }
    void cleanupModule();
public:
    std::shared_ptr<NaviUserResult>
    runGraph(GraphDef *graphDef, const RunGraphParams &params, const ResourceMap *resourceMap);
    std::shared_ptr<NaviUserResult> runGraphAsync(
        GraphDef *graphDef,
        const RunGraphParams &params,
        const ResourceMap *resourceMap,
        NaviUserResultClosure *closure);
    bool doRunStreamSession(NaviMessage *request,
                            const ArenaPtr &arena,
                            const RunParams &pbParams,
                            const std::shared_ptr<multi_call::GigStreamBase> &stream,
                            const std::shared_ptr<MemLeakCollector> &memLeakCollector);
    bool createResource(const std::string &bizName,
                        NaviPartId partCount,
                        NaviPartId partId,
                        const std::set<std::string> &resources,
                        ResourceMap &resourceMap);
    const std::shared_ptr<NaviThreadPool> &getDefaultThreadPool() const;
    void runStreamSession(const std::string &taskQueueName,
                          const SessionId &sessionId,
                          int64_t globalStartTimeUs,
                          TaskQueueScheduleItemBase *item,
                          std::shared_ptr<NaviLogger> &logger);
    NaviPerf *getNaviPerf() {
        return &_naviPerf;
    }
    const std::shared_ptr<NaviSymbolTable> &getNaviSymbolTable() const {
        return _naviSymbolTable;
    }
    std::shared_ptr<GraphVisDef> getInitVisProto() const;
    void fillSummaryToVis(const std::shared_ptr<GraphVisDef> &visProto,
                          const std::string &configStr) const;
    void updateProto(NaviSummaryDef &summaryDef) const;
    std::vector<std::string> getBizList() const;
    void setStopping(bool value) {
        _stopping = value;
    }
    bool waitLocalSubComplete();
    bool isStopping() const {
        return _stopping;
    }
    int64_t getGigSmoothIntervalUs() const {
        if (_config) {
            return _config->gigSmoothIntervalUs;
        } else {
            return multi_call::INVALID_SMOOTH_WEIGHT_INTERVAL;
        }
    }
    bool disableGigAllReplicaStopped() const {
        if (_config) {
            return _config->disableGigAllReplicaStopped;
        } else {
            return false;
        }
    }
public:
    void loop();
public:
    static NaviLoggerPtr getTlsLogger();
private:
    void initDefaultLogger();
    bool initLogger(InstanceId instanceId, const NaviLoggerPtr &defaultLogger);
    bool initTaskQueue();
    bool
    initSingleTaskQueue(const std::string &name, const ConcurrencyConfig &config, std::unique_ptr<TaskQueue> &entry);
    bool initBizManager(bool isRealNavi,
                        const ModuleManagerPtr &moduleManager,
                        multi_call::GigRpcServer *gigRpcServer,
                        const NaviSnapshotPtr &oldSnapshot,
                        const ResourceMap &rootResourceMap);
    void addBuildinExternalResource(multi_call::GigRpcServer *gigRpcServer, ResourceMap &rootResourceMap);
    bool initNaviPerf();
    bool initSymbolTable(const std::shared_ptr<NaviSymbolTable> &naviSymbolTable, const NaviSnapshotPtr &oldSnapshot);
    bool initBizResource(const ResourceMap &rootResourceMap);
    void collectInitResource();
    bool collectResultResource(const std::shared_ptr<NaviUserResult> &result,
                               ResourceMap &resourceMap, InitWaiter *waiter);
    NaviObjectLogger getObjectLogger(SessionId id);
    bool parseRunParams(const RunParams &pbParams,
                        RunGraphParams &params);
    void fillSnapshotResource(const RunGraphParams &params, NaviWorkerBase *session);
    void doSchedule(TaskQueue *taskQueue);
    TaskQueue *getTaskQueue(const std::string &taskQueueName);
    bool runGraphImpl(GraphDef *graphDef,
                      const RunGraphParams &params,
                      const ResourceMap *resourceMap,
                      std::shared_ptr<NaviUserResult> &userResult,
                      NaviUserResultClosure *closure);
    void logVisData(const std::shared_ptr<NaviResult> &naviResult) const;
    void writeSummary(const std::string &naviName,
                      const std::string &summaryStr) const;
    std::shared_ptr<NaviUserResult> getInitResult() const;
    void setInitResult(const std::shared_ptr<NaviUserResult> &initResult);
    void fillNaviSummary(NaviSummaryDef &summaryDef) const;
    void updateError(const std::shared_ptr<GraphVisDef> &visProto) const;
private:
    DECLARE_LOGGER();
    std::shared_ptr<NaviHostInfo> _hostInfo;
    NaviConfigPtr _config;
    std::shared_ptr<NaviSymbolTable> _naviSymbolTable;
    NaviLogManagerPtr _logManager;
    std::shared_ptr<kmonitor::MetricsReporter> _metricsReporter;
    std::map<std::string, std::unique_ptr<TaskQueue>> _extraTaskQueueMap;
    std::unique_ptr<TaskQueue> _defaultTaskQueue;
    BizManagerPtr _bizManager;
    autil::ThreadPtr _collectThread;
    std::shared_ptr<ResourceErrorInfoCollectorR> _errorInfoCollectorR;
    mutable autil::ThreadMutex _resultLock;
    std::shared_ptr<NaviUserResult> _initResult;
    InitWaiter _collectWaiter;
    std::shared_ptr<NaviThreadPool> _defaultThreadPool;
    NaviPerf _naviPerf;
    std::shared_ptr<EvTimer> _evTimer;
    TestMode _testMode;
    multi_call::GigRpcServer *_gigRpcServer;
    NaviRpcServerRPtr _naviRpcServer;
    volatile bool _stopping = false;
};

}
