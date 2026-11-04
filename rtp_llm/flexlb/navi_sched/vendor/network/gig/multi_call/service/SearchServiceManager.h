#pragma once
#include <atomic>
#include <functional>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/SubscribeClustersConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Closure.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/WorkerMetricReporter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HeartbeatClientManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ConnectionManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceProvider.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceSnapshot.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceSnapshotInBiz.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/SubscribeServiceManager.h"
#include "autil/Lock.h"
#include "autil/LoopThread.h"
#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"

namespace multi_call {

class SearchService;
class LatencyTimeSnapshot;
MULTI_CALL_TYPEDEF_PTR(LatencyTimeSnapshot);

class SearchServiceManager
{
public:
    SearchServiceManager(SearchService *searchService = nullptr);
    // virtual for ut
    virtual ~SearchServiceManager();

private:
    SearchServiceManager(const SearchServiceManager &);
    SearchServiceManager &operator=(const SearchServiceManager &);

public:
    // virtual for ut
    virtual bool init(const MultiCallConfig &mcConfig);
    SearchServiceSnapshotPtr getSearchServiceSnapshot() const;
    void setMetricReporterManager(const MetricReporterManagerPtr &metricReporterManager) {
        _metricReporterManager = metricReporterManager;
        _subscribeServiceManager->setMetricReporterManager(metricReporterManager);
    }
    void setLatencyTimeSnapshot(const LatencyTimeSnapshotPtr &snapshot) {
        _latencyTimeSnapshot = snapshot;
    }
    void setThisSpec(const Spec &spec);
    void stop();
    void startLogThread();
    void stopLogThread();
    SubscribeServiceManagerPtr getSubscribeServiceManager() const {
        return _subscribeServiceManager;
    }
    void notifyUpdate() const;
    bool setSnapshotChangeCallback(SnapshotChangeCallback *callback);
    void stealSnapshotChangeCallback();
    uint64_t addSnapshotChangeListener(const std::function<void()> &listener);
    void removeSnapshotChangeListener(uint64_t listenerId);

public:
    // for java client
    bool createSnapshot(const TopoNodeVec &topoNodeVec);
    void updateStaticParam();
    const MiscConfigPtr &getMiscConfig() const {
        return _miscConfig;
    }
    bool addSubscribeService(const SubscribeConfig &subServiceConf);
    bool updateSubscribeService(const SubscribeConfig &subServiceConf);
    bool addSubscribe(const SubscribeClustersConfig &gigSubConf);
    bool deleteSubscribe(const SubscribeClustersConfig &gigSubConf);
    const ConnectionManagerPtr &getConnectionManager() const {
        return _connectionManager;
    }

public:
    void reportMetrics();
    virtual void waitCreateSnapshotLoop(int64_t waitTime = UPDATE_TIME_INTERVAL);

private:
    // virtual for ut
    virtual void update();
    virtual bool initConnectionManager(const ConnectionConfig &conf, const MiscConfig &miscConf);
    bool initHeartbeat();
    virtual bool waitCreateSnapshot(int64_t waitTime);
    void setSearchServiceSnapshot(const SearchServiceSnapshotPtr &searchServiceSnapshotPtr);
    bool needReCreateSnapshot();
    void createSnapshot();
    SearchServiceSnapshotPtr constructSnapshot(const BizInfoMap &bizInfoMap,
                                               const SearchServiceSnapshotPtr &oldSnapshot);
    bool getSpecSet(std::set<std::string> &specSet) const;
    void syncConnection() const;
    void logThread();
    void logSnapshot(const SearchServiceSnapshotPtr &newSnapshot);
    void stopUpdateThread();
    void setUpdateThread(const autil::LoopThreadPtr &updateThread);
    autil::LoopThreadPtr getUpdateThread() const;
    void runSnapshotChangeCallback();

private:
    static void constructBizInfoMap(const TopoNodeVec &topoNodeVec, BizInfoMap &bizInfoMap);
    static ConnectionConfig addDefaultGrpcConfig(const ConnectionConfig &conf);

private:
    mutable autil::ReadWriteLock _commonLock;
    mutable autil::ReadWriteLock _snapshotLock;
    autil::ThreadCond _snapshotWaitCond;
    SearchServiceSnapshotPtr _searchServiceSnapshotPtr;
    SubscribeServiceManagerPtr _subscribeServiceManager;
    LatencyTimeSnapshotPtr _latencyTimeSnapshot;
    mutable autil::ThreadMutex _updateThreadLock;
    autil::LoopThreadPtr _updateThreadPtr;
    autil::LoopThreadPtr _snapshotLogThreadPtr;
    MetricReporterManagerPtr _metricReporterManager;
    ConnectionManagerPtr _connectionManager;
    std::string _snapshotThisPrefix;
    std::string _snapshotStr;
    std::string _newSnapshotStr;
    MiscConfigPtr _miscConfig;
    HeartbeatClientManagerPtr _heartbeatClientManager;
    bool _onlyHeartbeatSub;
    VersionTy _heartbeatVersion;
    mutable autil::ThreadMutex _callbackLock;
    SnapshotChangeCallback *_callback;
    std::map<uint64_t, std::function<void()>> _snapshotChangeListeners;
    std::atomic<uint64_t> _nextSnapshotChangeListenerId {0};

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(SearchServiceManager);

} // namespace multi_call
