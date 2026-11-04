#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/HostHeartbeatInfo.h"
#include "autil/LoopThread.h"

namespace multi_call {

class SearchService;
class SearchServiceSnapshot;
class StringTable;

typedef std::unordered_map<std::string, HostHeartbeatInfoPtr> HostHeartbeatInfoMap;
MULTI_CALL_TYPEDEF_PTR(HostHeartbeatInfoMap);

class HeartbeatClientManager;

class HeartbeatClientManagerNotifier
{
public:
    HeartbeatClientManagerNotifier(HeartbeatClientManager *manager) : _manager(manager) {
        atomic_set(&_version, 0);
    }

public:
    int64_t getVersion() const {
        return atomic_read(&_version);
    }
    void notify() {
        atomic_inc(&_version);
    }
    void notifyHeartbeatSuccess() const;
    void stealManager();

private:
    atomic64_t _version;
    mutable autil::ThreadMutex _managerLock;
    HeartbeatClientManager *_manager;
};

MULTI_CALL_TYPEDEF_PTR(HeartbeatClientManagerNotifier);

class SearchServiceManager;

class HeartbeatClientManager
{
public:
    HeartbeatClientManager(SearchService *searchService, SearchServiceManager *manager);
    ~HeartbeatClientManager();

private:
    HeartbeatClientManager(const HeartbeatClientManager &);
    HeartbeatClientManager &operator=(const HeartbeatClientManager &);

public:
    void setThisSpec(const Spec &spec);
    bool start(const MiscConfigPtr &miscConfig,
               const std::shared_ptr<SearchServiceSnapshot> &heartbeatSnapshot);
    void stop();
    bool updateSpecMap(const HeartbeatSpecMap &heartbeatSpecs);
    void notifyHeartbeatSuccess();
    bool fillBizInfoMap(BizInfoMap &bizInfoMap,
                        multi_call::TopoSignatureSetPtr &localSubSignatures);
    int64_t getVersion() const {
        return _notifier->getVersion();
    }
    void log();
    void getSpecSet(std::set<std::string> &specSet) const;

private:
    bool initThread();
    void loop();
    void setHostMap(const HostHeartbeatInfoMapPtr &newMap);
    HostHeartbeatInfoMapPtr getHostMap() const;
    void updateFlowControlHeartbeat(BizFlowControlInfoMapPtr newFlowControlInfoMap);
    std::string toString() const;

private:
    SearchService *_searchService;
    SearchServiceManager *_manager;
    std::shared_ptr<SearchServiceSnapshot> _heartbeatSnapshot;
    std::shared_ptr<StringTable> _stringTable;
    BizFlowControlInfoMapPtr _flowControlInfoMap;
    mutable autil::ThreadMutex _hostMapLock;
    HostHeartbeatInfoMapPtr _hostMap;
    HeartbeatClientManagerNotifierPtr _notifier;
    autil::LoopThreadPtr _thread;
    MiscConfigPtr _miscConfig;
    std::string _snapshotStr;
    std::shared_ptr<Spec> _thisSpec;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(HeartbeatClientManager);

} // namespace multi_call
