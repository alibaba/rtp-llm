#pragma once

#include "aios/navi_sched/resource/MasterStateR.h"
#include "autil/Log.h"
#include "navi/engine/Navi.h"
#include "navi/engine/NaviUserResult.h"
#include <autil/Lock.h>
#include <atomic>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace multi_call {
class SearchService;
class Spec;
typedef std::shared_ptr<SearchService> SearchServicePtr;
}

namespace autil {
class LoopThread;
}

namespace navi_sched {

class GraphVisRecorder;

class CollectorResultClosure : public navi::NaviUserResultClosure {
public:
    // The notification runs under the completion mutex and must not reenter this closure.
    CollectorResultClosure(const std::shared_ptr<GraphVisRecorder> &recorder,
                           std::function<void()> onCompleted);
    void run(navi::NaviUserResultPtr userResult,
             const std::shared_ptr<navi::ComputeScope> &) override;
    bool isCompleted() const;
    bool hasError() const;
    bool agentStopped() const;
    void waitAndSave() const;

private:
    void consumeToEof(const navi::NaviUserResultPtr &userResult);

private:
    std::shared_ptr<GraphVisRecorder> _recorder;
    std::function<void()> _onCompleted;
    mutable autil::ThreadCond _cond;
    navi::NaviUserResultPtr _userResult;
    bool _completed = false;
    bool _agentStopped = false;
};

class MasterCollectorManager {
public:
    MasterCollectorManager(const navi::NaviPtr &navi, const MasterStateRPtr &masterStateR,
                           const std::string &masterBizName, bool enableVis = false,
                           const std::vector<std::string> &taskQueueNames = {});
    ~MasterCollectorManager();

public:
    bool open(const std::string &agentId, const std::string &agentBizName);
    bool close(const std::string &agentId);
    bool startAutoCollect(const multi_call::SearchServicePtr &searchService);
    void stop();
    size_t size() const;

private:
    struct Collector {
        std::string collectorId;
        std::string agentId;
        std::string agentBizName;
        std::string address;
        std::string directHost;
        int32_t grpcStreamPort;
        bool automatic;
        std::string agentMeta;
        std::shared_ptr<CollectorResultClosure> closure;
        navi::NaviUserResultPtr userResult;
    };

private:
    static std::string makeRouteId(const std::string &bizName,
                                   const multi_call::Spec &spec);
    static bool isAgentOffline(const MasterStateRPtr &masterStateR,
                               const std::string &collectorId);
    static std::unique_ptr<navi::GraphDef> buildAgentCollectorGraph(
        const std::string &masterBizName, const std::string &agentBizName,
        const std::string &directHost, int32_t grpcStreamPort, const std::string &agentId,
        const std::string &collectorId);
    static void abortAndWait(const navi::NaviUserResultPtr &userResult,
                             const std::shared_ptr<CollectorResultClosure> &closure);
    bool openRoute(const std::string &routeId, const std::string &agentId,
                   const std::string &agentBizName, const std::string &address,
                   const std::string &directHost, int32_t grpcStreamPort,
                   const std::string &agentMeta, bool automatic);
    bool closeRoute(const std::string &routeId, bool markOffline);
    std::string makeCollectorId(const std::string &routeId);
    void requestRetry();
    void retryFailedCollectors();
    void refreshTopology();
    void syncRoutes(const std::map<std::string, std::string> &routeMetas);

private:
    navi::NaviPtr _navi;
    MasterStateRPtr _masterStateR;
    std::string _masterBizName;
    std::vector<std::string> _taskQueueNames;
    multi_call::SearchServicePtr _searchService;
    uint64_t _snapshotChangeListenerId = 0;
    std::atomic<bool> _stopping{false};
    std::shared_ptr<std::atomic<bool>> _retryRequested =
        std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<autil::LoopThread> _retryThread;
    mutable autil::ThreadMutex _topologyLock;
    mutable autil::ThreadMutex _mutex;
    std::map<std::string, Collector> _collectors;
    uint64_t _nextCollectorId = 0;
    std::shared_ptr<GraphVisRecorder> _visRecorder;

private:
    AUTIL_LOG_DECLARE();
};

NAVI_TYPEDEF_PTR(MasterCollectorManager);

}
