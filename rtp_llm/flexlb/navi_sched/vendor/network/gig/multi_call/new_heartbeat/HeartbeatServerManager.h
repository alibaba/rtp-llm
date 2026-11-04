#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/GigAgent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/Spec.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/new_heartbeat/ServerTopoMap.h"
#include "autil/LoopThread.h"

namespace multi_call {

class ServerTopoMap;
class HeartbeatServerStream;
class HeartbeatServerStreamQueue;

class HeartbeatServerManager
{
public:
    HeartbeatServerManager();
    ~HeartbeatServerManager();

private:
    HeartbeatServerManager(const HeartbeatServerManager &);
    HeartbeatServerManager &operator=(const HeartbeatServerManager &);

public:
    bool init(const GigAgentPtr &agent, const Spec &spec, size_t streamQueueCount = QUEUE_COUNT);
    void stop();
    bool updateSpec(const Spec &spec);
    bool publish(const ServerBizTopoInfo &info, SignatureTy &signature);
    bool publish(const std::vector<ServerBizTopoInfo> &infoVec,
                 std::vector<SignatureTy> &signatureVec);
    bool publishGroup(PublishGroupTy group, const std::vector<ServerBizTopoInfo> &infoVec,
                      std::vector<SignatureTy> &signatureVec);
    bool updateVolatileInfo(SignatureTy signature, const BizVolatileInfo &info);
    bool unpublish(SignatureTy signature);
    bool unpublish(const std::vector<SignatureTy> &signatureVec);
    void startAllBiz();
    void stopAllBiz();
    bool isAllReplicaStopped() const;
    std::vector<ServerBizTopoInfo> getPublishInfos() const;
    std::string getTopoInfoStr() const;
    void notify();

public:
    void addStream(const std::shared_ptr<HeartbeatServerStream> &stream);
    int64_t getNewId();
    const ServerTopoMapPtr &getServerTopoMap() const;

private:
    bool initLogThread(const std::string &logPrefix, const std::string &snapshotLogDir);
    void logThread();
    bool createTopoMap(const ServerBizTopoInfo &info, SignatureTy &signature,
                       BizTopoMapPtr &topoMapPtr);
    bool createTopoMap(const std::vector<ServerBizTopoInfo> &infoVec,
                       std::vector<SignatureTy> &signatureVec, BizTopoMapPtr &topoMapPtr);
    bool createReplaceTopoMap(PublishGroupTy group, const std::vector<ServerBizTopoInfo> &infoVec,
                              std::vector<SignatureTy> &signatureVec, BizTopoMapPtr &topoMapPtr);
    BizTopoPtr createBizTopo(PublishGroupTy group, const ServerBizTopoInfo &info);
    std::shared_ptr<BizStat> getBizStat(const std::string &bizName, PartIdTy partId);
    const std::shared_ptr<HeartbeatServerStreamQueue> &getQueue();
    void clearBiz();
    bool hasStream() const;

private:
    static constexpr size_t QUEUE_COUNT = 5;

private:
    atomic64_t _idCounter;
    atomic64_t _nextQueueId;
    GigAgentPtr _agent;
    std::shared_ptr<ServerTopoMap> _serverTopoMap;
    std::vector<std::shared_ptr<HeartbeatServerStreamQueue>> _streamQueues;
    std::string _logPrefix;
    std::string _snapshotLogDir;
    autil::LoopThreadPtr _logThread;
    std::string _snapshotStr;
    mutable autil::ThreadMutex _publishLock;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(HeartbeatServerManager);

} // namespace multi_call
