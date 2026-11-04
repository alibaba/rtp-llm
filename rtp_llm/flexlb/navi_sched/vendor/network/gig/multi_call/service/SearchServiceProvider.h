#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ControllerChain.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ControllerFeedBack.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CallBack.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/Connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ConnectionManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/DiffCounter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/RandomGenerator.h"
#include "autil/Lock.h"

namespace multi_call {

enum TagMatchLevel {
    TML_PREFER = 0,
    TML_REQUIRE = 1,
    TML_NOT_MATCH = 2,
};

struct TagInfo {
    VersionTy version;
    std::string date;
};

typedef std::unordered_map<std::string, TagInfo> TagInfoMap;
MULTI_CALL_TYPEDEF_PTR(TagInfoMap);

class HostHeartbeatStats;

class SearchServiceProvider
{
public:
    SearchServiceProvider(const ConnectionManagerPtr &connectionManager, const TopoNode &topoNode);
    virtual ~SearchServiceProvider();

private:
    SearchServiceProvider(const SearchServiceProvider &);
    SearchServiceProvider &operator=(const SearchServiceProvider &);

public:
    // virtual for ut
    virtual bool post(const RequestPtr &request, const CallBackPtr &callBack);
    WeightTy getWeight() const;
    WeightTy getTargetWeight() const;
    void setTargetWeight(WeightTy targetWeight, bool updateMeta = true);
    void setWeight(float weight);
    // virtual for ut
    virtual void updateWeight(ControllerFeedBack &feedBack);
    void updateLoadBalance(ControllerFeedBack &feedBack, bool updateWeight);
    ControllerChain *getControllerChain() {
        return &_controllerChain;
    }
    const TopoNodeMeta &getTopoNodeMeta() const {
        return _nodeMeta;
    }
    const NodeAttributeMap &getNodeAttributeMap() const {
        return _nodeMeta.attributes;
    }
    bool isHealth() const;
    bool isStarted() const;
    void updateTagsFromMap(const TagMap &tags);
    TagInfoMapPtr getTags() const;
    void updateHeartbeatMetas(const MetaMapPtr &metas);
    MetaMapPtr getHeartbeatMetas() const;
    void fillTagMaxVersion(MatchTagMap &matchTagMap);
    TagMatchLevel matchTags(const MatchTagMapPtr &matchTagMap) const;
    bool needProbe(WeightTy probePercent, RequestType &type);
    std::string getAnetSpec(ProtocolType type);
    std::string getSpecStr(ProtocolType type);
    void toString(const ControllerChain *bestChain, std::string &debugStr) const;
    void collectSpecs(std::set<std::string> &keepSpecs) const;
    const std::string &getNodeId() const {
        return _nodeId;
    }
    int64_t getTotalQueryCounter() const {
        return _totalNormalCounter.current() + _totalRetryCounter.current() +
               _totalProbeCounter.current();
    }
    void setClusterName(const std::string &clusterName) {
        _clusterName = clusterName;
    }
    const std::string &getServerBizName() const {
        return _serverBizName;
    }
    VersionTy getProtocalVersion() const {
        return _protocalVersion;
    }
    bool hasServerAgent() const {
        return _controllerChain.latencyController.hasServerAgent();
    }
    void setHasServerAgent(bool hasServerAgent) {
        _controllerChain.degradeRatioController.setHasServerAgent(hasServerAgent);
        _controllerChain.latencyController.setHasServerAgent(hasServerAgent);
        _controllerChain.errorRatioController.setHasServerAgent(hasServerAgent);
    }
    void updateNetLatency(int64_t netLatencyUs) {
        _controllerChain.latencyController.updateNetLatencyFromHeartbeat(netLatencyUs);
    }
    void updateClockDrift(int64_t drift) {
        _clockDrift.update(drift);
    }
    float getNetLatency() const {
        return _controllerChain.latencyController.netLatency();
    }
    void updateValidState(bool isValid);
    const Spec &getSpec() const {
        return _spec;
    }
    uint64_t getAgentId() const {
        return _agentId;
    }
    float serverValueDelayMs() const {
        return _controllerChain.latencyController.serverValueDelayMs();
    }
    const HbInfoPtr &getHeartbeatInfo() const;
    bool supportHeartbeat() const;
    ProtocolType getHeartbeatProtocol() const;
    std::string getHeartbeatSpec() const;
    void updateHeartbeatInfo(const HbMetaInfoPtr &hbMetaPtr);
    void updateHeartbeatTime(int64_t time);
    bool isHeartbeatHealth() const;
    void setSubscribeType(SubscribeType ssType) {
        _ssType = ssType;
    }
    MetaEnv getNodeMetaEnv() {
        autil::ScopedReadLock scope(_metaLock);
        return _targetMetaEnv;
    }
    void setNodeMetaEnv(const MetaEnv &metaEnv) {
        autil::ScopedWriteLock scope(_metaLock);
        _targetMetaEnv = metaEnv;
    }
    void updateRequestCounter(bool isRetry, RequestType requestType);
    void updateResponseCounter();
    void setHostStats(const std::shared_ptr<HostHeartbeatStats> &hostStats);
    void incStreamQueryCount();
    void incStreamResponseCount();

public:
    static bool isBetterThan(const ControllerChain *thisChain, const ControllerChain *bestChain,
                             const MetricLimits &metricLimits);
    static void updateStaticParam(const MiscConfigPtr &miscConfig);

private:
    float approachTargetWeight(float targetWeight);
    void updateWeightFactor(int64_t version, float newFactor);
    void updateCachedWeight(float minWeight = MIN_WEIGHT_FLOAT);
    ConnectionPtr getConnection(ProtocolType type);
    void doUpdateWeight(float diff, float minWeight = MIN_WEIGHT_FLOAT);
    void updateWeightBonus(float diff);
    void setCopy(bool isCopy) {
        _isCopy = isCopy;
    }
    bool isCopy() const {
        return _isCopy;
    }
    bool controllerValid() const;
    void updateTags(const TagInfoMapPtr &tags);
    std::shared_ptr<HostHeartbeatStats> getHostStats() const {
        autil::ScopedReadWriteLock lock(_heartbeatLock, 'r');
        return _hostStats;
    }
    std::string getAllProtocolStr() const;

private:
    static bool isLegal(const ControllerChain *candidateChain, const ControllerChain *baseChain,
                        const MetricLimits &metricLimits);
    static bool thisIsBest(const ControllerChain &thisChain, const ControllerChain &otherChain,
                           bool errorFirst);
    static std::string getCurrentTimeStrUs();
    static std::string getTimeStrUs(time_t second, int64_t usec);
    static std::string getTagVersionStr(int64_t version, int64_t currentTime);

private:
    // for ut
    int64_t getQueryCounter() const {
        return _controllerChain.latencyController.count();
    }
    void setQueryCounter(int64_t counter) {
        _controllerChain.latencyController.setFilterCounter(counter);
        _controllerChain.errorRatioController.setFilterCounter(counter);
        _controllerChain.degradeRatioController.setFilterCounter(counter);
    }
    void setAvgLatency(float latency) {
        _controllerChain.latencyController.setCurrent(latency);
    }
    float getAvgLatency() const {
        return _controllerChain.latencyController.controllerOutput();
    }

private:
    RandomGenerator _randomGenerator;

    ControllerChain _controllerChain;
    mutable DiffCounter _totalNormalCounter;
    mutable DiffCounter _totalRetryCounter;
    mutable DiffCounter _totalProbeCounter;
    mutable DiffCounter _totalCopyCounter;
    mutable DiffCounter _totalStreamQueryCounter;
    mutable DiffCounter _totalStreamResponseCounter;
    Filter _clockDrift;

    float _weightBonus;
    volatile FloatIntUnion _weight;
    volatile WeightTy _weightCached;
    volatile bool _useDefaultProbePercent;
    volatile float _loadBalanceWeightFactor;
    volatile atomic64_t _previousServerVersion;

    ConnectionPtr _connection[MC_PROTOCOL_UNKNOWN + 1];
    mutable autil::ReadWriteLock _connectionLock;
    ConnectionManagerPtr _connectionManager;
    Spec _spec;
    std::string _nodeId;
    uint64_t _agentId;
    std::string _clusterName;
    std::string _serverBizName;
    VersionTy _protocalVersion;
    bool _isCopy;
    bool _isValid;
    TopoNodeMeta _nodeMeta;
    HbInfoPtr _hbInfo;
    SubscribeType _ssType;
    autil::ReadWriteLock _metaLock;
    MetaEnv _targetMetaEnv;
    mutable autil::ReadWriteLock _heartbeatLock;
    TagInfoMapPtr _tags;
    MetaMapPtr _metas;
    std::string _subscribeTime;
    std::shared_ptr<HostHeartbeatStats> _hostStats;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(SearchServiceProvider);

typedef std::vector<SearchServiceProviderPtr> SearchServiceProviderVector;

bool operator==(const SearchServiceProvider &lhs, const SearchServiceProvider &rhs);

} // namespace multi_call
