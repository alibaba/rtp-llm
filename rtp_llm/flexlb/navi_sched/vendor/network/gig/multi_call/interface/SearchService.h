#pragma once

#include <functional>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/VersionInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/FlowControlConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/SubscribeClustersConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/BizNodeInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Closure.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Reply.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/RequestGenerator.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigClientStream.h"
#include "autil/legacy/json.h"

namespace multi_call {

class BizInfo;
struct CallParam;
typedef CallParam SearchParam;
class SearchServiceSnapshot;
class SearchServiceManager;
MULTI_CALL_TYPEDEF_PTR(SearchServiceManager);
class MetricReporterManager;
MULTI_CALL_TYPEDEF_PTR(MetricReporterManager);
class CallDelegationThread;
MULTI_CALL_TYPEDEF_PTR(CallDelegationThread);
class VersionSelectorFactory;
MULTI_CALL_TYPEDEF_PTR(VersionSelectorFactory);
class ProviderSelectorFactory;
MULTI_CALL_TYPEDEF_PTR(ProviderSelectorFactory);
class FlowConfigSnapshot;
MULTI_CALL_TYPEDEF_PTR(FlowConfigSnapshot);

class QuerySession;
MULTI_CALL_TYPEDEF_PTR(QuerySession);
class RetryLimitChecker;
MULTI_CALL_TYPEDEF_PTR(RetryLimitChecker);
class LatencyTimeSnapshot;
MULTI_CALL_TYPEDEF_PTR(LatencyTimeSnapshot);

class SearchService : public std::enable_shared_from_this<SearchService>
{
public:
    SearchService();
    virtual ~SearchService();

private:
    SearchService(const SearchService &);
    SearchService &operator=(const SearchService &);

public:
    virtual bool init(const autil::legacy::json::JsonMap &mcConfig);
    virtual bool init(const MultiCallConfig &mcConfig);

    virtual void search(const SearchParam &param, ReplyPtr &reply);
    virtual void search(const SearchParam &param, ReplyPtr &reply,
                        const QuerySessionPtr &querySession);

public: // config update
    virtual bool updateFlowConfig(const std::string &flowConfigStrategy,
                                  const autil::legacy::json::JsonMap *flowConf);
    virtual bool updateFlowConfig(const std::string &flowConfigStrategy,
                                  const FlowControlConfigPtr &flowControlConfig);
    virtual void
    updateFlowConfig(const std::map<std::string, FlowControlConfigPtr> &flowControlConfigMap);
    virtual bool updateDefaultFlowConfig(const FlowControlConfigPtr &flowControlConfig);

    
    
    virtual const MultiCallConfig &getMultiCallConfig() const {
        return _config;
    }
    FlowControlConfigPtr getFlowConfig(const std::string &flowConfigStrategy);
    void setThisSpec(const Spec &spec);

public: // subscribe
    bool addSubscribe(const SubscribeClustersConfig &gigSubConf);
    bool deleteSubscribe(const SubscribeClustersConfig &gigSubConf);

public: // biz & version & cluster
    virtual std::vector<std::string> getBizNames() const;
    virtual BizNodeInfoVec getBizNodeInfos(const std::string &bizName) const;
    virtual int32_t getBizPartCount(const std::string &bizName) const;
    bool hasCluster(const std::string &clusterName) const;
    virtual bool hasBiz(const std::string &bizName) const;
    virtual bool hasBiz(const std::string &clusterName, const std::string &bizName) const;
    bool hasVersion(const std::string &bizName, VersionTy version, VersionInfo &info) const;
    bool hasVersion(const std::string &clusterName, const std::string &bizName, VersionTy version,
                    VersionInfo &info) const;
    virtual std::vector<VersionTy> getBizVersion(const std::string &bizName) const;
    virtual std::set<VersionTy> getBizProtocalVersion(const std::string &bizName) const;
    bool getBizInfos(const std::string &bizName, std::vector<BizInfo> &bizInfos) const;
    bool getBizMetaInfos(SourceIdTy sourceId, bool singlePartMeta,
                         std::vector<BizMetaInfo> &bizMetaInfos,
                         multi_call::TopoSignatureSetPtr &localSubSignatures) const;
    std::string getAllFlowControlConfig();

public: // call in gig
    FlowConfigSnapshotPtr getFlowConfigSnapshot();
    std::shared_ptr<SearchServiceSnapshot> getSearchServiceSnapshot();
    const std::shared_ptr<VersionSelectorFactory> &getVersionSelectorFactory() const {
        return _versionSelectorFactory;
    }
    const std::shared_ptr<ProviderSelectorFactory> &getProviderSelectorFactory() const {
        return _providerSelectorFactory;
    }
    const RetryLimitCheckerPtr &getRetryLimitChecker() const {
        return _retryLimitChecker;
    }
    const LatencyTimeSnapshotPtr &getLatencyTimeSnapshot() const {
        return _latencyTimeSnapshot;
    }
    
    const std::shared_ptr<SearchServiceManager> &getSearchServiceManager() const {
        return _serviceManager;
    }
    const std::shared_ptr<MetricReporterManager> &getMetricReporterManager() const {
        return _metricReporterManager;
    }
    void disableSnapshotLog();

public:
    virtual bool restartCallThread(int64_t processInterval,
                                   uint64_t queueSize = std::numeric_limits<uint64_t>::max());
    virtual int64_t getCallProcessInterval() const;
    void waitCreateSnapshotLoop();
    bool bind(const GigClientStreamPtr &stream);
    bool bind(const GigClientStreamPtr &stream, const QuerySessionPtr &session);
    bool bind(const GigClientStreamPtr &stream,
              const std::shared_ptr<SearchServiceSnapshot> &searchSnapshot);
    bool setSnapshotChangeCallback(SnapshotChangeCallback *callback);
    void stealSnapshotChangeCallback();
    uint64_t addSnapshotChangeListener(const std::function<void()> &listener);
    void removeSnapshotChangeListener(uint64_t listenerId);

private:
    bool bind(const GigClientStreamPtr &stream, const QuerySessionPtr &session,
              const std::shared_ptr<SearchServiceSnapshot> &searchSnapshot);
    void setFlowConfigSnapshot(const FlowConfigSnapshotPtr &flowConfigSnapshot);
    void stop();

private:
    friend class HeartbeatClientManager;
    void
    updateFlowControlHeartbeat(const std::shared_ptr<BizFlowControlInfoMap> &flowControlInfoMap);

private:
    MultiCallConfig _config;

    autil::ReadWriteLock _snapshotLock;
    LatencyTimeSnapshotPtr _latencyTimeSnapshot;

    mutable autil::ThreadMutex _flowConfigUpdateLock;
    FlowConfigSnapshotPtr _flowConfigSnapshot;

    SearchServiceManagerPtr _serviceManager;
    MetricReporterManagerPtr _metricReporterManager;
    CallDelegationThreadPtr _callThread;
    RetryLimitCheckerPtr _retryLimitChecker;
    VersionSelectorFactoryPtr _versionSelectorFactory;
    ProviderSelectorFactoryPtr _providerSelectorFactory;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(SearchService);

} // namespace multi_call
