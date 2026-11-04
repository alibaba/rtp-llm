#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/VersionInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/WorkerNodeInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/MetricReporterManager.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/ReplyInfoCollector.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/FlowConfigSnapshot.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceProvider.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceResource.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceSnapshotInBiz.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/RandomGenerator.h"

namespace multi_call {

typedef std::pair<SearchServiceProviderPtr, SearchServiceReplicaPtr> ProviderReplicaPair;
typedef std::unordered_map<std::string, ProviderReplicaPair> ProviderMap;
typedef std::unordered_map<std::string, std::vector<SearchServiceProviderPtr>> ProviderAddressMap;
typedef std::map<std::string, SearchServiceSnapshotInVersionPtr> BizVersionSnapshotMap;
typedef std::map<std::string, std::vector<SearchServiceSnapshotInVersionPtr>>
    BizMultiVersionSnapshotMap;

class SearchServiceSnapshot;
MULTI_CALL_TYPEDEF_PTR(SearchServiceSnapshot);

class LatencyTimeSnapshot;
MULTI_CALL_TYPEDEF_PTR(LatencyTimeSnapshot);

class SearchServiceSnapshot
{
public:
    SearchServiceSnapshot(const ConnectionManagerPtr &connectionManager,
                          const MetricReporterManagerPtr &metricReporterManager,
                          const MiscConfigPtr &miscConfig);
    virtual ~SearchServiceSnapshot();

private:
    SearchServiceSnapshot(const SearchServiceSnapshot &);
    SearchServiceSnapshot &operator=(const SearchServiceSnapshot &);

public:
    bool init(const BizInfoMap &bizInfoMap, const SearchServiceSnapshotPtr &oldSnapshot);

    void getSpecSet(std::set<std::string> &specSet) const;
    const ProviderMap &getProviderMap() const;
    ProviderAddressMap &getAddressProviderMap(); // may update provider
    const BizSnapshotMap &getBizSnapshotMap() const;
    const MiscConfigPtr &getMiscConfig() const {
        return _miscConfig;
    }
    const ConnectionManagerPtr &getConnectionManager() const {
        return _connectionManager;
    }
    const std::vector<std::string> &getBizNames() const;
    bool hasCluster(const std::string &clusterName) const;
    bool hasBiz(const std::string &bizName) const;
    bool hasVersion(const std::string &bizName, VersionTy version, VersionInfo &info) const;
    void getBizMetaInfos(SourceIdTy sourceId, bool singlePartMeta,
                         std::vector<BizMetaInfo> &bizMetaInfos) const;
    void fillSnapshotInfo(SnapshotInfoCollector &collector);
    SourceIdTy getRandomSourceId();
    void toString(std::string &debugStr);
    SearchServiceSnapshotInBizPtr getBizSnapshot(const std::string &bizName);
    SearchServiceSnapshotInVersionPtr getBizVersionSnapshot(const std::string &bizName,
                                                            const VersionTy version);
    std::vector<VersionTy> getBizVersion(const std::string &bizName) const;
    std::set<VersionTy> getBizProtocalVersion(const std::string &bizName) const;
    bool disableBizNotExistLog() const;
    SearchServiceProviderPtr getProvider(const std::string &nodeId) const;
    void setLatencyTimeSnapshot(const LatencyTimeSnapshotPtr &snapshot) {
        _latencyTimeSnapshot = snapshot;
    }
    void setLocalSubSignatures(const multi_call::TopoSignatureSetPtr &signatures) {
        _localSubSignatures = signatures;
    }
    const multi_call::TopoSignatureSetPtr &getLocalSubSignatures() const {
        return _localSubSignatures;
    }

private:
    // virtual for ut
    virtual void getDiffBiz(const BizInfoMap &bizInfoMap, BizInfoMap &newBizInfoMap,
                            BizSnapshotMap &keepBizSnapshot);
    virtual bool constructFromBizMap(const BizInfoMap &newBizInfoMap,
                                     const ProviderMap &oldProviderMap);
    virtual void addBizSnapshot(const BizSnapshotMap &keepBizSnapshot,
                                const ProviderMap &oldProviderMap);
    bool addProvider(const TopoNode &topoNode, const ProviderMap &oldProviderMap);
    SearchServiceProviderPtr createProvider(const TopoNode &topoNode,
                                            const ProviderMap &oldProviderMap);
    bool constructConsistentHash();
    void finishCreateSnapshot(const BizInfoMap &bizInfoMap) const;
    void fillSortedBizNames();

private:
    void updateProviderStatus(const SearchServiceProviderPtr &provider, const TopoNode &topoNode);
    void updateProviderFromOld(const SearchServiceProviderPtr &oldProvider,
                               const SearchServiceProviderPtr &newProvider);

private:
    ProviderMap _providerMap;
    ProviderAddressMap _providerAddressMap;
    BizSnapshotMap _bizSnapshotMap;
    ConnectionManagerPtr _connectionManager;
    MetricReporterManagerPtr _metricReporterManager;
    LatencyTimeSnapshotPtr _latencyTimeSnapshot;
    MiscConfigPtr _miscConfig;
    RandomGenerator _randomGenerator;
    std::set<std::string> _clusterSet;
    std::vector<std::string> _bizNames;
    multi_call::TopoSignatureSetPtr _localSubSignatures;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call
