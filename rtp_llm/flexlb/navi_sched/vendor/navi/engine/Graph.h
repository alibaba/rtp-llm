#pragma once

#include "navi/engine/GraphBorder.h"
#include "navi/engine/TimeoutChecker.h"
#include "navi/engine/NaviError.h"
#include "navi/engine/ResourceGraphBuilder.h"
#include "navi/proto/GraphDef.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRpcInfo.h"

namespace navi {

enum GraphFinishType {
    GFT_PARENT_FINISH,
    GFT_THIS_FINISH,
    GFT_SUB_FINISH,
};

class SubGraphBase;
class GraphDomainFork;
class Node;
class ResourceGraphBuilder;
class PartInfoDefGraphMap;

class Graph
{
public:
    Graph(GraphParam *param, Graph *parent, bool rootGraph);
    ~Graph();
private:
    Graph(const Graph &);
    Graph &operator=(const Graph &);
public:
    ErrorCode init(GraphDef *graphDef, bool keepGraphDef = false);
    ErrorCode run();
    void notifyFinishEc(ErrorCode ec);
    void notifyFinish(const NaviErrorPtr &naviError);
    void notifyFinish(const NaviErrorPtr &error, GraphFinishType type);
    void notifyTimeout(bool notifyParent);
    bool checkTimeout(bool notifyParent);
    void setGraphDomain(GraphId graphId, const GraphDomainPtr &domain);
    void setForkInfo(GraphId graphId, const GraphDomainPtr &domain,
                     Node *forkNode, bool errorAsEof);
    NaviErrorPtr getError(ErrorCode ec) const;
    GraphParam *getGraphParam() const;
    void setResourceMap(const ResourceMap *resourceMap);
    void setSubResourceMap(const std::map<GraphId, ResourceMapPtr> &subResourceMaps);
    const std::string &getLocalBizName() const;
    NaviPartId getLocalPartId() const;
    void setErrorAsEof(bool errorAsEof);
    bool isTimeout() const;
    int64_t getRemainTimeMs() const;
    bool updateTimeoutMs(int64_t timeoutMs);
    bool refreshTimeout();
    const TimeoutChecker *getTimeoutChecker() const;
    GraphMetric *getGraphMetric() {
        return _param->graphResult->getGraphMetric();
    }
    Node *getForkNode() const {
        return _forkNode;
    }
    GraphDef *getCopyGraphDef() const {
        return _copyGraphDef;
    }
    void finishResult();
    bool needCollect() {
        return _param->runParams.getCollectFlags().needCollect();
    }
    void collectMetric();
private:
    size_t calcSubGraphKey(const SubGraphDef &subGraph);
    ErrorCode doInit(GraphDef *graphDef, bool keepGraphDef);
    ErrorCode initGraphDef(GraphDef *graphDef, bool keepGraphDef);
    ErrorCode rewrite(ResourceGraphBuilderPtr &builder, SubGraphDef *subGraphDef);
    RewriteInfoPtr prepareRewriteInfo(GraphId graphId, SubGraphDef *subGraphDef, NaviPartId partCount, NaviPartId partId);
    ErrorCode addResourceNode(SubGraphDef *subGraphDef, ResourceGraphBuilder &builder, const RewriteInfo &rewriteInfo);
    bool initGraphInlineResource();
    bool preInitGraphDomains(GraphDef *graphDef);
    bool postInitGraphDomains();
    bool validOverrideData() const;
    bool runGraphDomains();
    GraphDomainPtr createGraphDomain(GraphId graphId, const LocationDef &location);
    GraphDomainPtr createUserDomain(const std::string &bizName);
    GraphDomainPtr createLocalDomain(const std::string &bizName);
    GraphDomainPtr createRemoteDomain(const std::string &bizName);
    void bindGraphDomain(const GraphDomainPtr &domain,
                         SubGraphDef *subGraphDef);
    ErrorCode createSubGraphs(GraphDef *graphDef);
    SubGraphBase *createSubGraph(SubGraphDef *subGraphDef,
                                 const GraphDomainPtr &domain);
    bool validateInputResource(const BizPtr &biz) const;
    bool fillInputResource(GraphId graphId, const ResourceMapPtr &resourceMap);
    bool tryInitThisPartIdFromPartIds(SubGraphDef *subGraphDef,
                                      bool &isLocalGraph,
                                      NaviPartId &partCount,
                                      NaviPartId &partId);
    void addSubGraph(SubGraphBase *subGraph);
    ErrorCode initNaviUserResult();
    void initTimeoutChecker();
    bool initTimer();
    void stopTimer();
    bool initBorder();
    void terminateAll(ErrorCode ec);
    bool isLocal(const LocationDef &location) const;
    GraphDomainPtr getGraphDomain(GraphId graphId) const;
    GraphDomainFork *getForkDomain() const;
    BizPtr getBiz(const SubGraphDef &subGraphDef);
    BizPtr doGetBiz(const SubGraphDef &subGraphDef);
    ResourceMapPtr getSubResourceMap(GraphId graphId) const;
    bool doCollectMetric();
private:
    static const std::string &getBizName(const SubGraphDef &subGraphDef);
    static void initThisPartId(SubGraphDef *subGraphDef);
    static void timeoutCallback(void *data, bool isTimeout);
protected:
    DECLARE_LOGGER();
    std::unique_ptr<PartInfoDefGraphMap> _partInfoDefGraphMap;
    GraphParam *_param;
    GraphBorderPtr _border;
    NaviUnorderedMap<GraphId, RewriteInfoPtr> _rewriteInfoMap;
    GraphInlineBuildMapPtr _graphInlineBuildMap;
    ResourceMapPtr _inlineResourceMap;
    std::vector<SubGraphBase *> _subGraphs;
    std::set<GraphDomainPtr> _domainSet;
    GraphDomainMap _graphDomainMap;
    NaviUnorderedMap<SubGraphDef *, GraphDomainPtr> _subGraphDomainMap;
    Graph *_parent;
    Node *_forkNode;
    bool _errorAsEof;
    autil::RecursiveThreadMutex _terminateLock;
    std::atomic<bool> _terminated;
    autil::RecursiveThreadMutex _metricCollectLock;
    bool _metricCollected; // need_restore
    autil::ThreadMutex _timerLock;
    int64_t _timerRefreshTriggeredVersion = 0;
    atomic64_t _timerRefreshVersion;
    bool _timerStopped;
    bool _rootGraph;
    Timer *_timer;
    TimeoutChecker _timeoutChecker;
    ResourceMap _inputResourceMap;
    std::map<GraphId, ResourceMapPtr> _subResourceMaps;
    GraphDef *_copyGraphDef = nullptr;
private:
    std::string _localBizName;
    NaviPartId _localPartId = INVALID_NAVI_PART_ID;
};

}
