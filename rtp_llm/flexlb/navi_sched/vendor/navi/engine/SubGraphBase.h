#pragma once

#include "navi/common.h"
#include "navi/engine/Biz.h"
#include "navi/engine/GraphDomain.h"
#include "navi/engine/GraphParam.h"
#include "navi/engine/NaviWorkerBase.h"
#include "navi/engine/Port.h"
#include "navi/engine/SubGraphBorder.h"
#include "navi/engine/ReplaceInfoMap.h"
#include <atomic>

namespace navi {

struct LocalSubGraphInternal;

struct RewriteInfo {
    BizPtr biz;
    ResourceMapPtr resourceMap;
    ResourceMapPtr bizResourceMap;
    MultiLayerResourceMap multiLayerResourceMap;
    ReplaceInfoMapPtr replaceInfoMap;
    bool needCacheGraph = false;
    std::shared_ptr<LocalSubGraphInternal> cacheGraph;
    size_t graphKey = 0;
    size_t nodeCountBeforeAddResource = 0;
};

NAVI_TYPEDEF_PTR(RewriteInfo);

class Graph;

class SubGraphBase
{
public:
    SubGraphBase(SubGraphDef *subGraphDef,
                 GraphParam *param,
                 const GraphDomainPtr &domain,
                 const SubGraphBorderMapPtr &borderMap,
                 Graph *graph);
    virtual ~SubGraphBase();
private:
    SubGraphBase(const SubGraphBase &);
    SubGraphBase &operator=(const SubGraphBase &);
public:
    virtual ErrorCode init() = 0;
    virtual ErrorCode run() = 0;
    virtual int64_t checkTimeout(int64_t endTimeMs) {
        return -1;
    }
    // graph specialized terminate logic
    virtual void collectMetric(GraphMetric *graphMetric) {
    }
    virtual void collectForkGraphMetric() {
    }
    virtual void terminate(ErrorCode ec) = 0;
public:
    void terminateThis(ErrorCode ec);
    GraphId getGraphId() const;
    void setErrorCode(ErrorCode ec);
    Graph *getGraph() const;
    GraphParam *getParam() const;
    NaviWorkerBase *getWorker() const;
    bool terminated() const;
    const std::string &getBizName() const;
    NaviPartId getPartCount() const;
    NaviPartId getPartId() const;
public:
    static const std::string &getNodeName(SubGraphBase *graph,
                                          const NodePortDef &nodePortDef);

protected:
    BizPtr getBiz(const SubGraphDef *subGraphDef);
    ResourceMapPtr getResourceMap(const SubGraphDef *subGraphDef) const;
private:
    // do not call this directly, use notifyFinish instead
    void terminateAll(ErrorCode ec);
protected:
    DECLARE_LOGGER();
    SubGraphDef *_subGraphDef;
    GraphParam *_param;
    GraphDomainPtr _domain;
    SubGraphBorderMapPtr _borderMap;
    Graph *_graph;
    std::atomic<bool> _terminated;
};

}
