#pragma once

#include "navi/proto/GraphVis.pb.h"
#include "navi/engine/CollectFlags.h"
#include "navi/engine/GraphMetric.h"
#include "navi/engine/GraphResultBase.h"
#include "navi/engine/RpcGraphResult.h"
#include "navi/engine/StringHashTable.h"
#include "navi/log/TraceCollector.h"
#include "navi/engine/NaviResult.h"

namespace navi {

class NaviMessage;
class NaviHostInfo;
class Graph;
class GraphDomainUser;
class TopoNodeParam;

class SerializedResult {
public:
    SerializedResultDef resultDef;
};
NAVI_TYPEDEF_PTR(SerializedResult);

class GraphResult : public GraphResultBase
{
public:
    GraphResult();
    virtual ~GraphResult();
private:
    GraphResult(const GraphResult &) = delete;
    GraphResult &operator=(const GraphResult &) = delete;
public:
    void init(const NaviLoggerPtr &logger,
              const CollectFlags &collectFlags,
              const std::shared_ptr<TopoNodeParam> &topoParam,
              const std::string &formatPattern,
              const NaviSymbolTablePtr &naviSymbolTable,
              const NaviHostInfo *hostInfo);
    void setGraph(Graph *graph) {
        _graph = graph;
    }
    void setGraphDef(GraphDef *graphDef);
    GraphMetric *getGraphMetric() {
        return &_graphMetric;
    }
    const NaviHostInfo *getHostInfo() const {
        return _hostInfo;
    }
    void finishResult(const std::shared_ptr<GraphDomainUser> &userDomain);
    NaviErrorPtr makeError(const NaviLoggerPtr &logger, ErrorCode ec) const;
    NaviErrorPtr updateError(const NaviErrorPtr &error);
    void setError(const NaviLoggerPtr &logger, ErrorCode ec);
    NaviErrorPtr getError() const;
    CollectFlags getCollectFlags() const {
        return _collectFlags;
    }
    const std::shared_ptr<TopoNodeParam> &getTopoParam() const {
        return _topoParam;
    }
public:
    NaviResultPtr createNaviResult();
    std::shared_ptr<multi_call::GigStreamRpcInfoMap> getRpcInfoMap();
private:
    void doToProto(bool partial, NaviMessage &naviMessage, ResultToProtoParam *param) override;
    void markThisLocation(SingleResultLocationDef *singleLocation) const override;
    void doCollectTrace(TraceCollector &collector,
                        SymbolTableDef &symbolTableDef) override;
    void doToVisProto(GraphVisNodeDef *visDef, ResultToProtoParam *param) override;
private:
    void collectLoggerTrace();
    bool toGraphResultDef(bool partial,
                          GraphResultDef *inputGraphResultDef,
                          ResultToProtoParam *param);
    void fillHostInfo(GraphResultDef &graphResultDef);
    void collectSnapshot(GraphResultDef &graphResultDef);
private:
    NaviLoggerPtr _logger;
    CollectFlags _collectFlags;
    std::shared_ptr<TopoNodeParam> _topoParam;
    std::string _formatPattern;
    GraphMetric _graphMetric;
    const NaviHostInfo *_hostInfo = nullptr;
    GraphDef *_graphDef = nullptr;
    StringHashTablePtr _stringHashTable;
    Graph *_graph = nullptr;
    mutable autil::ThreadMutex _resultMutex;
    NaviErrorPtr _error;
    std::shared_ptr<multi_call::GigStreamRpcInfoMap> _rpcInfoMap;
};

NAVI_TYPEDEF_PTR(GraphResult);

}
