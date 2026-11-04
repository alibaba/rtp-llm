#pragma once

#include "navi/common.h"
#include "navi/proto/NaviStream.pb.h"
#include "navi/engine/RunGraphParams.h"

namespace navi {

class MallocTraceParam {
public:
    bool init(const std::string &paramStr);
    bool isValid() const;
    std::string toString() const;
    int64_t getMallocId(const std::string &node, int64_t scheduleId) const;
    void toProto(MallocTraceParamDef &paramDef) const;
    void fromProto(const MallocTraceParamDef &paramDef);
private:
    GraphId _graphId = INVALID_GRAPH_ID;
    std::string _node;
    std::string _kernel;
    int64_t _scheduleId = -1;
    int64_t _mallocId = -1;
};

class TopoNodeParam {
public:
    void addSummaryQuery(GraphId graphId, const std::string &edgeKey, const std::string &query);
    const std::string *getSummaryQuery(GraphId graphId, uint64_t edgeKeyHash) const;
    bool addBtFilterStr(const std::string &btFilterStr);
    bool addMallocTraceParamStr(const std::string &paramStr);
    void merge(const TopoNodeParam &other);
    void fillFilterParam(TraceAppender *traceAppender) const;
    const MallocTraceParam &getMallocTraceParam() const {
        return _mallocTraceParam;
    }
    void toProto(TopoNodeParamDef &paramDef) const;
    void fromProto(const TopoNodeParamDef &paramDef);
public:
    int32_t _threadLimit = -1;
    int64_t _timeoutMs = -1;
    int32_t _maxInline = -1;
    CollectFlags _collectFlags;
    std::string _traceLevel;
    std::vector<std::pair<std::string, int>> _traceBtFilterParams;
    NaviUnorderedMap<GraphId, NaviUnorderedMap<std::string, std::string>> _edgeSummaryMap;
    NaviUnorderedMap<GraphId, NaviUnorderedMap<uint64_t, std::string>> _edgeSummaryMapHash;
    MallocTraceParam _mallocTraceParam;
};

NAVI_TYPEDEF_PTR(TopoNodeParam);

class ParamTopoTreeNode;
NAVI_TYPEDEF_PTR(ParamTopoTreeNode);

class ParamTopoTreeNode {
public:
    ParamTopoTreeNode() {
    }
    ~ParamTopoTreeNode() {
    }
public:
    bool addParam(const std::string &paramStr);
    void toProto(ParamTopoTreeNodeDef *treeNodeDef) const;
    void fromProto(const ParamTopoTreeNodeDef &treeNodeDef);
    const TopoNodeParamPtr &getParam() const {
        return _param;
    }
    ParamTopoTreeNodePtr getNode(GraphId graphId, NaviPartId partId, const std::string &forkNode);
private:
    bool init(const std::string &valueStr, bool isLeaf);
    void initKey();
    void mergeParam(const TopoNodeParamPtr &otherParam);
    ParamTopoTreeNodePtr addChild(const ParamTopoTreeNodePtr &child);
    void mergeNode(const ParamTopoTreeNodePtr &other);
private:
    std::string _key;
    GraphId _graphId = INVALID_GRAPH_ID;
    NaviPartId _partId = INVALID_NAVI_PART_ID;
    std::string _forkNode;
    TopoNodeParamPtr _param;
    std::map<std::string, ParamTopoTreeNodePtr> _children;
};

}

