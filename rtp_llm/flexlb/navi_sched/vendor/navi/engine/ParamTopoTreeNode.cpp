#include "navi/engine/ParamTopoTreeNode.h"
#include "autil/StringTokenizer.h"
#include "autil/StringUtil.h"
#include "navi/log/NaviLogger.h"
#include "navi/log/TraceAppender.h"
#include "navi/util/CommonUtil.h"

namespace navi {

static const std::string TOPO_LOCATION_DELIM1 = "?";
static const std::string TOPO_LOCATION_DELIM2 = ";";
static const std::string TOPO_LOCATION_DELIM3 = "=";

bool MallocTraceParam::init(const std::string &paramStr) {
    auto splited =
        autil::StringTokenizer::tokenize(autil::StringView(paramStr),
                                         NAVI_DEBUG_DELIM3_STR,
                                         autil::StringTokenizer::TOKEN_TRIM);
    if (5 != splited.size()) {
        return false;
    }
    const auto &graphIdStr = splited[0];
    int64_t graphId = -1;
    if (!autil::StringUtil::fromString(graphIdStr, graphId)) {
        return false;
    }
    const auto &node = splited[1];
    const auto &kernel = splited[2];
    const auto &scheduleIdStr = splited[3];
    const auto &mallocIdStr = splited[4];
    int64_t scheduleId = -1;
    if (!autil::StringUtil::fromString(scheduleIdStr, scheduleId)) {
        return false;
    }
    int64_t mallocId = -1;
    if (!autil::StringUtil::fromString(mallocIdStr, mallocId)) {
        return false;
    }
    _graphId = graphId;
    _node = node;
    _kernel = kernel;
    _scheduleId = scheduleId;
    _mallocId = mallocId;
    return true;
}

bool MallocTraceParam::isValid() const {
    return _scheduleId >= 0 && _mallocId >= 0;
}

std::string MallocTraceParam::toString() const {
    return _node + NAVI_DEBUG_DELIM3_STR +
           autil::StringUtil::toString(_scheduleId) + NAVI_DEBUG_DELIM3_STR +
           autil::StringUtil::toString(_mallocId);
}

int64_t MallocTraceParam::getMallocId(const std::string &node,
                                      int64_t scheduleId) const {
    if (scheduleId == _scheduleId && node == _node) {
        return _mallocId;
    } else {
        return -1;
    }
}

void MallocTraceParam::toProto(MallocTraceParamDef &paramDef) const {
    paramDef.set_node(_node);
    paramDef.set_schedule_id(_scheduleId);
    paramDef.set_malloc_id(_mallocId);
}

void MallocTraceParam::fromProto(const MallocTraceParamDef &paramDef) {
    _node = paramDef.node();
    _scheduleId = paramDef.schedule_id();
    _mallocId = paramDef.malloc_id();
}

void TopoNodeParam::addSummaryQuery(GraphId graphId, const std::string &edgeKey, const std::string &query) {
    _edgeSummaryMap[graphId][edgeKey] = query;
    uint64_t edgeHash = CommonUtil::getPortKeyHash(edgeKey);
    _edgeSummaryMapHash[graphId][edgeHash] = query;
    _collectFlags.setCollectEdgeData(true);
}

const std::string *TopoNodeParam::getSummaryQuery(GraphId graphId, uint64_t edgeKeyHash) const {
    const NaviUnorderedMap<uint64_t, std::string> *edgeMap = nullptr;
    auto it = _edgeSummaryMapHash.find(graphId);
    if (_edgeSummaryMapHash.end() == it) {
        auto it2 = _edgeSummaryMapHash.find(INVALID_GRAPH_ID);
        if (_edgeSummaryMapHash.end() == it2) {
            return nullptr;
        }
        edgeMap = &it2->second;
    } else {
        edgeMap = &it->second;
    }
    {
        auto it = edgeMap->find(edgeKeyHash);
        if (edgeMap->end() != it) {
            return &it->second;
        }
    }
    {
        auto wildCardKey =
            CommonUtil::getPortKeyHash(std::string("*"), std::string());
        auto it = edgeMap->find(wildCardKey);
        if (edgeMap->end() != it) {
            return &it->second;
        }
    }
    return nullptr;
}

bool TopoNodeParam::addBtFilterStr(const std::string &btFilterStr) {
    auto splited = autil::StringTokenizer::tokenize(
        autil::StringView(btFilterStr), NAVI_DEBUG_DELIM3_STR);
    if (2 != splited.size()) {
        return false;
    }
    const auto &file = splited[0];
    const auto &lineStr = splited[1];
    int line = -1;
    if (!autil::StringUtil::fromString(lineStr, line)) {
        return false;
    }
    _traceBtFilterParams.push_back(std::make_pair(file, line));
    return true;
}

bool TopoNodeParam::addMallocTraceParamStr(const std::string &paramStr) {
    return _mallocTraceParam.init(paramStr);
}

void TopoNodeParam::merge(const TopoNodeParam &other) {
    if (other._threadLimit > 0) {
        _threadLimit = other._threadLimit;
    }
    if (other._timeoutMs >= 0) {
        _timeoutMs = other._timeoutMs;
    }
    if (other._maxInline >= 0) {
        _maxInline = other._maxInline;
    }
    _collectFlags.setFlags(_collectFlags.getFlags() | other._collectFlags.getFlags());
    if (!other._traceLevel.empty()) {
        _traceLevel = other._traceLevel;
    }
    _traceBtFilterParams.insert(
        _traceBtFilterParams.end(), other._traceBtFilterParams.begin(), other._traceBtFilterParams.end());
    if (other._mallocTraceParam.isValid()) {
        _mallocTraceParam = other._mallocTraceParam;
    }
    for (const auto &graphPair : other._edgeSummaryMap) {
        auto &edgeMap = _edgeSummaryMap[graphPair.first];
        auto &edgeMapHash = _edgeSummaryMapHash[graphPair.first];
        for (const auto &edgePair : graphPair.second) {
            edgeMap[edgePair.first] = edgePair.second;
            uint64_t edgeHash = CommonUtil::getPortKeyHash(edgePair.first);
            edgeMapHash[edgeHash] = edgePair.second;
        }
    }
}

void TopoNodeParam::fillFilterParam(TraceAppender *traceAppender) const {
    for (const auto &param : _traceBtFilterParams) {
        traceAppender->addBtFilter(LogBtFilterParam(param.first, param.second));
    }
}

void TopoNodeParam::toProto(TopoNodeParamDef &paramDef) const {
    paramDef.set_thread_limit(_threadLimit);
    paramDef.set_timeout_ms(_timeoutMs);
    paramDef.set_max_inline(_maxInline);
    paramDef.set_collect_flags(_collectFlags.getFlags());
    paramDef.set_trace_level(_traceLevel);
    for (const auto &pair : _traceBtFilterParams) {
        auto filter = paramDef.add_trace_bt_filters();
        filter->set_file(pair.first);
        filter->set_line(pair.second);
    }
    _mallocTraceParam.toProto(*paramDef.mutable_malloc_trace_param());
    auto &graphMap = *paramDef.mutable_edge_summary_map();
    for (const auto &graphPair : _edgeSummaryMap) {
        auto &edgeMap = *graphMap[graphPair.first].mutable_edge_map();
        for (const auto &edgePair : graphPair.second) {
            edgeMap[edgePair.first] = edgePair.second;
        }
    }
}

void TopoNodeParam::fromProto(const TopoNodeParamDef &paramDef) {
    _threadLimit = paramDef.thread_limit();
    _timeoutMs = paramDef.timeout_ms();
    _maxInline = paramDef.max_inline();
    _collectFlags.setFlags(paramDef.collect_flags());
    _traceLevel = paramDef.trace_level();
    for (const auto &filter : paramDef.trace_bt_filters()) {
        _traceBtFilterParams.push_back(std::make_pair(filter.file(), filter.line()));
    }
    _mallocTraceParam.fromProto(paramDef.malloc_trace_param());
    const auto &graphMap = paramDef.edge_summary_map();
    for (const auto &graphPair : graphMap) {
        auto &edgeMap = _edgeSummaryMap[graphPair.first];
        auto &edgeMapHash = _edgeSummaryMapHash[graphPair.first];
        for (const auto &edgePair : graphPair.second.edge_map()) {
            edgeMap[edgePair.first] = edgePair.second;
            uint64_t edgeHash = CommonUtil::getPortKeyHash(edgePair.first);
            edgeMapHash[edgeHash] = edgePair.second;
        }
    }
}

bool ParamTopoTreeNode::init(const std::string &valueStr, bool isLeaf) {
    const auto &splited2 = autil::StringTokenizer::tokenize(autil::StringView(valueStr), TOPO_LOCATION_DELIM2);
    std::string edgeKey;
    std::string edgeQuery;
    bool valid = false;
    auto param = std::make_shared<TopoNodeParam>();
    for (const auto &s : splited2) {
        const auto &splited3 = autil::StringTokenizer::tokenize(
            autil::StringView(s), TOPO_LOCATION_DELIM3, autil::StringTokenizer::TOKEN_TRIM);
        if (2u != splited3.size()) {
            NAVI_KERNEL_LOG(ERROR, "splited size is not 2: %s", s.c_str());
            return false;
        }
        const auto &key = splited3[0];
        const auto &value = splited3[1];
        if (key == "graph_id") {
            if (!autil::StringUtil::fromString(value, _graphId)) {
                NAVI_KERNEL_LOG(ERROR, "invalid graph id: %s", value.c_str());
                return false;
            }
        } else if (key == "part_id") {
            if (!autil::StringUtil::fromString(value, _partId)) {
                NAVI_KERNEL_LOG(ERROR, "invalid part id: %s", value.c_str());
                return false;
            }
        } else if (key == "fork_node") {
            _forkNode = value;
        } else if (key == "edge_key") {
            edgeKey = value;
        } else if (key == "edge_query") {
            edgeQuery = value;
        } else if (key == NAVI_DEBUG_METRIC_STR) {
            param->_collectFlags.setCollectMetric(autil::StringUtil::fromString<bool>(value));
            valid = true;
        } else if (key == NAVI_DEBUG_PERF_STR) {
            param->_collectFlags.setCollectPerf(autil::StringUtil::fromString<bool>(value));
            valid = true;
        } else if (key == NAVI_DEBUG_CONTEXT_SWITCH_STR) {
            param->_collectFlags.setCollectContextSwitch(autil::StringUtil::fromString<bool>(value));
            valid = true;
        } else if (key == NAVI_DEBUG_PAGE_FAULT_STR) {
            param->_collectFlags.setCollectPageFault(autil::StringUtil::fromString<bool>(value));
            valid = true;
        } else if (key == NAVI_DEBUG_TRACE_STR) {
            param->_traceLevel = value;
            valid = true;
        } else if (key == NAVI_DEBUG_MALLOC_STR) {
            param->_collectFlags.setCollectMalloc(autil::StringUtil::fromString<bool>(value));
            valid = true;
        } else if (key == NAVI_DEBUG_SNAPSHOT_STR) {
            param->_collectFlags.setCollectSnapshot(autil::StringUtil::fromString<bool>(value));
            valid = true;
        } else if (key == NAVI_DEBUG_BT_FILTER_STR) {
            if (param->addBtFilterStr(value)) {
                valid = true;
            }
        } else if (key == NAVI_DEBUG_MALLOC_TRACE_STR) {
            if (param->addMallocTraceParamStr(value)) {
                valid = true;
            }
        } else if (key == NAVI_DEBUG_TIMEOUT_STR) {
            auto timeout = autil::StringUtil::fromString<int64_t>(value);
            if (timeout > 0) {
                param->_timeoutMs = timeout;
                valid = true;
            }
        } else if (key == NAVI_DEBUG_THREAD_LIMIT_STR) {
            auto threadLimit = autil::StringUtil::fromString<int64_t>(value);
            if (threadLimit > 0) {
                param->_threadLimit = threadLimit;
                valid = true;
            }
        } else if (key == NAVI_DEBUG_MAX_INLINE_STR) {
            auto maxInline = autil::StringUtil::fromString<int64_t>(value);
            if (maxInline >= 0) {
                param->_maxInline = maxInline;
                valid = true;
            }
        }
    }
    if (isLeaf && !edgeKey.empty()) {
        param->addSummaryQuery(_graphId, edgeKey, edgeQuery);
        valid = true;
    }
    if (!isLeaf) {
        initKey();
    } else if (valid) {
        _param = param;
    }
    return true;
}

void ParamTopoTreeNode::initKey() {
    std::vector<std::string> paths;
    paths.push_back(std::string("graph_id=") +
                    autil::StringUtil::toString(_graphId));
    if (!_forkNode.empty() && _forkNode != "*") {
        paths.push_back(std::string("fork_node=") + _forkNode);
    }
    if (INVALID_NAVI_PART_ID != _partId) {
        paths.push_back(std::string("part_id=") + autil::StringUtil::toString(_partId));
    }

    _key = autil::StringUtil::join(paths, TOPO_LOCATION_DELIM2);
}

void ParamTopoTreeNode::mergeParam(const TopoNodeParamPtr &otherParam) {
    if (!otherParam) {
        return;
    }
    if (!_param) {
        _param = std::make_shared<TopoNodeParam>(*otherParam);
    } else {
        _param->merge(*otherParam);
    }
}

ParamTopoTreeNodePtr ParamTopoTreeNode::addChild(const ParamTopoTreeNodePtr &child) {
    if (!child) {
        return nullptr;
    }
    const auto &key = child->_key;
    if (key.empty()) {
        mergeParam(child->_param);
        return nullptr;
    } else {
        auto it = _children.find(key);
        if (_children.end() == it) {
            _children[key] = child;
            return child;
        } else {
            return it->second;
        }
    }
}

void ParamTopoTreeNode::mergeNode(const ParamTopoTreeNodePtr &other) {
    mergeParam(other->_param);
    for (const auto &pair : other->_children) {
        const auto &key = pair.first;
        const auto &child = pair.second;
        auto it = _children.find(key);
        if (_children.end() == it) {
            auto node = std::make_shared<ParamTopoTreeNode>(*child);
            node->_param = nullptr;
            node->_children.clear();
            node->mergeNode(child);
            _children[pair.first] = node;
        } else {
            it->second->mergeNode(child);
        }
    }
}

void ParamTopoTreeNode::toProto(ParamTopoTreeNodeDef *treeNodeDef) const {
    treeNodeDef->set_graph_id(_graphId);
    treeNodeDef->set_part_id(_partId);
    treeNodeDef->set_fork_node(_forkNode);
    if (_param) {
        _param->toProto(*treeNodeDef->mutable_param());
    }
    for (const auto &pair : _children) {
        pair.second->toProto(treeNodeDef->add_childs());
    }
}

void ParamTopoTreeNode::fromProto(const ParamTopoTreeNodeDef &treeNodeDef) {
    _graphId = treeNodeDef.graph_id();
    _partId = treeNodeDef.part_id();
    _forkNode = treeNodeDef.fork_node();
    if (treeNodeDef.has_param()) {
        auto param = std::make_shared<TopoNodeParam>();
        param->fromProto(treeNodeDef.param());
        _param = param;
    }
    initKey();
    auto childSize = treeNodeDef.childs_size();
    for (int32_t i = 0; i < childSize; i++) {
        auto node = std::make_shared<ParamTopoTreeNode>();
        node->fromProto(treeNodeDef.childs(i));
        _children[node->_key] = node;
    }
}

ParamTopoTreeNodePtr ParamTopoTreeNode::getNode(GraphId graphId, NaviPartId partId, const std::string &forkNode) {
    auto root = std::make_shared<ParamTopoTreeNode>();
    for (const auto &pair : _children) {
        const auto &node = pair.second;
        if ("*" != node->_forkNode && !node->_forkNode.empty() && forkNode != node->_forkNode) {
            continue;
        }
        if (INVALID_GRAPH_ID != node->_graphId && graphId != node->_graphId) {
            continue;
        }
        if (INVALID_NAVI_PART_ID != node->_partId && partId != node->_partId) {
            continue;
        }
        root->mergeNode(node);
    }
    if (!root->_children.empty() || root->_param) {
        return root;
    } else {
        return nullptr;
    }
}

bool ParamTopoTreeNode::addParam(const std::string &paramStr) {
    // do not ignore empty
    const auto &splited = autil::StringTokenizer::tokenize(
        autil::StringView(paramStr), TOPO_LOCATION_DELIM1, autil::StringTokenizer::TOKEN_TRIM);
    ParamTopoTreeNode *prev = nullptr;
    auto count = splited.size();
    for (size_t i = 0; i < count; i++) {
        const auto &value = splited[i];
        auto node = std::make_shared<ParamTopoTreeNode>();
        if (node->init(value, i == (count - 1))) {
            if (0 == i) {
                mergeParam(node->_param);
                prev = this;
                continue;
            }
            if (!prev) {
                prev = this;
            }
            node = prev->addChild(node);
            prev = node.get();
        } else {
            return false;
        }
    }
    return true;
}

}
