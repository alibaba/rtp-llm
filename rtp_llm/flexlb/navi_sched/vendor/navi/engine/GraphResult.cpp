#include "navi/engine/GraphResult.h"
#include "navi/engine/Graph.h"
#include "navi/engine/NaviHostInfo.h"
#include "navi/proto/NaviStream.pb.h"

namespace navi {

GraphResult::GraphResult() {
}

GraphResult::~GraphResult() {
    DELETE_AND_SET_NULL(_graphDef);
}

void GraphResult::init(const NaviLoggerPtr &logger,
                       const CollectFlags &collectFlags,
                       const std::shared_ptr<TopoNodeParam> &topoParam,
                       const std::string &formatPattern,
                       const NaviSymbolTablePtr &naviSymbolTable,
                       const NaviHostInfo *hostInfo)
{
    _stringHashTable = std::make_shared<StringHashTable>();
    _logger = logger;
    _collectFlags = collectFlags;
    _topoParam = topoParam;
    _formatPattern = formatPattern;
    _graphMetric.setNaviSymbolTable(naviSymbolTable);
    _hostInfo = hostInfo;
}

void GraphResult::setGraphDef(GraphDef *graphDef) {
    if (_graphDef) {
        delete _graphDef;
    }
    _graphDef = new GraphDef();
    _graphDef->CopyFrom(*graphDef);
}

void GraphResult::fillHostInfo(GraphResultDef &graphResultDef) {
    if (_hostInfo) {
        auto needAll = (_collectFlags.needHostInfo() || (LOG_LEVEL_DISABLE != _logger->getTraceLevel()));
        _hostInfo->fillHostInfo(needAll, graphResultDef.mutable_host_info());
    }
}

void GraphResult::collectSnapshot(GraphResultDef &graphResultDef) {
    if (unlikely(_collectFlags.collectSnapshot() && _hostInfo)) {
        auto visProto = _hostInfo->getInitVisProto(true);
        if (visProto && visProto->nodes_size() > 0) {
            auto snapshotVisDef = graphResultDef.mutable_snapshot();
            snapshotVisDef->CopyFrom(visProto->nodes(0));
            auto snapshotSymbolTableDef = snapshotVisDef->mutable_symbol_table();
            graphResultDef.mutable_symbol_table()->MergeFrom(*snapshotSymbolTableDef);
            snapshotVisDef->clear_symbol_table();
        }
    }
}

bool GraphResult::toGraphResultDef(bool partial,
                                   GraphResultDef *inputGraphResultDef,
                                   ResultToProtoParam *param) {
    auto &graphResultDef = *inputGraphResultDef;
    TraceCollector traceCollector;
    traceCollector.setFormatPattern(_formatPattern);
    if (partial) {
        _logger->collectTrace(traceCollector);
        if (!traceCollector.empty()) {
            graphResultDef.set_version(NAVI_VERSION);
            traceCollector.toProto(graphResultDef.mutable_traces(),
                                   _stringHashTable.get());
            if (_stringHashTable->needToProto()) {
                _stringHashTable->toProto(graphResultDef.mutable_symbol_table());
            }
            return true;
        } else {
            return false;
        }
    } else {
        graphResultDef.set_version(NAVI_VERSION);
        NaviError::fillSessionId(graphResultDef.mutable_session(),
                                 _logger->getLoggerId());
        graphResultDef.set_result_ptr((uint64_t)this);
        if (_error) {
            _error->toProto(*graphResultDef.mutable_error());
        }
        if (_graphDef) {
            if (unlikely(_collectFlags.collectMalloc())) {
                graphResultDef.mutable_graph()->ParseFromString(
                    _graphDef->SerializeAsString());
            } else {
                graphResultDef.mutable_graph()->Swap(_graphDef);
            }
            DELETE_AND_SET_NULL(_graphDef);
        }
        _graphMetric.toProto(graphResultDef.mutable_graph_metric(), _stringHashTable.get(), param);
        _logger->collectTrace(traceCollector);
        if (!traceCollector.empty()) {
            traceCollector.toProto(graphResultDef.mutable_traces(),
                                   _stringHashTable.get());
        }
        if (!param) {
            if (_stringHashTable->needToProto()) {
                _stringHashTable->toProto(
                    graphResultDef.mutable_symbol_table());
            }
        } else {
            param->stringHashTable = _stringHashTable;
        }
        fillHostInfo(graphResultDef);
        collectSnapshot(graphResultDef);
        return true;
    }
}

void GraphResult::doToProto(bool partial, NaviMessage &naviMessage, ResultToProtoParam *param) {
    auto beginTime = autil::TimeUtility::currentTime();
    GraphResultDef graphResultDef;
    if (!toGraphResultDef(partial, &graphResultDef, param)) {
        return;
    }
    SerializedResultDef *thisResult = nullptr;
    if (hasParent() || partial) {
        thisResult = naviMessage.add_sub_result();
    } else {
        thisResult = naviMessage.mutable_navi_result();
        if (graphResultDef.has_error()) {
            thisResult->mutable_error()->CopyFrom(graphResultDef.error());
        }
    }
    if (param) {
        param->thisResult = thisResult;
    }
    thisResult->set_result_str(graphResultDef.SerializeAsString());
    thisResult->set_has_trace(graphResultDef.has_traces());
    markLocation(thisResult->mutable_location());
    auto visMetric = thisResult->mutable_vis_fill_metric();
    visMetric->set_fill_begin_time(beginTime);
    visMetric->set_fill_end_time(autil::TimeUtility::currentTime());
}

void GraphResult::markThisLocation(SingleResultLocationDef *singleLocation) const {
    singleLocation->set_type(NRT_NORMAL);
}

void GraphResult::doCollectTrace(TraceCollector &collector,
                                 SymbolTableDef &symbolTableDef)
{
    _logger->collectTrace(collector);
}

void GraphResult::doToVisProto(GraphVisNodeDef *visDef, ResultToProtoParam *param) {
    auto beginTime = autil::TimeUtility::currentTime();
    ResultLocationDef location;
    markLocation(&location);
    auto thisDef = getVisNode(location, visDef);
    if (!toGraphResultDef(false, thisDef->mutable_result(), param)) {
        return;
    }
    if (thisDef->mutable_result()->has_symbol_table()) {
        auto resultSymbolTableDef = thisDef->mutable_result()->mutable_symbol_table();
        visDef->mutable_symbol_table()->MergeFrom(*resultSymbolTableDef);
        thisDef->mutable_result()->clear_symbol_table();
    }
    auto visMetric = thisDef->mutable_result()->mutable_vis_fill_metric();
    visMetric->set_fill_begin_time(beginTime);
    visMetric->set_fill_end_time(autil::TimeUtility::currentTime());
}

void GraphResult::setError(const NaviLoggerPtr &logger, ErrorCode ec) {
    updateError(makeError(logger, ec));
}

NaviErrorPtr GraphResult::makeError(const NaviLoggerPtr &logger, ErrorCode ec) const {
    auto error = NaviError::makeError(logger, ec);
    if (!error) {
        return error;
    }
    error->result = this;
    markLocation(&error->location);
    return error;
}

NaviErrorPtr GraphResult::updateError(const NaviErrorPtr &error) {
    autil::ScopedLock lock(_resultMutex);
    if (_error) {
        return _error;
    }
    _error = error;
    return _error;
}

NaviErrorPtr GraphResult::getError() const {
    autil::ScopedLock lock(_resultMutex);
    return _error;
}

void GraphResult::finishResult(const std::shared_ptr<GraphDomainUser> &userDomain) {
    if (!isFinished() && _graph) {
        _graph->finishResult();
        _graph = nullptr;
    }
}

NaviResultPtr GraphResult::createNaviResult() {

    autil::ScopedLock lock(_resultMutex);
    auto result = std::make_shared<NaviResult>(_logger);
    _graphMetric.fillResult(*result);
    result->id = _logger->getLoggerId();
    result->error = _error;
    return result;
}

std::shared_ptr<multi_call::GigStreamRpcInfoMap> GraphResult::getRpcInfoMap() {
    autil::ScopedLock lock(_resultMutex);
    if (_rpcInfoMap) {
        return _rpcInfoMap;
    }

    auto ret = std::make_shared<multi_call::GigStreamRpcInfoMap>();
    collectRpcInfo(*ret);
    _rpcInfoMap = ret;
    return _rpcInfoMap;
}

}

