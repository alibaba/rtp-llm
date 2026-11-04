#include "navi/engine/NaviResult.h"
#include "navi/engine/ComputeScope.h"
#include "navi/engine/GraphResult.h"
#include "navi/engine/ScheduleInfo.h"
#include "navi/perf/NaviSymbolTable.h"
#include "navi/util/CommonUtil.h"

namespace navi {

NaviResult::NaviResult(const NaviLoggerPtr &logger)
    : _logger(this, "navi result", logger)
{
}

bool NaviResult::hasError() const {
    return EC_NONE != getErrorCode();
}

ErrorCode NaviResult::getErrorCode() const {
    if (!error) {
        return EC_NONE;
    }
    return error->ec;
}

std::string NaviResult::getErrorMessage() const {
    if (!error) {
        return std::string();
    }
    if (!error->errorEvent) {
        return std::string();
    }
    return error->errorEvent->message;
}

std::string NaviResult::getErrorPrefix() const {
    if (!error) {
        return std::string();
    }
    if (!error->errorEvent) {
        return std::string();
    }
    return error->errorEvent->prefix;
}

std::string NaviResult::getErrorBt() const {
    if (!error) {
        return std::string();
    }
    if (!error->errorEvent) {
        return std::string();
    }
    return error->errorEvent->bt;
}

LoggingEventPtr NaviResult::getErrorEvent() const {
    if (!error) {
        return nullptr;
    }
    return error->errorEvent;
}

void NaviResult::collectTrace(std::vector<std::string> &traceVec) const {
    auto graphResult = getGraphResult();
    if (!graphResult) {
        return;
    }
    NaviLoggerScope scope(_logger);
    TraceCollector collector;
    SymbolTableDef symbolTableDef;
    graphResult->collectTrace(collector, symbolTableDef);
    collector.format(traceVec);
}

std::shared_ptr<GraphVisDef> NaviResult::getVisProto() {
    return getVisProto(nullptr);
}

std::shared_ptr<GraphVisDef> NaviResult::getVisProto(const std::shared_ptr<ComputeScope> &computeScope) {
    getRpcInfoMap();
    autil::ScopedLock lock(_resultMutex);
    if (_visProto) {
        return _visProto;
    }
    if (!_graphResult) {
        return nullptr;
    }
    NaviLoggerScope scope(_logger);
    std::shared_ptr<GraphVisDef> ret;
    GraphVisNodeDef *node = nullptr;
    {
        ThreadMemLeakDeduperScope deduperScope(
            computeScope ? computeScope->getThreadMemLeakCollector() : nullptr);
        ret = std::make_shared<GraphVisDef>();
        node = ret->add_nodes();
    }
    auto collectFlags = _graphResult->getCollectFlags();
    if (!collectFlags.collectMalloc()) {
        _graphResult->toVisProto(node, nullptr);
    } else {
        ResultToProtoParam param;
        std::string nodeStr;
        std::string stringHashStr;
        std::string errorStr;
        std::vector<std::pair<uint64_t, uint32_t>> addrVec;
        bool hasStringHashTable = false;
        if (!computeScope) {
            PerfEnableScope perfEnableScope(nullptr, collectFlags);
            auto initMetric = _graphResult->getGraphMetric()->getInitMetric();
            auto schedInfo = ScheduleInfo::makeInlineSchedInfo();
            auto memLeakCollector = std::make_shared<MemLeakCollector>();
            const auto &topoParam = _graphResult->getTopoParam();
            auto newComputeScope = std::make_shared<ComputeScope>(
                initMetric.get(),
                nullptr,
                0,
                nullptr,
                memLeakCollector.get(),
                *schedInfo,
                collectFlags,
                topoParam ? &topoParam->getMallocTraceParam() : nullptr);
            newComputeScope->setType(GET_GRAPH_GET_VIS_PROTO);
            {
                ThreadMemLeakDeduperScope deduperScope(
                    newComputeScope->getThreadMemLeakCollector());
                _graphResult->toVisProto(node, &param);
                nodeStr = node->SerializeAsString();
                node->Clear();
                if (error) {
                    NaviErrorDef errorDef;
                    error->toProto(errorDef);
                    errorStr = errorDef.SerializeAsString();
                    error.reset();
                }
                if (param.stackTable) {
                    addrVec = param.stackTable->toAddrVec();
                    param.stackTable.reset();
                }
                if (param.stringHashTable) {
                    SymbolTableDef symbolTableDef;
                    param.stringHashTable->toProto(&symbolTableDef);
                    stringHashStr = symbolTableDef.SerializeAsString();
                    param.stringHashTable.reset();
                    hasStringHashTable = true;
                }
                _userDomain.reset();
                _graphResult.reset();
            }
            newComputeScope->report();
        } else {
            {
                ThreadMemLeakDeduperScope deduperScope(
                    computeScope->getThreadMemLeakCollector());
                _graphResult->toVisProto(node, &param);
                nodeStr = node->SerializeAsString();
                node->Clear();
                if (error) {
                    NaviErrorDef errorDef;
                    error->toProto(errorDef);
                    errorStr = errorDef.SerializeAsString();
                    error.reset();
                }
                if (param.stackTable) {
                    addrVec = param.stackTable->toAddrVec();
                    param.stackTable.reset();
                }
                if (param.stringHashTable) {
                    SymbolTableDef symbolTableDef;
                    param.stringHashTable->toProto(&symbolTableDef);
                    stringHashStr = symbolTableDef.SerializeAsString();
                    param.stringHashTable.reset();
                    hasStringHashTable = true;
                }
                _userDomain.reset();
                _graphResult.reset();
            }
            computeScope->report();
        }
        if (!errorStr.empty()) {
            NaviErrorDef errorDef;
            errorDef.ParseFromString(errorStr);
            error = std::make_shared<NaviError>();
            error->fromProto(errorDef);
        }
        node->ParseFromString(nodeStr);
        auto graphResultDef = node->mutable_result();
        auto metricDef = graphResultDef->mutable_graph_metric();
        if (!addrVec.empty()) {
            param.stackTable = std::make_shared<StackTable>(nullptr);
            param.stackTable->setStackDef(metricDef->mutable_stack_table(),
                                          addrVec);
        }
        if (hasStringHashTable) {
            param.stringHashTable = std::make_shared<StringHashTable>();
            param.stringHashTable->fromStr(stringHashStr);
        }
        if (param.stackTable) {
            auto &stackTable = *param.stackTable;
            if (param.initMetric) {
                param.initMetric->fillProto(metricDef->add_metrics(),
                                            stackTable);
            }
            if (param.naviSymbolTable && param.stringHashTable) {
                param.naviSymbolTable->resolveSymbol(
                    stackTable,
                    metricDef->mutable_perf_symbol_table(),
                    param.stringHashTable.get());
            }
        }
        if (param.stringHashTable) {
            param.stringHashTable->toProto(node->mutable_symbol_table());
        }
    }
    node->set_version(NAVI_VERSION);
    _visProto = ret;
    return _visProto;
}

std::shared_ptr<multi_call::GigStreamRpcInfoMap> NaviResult::getRpcInfoMap() {
    autil::ScopedLock lock(_resultMutex);
    if (_rpcInfoMap) {
        return _rpcInfoMap;
    }
    if (!_graphResult) {
        return nullptr;
    }
    NaviLoggerScope scope(_logger);
    _rpcInfoMap = _graphResult->getRpcInfoMap();
    return _rpcInfoMap;
}

bool NaviResult::toProto(bool partial,
                         NaviMessage &naviMessage,
                         const std::shared_ptr<ComputeScope> &computeScope) {
    ResultToProtoParam paramValue;
    std::string resultStr;
    std::string graphResultStr;
    std::string stringHashStr;
    std::string errorStr;
    std::vector<std::pair<uint64_t, uint32_t>> addrVec;
    GraphResultDef graphResultDef;
    auto collectMalloc = NAVI_COLLECT_MALLOC();
    ResultToProtoParam *param = collectMalloc ? &paramValue : nullptr;
    bool hasStringHashTable = false;
    SerializedResultDef *thisResult = nullptr;
    {
        ThreadMemLeakDeduperScope deduperScope(
            computeScope ? computeScope->getThreadMemLeakCollector() : nullptr);
        {
            auto graphResult = getGraphResult();
            if (!graphResult) {
                return false;
            }
            _userDomain.reset();
            graphResult->toProto(false, naviMessage, param);
            setGraphResult(nullptr);
        }
        if (!param) {
            return true;
        }
        if (error) {
            NaviErrorDef errorDef;
            error->toProto(errorDef);
            errorStr = errorDef.SerializeAsString();
            error.reset();
        }
        thisResult = param->thisResult;
        if (thisResult) {
            graphResultStr = thisResult->result_str();
            thisResult->clear_result_str();
            resultStr = thisResult->SerializeAsString();
            thisResult->Clear();
        }
        if (param->stackTable) {
            addrVec = param->stackTable->toAddrVec();
            param->stackTable.reset();
        }
        if (param->stringHashTable) {
            SymbolTableDef symbolTableDef;
            param->stringHashTable->toProto(&symbolTableDef);
            stringHashStr = symbolTableDef.SerializeAsString();
            param->stringHashTable.reset();
            hasStringHashTable = true;
        }
    }
    if (computeScope) {
        computeScope->report();
    }
    if (!errorStr.empty()) {
        NaviErrorDef errorDef;
        errorDef.ParseFromString(errorStr);
        error = std::make_shared<NaviError>();
        error->fromProto(errorDef);
    }
    if (thisResult) {
        thisResult->ParseFromString(resultStr);
        graphResultDef.ParseFromString(graphResultStr);
    }
    auto metricDef = graphResultDef.mutable_graph_metric();
    if (!addrVec.empty()) {
        param->stackTable = std::make_shared<StackTable>(nullptr);
        param->stackTable->setStackDef(metricDef->mutable_stack_table(),
                                       addrVec);
    }
    if (hasStringHashTable) {
        param->stringHashTable = std::make_shared<StringHashTable>();
        param->stringHashTable->fromStr(stringHashStr);
    }
    if (param->stackTable) {
        auto &stackTable = *param->stackTable;
        if (param->initMetric) {
            param->initMetric->fillProto(metricDef->add_metrics(), stackTable);
        }
        if (param->naviSymbolTable && param->stringHashTable) {
            param->naviSymbolTable->resolveSymbol(
                stackTable,
                metricDef->mutable_perf_symbol_table(),
                param->stringHashTable.get());
        }
    }
    if (param->stringHashTable) {
        param->stringHashTable->toProto(graphResultDef.mutable_symbol_table());
    }
    if (thisResult) {
        thisResult->set_result_str(graphResultDef.SerializeAsString());
    }
    return true;
}

void NaviResult::setGraphResult(const GraphResultPtr &result) {
    GraphResultPtr tmp;
    {
        autil::ScopedLock lock(_resultMutex);
        tmp = _graphResult;
        _graphResult = result;
    }
}

const std::shared_ptr<GraphResult> &NaviResult::getGraphResult() const {
    autil::ScopedLock lock(_resultMutex);
    return _graphResult;
}

const char *NaviResult::getErrorString(ErrorCode ec) {
    return CommonUtil::getErrorString(ec);
}

}
