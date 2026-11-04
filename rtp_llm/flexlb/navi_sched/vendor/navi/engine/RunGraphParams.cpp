#include "navi/engine/RunGraphParams.h"
#include "navi/engine/GraphParam.h"
#include "navi/engine/SerializeData.h"
#include "navi/engine/ParamTopoTreeNode.h"
#include "navi/engine/ResourceCreator.h"
#include "navi/log/TraceAppender.h"
#include "navi/proto/NaviStream.pb.h"
#include "navi/engine/EdgeMetric.h"
#include "autil/StringTokenizer.h"
#include "autil/StringUtil.h"

namespace navi {

RunGraphParams::RunGraphParams()
    : _threadLimit(DEFAULT_THREAD_LIMIT)
    , _timeoutMs(DEFAULT_TIMEOUT_MS)
    , _globalStartTimeUs(0)
    , _traceLevel(LOG_LEVEL_DISABLE)
    , _traceFormatPattern(DEFAULT_LOG_PATTERN)
    , _asyncIo(true)
    , _disableGigDegrade(false)
    , _loadGraph(false)
    , _maxInline(DEFAULT_MAX_INLINE)
    , _resourceStage(RS_RUN_GRAPH_EXTERNAL)
{
}

RunGraphParams::~RunGraphParams() {
}

void RunGraphParams::setSessionId(SessionId id) {
    _id = id;
}

SessionId RunGraphParams::getSessionId() const {
    return _id;
}

void RunGraphParams::setThreadLimit(uint32_t limit) {
    _threadLimit = std::max(1u, limit);
}

uint32_t RunGraphParams::getThreadLimit() const {
    if (unlikely(_paramTopo && _paramTopo->_threadLimit > 0)) {
        return _paramTopo->_threadLimit;
    }
    return _threadLimit;
}

void RunGraphParams::setTimeoutMs(int64_t timeout) {
    _timeoutMs = timeout;
}

int64_t RunGraphParams::getTimeoutMs() const {
    if (unlikely(_paramTopo && _paramTopo->_timeoutMs >= 0)) {
        return _paramTopo->_timeoutMs;
    }
    return _timeoutMs;
}

void RunGraphParams::initGlobalStartTimeUs() {
    if (0 == _globalStartTimeUs) {
        _globalStartTimeUs = autil::TimeUtility::currentTime();
    }
}

void RunGraphParams::setGlobalStartTimeUs(int64_t startTimeUs) {
    if (0 == _globalStartTimeUs) {
        _globalStartTimeUs = startTimeUs;
    }
}

int64_t RunGraphParams::getGlobalStartTimeUs() const {
    return _globalStartTimeUs;
}

void RunGraphParams::setTaskQueueName(const std::string &taskQueueName) {
    _taskQueueName = taskQueueName;
}

const std::string &RunGraphParams::getTaskQueueName() const {
    return _taskQueueName;
}

void RunGraphParams::setTraceLevel(const std::string &traceLevelStr) {
    _traceLevel = getLevelByString(traceLevelStr);
}

LogLevel RunGraphParams::getTraceLevelWithoutTopo() const {
    return _traceLevel;
}

LogLevel RunGraphParams::getTraceLevel() const {
    if (unlikely(_paramTopo != nullptr)) {
        auto level = getLevelByString(_paramTopo->_traceLevel);
        if (LOG_LEVEL_DISABLE != level) {
            return level;
        }
    }
    return _traceLevel;
}

bool RunGraphParams::needLogPrefix() const {
    return _traceLevel != LOG_LEVEL_DISABLE;
}

void RunGraphParams::setTraceFormatPattern(const std::string &pattern) {
    _traceFormatPattern = pattern;
}

const std::string &RunGraphParams::getTraceFormatPattern() const {
    return _traceFormatPattern;
}

void RunGraphParams::addTraceBtFilter(const std::string &file, int line) {
    _traceBtFilterParams.emplace_back(file, line);
}

void RunGraphParams::fillFilterParam(TraceAppender *traceAppender) const {
    if (!traceAppender) {
        return;
    }
    if (unlikely(_paramTopo != nullptr)) {
        _paramTopo->fillFilterParam(traceAppender);
    }
    for (const auto &param : _traceBtFilterParams) {
        traceAppender->addBtFilter(LogBtFilterParam(param.first, param.second));
    }
}

const std::vector<std::pair<std::string, int>> &RunGraphParams::getTraceBtFilter() const {
    return _traceBtFilterParams;
}

void RunGraphParams::setCollectMetric(bool collect) {
    _collectFlags.setCollectMetric(collect);
}

void RunGraphParams::setCollectPerf(bool collect) {
    _collectFlags.setCollectPerf(collect);
}

void RunGraphParams::setCollectContextSwitch(bool collect) {
    _collectFlags.setCollectContextSwitch(collect);
}

void RunGraphParams::setCollectPageFault(bool collect) {
    _collectFlags.setCollectPageFault(collect);
}

void RunGraphParams::setCollectMalloc(bool collect) {
    _collectFlags.setCollectMalloc(collect);
}

void RunGraphParams::setCollectFlags(uint64_t flags) {
    _collectFlags.setFlags(flags);
}

CollectFlags RunGraphParams::getCollectFlags() const {
    if (unlikely(_paramTopo != nullptr)) {
        return CollectFlags(_paramTopo->_collectFlags.getFlags() | _collectFlags.getFlags());
    }
    return _collectFlags;
}

CollectFlags RunGraphParams::getCollectFlags(const std::shared_ptr<ParamTopoTreeNode> &paramTopoTreeNode) const {
    if (unlikely(paramTopoTreeNode != nullptr)) {
        const auto &paramTopo = paramTopoTreeNode->getParam();
        if (paramTopo) {
            return CollectFlags(paramTopo->_collectFlags.getFlags() | _collectFlags.getFlags());
        } else {
            return _collectFlags;
        }
    }
    return _collectFlags;
}

void RunGraphParams::setAsyncIo(bool async) {
    _asyncIo = async;
}

bool RunGraphParams::asyncIo() const {
    return _asyncIo;
}

void RunGraphParams::setDisableGigDegrade(bool disable) {
    _disableGigDegrade = disable;
}

bool RunGraphParams::disableGigDegrade() const {
    return _disableGigDegrade;
}

int32_t RunGraphParams::getMaxInline() const {
    if (unlikely(_paramTopo && _paramTopo->_maxInline >= 0)) {
        return _paramTopo->_maxInline;
    }
    return _maxInline;
}

void RunGraphParams::setMaxInline(int32_t maxInline) {
    _maxInline = maxInline;
}

bool RunGraphParams::overrideEdgeData(const OverrideData &data) {
    if (INVALID_GRAPH_ID == data.graphId) {
        return false;
    }
    std::string typeId;
    for (const auto &d : data.datas) {
        if (!d) {
            continue;
        }
        const auto &id = d->getTypeId();
        if (id.empty()) {
            NAVI_KERNEL_LOG(
                ERROR,
                "override data has empty type, data [%p] graphId [%d] "
                "outputNode [%s] outputPort [%s]",
                d.get(), data.graphId, data.outputNode.c_str(),
                data.outputPort.c_str());
            return false;
        }
        if (typeId.empty()) {
            typeId = id;
        } else if (typeId != id) {
            NAVI_KERNEL_LOG(ERROR,
                            "override data has diff type, [%s] and [%s], "
                            "graphId [%d] outputNode [%s] outputPort [%s]",
                            typeId.c_str(), id.c_str(), data.graphId,
                            data.outputNode.c_str(), data.outputPort.c_str());
            return false;
        }
    }
    _overrideDatas.emplace_back(data);
    return true;
}

const std::vector<OverrideData> &RunGraphParams::getOverrideDatas() const {
    return _overrideDatas;
}

void RunGraphParams::clearOverrideDatas() {
    _overrideDatas.clear();
}

bool RunGraphParams::addNamedData(const NamedData &data) {
    if (INVALID_GRAPH_ID == data.graphId) {
        return false;
    }
    auto ret = _namedDatas[data.graphId].emplace(data.name, data);
    return ret.second;
}

void RunGraphParams::clearNamedData() {
    _namedDatas.clear();
}

const SubNamedDataMap *RunGraphParams::getSubNamedDataMap(GraphId graphId) const {
    auto it = _namedDatas.find(graphId);
    if (_namedDatas.end() == it) {
        return nullptr;
    }
    return &it->second;
}

const NamedDataMap &RunGraphParams::getNamedDatas() const {
    return _namedDatas;
}

void RunGraphParams::setResourceStage(ResourceStage stage) {
    _resourceStage = stage;
}

ResourceStage RunGraphParams::getResourceStage() const {
    return _resourceStage;
}

const std::shared_ptr<ParamTopoTreeNode> &RunGraphParams::getParamTopoTree() const {
    return _paramTopoTreeNode;
}

const std::shared_ptr<TopoNodeParam> &RunGraphParams::getTopoParam() const {
    return _paramTopo;
}

void RunGraphParams::initTopoParam() {
    if (!_paramTopoTreeNode) {
        _paramTopo = nullptr;
        return;
    }
    _paramTopo = _paramTopoTreeNode->getParam();
}

ParamTopoTreeNodePtr
RunGraphParams::getParamNode(GraphId graphId, NaviPartId partId, const std::string &forkNode) const {
    if (likely(!_paramTopoTreeNode)) {
        return nullptr;
    }
    return _paramTopoTreeNode->getNode(graphId, partId, forkNode);
}

void RunGraphParams::setParamNode(const ParamTopoTreeNodePtr &node) {
    _paramTopoTreeNode = node;
    initTopoParam();
}

void RunGraphParams::setLoadGraph(bool value) {
    _loadGraph = value;
}

bool RunGraphParams::isLoadGraph() const {
    return _loadGraph;
}

void RunGraphParams::setResourceBuildMode(ResourceBuildMode mode) {
    _resourceBuildMode = mode;
}

ResourceBuildMode RunGraphParams::getResourceBuildMode() const {
    return _resourceBuildMode;
}

ResourceBuildMode RunGraphParams::getResourceBuildMode(const std::string &name) const {
    auto it = _resourceBuildModeMap.find(name);
    if (_resourceBuildModeMap.end() != it) {
        return it->second;
    } else {
        return _resourceBuildMode;
    }
}

void RunGraphParams::setResourceBuildMode(const std::string &name, ResourceBuildMode mode) {
    _resourceBuildModeMap[name] = mode;
}

const NaviUnorderedMap<std::string, ResourceBuildMode> &RunGraphParams::getResourceBuildModeMap() const {
    return _resourceBuildModeMap;
}

bool RunGraphParams::fromProto(const RunParams &pbParams,
                               CreatorManager *creatorManager,
                               RunGraphParams &params)
{
    NAVI_KERNEL_LOG(DEBUG, "pb run params: %s", pbParams.DebugString().c_str());
    SessionId sessionId;
    const auto &pbSessionId = pbParams.id();
    sessionId.instance = pbSessionId.instance();
    sessionId.queryId = pbSessionId.query_id();
    params.setSessionId(sessionId);

    params.setTaskQueueName(pbParams.task_queue_name());
    params.setTraceLevel(pbParams.trace_level());
    params.setThreadLimit(pbParams.thread_limit());
    params.setTimeoutMs(pbParams.timeout_ms());
    params.setGlobalStartTimeUs(pbParams.global_start_time_us());
    params.setCollectFlags(pbParams.collect_flags());
    params.setMaxInline(pbParams.max_inline());
    params.setResourceStage(pbParams.resource_stage());
    params.setDisableGigDegrade(pbParams.disable_gig_degrade());
    deserializeParamTopoTree(pbParams, params);
    deserializeResourceBuildMode(pbParams, params);
    deserializeTraceBtFilter(pbParams, params);

    if (!deserializeOverrideData(pbParams, creatorManager, params)) {
        NAVI_KERNEL_LOG(ERROR, "deserialize override data failed");
        return false;
    }
    if (!deserializeNamedData(pbParams, creatorManager, params)) {
        NAVI_KERNEL_LOG(ERROR, "deserialize named data failed");
        return false;
    }
    return true;
}

void RunGraphParams::deserializeResourceBuildMode(const RunParams &pbParams, RunGraphParams &params) {
    params.setResourceBuildMode(pbParams.resource_build_mode());
    const auto &pbMap = pbParams.resource_build_mode_map();
    for (const auto &pair : pbMap) {
        params.setResourceBuildMode(pair.first, pair.second);
    }
}

void RunGraphParams::deserializeTraceBtFilter(const RunParams &pbParams,
                                              RunGraphParams &params)
{
    auto filterCount = pbParams.trace_bt_filters_size();
    for (int32_t i = 0; i < filterCount; i++) {
        const auto &filter = pbParams.trace_bt_filters(i);
        params.addTraceBtFilter(filter.file(), filter.line());
    }
}

bool RunGraphParams::parseOverrideDataValues(const RunParams &pbParams,
                                             CreatorManager *creatorManager,
                                             NaviUnorderedMap<int32_t, DataPtr> &dataMap) {
    const auto &overrideDataDef = pbParams.override_data();
    auto valueCount = overrideDataDef.datas_size();
    for (int32_t i = 0; i < valueCount; i++) {
        DataPtr data;
        const auto &valueDef = overrideDataDef.datas(i);
        const auto &dataStr = valueDef.data();
        if (!dataStr.empty()) {
            const auto &typeId = valueDef.type();
            auto dataType = creatorManager->getType(typeId);
            if (!dataType) {
                NAVI_KERNEL_LOG(ERROR, "deserialize override data failed, type [%s] not registered", typeId.c_str());
                return false;
            }
            if (!SerializeData::deserializeFromStr(NAVI_TLS_LOGGER, dataStr, dataType, data)) {
                NAVI_KERNEL_LOG(ERROR, "deserialize override data failed, type [%s]", typeId.c_str());
                return false;
            }
        }
        dataMap[i] = data;
    }
    return true;
}

bool RunGraphParams::deserializeOverrideData(const RunParams &pbParams,
                                             CreatorManager *creatorManager,
                                             RunGraphParams &params)
{
    const auto &overrideDataDef = pbParams.override_data();
    NaviUnorderedMap<int32_t, DataPtr> dataMap;
    if (!parseOverrideDataValues(pbParams, creatorManager, dataMap)) {
        return false;
    }
    auto edgeCount = overrideDataDef.edge_overrides_size();
    for (int32_t i = 0; i < edgeCount; i++) {
        const auto &edgeOverride = overrideDataDef.edge_overrides(i);
        OverrideData overrideData;
        auto dataCount = edgeOverride.indexs_size();
        for (int32_t j = 0; j < dataCount; j++) {
            auto index = edgeOverride.indexs(j);
            auto it = dataMap.find(index);
            if (dataMap.end() == it) {
                NAVI_KERNEL_LOG(ERROR, "deserialize failed, data index [%d] not found", index);
                return false;
            }
            overrideData.datas.emplace_back(it->second);
        }
        overrideData.graphId = edgeOverride.graph_id();
        const auto &outputPortDef = edgeOverride.output_port();
        overrideData.outputNode = outputPortDef.node_name();
        overrideData.outputPort = outputPortDef.port_name();
        params.overrideEdgeData(overrideData);
    }
    return true;
}

ErrorCode RunGraphParams::toProto(const RunGraphParams &params,
                                  GraphId matchGraphId,
                                  CreatorManager *creatorManager,
                                  NaviPartId partId,
                                  int64_t timeoutMs,
                                  const ParamTopoTreeNodePtr &paramTopoTreeNode,
                                  RunParams &pbParams) {
    auto sessionId = params.getSessionId();
    auto pbSessionId = pbParams.mutable_id();
    pbSessionId->set_instance(sessionId.instance);
    pbSessionId->set_query_id(sessionId.queryId);

    pbParams.set_thread_limit(params._threadLimit);
    pbParams.set_timeout_ms(timeoutMs);
    pbParams.set_global_start_time_us(params.getGlobalStartTimeUs());
    pbParams.set_task_queue_name(params.getTaskQueueName());
    pbParams.set_trace_level(LoggingEvent::levelStrByLevel(params._traceLevel));
    pbParams.set_collect_flags(params._collectFlags.getFlags());
    pbParams.set_max_inline(params._maxInline);
    pbParams.set_resource_stage(params.getResourceStage());
    pbParams.set_disable_gig_degrade(params.disableGigDegrade());
    serializeParamTopoTree(paramTopoTreeNode, pbParams);
    serializeResourceBuildMode(params, pbParams);
    serializeTraceBtFilter(params, pbParams);

    if (!serializeOverrideData(params.getOverrideDatas(), matchGraphId,
                               creatorManager, partId, pbParams))
    {
        NAVI_KERNEL_LOG(ERROR, "serialize override data failed");
        return EC_SERIALIZE_OVERRIDE_DATA;
    }
    if (!serializeNamedData(params.getNamedDatas(), matchGraphId, creatorManager,
                            pbParams))
    {
        NAVI_KERNEL_LOG(ERROR, "serialize named data failed");
        return EC_SERIALIZE_NAMED_DATA;
    }
    return EC_NONE;
}

void RunGraphParams::serializeParamTopoTree(const ParamTopoTreeNodePtr &paramTopoTreeNode, RunParams &pbParams) {
    if (!paramTopoTreeNode) {
        return;
    }
    paramTopoTreeNode->toProto(pbParams.mutable_param_topo_node());
}

void RunGraphParams::deserializeParamTopoTree(const RunParams &pbParams, RunGraphParams &params) {
    if (!pbParams.has_param_topo_node()) {
        return;
    }
    auto paramTopoTreeNode = std::make_shared<ParamTopoTreeNode>();
    paramTopoTreeNode->fromProto(pbParams.param_topo_node());
    params._paramTopoTreeNode = paramTopoTreeNode;
    params.initTopoParam();
}

void RunGraphParams::serializeResourceBuildMode(const RunGraphParams &params, RunParams &pbParams) {
    pbParams.set_resource_build_mode(params.getResourceBuildMode());
    const auto &modeMap = params.getResourceBuildModeMap();
    auto pbMap = pbParams.mutable_resource_build_mode_map();
    for (const auto &pair : modeMap) {
        (*pbMap)[pair.first] = pair.second;
    }
}

void RunGraphParams::serializeTraceBtFilter(const RunGraphParams &params,
                                            RunParams &pbParams)
{
    const auto &traceBtFilter = params._traceBtFilterParams;
    for (const auto &pair : traceBtFilter) {
        auto filter = pbParams.add_trace_bt_filters();
        filter->set_file(pair.first);
        filter->set_line(pair.second);
    }
}

bool RunGraphParams::serializeOverrideData(
    const std::vector<OverrideData> &overrideEdgeDatas,
    GraphId matchGraphId,
    CreatorManager *creatorManager,
    NaviPartId partId,
    RunParams &pbParams)
{
    auto &pbOverrideData = *pbParams.mutable_override_data();
    std::map<DataPtr, int32_t> dataMap;
    size_t totalSize = 0;
    for (const auto &overrideData : overrideEdgeDatas) {
        auto graphId = overrideData.graphId;
        if (graphId != matchGraphId) {
            continue;
        }
        auto overridePartId = overrideData.partId;
        if (INVALID_NAVI_PART_ID != overridePartId && partId != overridePartId) {
            continue;
        }
        auto &edgeOverrides = *pbOverrideData.add_edge_overrides();
        for (const auto &data : overrideData.datas) {
            int32_t index = 0;
            auto it = dataMap.find(data);
            if (dataMap.end() != it) {
                edgeOverrides.add_indexs(it->second);
                continue;
            } else {
                index = pbOverrideData.datas_size();
                edgeOverrides.add_indexs(index);
            }
            dataMap[data] = index;
            auto dataValueDef = pbOverrideData.add_datas();
            if (data) {
                const auto &typeId = data->getTypeId();
                auto dataType = creatorManager->getType(typeId);
                if (!dataType) {
                    NAVI_KERNEL_LOG(ERROR,
                                    "serialize override data failed, type "
                                    "[%s] not registered, graphId [%d] "
                                    "outputNode [%s] outputPort [%s]",
                                    typeId.c_str(),
                                    graphId,
                                    overrideData.outputNode.c_str(),
                                    overrideData.outputPort.c_str());
                    return false;
                }
                auto mutableStr = dataValueDef->mutable_data();
                if (!SerializeData::serializeToStr(NAVI_TLS_LOGGER, dataType,
                                                   data, *mutableStr))
                {
                    NAVI_KERNEL_LOG(ERROR,
                                    "serialize override data failed, type [%s] data [%p] "
                                    "graphId [%d] outputNode [%s] outputPort [%s]",
                                    typeId.c_str(),
                                    data.get(),
                                    graphId,
                                    overrideData.outputNode.c_str(),
                                    overrideData.outputPort.c_str());
                    return false;
                }
                totalSize += mutableStr->size();
                if (totalSize > 512 * 1024 * 1024) {
                    NAVI_KERNEL_LOG(ERROR, "override data too large [%lu]", totalSize);
                    return false;
                }
                dataValueDef->set_type(typeId);
            } else {
                dataValueDef->set_data("");
                dataValueDef->set_type("");
            }
        }
        edgeOverrides.set_graph_id(graphId);
        auto outputPortDef = edgeOverrides.mutable_output_port();
        outputPortDef->set_node_name(overrideData.outputNode);
        outputPortDef->set_port_name(overrideData.outputPort);
    }
    return true;
}

void RunGraphParams::setQuerySession(std::shared_ptr<multi_call::QuerySession> querySession) {
    _querySession = std::move(querySession);
}

const std::shared_ptr<multi_call::QuerySession> &RunGraphParams::getQuerySession() const {
    return _querySession;
}

bool RunGraphParams::serializeNamedData(const NamedDataMap &namedDataMap,
                                        GraphId matchGraphId,
                                        CreatorManager *creatorManager,
                                        RunParams &pbParams)
{
    for (const auto &pair : namedDataMap) {
        auto graphId = pair.first;
        if (graphId != matchGraphId) {
            continue;
        }
        const auto &dataMap = pair.second;
        for (const auto &dataPair : dataMap) {
            const auto &namedData = dataPair.second;
            const auto &name = namedData.name;
            const auto &data = namedData.data;
            auto &pbNamedData = *pbParams.add_named_datas();
            pbNamedData.set_graph_id(graphId);
            pbNamedData.set_name(name);
            if (data) {
                const auto &typeId = data->getTypeId();
                auto dataType = creatorManager->getType(typeId);
                if (!dataType) {
                    NAVI_KERNEL_LOG(
                        ERROR,
                        "serialize named data failed, type [%s] not "
                        "registered, data [%p] graphId [%d] name [%s]",
                        typeId.c_str(), data.get(), graphId, name.c_str());
                    return false;
                }
                pbNamedData.set_type(typeId);
                auto &pbSerializeStr = *pbNamedData.mutable_data();
                if (!SerializeData::serializeToStr(NAVI_TLS_LOGGER, dataType,
                                                   data, pbSerializeStr))
                {
                    NAVI_KERNEL_LOG(ERROR,
                                    "serialize named data failed, type [%s] "
                                    "data [%p] graphId [%d] name [%s]",
                                    typeId.c_str(), data.get(), graphId,
                                    name.c_str());
                    return false;
                }
            }
        }
    }
    return true;
}

bool RunGraphParams::deserializeNamedData(const RunParams &pbParams,
                                          CreatorManager *creatorManager,
                                          RunGraphParams &params)
{
    auto namedDataCount = pbParams.named_datas_size();
    for (int32_t i = 0; i < namedDataCount; i++) {
        NamedData namedData;
        const auto &pbNamedData = pbParams.named_datas(i);
        const auto &typeId = pbNamedData.type();
        const auto &dataStr = pbNamedData.data();
        DataPtr data;
        if (!dataStr.empty()) {
            auto dataType = creatorManager->getType(typeId);
            if (!dataType) {
                NAVI_KERNEL_LOG(ERROR,
                                "deserialize named data failed, type [%s] not "
                                "registered, graphId [%d] name [%s]",
                                typeId.c_str(), pbNamedData.graph_id(),
                                pbNamedData.name().c_str());
                return false;
            }
            if (!SerializeData::deserializeFromStr(NAVI_TLS_LOGGER, dataStr, dataType, data)) {
                NAVI_KERNEL_LOG(ERROR,
                                "deserialize named data failed, type [%s] "
                                "graphId [%d] name [%s]",
                                typeId.c_str(), pbNamedData.graph_id(),
                                pbNamedData.name().c_str());
                return false;
            }
        }
        namedData.graphId = pbNamedData.graph_id();
        namedData.name = pbNamedData.name();
        namedData.data = std::move(data);
        if (!params.addNamedData(std::move(namedData))) {
            NAVI_KERNEL_LOG(ERROR,
                            "named data name [%s] conflict, graphId [%d]",
                            namedData.name.c_str(), namedData.graphId);
            return false;
        }
    }
    return true;
}

void RunGraphParams::setDebugParamStr(const std::string &debugParamStr) {
    auto params = autil::StringTokenizer::tokenize(autil::StringView(debugParamStr), NAVI_DEBUG_DELIM1_STR);
    for (const auto &param : params) {
        auto pos = param.find(NAVI_DEBUG_DELIM2_STR);
        if (std::string::npos == pos) {
            continue;
        }
        const auto &key = param.substr(0, pos);
        const auto &value = param.substr(pos + 1);
        if (key == NAVI_DEBUG_METRIC_STR) {
            setCollectMetric(autil::StringUtil::fromString<bool>(value));
        } else if (key == NAVI_DEBUG_PERF_STR) {
            setCollectPerf(autil::StringUtil::fromString<bool>(value));
        } else if (key == NAVI_DEBUG_CONTEXT_SWITCH_STR) {
            setCollectContextSwitch(autil::StringUtil::fromString<bool>(value));
        } else if (key == NAVI_DEBUG_PAGE_FAULT_STR) {
            setCollectPageFault(autil::StringUtil::fromString<bool>(value));
        } else if (key == NAVI_DEBUG_TRACE_STR) {
            setTraceLevel(value);
        } else if (key == NAVI_DEBUG_MALLOC_STR) {
            setCollectMalloc(autil::StringUtil::fromString<bool>(value));
        } else if (key == NAVI_DEBUG_BT_FILTER_STR) {
            addBtFilterStr(value);
        } else if (key == NAVI_DEBUG_TIMEOUT_STR) {
            auto timeout = autil::StringUtil::fromString<size_t>(value);
            if (timeout > 0) {
                setTimeoutMs(timeout);
            }
        } else if (key == NAVI_DEBUG_RESOURCE_BUILD_MODE) {
            addResourceBuildModeStr(value);
        } else if (key == NAVI_DEBUG_PARAM_TOPO_TREE_STR) {
            addParamTopoTreeStr(value);
        } else if (key == NAVI_DEBUG_DISABLE_GIG_DEGRADE) {
            setDisableGigDegrade(autil::StringUtil::fromString<bool>(value));
        }
    }
}

void RunGraphParams::addBtFilterStr(const std::string &btFilterStr) {
    auto splited = autil::StringTokenizer::tokenize(
        autil::StringView(btFilterStr), NAVI_DEBUG_DELIM3_STR);
    if (2 != splited.size()) {
        return;
    }
    const auto &file = splited[0];
    const auto &lineStr = splited[1];
    int line = -1;
    if (!autil::StringUtil::fromString(lineStr, line)) {
        return;
    }
    addTraceBtFilter(file, line);
}

void RunGraphParams::addParamTopoTreeStr(const std::string &paramStr) {
    if (!_paramTopoTreeNode) {
        _paramTopoTreeNode = std::make_shared<ParamTopoTreeNode>();
    }
    _paramTopoTreeNode->addParam(paramStr);
    initTopoParam();
}

void RunGraphParams::addResourceBuildModeStr(const std::string &buildModeStr) {
    auto splited = autil::StringTokenizer::tokenize(
        autil::StringView(buildModeStr), NAVI_DEBUG_DELIM3_STR);
    if (1 == splited.size()) {
        setResourceBuildMode(ResourceCreator::parseResourceBuildMode(splited[0]));
        return;
    } else if (2 == splited.size()) {
        _resourceBuildModeMap[splited[0]] = ResourceCreator::parseResourceBuildMode(splited[1]);
        return;
    }
}

}
