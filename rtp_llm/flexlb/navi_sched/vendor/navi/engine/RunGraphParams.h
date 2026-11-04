#pragma once

#include "navi/common.h"
#include "navi/engine/Data.h"
#include "navi/engine/NamedData.h"
#include "navi/engine/CollectFlags.h"
#include "navi/proto/KernelDef.pb.h"
#include <set>

namespace multi_call {
class QuerySession;
} // namespace multi_call

namespace navi {

class RunParams;
class OverrideDataDef;
class CreatorManager;
class TraceAppender;
class ParamTopoTreeNode;
class TopoNodeParam;

struct OverrideData {
public:
    OverrideData()
        : graphId(INVALID_GRAPH_ID)
        , partId(INVALID_NAVI_PART_ID)
    {
    }
public:
    GraphId graphId;
    NaviPartId partId;
    std::string outputNode;
    std::string outputPort;
    std::vector<DataPtr> datas;
    std::vector<Data *> dataPointers;
};

class RunGraphParams
{
public:
    RunGraphParams();
    ~RunGraphParams();
public:
    // for user
    void setSessionId(SessionId id);
    SessionId getSessionId() const;
    void setThreadLimit(uint32_t limit);
    uint32_t getThreadLimit() const;
    void setTimeoutMs(int64_t timeout);
    int64_t getTimeoutMs() const;
    void setGlobalStartTimeUs(int64_t startTimeUs);
    int64_t getGlobalStartTimeUs() const;
    void setTaskQueueName(const std::string &taskQueueName);
    const std::string &getTaskQueueName() const;
    void setTraceLevel(const std::string &traceLevelStr);
    LogLevel getTraceLevel() const;
    void setTraceFormatPattern(const std::string &pattern);
    const std::string &getTraceFormatPattern() const;
    void addTraceBtFilter(const std::string &file, int line);
    const std::vector<std::pair<std::string, int>> &getTraceBtFilter() const;
    void setCollectMetric(bool collect);
    void setCollectPerf(bool collect);
    void setCollectContextSwitch(bool collect);
    void setCollectPageFault(bool collect);
    void setCollectMalloc(bool collect);
    void setDebugParamStr(const std::string &debugParamStr);
    void setCollectFlags(uint64_t flags);
    CollectFlags getCollectFlags() const;
    void setAsyncIo(bool async);
    bool asyncIo() const;
    void setDisableGigDegrade(bool disable);
    bool disableGigDegrade() const;
    int32_t getMaxInline() const;
    void setMaxInline(int32_t maxInline);
    bool overrideEdgeData(const OverrideData &data);
    const std::vector<OverrideData> &getOverrideDatas() const;
    void clearOverrideDatas();
    void setQuerySession(std::shared_ptr<multi_call::QuerySession> querySession);
    const std::shared_ptr<multi_call::QuerySession> &getQuerySession() const;
    bool addNamedData(const NamedData &data);
    void clearNamedData();
    const SubNamedDataMap *getSubNamedDataMap(GraphId graphId) const;
    const NamedDataMap &getNamedDatas() const;
    void setResourceStage(ResourceStage stage);
    ResourceStage getResourceStage() const;
    void addParamTopoTreeStr(const std::string &paramStr);
    void setResourceBuildMode(ResourceBuildMode mode);
    ResourceBuildMode getResourceBuildMode() const;
    ResourceBuildMode getResourceBuildMode(const std::string &name) const;
    void setResourceBuildMode(const std::string &name, ResourceBuildMode mode);
    const NaviUnorderedMap<std::string, ResourceBuildMode> &getResourceBuildModeMap() const;
public:
    // internal
    void initGlobalStartTimeUs();
    bool needLogPrefix() const;
    LogLevel getTraceLevelWithoutTopo() const;
    void fillFilterParam(TraceAppender *traceAppender) const;
    const std::shared_ptr<ParamTopoTreeNode> &getParamTopoTree() const;
    const std::shared_ptr<TopoNodeParam> &getTopoParam() const;
    void initTopoParam();
    std::shared_ptr<ParamTopoTreeNode>
    getParamNode(GraphId graphId, NaviPartId partId, const std::string &forkNode) const;
    void setParamNode(const std::shared_ptr<ParamTopoTreeNode> &node);
    CollectFlags getCollectFlags(const std::shared_ptr<ParamTopoTreeNode> &paramTopoTreeNode) const;
    void setLoadGraph(bool value);
    bool isLoadGraph() const;
public:
    // internal
    static bool fromProto(const RunParams &pbParams,
                          CreatorManager *creatorManager,
                          RunGraphParams &params);
    static ErrorCode toProto(const RunGraphParams &params,
                             GraphId matchGraphId,
                             CreatorManager *creatorManager,
                             NaviPartId partId,
                             int64_t timeoutMs,
                             const std::shared_ptr<ParamTopoTreeNode> &paramTopoTreeNode,
                             RunParams &pbParams);
private:
    static void deserializeParamTopoTree(const RunParams &pbParams, RunGraphParams &params);
    static void deserializeTraceBtFilter(const RunParams &pbParams,
                                         RunGraphParams &params);
    static void serializeParamTopoTree(const std::shared_ptr<ParamTopoTreeNode> &paramTopoTreeNode,
                                       RunParams &pbParams);
    static void deserializeResourceBuildMode(const RunParams &pbParams, RunGraphParams &params);
    static void serializeResourceBuildMode(const RunGraphParams &params, RunParams &pbParams);
    static void serializeTraceBtFilter(const RunGraphParams &params,
                                       RunParams &pbParams);
    static bool parseOverrideDataValues(const RunParams &pbParams,
                                        CreatorManager *creatorManager,
                                        NaviUnorderedMap<int32_t, DataPtr> &dataMap);
    static bool deserializeOverrideData(const RunParams &pbParams,
                                        CreatorManager *creatorManager,
                                        RunGraphParams &params);
    static bool serializeOverrideData(
            const std::vector<OverrideData> &overrideEdgeDatas,
            GraphId matchGraphId,
            CreatorManager *creatorManager,
            NaviPartId partId,
            RunParams &pbParams);
    static bool serializeNamedData(const NamedDataMap &namedDataMap,
                                   GraphId matchGraphId,
                                   CreatorManager *creatorManager,
                                   RunParams &pbParams);
    static bool deserializeNamedData(const RunParams &pbParams,
                                     CreatorManager *creatorManager,
                                     RunGraphParams &params);
private:
    void addBtFilterStr(const std::string &btFilterStr);
    void addResourceBuildModeStr(const std::string &buildModeStr);
private:
    SessionId _id;
    uint32_t _threadLimit;
    int64_t _timeoutMs;
    int64_t _globalStartTimeUs;
    std::string _taskQueueName;
    LogLevel _traceLevel;
    std::string _traceFormatPattern;
    std::vector<std::pair<std::string, int>> _traceBtFilterParams;
    bool _asyncIo;
    bool _disableGigDegrade;
    bool _loadGraph;
    CollectFlags _collectFlags;
    int32_t _maxInline;
    std::vector<OverrideData> _overrideDatas;
    std::shared_ptr<multi_call::QuerySession> _querySession;
    NamedDataMap _namedDatas;
    ResourceStage _resourceStage;
    std::shared_ptr<ParamTopoTreeNode> _paramTopoTreeNode;
    std::shared_ptr<TopoNodeParam> _paramTopo;
    ResourceBuildMode _resourceBuildMode = RBM_UNKNOWN;
    NaviUnorderedMap<std::string, ResourceBuildMode> _resourceBuildModeMap;
};

}
