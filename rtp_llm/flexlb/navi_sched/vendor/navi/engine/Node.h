#pragma once

#include "navi/common.h"
#include "navi/engine/Edge.h"
#include "navi/engine/Data.h"
#include "navi/engine/Kernel.h"
#include "navi/engine/KernelCreator.h"
#include "navi/engine/KernelMetric.h"
#include "navi/engine/NodeAsyncInfo.h"
#include "navi/engine/ComputeScope.h"
#include "navi/engine/KernelWorkItem.h"
#include "navi/ops/ResourceKernelDef.h"
#include "navi/proto/GraphDef.pb.h"
#include "navi/proto/KernelDef.pb.h"
#include <map>
#include <string>
#include <vector>

namespace navi {

class Node;

#define NAVI_NODE_LOG(level, format, args...)                                                                          \
    {                                                                                                                  \
        static_assert(std::is_same<std::remove_cv<std::remove_pointer_t<decltype(this)>>::type, Node>::value,          \
                      "type must be Edge");                                                                            \
        NAVI_OBJECT_LOG(this, LOT_NODE, level, format, ##args);                                                        \
    }

class InputPortGroup;
struct InputPortGroupInternal;
class OutputPortGroup;
struct OutputPortGroupInternal;
class LocalSubGraph;
class Biz;
class Graph;
class GraphDomainFork;
class TimeoutChecker;
class ScopeInfo;
class LocalPortBase;

struct InputSnapshot {
public:
    InputSnapshot() {
        reset();
    }
    void reset() {
        peeked = false;
        cleared = false;
        data.reset(false);
    }
public:
    bool cleared;
    bool peeked;
    StreamData data;
};

union ExpensiveFlag {
    uint32_t value;
    struct {
        uint16_t expensive;
        uint16_t valid;
    };
};

struct NodeInternal {
    NodeInternal(int32_t id,
                 const NodeDef &nodeDef,
                 LocalSubGraph *graph,
                 Biz *biz,
                 bool isResourceNode)
        : _id(id)
        , _graph(graph)
        , _nodeDef(&nodeDef)
        , _name(nodeDef.name())
        , _kernelName(nodeDef.kernel_name())
        , _biz(biz)
        , _resourceInputGroupIndex(INVALID_INDEX)
        , _controlInputGroupIndex(INVALID_INDEX) {
        _isResourceNode = isResourceNode;
        _inlineConfigAndInit = false;
        _hasInput = false;
        _hasOutput = false;
        _started = false;
        _expensive.expensive = 1;
        _expensive.valid = 0;
        if (_isResourceNode) {
            const auto &binaryAttrMap = nodeDef.binary_attrs();
            auto it = binaryAttrMap.find(RESOURCE_ATTR_NAME);
            if (binaryAttrMap.end() != it) {
                _resourceName = it->second;
            }
        }
    }
    ~NodeInternal();
public:
    int32_t getId() const { return _id; }
    const std::string &getName() const { return _name; }
    const std::string &getKernelName() const { return _kernelName; }
    const std::string &getResourceName() const { return _resourceName; }
    bool isResourceCreateKernel() const { return _isResourceNode; }
    bool init(const NodeDef &def, KernelMemEstimator &estimator);
    bool bindPort(Port *port);
    void initIOGroup();
    void setHasOutput();
    bool initCreatorStats(const NodeDef &def);
    bool addOutput(EdgeInternal *output,
                   const std::string &portName,
                   std::string &dataType);
    PortIndex doAddOutput(const PortInfo &portInfo, EdgeInternal *output);
    PortIndex doAddOutputGroup(const PortInfo &portInfo, EdgeInternal *output);
    bool addInput(const std::string &portName,
                  EdgeOutputInfo &edgeOutputInfo,
                  PortIndex &inputIndex,
                  std::string &dataType);
    PortIndex doAddInput(const PortInfo &portInfo,
                         EdgeOutputInfo &edgeOutputInfo);
    PortIndex doAddInputGroup(const PortInfo &portInfo,
                              EdgeOutputInfo &edgeOutputInfo);
    ActivateStrategy getInputActivateStrategy(const InputDef *inputDef);
    bool checkOrFinishInput();
    bool checkInputGroup() const;
    bool postInit();
    bool initInputSnapshotVec();
    void initOutputSnapshotVec();
    void initOutputDegree();
    size_t getResourceInputGroupCount();
    bool initResourceIndexMap();
    InputPortGroupInternal *addControlInputGroup();
    InputPortGroupInternal *addResourceInputGroup();
    void setErrorCode(ErrorCode ec, const ComputeScopePtr &computeScope);
    void setHasInput();
    bool getInputInfo(PortIndex index,
                      EdgeOutputInfo &info,
                      IndexType &inputIndex) const;
    bool getOutputEdge(PortIndex index, EdgeInternal *&edge) const;
    bool getIORange(IoType ioType,
                    const std::string &name,
                    PortIndex &start,
                    PortIndex &end) const;
public:
    const int32_t _id;
    LocalSubGraph *_graph;
    const NodeDef *_nodeDef = nullptr;
    const std::string _name;
    const std::string _kernelName;
    std::string _resourceName;
    Biz *_biz;
    const KernelCreator *_kernelCreator = nullptr;
    NaviVector<EdgeOutputInfo, 4> _inputs;
    NaviVector<IndexType, 4> _disconnectInputs;
    NaviVector<EdgeInternal *, 4> _outputs;
    NaviVector<EdgeInternal *, 4> _controlOutputs;
    NaviVector<InputPortGroupInternal *, 4> _inputGroups;
    NaviVector<OutputPortGroupInternal *, 4> _outputGroups;
    CreatorStats *_creatorStats = nullptr;
    KernelConfigContext _ctx;
    IndexType _resourceInputGroupIndex;
    IndexType _controlInputGroupIndex;
    IndexType _inputSnapshotVecSize = 0;
    int32_t _scopeIndex = -1;
    NaviUnorderedMap<std::string, std::pair<IndexType, ResourceStage>> _resourceIndexMap;
    NaviVector<NaviVector<int32_t, 2>, 4> *_selectIndexVec = nullptr;
    NaviVector<IndexType, 4> _flushIndexVec;
    uint32_t _outputDegree;
    bool _isResourceNode : 1;
    bool _inlineConfigAndInit : 1;
    bool _hasInput : 1;
    bool _hasOutput : 1;
    bool _started : 1;
    ExpensiveFlag _expensive;
    size_t _runStateOffset = 0;
    size_t _memOffset = 0;
    size_t _memSize = 0;
};

struct NodeRunState {
public:
    NodeRunState(NaviWorkerBase *worker, const CollectFlags &collectFlags, Node *node)
        : kernelWorkItem(worker, collectFlags, node)
    {
    }
    ~NodeRunState() {
        if (jsonAttrsDocument) {
            delete jsonAttrsDocument;
        }
    }
public:
    bool inited = false;
    KernelWorkItem kernelWorkItem;
    Kernel *kernel = nullptr;
    ResourceMapPtr resourceMap;
    autil::legacy::RapidDocument *jsonAttrsDocument = nullptr;
    NaviVector<int32_t, 1> selectorResult;
    std::vector<bool> *startBitMap = nullptr;
    NodeAsync *nodeAsync = nullptr;
    mutable autil::RecursiveThreadMutex forkLock;
    GraphDef *forkGraphDef = nullptr;
    Graph *forkGraph = nullptr;
};

class Node
{
private:
    static constexpr int64_t INPUT_PENDING_WORK_FLAG = (1LL << 62);
    static constexpr int64_t OUTPUT_PENDING_WORK_FLAG = (1LL << 61);
    static constexpr int64_t STOP_PENDING_WORK_FLAG = (1LL << 60);
    static constexpr int64_t DOING_PENDING_WORK_FLAG = (1LL << 59);
    static constexpr int64_t PENDING_WORK_MASK =
        (INPUT_PENDING_WORK_FLAG | OUTPUT_PENDING_WORK_FLAG);

private:
    enum SS_STATE : int64_t {
        SS_INIT = 0,
        SS_IDLE = 1,
        SS_SCHEDULING = 2,
        SS_RUNNING = 3,
        SS_FINISH = 4,
    };
public:
    Node(const NodeDef &nodeDef,
         LocalSubGraph *graph,
         NodeInternal *internal,
         char *baseAddr,
         bool needCollect);
    ~Node();

private:
    Node(const Node &);
    Node& operator=(const Node &);
public:
    const std::string &getName() const { return _def->name(); }
    const std::string &getKernelName() const { return _internal->getKernelName(); }
    const NaviObjectLogger &getLogger() const {
        return getScopeInfo()->objectLogger;
    }
    const NodeDef *getDef() const {
        return _def;
    }
    int32_t getId() const {
        return _internal->getId();
    }
    // kernel context interface
    bool getInput(PortIndex index, EdgeGetType type, StreamData &streamData, bool clear);
    bool getInputGroupSize(IndexType group, size_t &size);
    size_t getInputGroupCount() const;
    bool setInputEof(PortIndex index);
    bool getOutputGroupSize(IndexType group, size_t &size);
    bool snapshotGroupInput(IndexType group, EdgeGetType type, InputSnapshot *&begin, InputSnapshot *&end);
    bool setOutput(PortIndex index,
                   const DataPtr &data,
                   Data *dataPointer,
                   bool eof);
    bool outputFinish() const;
    bool inputFinish() const;
    bool inputHasReady() const;
    bool inputFinishConsumed() const;
    bool getIORange(IoType ioType, const std::string &name, PortIndex &start,
                    PortIndex &end) const;
    bool getInputNode(PortIndex index, std::string &node) const;
    bool inputConnected(PortIndex index);
    bool outputConnected(PortIndex index);
    IndexType getResourceInputGroupIndex() const {
        return _internal->_resourceInputGroupIndex;
    }
    bool isResourceCreateKernel() const {
        return _internal->isResourceCreateKernel();
    }
    bool isBorder() const {
        return _isInputBorder || _isOutputBorder;
    }
    NaviErrorPtr getScopeError() const;
    ErrorHandleStrategy getScopeErrorHandleStrategy() const;
    int64_t getRemainTimeMs() const;
    const TimeoutChecker *getTimeoutChecker() const;
public:
    // engine interface
    bool bindPort(Port *port);
    void setHasInput();
    bool hasInput() const;
    void setHasOutput();
    bool hasOutput() const;
    const std::string &getConfigPath() const;
    Port *getPort() const;
    bool getPortTypeStr(const std::string &portName, IoType ioType,
                        std::string &typeStr) const;
    bool isStarted() const {
        return _scheduleStat != SS_INIT;
    }
    void setErrorCode(ErrorCode ec, const ComputeScopePtr &computeScope);
    void reportError(const NaviErrorPtr &error);
    void notifyFinish(const NaviErrorPtr &error);
    bool startSchedule(const Node *callNode,
                       bool ignoreFrozen,
                       bool graphStart);
    bool schedule(const Node *callNode, bool ignoreFrozen = false);
    KernelWorkItem *getKernelWorkItem() const {
        return &_runState->kernelWorkItem;
    }
    bool isInline() const;
    size_t outputDegree() const;
    ComputeScopePtr makeComputeScope(const ScheduleInfo &schedInfo, NaviPerfThread *perfThread);
    void compute(const ComputeScopePtr &computeScope);
    inline void incNodeSnapshot() {
        // snapshot can't be equal
        atomic_inc(&_nodeSnapshot);
    }
    inline int64_t readNodeSnapshot() const {
        return atomic_read(&_nodeSnapshot) &
               (~(PENDING_WORK_MASK | STOP_PENDING_WORK_FLAG));
    }
    bool setInputPendingWork(const Node *callNode) {
        return setPendingWork(INPUT_PENDING_WORK_FLAG, callNode);
    }
    bool setOutputPendingWork(const Node *callNode) {
        return setPendingWork(OUTPUT_PENDING_WORK_FLAG, callNode);
    }
    void setStopPendingWork(const Node *callNode) {
        setPendingWork(STOP_PENDING_WORK_FLAG, callNode);
    }
    LocalSubGraph *getGraph() const;
    const KernelCreator *getKernelCreator() const;
    ErrorCode fork(GraphDef *graphDef, const ForkGraphParam &param);
    void stopSchedule();
    int64_t getScheduleCount() const;
    void updateTraceLevel(const std::string &levelStr);
    bool updateTimeoutMs(int64_t timeoutMs);
    void updateThreadLimit(uint32_t threadLimit);
    void updateCollect(bool collectMetric, bool collectPerf) const;
    void collectTrace(std::vector<std::string> &traceVec) const;
    std::shared_ptr<multi_call::GigStreamRpcInfoMap> getRpcInfoMap() const;
    void terminate(ErrorCode ec);
    inline int64_t getComputeId() const {
        return _scheduleCount;
    }
    void collectState();
    const KernelMetricPtr &getMetric() const;
    void collectForkGraphMetric() const;
    KernelConfigContext getConfigContext();
    bool getConfigContext(const std::string &dependName, KernelConfigContext &ctx);
    const std::string &getBizName() const;
    bool getBizPartInfo(const std::string &bizName, NaviPartId &partCount,
                        std::vector<NaviPartId> &partIds) const;
    bool getBinaryAttr(const std::string &attr, std::string &value) const;
    ResourcePtr buildR(const std::string &name, KernelConfigContext *ctx, ResourceMap &inputResourceMap);
    void cacheExpensive(bool value);
    const ReplaceInfoMap *getReplaceInfoMap() const {
        return _graph->getReplaceInfoMap();
    }

    // bitmap interface
    AsyncPipePtr createAsyncPipe(ActivateStrategy activateStrategy);
    AsyncPipePtr doCreateAsyncPipe(ActivateStrategy activateStrategy);
    bool registerPeriodicPipe(
        const AsyncPipePtr &asyncPipe,
        int64_t periodMs,
        int64_t repeat);
    bool getPeriodicPipeActivationCount(
        const AsyncPipePtr &asyncPipe,
        int64_t &activatedCount,
        int64_t &remainingCount,
        int64_t &periodMs,
        int64_t &nextTickTimeMs) const;
    bool stopPeriodicAsyncPipe(const AsyncPipePtr &asyncPipe);
    int64_t checkTimeoutAsyncPipes(int64_t endTimeMs);
    void notifyPipeTerminate();
    int32_t getScope() const {
        return _internal->_scopeIndex;
    }
    const ScopeInfo *getScopeInfo() const {
        return _graph->getScopeInfo(_internal->_scopeIndex);
    }
    void setForceStopFlag(Node *node);
    void forceStop();
    std::string getLogPrefix() const;
    // for test
    Kernel *getKernel() const;
    void setFrozen(bool frozen);
    bool isFinished() const;
private:
    bool collectInputResource(ResourceMap &resourceMap);
    bool bindResource(void *castPtr);
    bool bindNamedData(void *castPtr);
    void initIOGroup();
    bool initCreatorStats();
    void initOutputDegree();
    void initResourceIndexMap();
    bool createKernel(const ComputeScopePtr &computeScope);
    bool initKernel(const ComputeScopePtr &computeScope);
    bool initAttribute();
    bool initConfigContext(bool parse, KernelConfigContext &ctx);
    bool createSelectResult();
    void deleteKernel(bool inDestruct = false);
    InputPortGroup *addControlInputGroup();
    InputPortGroup *addResourceInputGroup();
    bool inputOk() const;
    bool outputOk() const;
    bool resourceInputOk() const;
    bool getInputInfo(PortIndex index, EdgeOutputInfo &info, IndexType &inputIndex) const;
    bool getOutputEdgeInternal(PortIndex index, EdgeInternal *&edge) const;
    bool getOutputEdge(PortIndex index, Edge *&edge);
    bool checkForkGraph(const GraphDef *graphDef) const;
    bool bindDomain(GraphDomainFork *domain);
    void reportForkError(const KernelMetricPtr &forkMetric,
                         const ComputeScopePtr &computeScope,
                         ErrorCode ec) const;
    void destructData(DataPtr &&data) const;
    void notifyFinish(ErrorCode ec);
    bool setPendingWork(int64_t mask, const Node *callNode);
    void finishPendingWork();
    void doFinishPendingWork(int64_t snapshot);
    void collectInputStates(KernelInputStatesDef *states) const;
    void collectOutputStates(KernelOutputStatesDef *states);
private:
    bool scheduleLock(const Node *callNode);
    bool doSchedule(const Node *callNode, bool ignoreFrozen, bool &resched, bool &firstTry);
    ErrorCode doCompute(const ComputeScopePtr &computeScope);
    void clearControlInput();
    InputSnapshot &doFillInput(const EdgeOutputInfo &info,
                               IndexType inputIndex,
                               EdgeGetType type);
    void resetIOSnapshot();
    bool checkFinish(ErrorCode ec, const Node *callNode);
    void propagateEof();
    void doForceStop();
    size_t mySwapPoolMallocSize(size_t size);
    size_t mySwapPoolMallocCount(size_t count);
    void startKernelResourceInput() const;
    void startNormalResourceInput(bool graphStart) const;
    void startSelectResourceInput() const;
    void setAsyncInfo(const NodeAsyncInfoPtr &asyncInfo);
    NodeAsyncInfoPtr getAsyncInfo() const;
    void recycleTimeoutNode(const NodeAsyncInfoPtr &asyncInfo);
    bool hasActivePeriodicPipe() const;
private:
    const NodeDef *_def;
    LocalSubGraph *_graph;
    NodeInternal *_internal;
    char *_baseAddr;
    KernelMetricPtr _metric;
    std::vector<InputSnapshot> _inputSnapshotVec;
    int64_t _scheduleCount;
    atomic64_t _nodeSnapshot;
    volatile SS_STATE _scheduleStat;
    NodeRunState *_runState = nullptr;
    std::atomic<bool> _forceStopped; // need_restore
    bool _hasInput : 1;
    bool _hasOutput : 1;
    bool _skipConfig : 1;
    bool _skipInit : 1;
    bool _stopAfterInit : 1;
    bool _skipDeleteKernel : 1;
    bool _isInputBorder : 1;
    bool _isOutputBorder : 1;
    bool _kernelInited;
    bool _allInputEof;
    bool _frozen;
    bool _stopSchedule;
    bool _forceStop;
    const Node *_forceStopNode;
    std::atomic<bool> _timeoutNodeRegistered{false};
    LocalPortBase *_port = nullptr;
private:
    static constexpr float INLINE_LATENCY_THRESHOLD_US = 100.0f;
private:
    friend struct NodeResourceBuilder;
    friend class LocalSubGraph;
};

}
