#include "navi/engine/Node.h"
#include "autil/TimeUtility.h"
#include "navi/engine/Biz.h"
#include "navi/engine/Edge.h"
#include "navi/engine/Graph.h"
#include "navi/engine/GraphDomainFork.h"
#include "navi/engine/InputPortGroup.h"
#include "navi/engine/KernelWorkItem.h"
#include "navi/engine/LocalPortBase.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/OutputPortGroup.h"
#include "navi/engine/ScopeTerminatorKernel.h"
#include "navi/log/NaviLogger.h"
#include "navi/ops/ResourceData.h"
#include "navi/proto/GraphVis.pb.h"
#include "navi/util/CommonUtil.h"
#include <iostream>
#include <limits>
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"

namespace navi {

class DataDestructItem : public NaviNoDropWorkerItem
{
public:
    DataDestructItem(NaviWorkerBase *worker, const CollectFlags &collectFlags, DataPtr &&data)
        : NaviNoDropWorkerItem(worker, collectFlags)
        , _data(data)
    {
    }
public:
    void doProcess(NaviPerfThread *perfThread) override {
        if (_data) {
            _data.reset();
        }
    }
private:
    DataPtr _data;
};

class KernelDestructItem : public NaviNoDropWorkerItem
{
public:
    KernelDestructItem(NaviWorkerBase *worker,
                       const CollectFlags &collectFlags,
                       LocalSubGraph *graph,
                       KernelMetric *metric,
                       Kernel *kernel)
        : NaviNoDropWorkerItem(worker, collectFlags)
        , _graph(graph)
        , _metric(metric)
        , _kernel(kernel)
    {
    }
public:
    void doProcess(NaviPerfThread *perfThread) override {
        if (_kernel) {
            auto graphParam = _graph->getParam();
            auto collectFlags = graphParam->runParams.getCollectFlags();
            if (collectFlags.needCollect()) {
                const auto &topoParam = graphParam->runParams.getTopoParam();
                ComputeScope scope(_metric,
                                   nullptr,
                                   _graph->getPartId(),
                                   perfThread,
                                   graphParam->memLeakCollector.get(),
                                   _schedInfo,
                                   collectFlags,
                                   topoParam ? &topoParam->getMallocTraceParam()
                                             : nullptr);
                scope.setType(GET_KERNEL_DELETE_KERNEL);
                _kernel->~Kernel();
                _kernel = nullptr;
            } else {
                _kernel->~Kernel();
                _kernel = nullptr;
            }
        }
    }
private:
    LocalSubGraph *_graph;
    KernelMetric *_metric;
    Kernel *_kernel;
};

NodeInternal::~NodeInternal() {
    for (const auto &input : _inputGroups) {
        delete input;
    }
    for (const auto &output : _outputGroups) {
        delete output;
    }
    DELETE_AND_SET_NULL(_selectIndexVec);
}

Node::Node(const NodeDef &nodeDef,
           LocalSubGraph *graph,
           NodeInternal *internal,
           char *baseAddr,
           bool needCollect)
    : _def(&nodeDef)
    , _graph(graph)
    , _internal(internal)
    , _baseAddr(baseAddr)
    , _scheduleCount(0)
    , _forceStopped(false)
{
    _hasInput = _internal->_hasInput;
    _hasOutput = _internal->_hasOutput;
    _isInputBorder = false;
    _isOutputBorder = false;
    _kernelInited = false;
    _allInputEof = false;
    atomic_set(&_nodeSnapshot, 0);
    _scheduleStat = _internal->_started ? SS_IDLE : SS_INIT;
    _frozen = false;
    _stopSchedule = false;
    _forceStop = false;
    _forceStopNode = nullptr;
    if (unlikely(needCollect)) {

        _metric = std::make_shared<KernelMetric>(_graph->getGraphId(), nodeDef.name(), nodeDef.kernel_name());
    }
    if (unlikely(_def->has_buildin_attrs())) {
        const auto &buildinAttrs = _def->buildin_attrs();
        _skipConfig = buildinAttrs.skip_config();
        _skipInit = buildinAttrs.skip_init();
        _stopAfterInit = buildinAttrs.stop_after_init();
        _skipDeleteKernel = buildinAttrs.skip_delete_kernel();
    } else {
        _skipConfig = false;
        _skipInit = false;
        _stopAfterInit = false;
        _skipDeleteKernel = false;
    }
}

Node::~Node() {
    deleteKernel(true);
    if (_runState) {
        DELETE_AND_SET_NULL(_runState->forkGraph);
        multi_call::freeProtoMessage(_runState->forkGraphDef);
        DELETE_AND_SET_NULL(_runState->startBitMap);
        DELETE_AND_SET_NULL(_runState->nodeAsync);
        _runState->~NodeRunState();
    }
}

bool NodeInternal::init(const NodeDef &def, KernelMemEstimator &estimator) {
    if (getName().empty()) {
        NAVI_KERNEL_LOG(ERROR,
                        "node name can't be empty, def: %s",
                        def.DebugString().c_str());
        return false;
    }
    const auto &kernelName = getKernelName();
    _kernelCreator = _biz->getKernelCreator(kernelName);
    if (!_kernelCreator) {
        NAVI_KERNEL_LOG(ERROR, "kernel not supported");
        return false;
    }
    if (_kernelCreator->isInlineConfigAndInit() || _isResourceNode ||
        kernelName == DEFAULT_SPLIT_KERNEL ||
        kernelName == DEFAULT_MERGE_KERNEL ||
        kernelName == TERMINATOR_KERNEL_NAME) {
        _inlineConfigAndInit = true;
    }
    _inputs.resize(_kernelCreator->inputSize());
    _outputs.resize(_kernelCreator->outputSize(), nullptr);
    _controlOutputs.resize(_kernelCreator->controlOutputSize(), nullptr);
    initIOGroup();
    if (!_outputs.empty() || !_outputGroups.empty()) {
        setHasOutput();
    }
    _ctx = KernelConfigContext(_biz->getConfigPath(),
                               nullptr,
                               _biz->getKernelConfig(kernelName),
                               nullptr,
                               _kernelCreator->getJsonAttrs());
    if (!initCreatorStats(def)) {
        return false;
    }
    _runStateOffset = estimator.add(sizeof(NodeRunState), alignof(NodeRunState), 1);
    _memOffset = _kernelCreator->collectMemInfo(estimator);
    _memSize = estimator.getBaseAddr() - _memOffset;
    return true;
}

void NodeInternal::initIOGroup() {
    const auto *kernelDef = _kernelCreator->def();
    for (auto i = 0; i < kernelDef->input_groups().size(); ++i) {
        const auto &inputGroupDef = kernelDef->input_groups(i);
        auto group = new InputPortGroupInternal(inputGroupDef.type(), inputGroupDef.group_type());
        _inputGroups.push_back(group);
        NAVI_KERNEL_LOG(SCHEDULE2, "add IO inputGroup[%s]", autil::StringUtil::toString(group).c_str());
    }
    for (auto i = 0; i < kernelDef->output_groups().size(); ++i) {
        auto group = new OutputPortGroupInternal();
        _outputGroups.push_back(group);
    }
}

bool NodeInternal::initCreatorStats(const NodeDef &def) {
    if (!isResourceCreateKernel()) {
        _creatorStats = &_kernelCreator->getCreatorStats();
    } else {
        const auto &binaryAttrMap = def.binary_attrs();
        auto it = binaryAttrMap.find(RESOURCE_ATTR_NAME);
        if (binaryAttrMap.end() == it) {
            NAVI_KERNEL_LOG(ERROR, "can't find resource name in node binary attr");
            return false;
        }
        auto resourceCreator = _biz->getResourceCreator(it->second);
        if (!resourceCreator) {
            return true;
        }
        _creatorStats = &resourceCreator->getCreatorStats();
    }
    return true;
}

bool NodeInternal::addOutput(EdgeInternal *output,
                             const std::string &portName,
                             std::string &dataType) {
    if (!output) {
        NAVI_KERNEL_LOG(ERROR, "input is null");
        return false;
    }
    auto portInfo = _kernelCreator->getOutputPortInfo(portName);
    if (!portInfo.isValid()) {
        NAVI_KERNEL_LOG(ERROR, "no output port named [%s], kernel def[%s]",
                 portName.c_str(),
                 _kernelCreator->def()->DebugString().c_str());
        return false;
    }
    PortIndex outputIndex;
    if (portInfo.isGroup()) {
        outputIndex = doAddOutputGroup(portInfo, output);
        dataType = static_cast<const OutputGroupDef *>(portInfo.def)->data_type().name();
    } else {
        outputIndex = doAddOutput(portInfo, output);
        dataType = static_cast<const OutputDef *>(portInfo.def)->data_type().name();
    }
    if (outputIndex.isValid()) {
        return true;
    } else {
        return false;
    }
}

PortIndex NodeInternal::doAddOutput(const PortInfo &portInfo, EdgeInternal *output) {
    auto portIndex = portInfo.index;
    if (portIndex.isControl()) {
        assert(portIndex.index() < _controlOutputs.size());
        if (_controlOutputs[portIndex.index()]) {
            NAVI_KERNEL_LOG(
                ERROR,
                "got multiple control link, control output port index [%u]",
                portIndex.index());
            return PortIndex();
        }
        _controlOutputs[portIndex.index()] = output;
        return portIndex;
    } else {
        assert(portIndex.index() < _outputs.size());
        assert(!_outputs[portIndex.index()]);
        _outputs[portIndex.index()] = output;
        return portIndex;
    }
}

PortIndex NodeInternal::doAddOutputGroup(const PortInfo &portInfo,
                                         EdgeInternal *output) {
    auto groupIndex = portInfo.index;
    assert(groupIndex.group() < _outputGroups.size());
    _outputGroups[groupIndex.group()]->addOutput(groupIndex.index(), output);
    return groupIndex;
}

bool NodeInternal::addInput(const std::string &portName,
                            EdgeOutputInfo &edgeOutputInfo,
                            PortIndex &inputIndex,
                            std::string &dataType) {
    autil::StringView port;
    IndexType index = 0;
    if (!CommonUtil::parseGroupPort(portName, port, index)) {
        NAVI_KERNEL_LOG(ERROR,
                 "parse portName [%s] failed, kernel def[%s]",
                 portName.c_str(),
                 _kernelCreator->def()->DebugString().c_str());
        setErrorCode(EC_INIT_GRAPH, nullptr);
        return false;
    }
    if (autil::StringView(NODE_CONTROL_INPUT_PORT) == port) {
        auto controlGroup = addControlInputGroup();
        auto offset = controlGroup->getInputCount();
        controlGroup->addInput(offset, edgeOutputInfo);
        setHasInput();
        inputIndex.setGroup(_controlInputGroupIndex);
        inputIndex.setIndex(offset);
        return true;
    }
    if (autil::StringView(NODE_RESOURCE_INPUT_PORT) == port) {
        auto resourceGroup = addResourceInputGroup();
        auto offset = resourceGroup->getInputCount();
        resourceGroup->addInput(offset, edgeOutputInfo);
        setHasInput();
        inputIndex.setGroup(_resourceInputGroupIndex);
        inputIndex.setIndex(offset);
        dataType = "";
        return true;
    }
    assert(edgeOutputInfo.isValid());
    auto portInfo = _kernelCreator->getInputPortInfo(port, index);
    if (!portInfo.isValid()) {
        NAVI_KERNEL_LOG(ERROR,
                 "no input port named [%s], check port name or group index, "
                 "kernel def[%s]",
                 portName.c_str(),
                 _kernelCreator->def()->DebugString().c_str());
        setErrorCode(EC_INIT_GRAPH, nullptr);
        return false;
    }
    if (portInfo.isGroup()) {
        inputIndex = doAddInputGroup(portInfo, edgeOutputInfo);
        dataType = static_cast<const InputGroupDef *>(portInfo.def)->data_type().name();
    } else {
        inputIndex = doAddInput(portInfo, edgeOutputInfo);
        dataType = static_cast<const InputDef *>(portInfo.def)->data_type().name();
    }
    if (inputIndex.isValid()) {
        setHasInput();
        return true;
    } else {
        NAVI_KERNEL_LOG(ERROR, "invalid index");
        setErrorCode(EC_INIT_GRAPH, nullptr);
        return false;
    }
}

PortIndex NodeInternal::doAddInput(const PortInfo &portInfo,
                                   EdgeOutputInfo &edgeOutputInfo) {
    auto portIndex = portInfo.index;
    auto inputDef = (InputDef *)portInfo.def;
    assert(inputDef);
    assert(portIndex.index() < _inputs.size());

    auto &outputInfo = _inputs[portIndex.index()];
    if (outputInfo.isValid()) {
        NAVI_KERNEL_LOG(ERROR, "multiple input, edge [%s] and edge [%s]",
                 outputInfo.getDebugName(nullptr).c_str(),
                 edgeOutputInfo.getDebugName(nullptr).c_str());
        return PortIndex();
    }
    outputInfo = edgeOutputInfo;
    if (AS_OPTIONAL == getInputActivateStrategy(inputDef)) {
        edgeOutputInfo.setInternlRequire(false);
    }
    return portIndex;
}

PortIndex NodeInternal::doAddInputGroup(const PortInfo &portInfo,
                                        EdgeOutputInfo &edgeOutputInfo) {
    auto groupIndex = portInfo.index;
    assert(groupIndex.group() < _inputGroups.size());
    auto retIndex = _inputGroups[groupIndex.group()]->addInput(groupIndex.index(),
                                                               edgeOutputInfo);
    NAVI_KERNEL_LOG(SCHEDULE2,
                    "add input group[%d,%d], retIndex[%d]",
                    groupIndex.group(),
                    groupIndex.index(),
                    retIndex);
    groupIndex.setIndex(retIndex);
    return groupIndex;
}

void NodeInternal::setErrorCode(ErrorCode ec, const ComputeScopePtr &computeScope) {
    NaviLoggerPtr tlsLogger;
    if (NAVI_TLS_LOGGER && NAVI_TLS_LOGGER->logger) {
        tlsLogger = NAVI_TLS_LOGGER->logger;
    }
    _graph->getParam()->graphResult->setError(tlsLogger, ec);
    if (computeScope) {
        computeScope->setErrorCode(ec);
    }
}

bool Node::getBinaryAttr(const std::string &attr, std::string &value) const {
    const auto &binaryAttrMap = _def->binary_attrs();
    auto it = binaryAttrMap.find(attr);
    if (binaryAttrMap.end() == it) {
        NAVI_NODE_LOG(SCHEDULE3, "can't find attr [%s] in node binary attr", attr.c_str());
        return false;
    }
    value = it->second;
    return true;
}

bool NodeInternal::postInit() {
    if (!initInputSnapshotVec()) {
        return false;
    }
    initOutputSnapshotVec();
    initOutputDegree();
    if (!initResourceIndexMap()) {
        return false;
    }
    if (!checkOrFinishInput()) {
        return false;
    }
    if (!checkInputGroup()) {
        return false;
    }
    return true;
}

bool NodeInternal::initInputSnapshotVec() {
    IndexType start = _inputs.size();
    uint32_t count = _inputGroups.size();
    for (uint32_t index = 0; index < count; index++) {
        auto group = _inputGroups[index];
        if (!group->postInit()) {
            NAVI_KERNEL_LOG(ERROR, "invalid input group index");
            return false;
        }
        group->setInputIndexStart(start);
        start += group->getInputCount();
    }
    _inputSnapshotVecSize = start;
    return true;
}

void NodeInternal::initOutputSnapshotVec() {
    for (auto group : _outputGroups) {
        group->postInit();
    }
}

void NodeInternal::initOutputDegree() {
    size_t outputDegree = 0;
    for (auto edge : _outputs) {
        if (edge) {
            outputDegree += edge->getOutputDegree();
        }
    }
    for (auto group : _outputGroups) {
        outputDegree += group->getOutputDegree();
    }
    _outputDegree = outputDegree;
}

size_t NodeInternal::getResourceInputGroupCount() {
    if (INVALID_INDEX == _resourceInputGroupIndex) {
        return 0;
    }
    return _inputGroups[_resourceInputGroupIndex]->getInputCount();
}

bool NodeInternal::initResourceIndexMap() {
    if (INVALID_INDEX == _resourceInputGroupIndex) {
        return true;
    }
    auto resourceGroup = _inputGroups[_resourceInputGroupIndex];
    const auto &inputVec = resourceGroup->getInputVec();
    auto inputCount = inputVec.size();
    for (IndexType index = 0; index < inputCount; index++) {
        const auto &edgeInfo = inputVec[index].second;
        auto inputNode = edgeInfo.getInputNode();
        if (!inputNode) {
            NAVI_KERNEL_LOG(ERROR, "unknown error, null resource input node");
            return false;
        }
        const auto &resourceName = inputNode->getResourceName();
        if (resourceName.empty()) {
            _flushIndexVec.push_back(index);
            continue;
        }
        auto selectIndexVec = edgeInfo.getSelectIndexVec();
        if (!selectIndexVec) {
            _resourceIndexMap.emplace(resourceName, std::make_pair(index, _biz->getResourceStage(resourceName)));
        } else {
            for (auto selectIndex : *selectIndexVec) {
                if (selectIndex < 0) {
                    _resourceIndexMap.emplace(resourceName,
                                              std::make_pair(index, _biz->getResourceStage(resourceName)));
                } else {
                    if (!_selectIndexVec) {
                        _selectIndexVec = new NaviVector<NaviVector<int32_t, 2>, 4>();
                    }
                    if (_selectIndexVec->size() <= selectIndex) {
                        _selectIndexVec->resize(selectIndex + 1);
                    }
                    (*_selectIndexVec)[selectIndex].push_back(index);
                }
            }
        }
    }
    return true;
}

void Node::initResourceIndexMap() {
    if (INVALID_INDEX != _internal->_resourceInputGroupIndex &&
        _internal->_selectIndexVec) {
        auto inputCount = _internal->getResourceInputGroupCount();
        _runState->startBitMap = new std::vector<bool>(inputCount, false);
        for (auto index : _internal->_flushIndexVec) {
            (*_runState->startBitMap)[index] = true;
        }
        for (const auto &pair : _internal->_resourceIndexMap) {
            (*_runState->startBitMap)[pair.second.first] = true;
        }
    }
}

void Node::startNormalResourceInput(bool graphStart) const {
    if (INVALID_INDEX == _internal->_resourceInputGroupIndex) {
        return;
    }
    auto resourceGroup =
        _internal->_inputGroups[_internal->_resourceInputGroupIndex];
    for (const auto &pair : _internal->_resourceIndexMap) {
        if (_internal->_isResourceNode || (RS_KERNEL != pair.second.second)) {
            resourceGroup->startUpStreamSchedule(_graph, pair.second.first, this, graphStart);
        }
    }
    for (auto index : _internal->_flushIndexVec) {
        resourceGroup->startUpStreamSchedule(_graph, index, this, graphStart);
    }
}

void Node::startKernelResourceInput() const {
    if (INVALID_INDEX == _internal->_resourceInputGroupIndex) {
        return;
    }
    if (_internal->_isResourceNode) {
        return;
    }
    auto resourceGroup =
        _internal->_inputGroups[_internal->_resourceInputGroupIndex];
    for (const auto &pair : _internal->_resourceIndexMap) {
        if (RS_KERNEL == pair.second.second) {
            resourceGroup->startUpStreamSchedule(_graph, pair.second.first, this, false);
        }
    }
}

bool Node::createSelectResult() {
    bool ret = true;
    const auto &selectBindVec =
        _internal->_kernelCreator->getBindInfos().selectBindVec;
    auto total = selectBindVec.size();
    if (0 == total) {
        return ret;
    }
    KernelConfigContext thisCtx;
    initConfigContext(false, thisCtx);
    for (size_t i = 0; i < total; i++) {
        const auto &selector = selectBindVec[i];
        auto tmpCtx = thisCtx;
        int32_t index = -1;
        try {
            index = selector.selector(tmpCtx);
        } catch (const autil::legacy::ExceptionBase &e) {
            NAVI_NODE_LOG(ERROR,
                     "compute select resource failed, selector [%lu], throw exception: [%s]",
                     i,
                     e.GetMessage().c_str());
            NAVI_NODE_LOG(DEBUG,
                          "exception bt: %s",
                          getLogger().logger->getExceptionBt(e).c_str());
            ret = false;
            continue;
        }
        int32_t candidateCount = selector.candidates.size();
        if (index >= candidateCount) {
            NAVI_NODE_LOG(ERROR,
                     "selector index [%d] overflow, total count [%d], selectResource [%lu]",
                     index,
                     candidateCount,
                     i);
            ret = false;
            continue;
        }
        NAVI_NODE_LOG(SCHEDULE3,
                 "selected resource [%d] [%s] ok, selector [%lu]",
                 index,
                 index >= 0 ? selector.candidates[index].c_str() : "",
                 i);
        _runState->selectorResult.push_back(index);
    }
    return ret;
}

void Node::startSelectResourceInput() const {
    if (INVALID_INDEX == _internal->_resourceInputGroupIndex) {
        return;
    }
    if (!_internal->_selectIndexVec) {
        return;
    }
    auto resourceGroup =
        _internal->_inputGroups[_internal->_resourceInputGroupIndex];
    const auto &selectBindVec =
        _internal->_kernelCreator->getBindInfos().selectBindVec;
    auto total = selectBindVec.size();
    int32_t selectIndex = 0;
    for (size_t i = 0; i < total; i++) {
        const auto &selector = selectBindVec[i];
        auto index = _runState->selectorResult[i];
        auto candidateCount = selector.candidates.size();
        for (int32_t j = 0; j < candidateCount; j++) {
            if (selectIndex >= _internal->_selectIndexVec->size()) {
                selectIndex++;
                continue;
            }
            const auto &offsetVec = (*_internal->_selectIndexVec)[selectIndex++];
            bool selected = (j == index);
            for (auto offset : offsetVec) {
                if (selected) {
                    resourceGroup->startUpStreamSchedule(
                        _graph, offset, this, false);
                    if (_runState->startBitMap) {
                        (*_runState->startBitMap)[offset] = true;
                    }
                }
            }
        }
    }
    if (_runState->startBitMap) {
        int32_t count = _runState->startBitMap->size();
        for (int32_t i = 0; i < count; i++) {
            if (!(*_runState->startBitMap)[i]) {
                resourceGroup->setEof(_graph, i, this);
            }
        }
    }
}

ResourcePtr Node::buildR(const std::string &name, KernelConfigContext *ctx, ResourceMap &inputResourceMap) {
    const auto &enableResourceSet =
        _internal->_kernelCreator->getEnableBuildResourceSet();
    if (enableResourceSet.end() == enableResourceSet.find(name)) {
        NAVI_NODE_LOG(ERROR, "can't build resource [%s], not enabled", name.c_str());
        return nullptr;
    }
    MultiLayerResourceMap multiLayerResourceMap;
    _graph->collectResourceMap(getScope(), multiLayerResourceMap);
    if (_runState->resourceMap) {
        multiLayerResourceMap.addResourceMap(_runState->resourceMap.get());
    }
    auto resource = _graph->buildResourceRecur(
        name, RS_KERNEL, ctx, multiLayerResourceMap, getReplaceInfoMap(), this, inputResourceMap);
    if (resource) {
        return resource->shared_from_this();
    } else {
        return nullptr;
    }
}

NodeAsyncInfoPtr Node::getAsyncInfo() const {
    if (!_runState || !_runState->nodeAsync) {
        return NodeAsyncInfoPtr();
    }
    autil::ScopedLock lock(_runState->nodeAsync->asyncMutex);
    return _runState->nodeAsync->asyncInfo;
}

void Node::setAsyncInfo(const NodeAsyncInfoPtr &asyncInfo) {
    if (!_runState->nodeAsync) {
        _runState->nodeAsync = new NodeAsync();
    }
    autil::ScopedLock lock(_runState->nodeAsync->asyncMutex);
    _runState->nodeAsync->asyncInfo = asyncInfo;
}

AsyncPipePtr __attribute__((weak)) Node::createAsyncPipe(ActivateStrategy activateStrategy) {
    return doCreateAsyncPipe(activateStrategy);
}

AsyncPipePtr Node::doCreateAsyncPipe(ActivateStrategy activateStrategy) {
    if (!_runState) {
        return AsyncPipePtr();
    }
    autil::ScopedLock lock(_runState->forkLock);
    if (_graph->terminated()) {
        NAVI_NODE_LOG(ERROR, "create async pipe failed, graph terminated");
        return nullptr;
    }
    // ATTENTION: `_forkLock` && `_graph->terminated check` can GUARANTEE THAT:
    // `createAsyncPipe` will not be executed concurrently with `terminate`,
    // so no lock is needed for member variable `_asyncPipe`

    auto asyncInfo = std::make_shared<NodeAsyncInfo>(this, getAsyncInfo());
    recycleTimeoutNode(asyncInfo);
    auto pipe = asyncInfo->addOne(activateStrategy);
    setAsyncInfo(asyncInfo);
    return pipe;
}

bool Node::registerPeriodicPipe(
        const AsyncPipePtr &asyncPipe,
        int64_t periodMs,
        int64_t repeat)
{
    if (!asyncPipe || periodMs <= 0 || _graph->terminated()) {
        NAVI_NODE_LOG(
            ERROR,
            "invalid periodic async pipe config, pipe [%p], period [%ldms], "
            "repeat [%ld], graph terminated [%d]",
            asyncPipe.get(),
            periodMs,
            repeat,
            _graph->terminated());
        return false;
    }
    auto timeoutChecker = _graph->getGraph()->getTimeoutChecker();
    auto elapsedMs = (autil::TimeUtility::currentTime() -
        timeoutChecker->beginTime()) / 1000;
    if (!_runState) {
        return false;
    }
    auto asyncInfo = getAsyncInfo();
    if (!asyncInfo || !asyncInfo->registerPeriodicPipe(
            asyncPipe, periodMs, elapsedMs + periodMs, repeat)) {
        return false;
    }
    bool expected = false;
    if (!_timeoutNodeRegistered.compare_exchange_strong(expected, true)) {
        if (_graph->getGraph()->refreshTimeout()) {
            return true;
        }
        asyncInfo->stopPeriodicAsyncPipe(asyncPipe);
        return false;
    }
    if (_graph->registerTimeoutNode(this)) {
        if (!_timeoutNodeRegistered.load()) {
            _graph->unregisterTimeoutNode(this);
        }
        return true;
    }
    asyncInfo->stopPeriodicAsyncPipe(asyncPipe);
    _timeoutNodeRegistered.store(false);
    _graph->unregisterTimeoutNode(this);
    return false;
}

bool Node::getPeriodicPipeActivationCount(
        const AsyncPipePtr &asyncPipe,
        int64_t &activatedCount,
        int64_t &remainingCount,
        int64_t &periodMs,
        int64_t &nextTickTimeMs) const
{
    if (!asyncPipe) {
        return false;
    }
    if (!_runState) {
        return false;
    }
    auto asyncInfo = getAsyncInfo();
    return asyncInfo && asyncInfo->getPeriodicPipeActivationCount(
        asyncPipe,
        activatedCount,
        remainingCount,
        periodMs,
        nextTickTimeMs);
}

bool Node::stopPeriodicAsyncPipe(const AsyncPipePtr &asyncPipe)
{
    if (!asyncPipe) {
        return false;
    }
    if (!_runState) {
        return false;
    }
    auto asyncInfo = getAsyncInfo();
    if (!asyncInfo || !asyncInfo->stopPeriodicAsyncPipe(asyncPipe)) {
        return false;
    }
    return _graph->terminated() || _graph->getGraph()->refreshTimeout();
}

int64_t Node::checkTimeoutAsyncPipes(int64_t endTimeMs)
{
    std::vector<AsyncPipePtr> activatedPipes;
    if (!_runState) {
        return -1;
    }
    auto asyncInfo = getAsyncInfo();
    if (!asyncInfo) {
        return -1;
    }
    auto nextTimeoutMs =
        asyncInfo->checkTimeoutAsyncPipes(endTimeMs, activatedPipes);
    for (const auto &asyncPipe : activatedPipes) {
        asyncPipe->setDataOverride(nullptr, false, true);
    }
    return nextTimeoutMs;
}

void Node::notifyPipeTerminate() {
    if (!_runState) {
        return;
    }
    autil::ScopedLock lock(_runState->forkLock);
    if (_graph->terminated()) {
        return;
    }
    auto oldInfo = getAsyncInfo();
    if (!oldInfo) {
        return;
    }
    auto asyncInfo = std::make_shared<NodeAsyncInfo>(this, oldInfo);
    if (asyncInfo->consolidate()) {
        setAsyncInfo(asyncInfo);
    } else {
        setAsyncInfo(nullptr);
    }
}

void Node::recycleTimeoutNode(
        const NodeAsyncInfoPtr &asyncInfo)
{
    if (!asyncInfo) {
        return;
    }
    if (!asyncInfo->hasActivePeriodicPipe() &&
        _timeoutNodeRegistered.exchange(false)) {
        _graph->unregisterTimeoutNode(this);
    }
}

bool Node::hasActivePeriodicPipe() const
{
    auto asyncInfo = getAsyncInfo();
    return asyncInfo && asyncInfo->hasActivePeriodicPipe();
}

bool NodeInternal::bindPort(Port *port) {
    auto localPort = static_cast<LocalPortBase *>(port);
    auto &partInfo = localPort->getPartInfo();
    bool valid = (partInfo.getFullPartCount() > 0);
    if (!valid) {
        NAVI_KERNEL_LOG(ERROR,
                        "invalid port part count [%d], port: %s",
                        partInfo.getFullPartCount(),
                        port->getPortNodeName().c_str());
        return false;
    }
    if (IOT_INPUT == localPort->getIoType()) {
        assert(0 < _inputGroups.size());
        _inputGroups[0]->setValidPortGroup(valid);
    }
    return true;
}

bool Node::bindPort(Port *port) {
    auto localPort = static_cast<LocalPortBase *>(port);
    auto &partInfo = localPort->getPartInfo();
    if (IOT_INPUT == localPort->getIoType()) {
        assert(0 < _internal->_inputGroups.size());
        auto readyBitMap = _internal->_inputGroups[0]->createReadyBitMap(partInfo);
        localPort->setNode(this, readyBitMap);
        setHasInput();
        _isInputBorder = true;
    } else {
        assert(0 < _internal->_outputGroups.size());
        auto readyBitMap = _internal->_outputGroups[0]->createReadyBitMap(partInfo);
        localPort->setNode(this, readyBitMap);
        setHasOutput();
        _isOutputBorder = true;
    }
    _port = localPort;
    return true;
}

InputPortGroupInternal *NodeInternal::addControlInputGroup() {
    if (INVALID_INDEX == _controlInputGroupIndex) {
        auto group = new InputPortGroupInternal(IT_REQUIRE, IT_REQUIRE);
        _controlInputGroupIndex = _inputGroups.size();
        _inputGroups.push_back(group);
        NAVI_KERNEL_LOG(SCHEDULE2, "add control inputGroup[%s]", autil::StringUtil::toString(group).c_str());
    }
    return _inputGroups[_controlInputGroupIndex];
}

InputPortGroupInternal *NodeInternal::addResourceInputGroup() {
    if (INVALID_INDEX == _resourceInputGroupIndex) {
        auto group = new InputPortGroupInternal(IT_REQUIRE, IT_REQUIRE);
        _resourceInputGroupIndex = _inputGroups.size();
        _inputGroups.push_back(group);
        NAVI_KERNEL_LOG(SCHEDULE2, "add resource inputGroup[%s]", autil::StringUtil::toString(group).c_str());
    }
    return _inputGroups[_resourceInputGroupIndex];
}

void NodeInternal::setHasInput() {
    _hasInput = true;
}

void Node::setHasInput() {
    _hasInput = true;
}

bool Node::hasInput() const {
    return _hasInput || (getAsyncInfo() != nullptr);
}

void Node::setHasOutput() {
    _hasOutput = true;
}

void NodeInternal::setHasOutput() {
    _hasOutput = true;
}

bool Node::hasOutput() const {
    return _hasOutput;
}

ActivateStrategy
NodeInternal::getInputActivateStrategy(const InputDef *inputDef) {
    auto type = inputDef->type();
    if (IT_OPTIONAL & type) {
        return AS_OPTIONAL;
    } else {
        return AS_REQUIRED;
    }
}

bool Node::getPortTypeStr(const std::string &portName,
                          IoType ioType,
                          std::string &typeStr) const
{
    if (!_internal->_kernelCreator) {
        return false;
    }
    return _internal->_kernelCreator->getPortTypeStr(portName, ioType, typeStr);
}

bool NodeInternal::checkOrFinishInput() {
    for (IndexType i = 0; i < _inputs.size(); i++) {
        if (!_inputs[i].isValid()) {
            auto inputDef = _kernelCreator->getInputDef(i);
            if (inputDef) {
                if (IT_ALLOW_DISCONNECT & inputDef->type()) {
                    _disconnectInputs.push_back(i);
                    NAVI_KERNEL_LOG(SCHEDULE3,
                                    "input [%s] [%d] disconnected, allowed",
                                    inputDef->name().c_str(),
                                    i);
                } else {
                    NAVI_KERNEL_LOG(ERROR,
                                    "input [%s] [%d] disconnected",
                                    inputDef->name().c_str(),
                                    i);
                    return false;
                }
            } else {
                NAVI_KERNEL_LOG(
                    ERROR, "unknown error, input index [%d] overflow", i);
                return false;
            }
        }
    }
    return true;
}

bool NodeInternal::checkInputGroup() const {
    for (IndexType i = 0; i < _inputGroups.size(); i++) {
        auto *inputGroup = _inputGroups[i];
        if (!inputGroup->isValid()) {
            const auto &portName = _kernelCreator->getInputPortName(
                PortIndex(INVALID_INDEX, i));
            if (portName.empty()) {
                NAVI_KERNEL_LOG(
                    ERROR, "unknown error, input group index [%d] overflow", i);
                return false;
            }
            if (IT_ALLOW_DISCONNECT & inputGroup->getGroupType()) {
                NAVI_KERNEL_LOG(SCHEDULE3,
                                "input group [%s] [%d] disconnected, allowed",
                                portName.c_str(),
                                i);
            } else {
                NAVI_KERNEL_LOG(
                    ERROR,
                    "input group name[%s] not connected, group[%s], node [%s] "
                    "kernel [%s]",
                    portName.c_str(),
                    autil::StringUtil::toString(*inputGroup).c_str(),
                    _name.c_str(),
                    _kernelName.c_str());
                return false;
            }
        }
    }
    return true;
}

bool Node::collectInputResource(ResourceMap &resourceMap) {
    InputSnapshot *begin = nullptr;
    InputSnapshot *end = nullptr;
    if (!snapshotGroupInput(
            _internal->_resourceInputGroupIndex, EGT_GET, begin, end)) {
        NAVI_NODE_LOG(ERROR, "read resource input failed index [%u] size [%lu]",
                 _internal->_resourceInputGroupIndex, _internal->_inputGroups.size());
        return false;
    }
    for (; begin != end; begin++) {
        DataPtr data(std::move(begin->data.data));
        auto resourceData = std::dynamic_pointer_cast<ResourceData>(data);
        if (!resourceData) {
            continue;
        }
        const auto &resource = resourceData->_resource;
        resourceMap.addResource(resourceData->_name, resource);
    }
    return true;
}

struct NodeResourceBuilder {
public:
    bool init() {
        if (INVALID_INDEX != node->_internal->_resourceInputGroupIndex) {
            if (!node->_runState->resourceMap) {
                node->_runState->resourceMap = std::make_shared<ResourceMap>();
            }
            if (!node->collectInputResource(*node->_runState->resourceMap)) {
                return false;
            }
        }
        node->_graph->collectResourceMap(node->getScope(),
                                         multiLayerResourceMap);
        if (node->_runState->resourceMap) {
            multiLayerResourceMap.addResourceMap(node->_runState->resourceMap.get());
        }
        replaceInfoMap = node->getReplaceInfoMap();
        return true;
    }
    Resource *getResource(const std::string &resourceName) {
        const std::string *namePtr = replaceInfoMap ? &replaceInfoMap->getReplaceRTo(resourceName) : &resourceName;
        bool exist = false;
        auto resource = multiLayerResourceMap.getResource(*namePtr, exist);
        if (exist) {
            return resource;
        }
        if (!node->_runState->resourceMap) {
            node->_runState->resourceMap = std::make_shared<ResourceMap>();
        }
        return node->_graph->buildResourceRecur(resourceName,
                                                RS_KERNEL,
                                                nullptr,
                                                multiLayerResourceMap,
                                                replaceInfoMap,
                                                node,
                                                *node->_runState->resourceMap);
    }
public:
    Node *node = nullptr;
    MultiLayerResourceMap multiLayerResourceMap;
    const ReplaceInfoMap *replaceInfoMap = nullptr;
};

const std::string &Node::getConfigPath() const {
    return _internal->_biz->getConfigPath();
}

Port *Node::getPort() const {
    return _port;
}

bool Node::createKernel(const ComputeScopePtr &computeScope) {
    if (computeScope) {
        computeScope->setType(GET_KERNEL_CREATE);
    }
    NAVI_NODE_LOG(SCHEDULE1, "create kernel begin");
    auto kernel = _internal->_kernelCreator->create(_baseAddr + _internal->_memOffset);
    if (!kernel) {
        NAVI_NODE_LOG(ERROR, "create kernel failed, ec[%s]",
                 CommonUtil::getErrorString(EC_CREATE_KERNEL));
        setErrorCode(EC_CREATE_KERNEL, computeScope);
        return false;
    }
    _runState->kernel = kernel;
    kernel->setNodeDef(_def);
    auto nodeType = _def->type();
    if (NT_SCOPE_TERMINATOR == nodeType) {
        if (!dynamic_cast<ScopeTerminatorKernel *>(kernel)) {
            NAVI_NODE_LOG(ERROR, "scope kernel must inherit ScopeTerminatorKernel");
            setErrorCode(EC_CREATE_KERNEL, computeScope);
            return false;
        }
    }
    if (!initAttribute()) {
        setErrorCode(EC_INVALID_ATTRIBUTE, computeScope);
        return false;
    }
    NAVI_NODE_LOG(SCHEDULE1, "create kernel success");
    return true;
}

bool Node::initKernel(const ComputeScopePtr &computeScope) {
    if (computeScope) {
        computeScope->setType(GET_KERNEL_INIT);
    }
    _kernelInited = true;
    if (unlikely(_skipInit)) {
        NAVI_NODE_LOG(DEBUG, "kernel init skipped");
        return true;
    }
    NAVI_NODE_LOG(SCHEDULE1, "init kernel begin");
    void *castPtr = nullptr;
    auto castOffset = _internal->_kernelCreator->getCastOffsetValue();
    if (NAVI_INVALID_CAST_OFFSET != castOffset) {
        castPtr = ((char *)_runState->kernel) + castOffset;
    }
    if (!bindResource(castPtr)) {
        setErrorCode(EC_LACK_RESOURCE, computeScope);
        return false;
    }
    if (!bindNamedData(castPtr)) {
        setErrorCode(EC_BIND_NAMED_DATA, computeScope);
        return false;
    }
    ErrorCode ec = EC_NONE;
    {
        KernelInitContext ctx(this);
        try {
            ec = _runState->kernel->init(ctx);
        } catch (const autil::legacy::ExceptionBase &e) {
            NAVI_NODE_LOG(ERROR, "kernel init failed, throw exception: [%s]", e.GetMessage().c_str());
            NAVI_NODE_LOG(DEBUG,
                          "exception bt: %s",
                          getLogger().logger->getExceptionBt(e).c_str());
            ec = EC_ABORT;
        }
        if (ec == EC_NONE && ctx.forkError()) {
            ec = EC_FORK;
        }
    }
    if (EC_NONE != ec) {
        NAVI_NODE_LOG(ERROR, "kernel init failed");
        setErrorCode(ec, computeScope);
        return false;
    }
    NAVI_NODE_LOG(SCHEDULE1, "init kernel success");
    return true;
}

bool Node::initAttribute() {
    if (unlikely(_skipConfig)) {
        NAVI_NODE_LOG(DEBUG, "kernel config skipped");
        return true;
    }
    KernelConfigContext ctx;
    if (!initConfigContext(true, ctx)) {
        return false;
    }
    if (!createSelectResult()) {
        return false;
    }
    NAVI_MEMORY_BARRIER();
    startKernelResourceInput();
    startSelectResourceInput();
    try {
        return _runState->kernel->config(ctx);
    } catch (const autil::legacy::ExceptionBase &e) {
        NAVI_NODE_LOG(
            ERROR, "biz[%s], init kernel attribute failed, msg [%s]", _internal->_biz->getName().c_str(), e.GetMessage().c_str());
        NAVI_NODE_LOG(DEBUG,
                      "json attr [%s], exception bt: %s",
                      autil::StringUtil::toString(ctx).c_str(),
                      getLogger().logger->getExceptionBt(e).c_str());
        return false;
    }
}

bool Node::initConfigContext(bool parse, KernelConfigContext &ctx) {
    ctx = _internal->_ctx;
    if (!_def->json_attrs().empty()) {
        if (parse) {
            _runState->jsonAttrsDocument = new autil::legacy::RapidDocument();
            if (!NaviConfig::parseToDocument(_def->json_attrs(),
                                             *_runState->jsonAttrsDocument)) {
                NAVI_NODE_LOG(ERROR,
                         "invalid kernel attr, biz [%s]",
                         _internal->_biz->getName().c_str());
                return false;
            }
        }
        ctx._jsonAttrs = _runState->jsonAttrsDocument;
    }
    ctx._nodeDef = _def;
    return true;
}

KernelConfigContext Node::getConfigContext() {
    KernelConfigContext ctx;
    initConfigContext(false, ctx);
    return ctx;
}

bool Node::getConfigContext(const std::string &dependName, KernelConfigContext &ctx) {
    KernelConfigContext thisCtx;
    initConfigContext(false, thisCtx);
    const auto &configKeyMap = _internal->_kernelCreator->getConfigKeyMap();
    auto it = configKeyMap.find(dependName);
    if (configKeyMap.end() == it) {
        ctx = thisCtx;
        return true;
    }
    const auto &configKey = it->second;
    if (!thisCtx.hasKey(configKey)) {
        NAVI_NODE_LOG(ERROR, "lack config key [%s], depend resource [%s]", configKey.c_str(), dependName.c_str());
        return false;
    }
    ctx = thisCtx.enter(configKey);
    return true;
}

const std::string &Node::getBizName() const {
    const auto &bizName = _graph->getBizName();
    if (bizName.empty() || bizName == NAVI_FORK_BIZ) {
        return _internal->_biz->getName();
    } else {
        return bizName;
    }
}

bool Node::getBizPartInfo(const std::string &bizName, NaviPartId &partCount, std::vector<NaviPartId> &partIds) const {
    return _graph->getParam()->bizManager->getBizPartInfo(bizName, partCount, partIds);
}

bool Node::bindResource(void *castPtr) {
    if (isResourceCreateKernel()) {
        return true;
    }
    NodeResourceBuilder nodeResourceBuilder;
    nodeResourceBuilder.node = this;
    if (!nodeResourceBuilder.init()) {
        return false;
    }
    const auto &bindInfos = _internal->_kernelCreator->getBindInfos();
    const auto &dynamicInfoMap = _internal->_kernelCreator->getDynamicGroupInfoMap();
    if (!ResourceManager::bindResource(getKernelName(),
                                       _runState->kernel,
                                       castPtr,
                                       nodeResourceBuilder,
                                       dynamicInfoMap,
                                       bindInfos,
                                       _runState->selectorResult,
                                       "kernel")) {
        NAVI_NODE_LOG(ERROR, "bind kernel resource failed");
        return false;
    }
    return true;
}

bool Node::bindNamedData(void *castPtr) {
    const auto &bindInfos = _internal->_kernelCreator->getBindInfos();
    const auto &runGraphParams = _graph->getParam()->runParams;
    auto graphId = _graph->getGraphId();
    auto namedDataMap = runGraphParams.getSubNamedDataMap(graphId);
    if (!ResourceManager::bindNamedData(getKernelName(),
                                        *_runState->kernel,
                                        castPtr,
                                        namedDataMap,
                                        bindInfos,
                                        "kernel")) {
        NAVI_NODE_LOG(ERROR, "bind kernel named data failed");
        return false;
    }
    return true;
}

void Node::deleteKernel(bool inDestruct) {
    if (!_runState || !_runState->kernel) {
        return;
    }
    auto collectFlags = _graph->getParam()->runParams.getCollectFlags();
    if (collectFlags.collectMalloc() && !inDestruct) {
        return;
    }
    if (_internal->_inlineConfigAndInit || inDestruct) {
        _runState->kernel->~Kernel();
        _runState->kernel = nullptr;
    } else {
        auto worker = _graph->getWorker();
        auto item = new KernelDestructItem(
            worker, collectFlags, _graph, _metric.get(), _runState->kernel);
        item->setDelayDestruct(true);
        item->setTailCompute(true);
        _runState->kernel = nullptr;
        worker->schedule(item);
    }
}

bool Node::inputOk() const {
    auto asyncInfo = getAsyncInfo();
    if (asyncInfo) {
        if (asyncInfo->independentReady()) {
            return true;
        }
        if (!asyncInfo->isOk()) {
            return false;
        }
    }
    for (const auto &input : _internal->_inputs) {
        if (input.isValid() && !input.isOutputOk(_graph)) {
            return false;
        }
    }
    for (auto group : _internal->_inputGroups) {
        if (!group->isOk(_port, _graph) && (group->getGroupType() & IT_OPTIONAL) == 0) {
            return false;
        }
    }
    if (hasInput()) {
        return inputHasReady() || inputFinish();
    } else {
        return true;
    }
}

bool Node::resourceInputOk() const {
    if (INVALID_INDEX == _internal->_resourceInputGroupIndex) {
        return true;
    }
    return _internal->_inputGroups[_internal->_resourceInputGroupIndex]->isOk(
        nullptr, _graph);
}

bool Node::inputFinish() const {
    for (const auto &input : _internal->_inputs) {
        if (input.isValid() && !input.isEofOrFinish(_graph)) {
            return false;
        }
    }
    for (const auto &inputGroup : _internal->_inputGroups) {
        if (!inputGroup->isEofOrFinish(_port, _graph)) {
            return false;
        }
    }
    auto asyncInfo = getAsyncInfo();
    if (asyncInfo && !asyncInfo->isFinish()) {
        return false;
    }
    return true;
}

bool Node::inputHasReady() const {
    for (const auto &input : _internal->_inputs) {
        if (input.isValid() && input.slotHasData(_graph)) {
            return true;
        }
    }
    for (const auto &inputGroup : _internal->_inputGroups) {
        if (inputGroup->hasReady(_port, _graph)) {
            return true;
        }
    }
    auto asyncInfo = getAsyncInfo();
    if (asyncInfo && asyncInfo->hasReady()) {
        return true;
    }
    return false;
}

bool Node::inputFinishConsumed() const {
    if (_inputSnapshotVec.size() != _internal->_inputSnapshotVecSize) {
        return false;
    }
    for (const auto &snapshot : _inputSnapshotVec) {
        if (!snapshot.data.eof) {
            return false;
        }
    }
    auto asyncInfo = getAsyncInfo();
    if (asyncInfo && !asyncInfo->isFinishConsumed()) {
        return false;
    }
    return true;
}

bool Node::outputOk() const {
    for (const auto &output : _internal->_outputs) {
        if (output) {
            if (!_graph->transformEdgeFromInternal(output)->isInputOk()) {
                return false;
            }
        }
    }
    for (auto outputGroup : _internal->_outputGroups) {
        if (!outputGroup->isOk(_port, _graph)) {
            return false;
        }
    }
    return true;
}

bool Node::outputFinish() const {
    for (const auto &output : _internal->_outputs) {
        if (output) {
            if (!_graph->transformEdgeFromInternal(output)->isEofOrFinish()) {
                return false;
            }
        }
    }
    for (const auto &outputGroup : _internal->_outputGroups) {
        if (!outputGroup->isEofOrFinish(_port, _graph)) {
            return false;
        }
    }
    return true;
}

bool Node::scheduleLock(const Node *callNode) {
    while (1) {
        if (_scheduleStat == SS_INIT) {
            NAVI_NODE_LOG(SCHEDULE2,
                     "node not started, schedule by [%p] [%s]",
                     callNode,
                     callNode ? callNode->getName().c_str() : "");
            return false;
        }
        if (_scheduleStat == SS_FINISH) { // schedule is checked after compute
            if (unlikely((atomic_read(&_nodeSnapshot) & PENDING_WORK_MASK))) {
                finishPendingWork();
            }
            NAVI_NODE_LOG(SCHEDULE2, "schedule by [%p] [%s], failed for finish",
                     callNode, callNode ? callNode->getName().c_str() : "");
            return false;
        }
        if (_scheduleStat == SS_RUNNING) { // schedule is checked after compute
            NAVI_NODE_LOG(SCHEDULE2, "schedule by [%p] [%s], failed for running",
                     callNode, callNode ? callNode->getName().c_str() : "");
            return false;
        }
        if (_scheduleStat == SS_SCHEDULING) {
            NAVI_NODE_LOG(SCHEDULE2, "failed for scheduling, snapshot [%ld]", readNodeSnapshot());
            return false;
        }
        if (cmpxchg(&_scheduleStat, SS_IDLE, SS_SCHEDULING)) {
            return true;
        }
    }
}

bool Node::schedule(const Node *callNode, bool ignoreFrozen) {
    bool firstTry = true;
    NAVI_MEMORY_BARRIER();
    while (true) {
        bool resched = true;
        auto ret = doSchedule(callNode, ignoreFrozen, resched, firstTry);
        if (ret) {
            return ret;
        }
        if (firstTry) {
            firstTry = false;
            incNodeSnapshot();
            continue;
        }
        if (resched) {
            continue;
        }
        return ret;
    }
}

bool Node::startSchedule(const Node *callNode, bool ignoreFrozen, bool graphStart) {
    if (cmpxchg(&_scheduleStat, SS_INIT, SS_IDLE)) {
        NAVI_NODE_LOG(SCHEDULE3, "start schedule by [%s]", callNode ? callNode->getName().c_str() : "");
        for (const auto &edgeOutputInfo : _internal->_inputs) {
            if (edgeOutputInfo.isValid()) {
                edgeOutputInfo.startUpStreamSchedule(_graph, this, graphStart);
            }
        }
        IndexType groupCount = _internal->_inputGroups.size();
        for (IndexType index = 0; index < groupCount; index++) {
            if (_internal->_resourceInputGroupIndex == index) {
                continue;
            }
            _internal->_inputGroups[index]->startUpStreamSchedule(
                _graph, this, graphStart);
        }
        startNormalResourceInput(graphStart);
        if (graphStart) {
            _internal->_started = true;
        }
        return schedule(callNode, ignoreFrozen);
    }
    return true;
}

/*
lock  -> checkReady  -> running -> checkFinish ->reschedule
finish must be checked after compute
 */
bool Node::doSchedule(const Node *callNode, bool ignoreFrozen, bool &resched, bool &firstTry) {
    if (!ignoreFrozen && _frozen) {
        resched = false;
        return false;
    }
    if (!scheduleLock(callNode)) {
        resched = false;
        return false;
    }
    firstTry = false;
    auto snapshot = readNodeSnapshot();
    if (unlikely(_metric != nullptr)) {
        _metric->incTryScheduleCount();
    }
    NAVI_MEMORY_BARRIER();
    // critical region under SS_SCHEDULING stat;
    assert(SS_SCHEDULING == _scheduleStat);
    if (checkFinish(EC_NONE, callNode)) {
        resched = false;
        return false;
    }
    bool needCreateKernel = !_runState || !_runState->kernel;
    bool needInitKernel = !needCreateKernel && !_kernelInited;
    bool ret = false;
    if (_runState && _runState->forkGraph) {
        ret = false;
    } else if (_internal->_inlineConfigAndInit) {
        ret = resourceInputOk() && inputOk() && outputOk();
    } else if (needCreateKernel) {
        ret = true;
    } else if (needInitKernel) {
        ret = resourceInputOk();
    } else {
        ret = (inputOk() && outputOk());
    }
    NAVI_NODE_LOG(SCHEDULE1,
             "schedule by node [%p] [%s], ret [%d], forGraph [%p], needCreate [%d], needInit "
             "[%d], resourceInputOk [%d], inputOk [%d], outputOk [%d], "
             "inputFinish [%d], outputFinish [%d], inputHasReady [%d], hasInput[%d], "
             "hasOutput[%d], stat [%ld], snapshot: %ld, computeId: %ld",
             callNode, callNode ? callNode->getName().c_str() : "", ret,
             _runState ? _runState->forkGraph : nullptr, needCreateKernel, needInitKernel, resourceInputOk(), inputOk(),
             outputOk(), inputFinish(), outputFinish(), inputHasReady(), hasInput(),
             hasOutput(), _scheduleStat, snapshot, getComputeId());
    if (ret) {
        _scheduleStat = SS_RUNNING;
        if (!_runState) {
            auto param = _graph->getParam();
            auto ptr = (void *)(_baseAddr + _internal->_runStateOffset);
            new (ptr) NodeRunState(param->worker, param->runParams.getCollectFlags(), this);
            _runState = (NodeRunState *)ptr;
        }
        if (!_graph->schedule(this)) {
            _scheduleStat = SS_IDLE;
        }
        resched = false;
    } else {
        cmpxchg(&_scheduleStat, SS_SCHEDULING, SS_IDLE);
        auto newSnapshot = readNodeSnapshot();
        resched = (newSnapshot != snapshot);
        NAVI_NODE_LOG(SCHEDULE2,
                 "schedule by [%p] [%s], failed, old snapshot [%ld], new snapshot [%ld], resched [%d]",
                 callNode,
                 callNode ? callNode->getName().c_str() : "",
                 snapshot,
                 newSnapshot,
                 resched);
    }
    return ret;
}

void Node::stopSchedule() {
    _stopSchedule = true;
}

void Node::setForceStopFlag(Node *node) {
    _forceStopNode = node;
    _forceStop = true;
}

void Node::forceStop() {
    doForceStop();
    schedule(this);
}

bool Node::checkFinish(ErrorCode ec, const Node *callNode) {
    bool finish = false;
    if (_stopSchedule) {
        finish = true;
    } else if (EC_NONE != ec) {
        finish = true;
    } else if (hasOutput()) {
        finish = outputFinish();
    } else {
        finish = inputFinish() && inputFinishConsumed();
    }
    if (!finish && _forceStop) {
        NAVI_NODE_LOG(SCHEDULE2,
                 "node [%s] force stopped by node [%s]",
                 getName().c_str(),
                 _forceStopNode ? _forceStopNode->getName().c_str() : "");
        finish = true;
    }
    NAVI_NODE_LOG(SCHEDULE2,
             "check finish [%d], forceStop [%d], stopSchedule [%d], computeId: %ld, hasOutput [%d], outputFinish "
             "[%d], inputFinish [%d], inputFinishConsumed [%d]",
             finish,
             _forceStop,
             _stopSchedule,
             getComputeId(),
             hasOutput(),
             outputFinish(),
             inputFinish(),
             inputFinishConsumed());
    if (finish) {
        _scheduleStat = SS_FINISH;
        NAVI_MEMORY_BARRIER();
        auto snapshot = atomic_read(&_nodeSnapshot);
        if (unlikely((snapshot & PENDING_WORK_MASK))) {
            finishPendingWork();
        }
        if (unlikely(_metric != nullptr)) {
            if (!_forceStopNode) {
                _forceStopNode = callNode;
            }
            _metric->setFinish();
        }
        // the last edge is finish edge, see creator
        auto finishEdge = _graph->transformEdgeFromInternal(
            _internal->_controlOutputs[CONTROL_PORT_FINISH_INDEX]);
        if (finishEdge) {
            if (_forceStop) {
                finishEdge->forceEof(this);
            } else {
                finishEdge->setEof(this);
            }
        }
        propagateEof();
        if (likely(!_skipDeleteKernel)) {
            deleteKernel();
        }
    } else {
        auto snapshot = atomic_read(&_nodeSnapshot);
        if (unlikely((snapshot & PENDING_WORK_MASK))) {
            finishPendingWork();
        }
    }
    return finish;
}

bool Node::setPendingWork(int64_t mask, const Node *callNode) {
    int64_t snapshot;
    int64_t newSnapshot;
    do {
        snapshot = atomic_read(&_nodeSnapshot);
        if (snapshot & STOP_PENDING_WORK_FLAG) {
            return false;
        }
        if (snapshot & mask) {
            return true;
        }
        newSnapshot = (snapshot | mask);
    } while (!compare_and_swap(&_nodeSnapshot, snapshot, newSnapshot));
    schedule(callNode);
    return true;
}

void Node::finishPendingWork() {
    bool reCheck = false;
    do {
        int64_t snapshot;
        int64_t newSnapshot;
        for (;;) {
            snapshot = atomic_read(&_nodeSnapshot);
            if (!(snapshot & PENDING_WORK_MASK)) {
                if (reCheck) {
                    newSnapshot = (snapshot & (~DOING_PENDING_WORK_FLAG));
                    if (compare_and_swap(&_nodeSnapshot, snapshot, newSnapshot)) {
                        return;
                    } else {
                        continue;
                    }
                } else {
                    return;
                }
            }
            if (!reCheck && (snapshot & DOING_PENDING_WORK_FLAG)) {
                return;
            }
            newSnapshot = ((snapshot & (~PENDING_WORK_MASK)) | DOING_PENDING_WORK_FLAG);
            if (compare_and_swap(&_nodeSnapshot, snapshot, newSnapshot)) {
                break;
            }
        }
        doFinishPendingWork(snapshot);
        reCheck = true;
    } while (true);
}

void Node::doFinishPendingWork(int64_t snapshot) {
    if (snapshot & INPUT_PENDING_WORK_FLAG) {
        for (const auto &edgeOutputInfo : _internal->_inputs) {
            if (edgeOutputInfo.isValid()) {
                edgeOutputInfo.finishOutputWork(_graph, this);
            }
        }
        for (auto inputGroup : _internal->_inputGroups) {
            inputGroup->finishOutputWork(_graph, this);
        }
    }
    if (snapshot & OUTPUT_PENDING_WORK_FLAG) {
        for (auto edge : _internal->_outputs) {
            if (edge) {
                if (!_graph->transformEdgeFromInternal(edge)->finishInputWork(this)) {
                    notifyFinish(EC_ABORT);
                    break;
                }
            }
        }
        for (auto group : _internal->_outputGroups) {
            if (!group->finishInputWork(_graph, this)) {
                notifyFinish(EC_ABORT);
                break;
            }
        }
    }
}

void Node::propagateEof() {
    if (!_allInputEof) {
        for (const auto &edgeOutputInfo : _internal->_inputs) {
            if (edgeOutputInfo.isValid()) {
                edgeOutputInfo.setEof(_graph, false, this);
            }
        }
        for (auto inputGroup : _internal->_inputGroups) {
            inputGroup->setEof(_graph, this);
        }
    }
    if (_port) {
        if (_port->getIoType() == IOT_INPUT) {
            _port->setForceEofAll();
        } else {
            _port->setEofAll();
        }
    }
    doForceStop();
}

void Node::doForceStop() {
    if (_forceStop) {
        bool expect = false;
        if (!_forceStopped.compare_exchange_weak(expect, true)) {
            return;
        }
        for (auto edge : _internal->_outputs) {
            if (edge) {
                _graph->transformEdgeFromInternal(edge)->forceEof(this);
            }
        }
        for (auto group : _internal->_outputGroups) {
            group->forceEof(_graph, this);
        }
    }
}

void Node::cacheExpensive(bool value) {
    ExpensiveFlag newValue;
    newValue.expensive = (uint16_t)value;
    newValue.valid = 1;
    if (newValue.value != _internal->_expensive.value) {
        _internal->_expensive.value = newValue.value;
    }
}

bool Node::isInline() const {
    auto expensive = _internal->_expensive;
    if (expensive.valid) {
        return (0 == expensive.expensive);
    }
    if (unlikely(!_internal->_creatorStats)) {
        return true;
    }
    if (_internal->_inlineConfigAndInit) {
        if (!_runState || !_runState->kernel) {
            return true;
        } else {
            return !_runState->kernel->isExpensive();
        }
    } else {
        if (!_runState || !_runState->kernel) {
            auto createLatency =
                _internal->_creatorStats->getCreateAvgLatency();
            return createLatency < INLINE_LATENCY_THRESHOLD_US;
        } else if (!_kernelInited) {
            auto initLatency = _internal->_creatorStats->getInitAvgLatency();
            return initLatency < INLINE_LATENCY_THRESHOLD_US;
        } else {
            return !_runState->kernel->isExpensive();
        }
    }
}

size_t Node::outputDegree() const {
    return _internal->_outputDegree;
}

ComputeScopePtr Node::makeComputeScope(const ScheduleInfo &schedInfo, NaviPerfThread *perfThread) {
    _scheduleCount++;
    auto graphParam = _graph->getParam();
    auto collectFlags = graphParam->runParams.getCollectFlags();
    if (unlikely(collectFlags.needCollect())) {
        const auto &topoParam = graphParam->runParams.getTopoParam();
        return std::make_shared<ComputeScope>(
            _metric.get(),
            _internal->_creatorStats,
            _graph->getPartId(),
            perfThread,
            graphParam->memLeakCollector.get(),
            schedInfo,
            collectFlags,
            topoParam ? &topoParam->getMallocTraceParam() : nullptr);
    } else {
        if (unlikely(_metric != nullptr)) {
            _metric->incScheduleCount();
        }
        return nullptr;
    }
}

void Node::compute(const ComputeScopePtr &computeScope) {
    if (!_runState->inited) {
        _runState->inited = true;
        initResourceIndexMap();
        _inputSnapshotVec.resize(_internal->_inputSnapshotVecSize);
        for (auto i : _internal->_disconnectInputs) {
            auto &inputSnapshot = _inputSnapshotVec[i];
            inputSnapshot.data.eof = true;
            inputSnapshot.cleared = true;
        }
    }
    const auto &scopeLogger = getLogger();
    NaviLoggerObjectScope logObjectScope(LOT_NODE, this);
    NaviLoggerScope scope(scopeLogger);
    // critical region under SS_RUNNING stat;
    assert(SS_RUNNING == _scheduleStat);
    auto *graph = _graph->getGraph();
    if (graph->isTimeout()) {
        if (computeScope) {
            computeScope->setType(GET_KERNEL_TIMEOUT);
        }
        NAVI_NODE_LOG(ERROR, "compute failed, graph[%p] timeout", graph);
        graph->notifyTimeout(true);
        return;
    }
    auto snapshot = readNodeSnapshot();
    auto computeId = getComputeId();
    ErrorCode ec = EC_NONE;
    bool needCreateKernel = !_runState->kernel;
    bool needInitKernel = _runState->kernel && !_kernelInited;
    if (_internal->_inlineConfigAndInit) {
        needInitKernel = !_kernelInited;
        if (needCreateKernel && !createKernel(computeScope)) {
            ec = EC_CREATE_KERNEL;
            NAVI_NODE_LOG(ERROR, "create kernel failed, error [%d], abort kernel", ec);
        } else if (needInitKernel && !initKernel(computeScope)) {
            ec = EC_INIT_GRAPH;
            NAVI_NODE_LOG(ERROR, "init kernel failed, error [%d], abort kernel", ec);
        } else {
            NAVI_NODE_LOG(SCHEDULE1, "begin compute, snapshot: %ld, compute id: %ld",
                     snapshot, computeId);
            ec = doCompute(computeScope);
            NAVI_NODE_LOG(SCHEDULE1, "end compute, compute id: %ld", computeId);
        }
    } else {
        if (needCreateKernel) {
            if (!createKernel(computeScope)) {
                ec = EC_CREATE_KERNEL;
                NAVI_NODE_LOG(ERROR, "create kernel failed, error [%d], abort kernel", ec);
            }
            if (resourceInputOk()) {
                _scheduleCount++;
                needInitKernel = true;
                if (!initKernel(computeScope)) {
                    ec = EC_INIT_GRAPH;
                    NAVI_NODE_LOG(ERROR,
                             "init kernel failed, error [%d], abort kernel",
                             ec);
                }
            }
        } else if (needInitKernel) {
            if (!initKernel(computeScope)) {
                ec = EC_INIT_GRAPH;
                NAVI_NODE_LOG(ERROR, "init kernel failed, error [%d], abort kernel", ec);
            }
        } else {
            NAVI_NODE_LOG(SCHEDULE1, "begin compute, snapshot: %ld, compute id: %ld", snapshot, computeId);
            ec = doCompute(computeScope);
            NAVI_NODE_LOG(SCHEDULE1, "end compute, compute id: %ld", computeId);
        }
    }
    if (EC_NONE != ec) {
        NAVI_NODE_LOG(ERROR,
                 "compute error, error code [%s], compute id: %ld, abort kernel",
                 CommonUtil::getErrorString(ec), computeId);
        if (!NAVI_COLLECT_MALLOC()) {
            if (computeScope) {
                computeScope->report();
            }
        }
        notifyFinish(ec);
        checkFinish(ec, this);
        return;
    }
    if (unlikely(_stopAfterInit && needInitKernel)) {
        setFrozen(true);
    }
    // critical region
    if (!checkFinish(ec, this)) {
        // block other thread and check snapshot id
        cmpxchg(&_scheduleStat, SS_RUNNING, SS_IDLE);
        auto newSnapshot = readNodeSnapshot();
        auto needSchedule =
            (newSnapshot != snapshot) || needCreateKernel || needInitKernel;
        if (needSchedule) {
            NAVI_NODE_LOG(
                SCHEDULE2,
                "reschedule, prev snapshot [%ld], current snapshot [%ld]",
                snapshot, newSnapshot);
            schedule(this);
        } else {
            NAVI_NODE_LOG(SCHEDULE2, "no need reschedule");
        }
    } else {
        auto nodeType = _def->type();
        if (NT_SCOPE_TERMINATOR == nodeType) {
            auto error =
                _graph->getParam()->graphResult->makeError(scopeLogger.logger, ec);
            _graph->terminateScope(this, getScope(), error);
        }
    }
}

ErrorCode Node::doCompute(const ComputeScopePtr &computeScope) {
    if (computeScope) {
        computeScope->setType(GET_KERNEL_COMPUTE);
    }
    clearControlInput();
    KernelComputeContext ctx(this);
    ErrorCode ec = EC_NONE;
    {
        try {
            ec = _runState->kernel->compute(ctx);
        } catch (const autil::legacy::ExceptionBase &e) {
            NAVI_NODE_LOG(ERROR, "kernel compute failed, throw exception: [%s]", e.GetMessage().c_str());
            NAVI_NODE_LOG(DEBUG,
                          "exception bt: %s",
                          getLogger().logger->getExceptionBt(e).c_str());
            ec = EC_ABORT;
        }
        if (ec == EC_NONE && !getAsyncInfo() && ctx.deadLock()) {
            ec = EC_DEADLOCK;
        }
        if (ec == EC_NONE && ctx.forkError()) {
            ec = EC_FORK;
        }
        if (computeScope) {
            computeScope->setErrorCode(ec);
        }
    }
    resetIOSnapshot();
    return ec;
}

void Node::clearControlInput() {
    if (INVALID_INDEX == _internal->_controlInputGroupIndex) {
        return;
    }
    InputSnapshot *begin = nullptr;
    InputSnapshot *end = nullptr;
    snapshotGroupInput(_internal->_controlInputGroupIndex, EGT_CLEAR, begin, end);
    for (; begin != end; begin++) {
        if (begin->data.data.use_count() == 1) {
            destructData(std::move(begin->data.data));
        }
        begin->data.reset();
    }
}

void Node::resetIOSnapshot() {
    bool allEof = true;
    for (auto &input: _inputSnapshotVec) {
        if (input.data.data.use_count() == 1) {
            destructData(std::move(input.data.data));
        }
        input.reset();
        if (!input.data.eof) {
            allEof = false;
        }
    }
    if (allEof) {
        _allInputEof = true;
    }
}

LocalSubGraph *Node::getGraph() const {
    return _graph;
}

void Node::destructData(DataPtr &&data) const {
    auto collectFlags = _graph->getParam()->runParams.getCollectFlags();
    if (_internal->_inlineConfigAndInit || collectFlags.collectMalloc()) {
        data.reset();
    } else {
        auto worker = _graph->getWorker();
        auto item = new DataDestructItem(worker, collectFlags, std::move(data));
        item->setDelayDestruct(true);
        item->setTailCompute(true);
        worker->schedule(item);
    }
}

const KernelCreator *Node::getKernelCreator() const {
    return _internal->_kernelCreator;
}

const KernelMetricPtr &Node::getMetric() const {
    return _metric;
}

void Node::collectInputStates(KernelInputStatesDef *states) const {
    states->set_ok(inputOk());
    states->set_resource_ok(resourceInputOk());
    states->set_has_ready(inputHasReady());
    states->set_finish(inputFinish());
    states->set_finish_consumed(inputFinishConsumed());

    // input
    for (size_t i = 0; i < _internal->_inputs.size(); i++) {
        const auto &edgeOutputInfo = _internal->_inputs[i];
        auto inputDef = states->add_inputs();
        if (edgeOutputInfo.isValid()) {
            edgeOutputInfo.fillState(_graph, inputDef);
        } else {
            inputDef->set_port(_internal->_kernelCreator->getInputPortName(i));
            inputDef->set_disconnected(true);
            inputDef->set_finish(true);
        }
    }

    // input groups
    const auto *kernelDef = _internal->_kernelCreator->def();
    for (size_t i = 0; i < _internal->_inputGroups.size(); i++) {
        auto state = states->add_input_groups();
        _internal->_inputGroups[i]->fillState(_port, _graph, state);
        if (_internal->_controlInputGroupIndex == i) {
            state->set_name("navi.control_input_group");
        } else if (_internal->_resourceInputGroupIndex == i) {
            state->set_name("navi.resource_input_group");
        } else if (i < kernelDef->input_groups().size()) {
            state->set_name(kernelDef->input_groups(i).name());
        }
    }

    // async pipe
    {
        auto asyncInfo = getAsyncInfo();
        if (asyncInfo) {
            asyncInfo->fillState(states->mutable_async_pipes());
        }
    }

    if (_port && _port->getIoType() == IOT_INPUT) {
        _port->fillOutputState(states->mutable_port());
    }
}

void Node::collectOutputStates(KernelOutputStatesDef *states) {
    states->set_ok(outputOk());
    states->set_finish(outputFinish());

    // output
    for (IndexType i = 0; i < _internal->_outputs.size(); i++) {
        auto state = states->add_outputs();
        auto output = _internal->_outputs[i];
        if (output) {
            _graph->transformEdgeFromInternal(output)->fillUpstreamState(state);
        } else {
            state->set_port(_internal->_kernelCreator->getOutputPortName(i));
            state->set_index(i);
            state->set_disconnected(true);
        }
    }
    // control output
    for (IndexType i = 0; i < _internal->_controlOutputs.size(); i++) {
        auto output = _internal->_controlOutputs[i];
        if (output) {
            auto state = states->add_outputs();
            state->set_is_control(true);
            _graph->transformEdgeFromInternal(output)->fillUpstreamState(state);
        }
    }
    // output groups
    const auto *kernelDef = _internal->_kernelCreator->def();
    for (size_t i = 0; i < _internal->_outputGroups.size(); i++) {
        auto state = states->add_output_groups();
        _internal->_outputGroups[i]->fillState(_port, _graph, state);
        state->set_name(kernelDef->output_groups(i).name());
    }

    if (_port && _port->getIoType() == IOT_OUTPUT) {
        _port->fillInputState(states->mutable_port());
    }
}

void Node::collectState() {
    if (likely(!_metric)) {
        return;
    }
    auto states = new KernelStatesDef();
    states->set_ss_state((NodeSchedStateDef)_scheduleStat);
    {
        auto snapshot = atomic_read(&_nodeSnapshot);
        auto pendingStateDef = states->mutable_pending_state();
        pendingStateDef->set_input_pending(
            (INPUT_PENDING_WORK_FLAG & snapshot) != 0);
        pendingStateDef->set_output_pending(
            (OUTPUT_PENDING_WORK_FLAG & snapshot) != 0);
        pendingStateDef->set_stop_pending((STOP_PENDING_WORK_FLAG & snapshot) !=
                                          0);
    }
    if (_forceStopNode) {
        states->set_force_stop_node(_forceStopNode->getName());
    }
    if (_internal) {
        auto expensive = _internal->_expensive;
        auto expensiveDef = states->mutable_expensive_cache();
        expensiveDef->set_valid(expensive.valid);
        expensiveDef->set_expensive(expensive.expensive);
    }
    collectInputStates(states->mutable_inputs());
    if (hasOutput()) {
        collectOutputStates(states->mutable_outputs());
    }
    _metric->setKernelStates(states);
}

void Node::collectForkGraphMetric() const {
    if (!_runState || !_runState->forkGraph) {
        return;
    }
    if (!_runState->forkGraph->needCollect()) {
        return;
    }
    autil::ScopedLock lock(_runState->forkLock);
    _runState->forkGraph->collectMetric();
}

ErrorCode Node::fork(GraphDef *graphDef, const ForkGraphParam &param) {
    if (!graphDef) {
        NAVI_NODE_LOG(ERROR, "null graph def");
        return EC_FORK;
    }
    auto keepGraphDef = param.keepGraphDef;
    ErrorCode ec = EC_NONE;
    auto errorAsEof = param.errorAsEof;
    KernelMetricPtr forkMetric;
    std::unique_ptr<PerfEnableScope> perfEnableScope;
    ComputeScopePtr computeScope;
    auto graphParam = _graph->getParam();
    const auto &runParams = graphParam->runParams;
    auto paramTopoTreeNode = runParams.getParamNode(_graph->getGraphId(), _graph->getPartId(), getName());
    auto collectFlags = runParams.getCollectFlags(paramTopoTreeNode);
    MemLeakCollectorPtr memLeakCollector;
    if (collectFlags.needCollect()) {
        auto worker = _graph->getWorker();
        perfEnableScope = std::make_unique<PerfEnableScope>(worker->getNaviPerf(), collectFlags);
        if (unlikely(collectFlags.collectMalloc())) {
            if (!graphParam->memLeakCollector) {
                memLeakCollector = std::make_shared<MemLeakCollector>();
            } else {
                memLeakCollector = graphParam->memLeakCollector;
            }
        }
        auto schedInfo = worker->makeInlineSchedInfo();
        forkMetric = std::make_shared<KernelMetric>(INVALID_GRAPH_ID, "navi.fork_graph_init", "");
        auto topoParam = paramTopoTreeNode ? paramTopoTreeNode->getParam() : nullptr;
        computeScope = std::make_shared<ComputeScope>(
            forkMetric.get(),
            nullptr,
            _graph->getPartId(),
            perfEnableScope->getPerfThread(),
            memLeakCollector.get(),
            *schedInfo,
            collectFlags,
            topoParam ? &topoParam->getMallocTraceParam() : nullptr);
        computeScope->setType(GET_GRAPH_INIT);
    }
    {
        NAVI_NODE_LOG(SCHEDULE1, "fork graph: %s", graphDef->DebugString().c_str());
        autil::ScopedLock lock(_runState->forkLock);
        if (_graph->terminated()) {
            // see terminate
            if (!keepGraphDef) {
                multi_call::freeProtoMessage(graphDef);
            }
            NAVI_NODE_LOG(ERROR, "fork failed, graph terminated");
            if (computeScope) {
                computeScope->setErrorCode(EC_FORK);
            }
            return EC_FORK;
        }
        if (_runState->forkGraph) {
            if (!keepGraphDef) {
                multi_call::freeProtoMessage(graphDef);
            }
            NAVI_NODE_LOG(ERROR, "double fork");
            if (computeScope) {
                computeScope->setErrorCode(EC_FORK);
            }
            return EC_FORK;
        }
        if (!checkForkGraph(graphDef)) {
            if (!keepGraphDef) {
                multi_call::freeProtoMessage(graphDef);
            }
            NAVI_NODE_LOG(ERROR, "invalid fork graph def");
            if (computeScope) {
                computeScope->setErrorCode(EC_FORK);
            }
            return EC_FORK;
        }
        auto parentGraph = _graph->getGraph();
        auto forkGraphParam = new GraphParam();
        ec = forkGraphParam->init(_graph->getGraphId(),
                                  _graph->getPartId(),
                                  _graph->getBizName(),
                                  getName(),
                                  *graphParam,
                                  paramTopoTreeNode,
                                  memLeakCollector,
                                  getLogger().logger,
                                  param,
                                  getTimeoutChecker()->endTime());
        if (EC_NONE != ec) {
            if (!keepGraphDef) {
                multi_call::freeProtoMessage(graphDef);
            }
            delete forkGraphParam;
            NAVI_NODE_LOG(ERROR, "init fork graph param failed");
            if (computeScope) {
                computeScope->setErrorCode(ec);
            }
            return ec;
        }
        _runState->forkGraph = new Graph(forkGraphParam, parentGraph, false);
        if (forkMetric) {

            _runState->forkGraph->getGraphMetric()->addKernelMetricPtr(forkMetric);
        }
        auto forkDomain = new GraphDomainFork(
            _runState->forkGraph, _graph->getBiz(), this, _graph->getPartCount(), _graph->getPartId(), errorAsEof);
        GraphDomainPtr domain(forkDomain);
        _runState->forkGraph->setForkInfo(PARENT_GRAPH_ID, domain, this, errorAsEof);
        _runState->forkGraph->setSubResourceMap(param.subResourceMaps);
        ec = _runState->forkGraph->init(graphDef, keepGraphDef);
        if (keepGraphDef) {
            _runState->forkGraphDef = _runState->forkGraph->getCopyGraphDef();
        } else {
            _runState->forkGraphDef = graphDef;
        }
        if (EC_NONE != ec) {
            NAVI_NODE_LOG(ERROR, "fork graph init failed [%s]", CommonUtil::getErrorString(ec));
            graphParam->graphResult->addSubResult(forkGraphParam->graphResult);
            reportForkError(forkMetric, computeScope, ec);
            return ec;
        }
        if (!bindDomain(forkDomain)) {
            graphParam->graphResult->addSubResult(forkGraphParam->graphResult);
            reportForkError(forkMetric, computeScope, EC_FORK);
            return EC_FORK;
        }
        graphParam->graphResult->addSubResult(forkGraphParam->graphResult);
    }
    ec = _runState->forkGraph->run();
    if (forkMetric) {
        forkMetric->setFinish();
    }
    if (EC_NONE != ec) {
        if (!errorAsEof) {
            reportForkError(forkMetric, computeScope, ec);
            return ec;
        } else {
            _runState->forkGraph->notifyFinish(nullptr, GFT_PARENT_FINISH);
            if (forkMetric) {
                forkMetric->setFinish();
            }
            NAVI_NODE_LOG(DEBUG, "fork graph failed [%s], error as eof",
                     CommonUtil::getErrorString(ec));
            return EC_NONE;
        }
    } else {
        if (forkMetric) {
            forkMetric->setFinish();
        }
        NAVI_NODE_LOG(DEBUG, "fork graph success");
        return EC_NONE;
    }
}

void Node::reportForkError(const KernelMetricPtr &forkMetric,
                           const ComputeScopePtr &computeScope,
                           ErrorCode ec) const
{
    if (forkMetric) {
        forkMetric->setFinish();
    }
    auto error = _runState->forkGraph->getError(ec);
    _runState->forkGraph->notifyFinish(error, GFT_PARENT_FINISH);
    if (computeScope) {
        computeScope->setErrorCode(ec);
        computeScope->report();
    }
    _graph->setScopeError(getScope(), error);
}

bool Node::checkForkGraph(const GraphDef *graphDef) const {
    bool hasParentGraph = false;
    auto subCount = graphDef->sub_graphs_size();
    for (int i = 0; i < subCount; i++) {
        auto subGraphId = graphDef->sub_graphs(i).graph_id();
        if (USER_GRAPH_ID == subGraphId) {
            NAVI_NODE_LOG(ERROR, "fork graph can't has graph output");
            return false;
        }
        if (PARENT_GRAPH_ID == subGraphId) {
            hasParentGraph = true;
            break;
        }
    }
    if (hasParentGraph) {
        return true;
    } else {
        NAVI_NODE_LOG(ERROR,
                 "invalid fork graph, no fromInput or toOutput defined ");
        return false;
    }
}

void Node::terminate(ErrorCode ec) {
    if (!_runState) {
        return;
    }
    {
        // _graph->terminated is true,
        // lock success means no forking process
        autil::ScopedLock lock(_runState->forkLock);
    }
    auto asyncInfo = getAsyncInfo();
    if (asyncInfo) {
        setAsyncInfo(nullptr);
        asyncInfo->terminate();
    }
    if (_runState && _runState->forkGraph) {
        _runState->forkGraph->notifyFinish(nullptr, GFT_PARENT_FINISH);
    }
}

bool Node::bindDomain(GraphDomainFork *domain) {
    // replace output first
    bool ret = true;
    for (size_t i = 0; i < _internal->_outputs.size(); i++) {
        auto edge = _internal->_outputs[i];
        const auto &portName = _internal->_kernelCreator->getOutputPortName(i);
        auto peer = domain->getOutputPeer(portName);
        if (peer.first == nullptr or peer.second == INVALID_INDEX) {
            if (edge &&
                !_graph->transformEdgeFromInternal(edge)->isEofOrFinish(i)) {
                NAVI_NODE_LOG(ERROR,
                         "unfinished output must be bound by fork, output: %s",
                         portName.c_str());
                ret = false;
            }
            continue;
        }
        if (!edge) {
            peer.first->setOutputSlotEof(peer.second, false, false, false, this);
            continue;
        }
        _graph->transformEdgeFromInternal(edge)->bindInputNode(peer);
    }
    for (IndexType index = 0; index < _internal->_outputGroups.size(); index++) {
        auto group = _internal->_outputGroups[index];
        PortIndex groupIndex(INVALID_INDEX, index);
        auto groupName = _internal->_kernelCreator->getOutputPortName(groupIndex);
        auto groupSize = group->getOutputCount();
        for (size_t i = 0; i < groupSize; i++) {
            auto edgeInternal = group->getEdge(i);
            auto portName = CommonUtil::getGroupPortName(groupName, i);
            auto peer = domain->getOutputPeer(portName);
            if (peer.first == nullptr or peer.second == INVALID_INDEX) {
                if (edgeInternal && !group->isEofOrFinish(_port, _graph, i)) {
                    NAVI_NODE_LOG(ERROR,
                             "unfinished group output must be bound by fork, output: %s, index: %lu",
                             portName.c_str(),
                             i);
                    ret = false;
                }
                continue;
            }
            if (!edgeInternal) {
                peer.first->setOutputSlotEof(peer.second, false, false, false, this);
                continue;
            }
            _graph->transformEdgeFromInternal(edgeInternal)->bindInputNode(peer);
        }
    }
    for (size_t i = 0; i < _internal->_inputs.size(); i++) {
        auto portName = _internal->_kernelCreator->getInputPortName(i);
        auto peer = domain->getInputPeer(portName);
        const auto &edgeOutputInfo = _internal->_inputs[i];
        if (peer) {
            if (!edgeOutputInfo.isValid()) {
                peer->finishAsDownStream();
                continue;
            } else {
                auto edge = edgeOutputInfo.getEdge();
                auto slotId = edgeOutputInfo.getSlotId();
                _graph->transformEdgeFromInternal(edge)->bindOutputNode(slotId,
                                                                        peer);
            }
        } else {
            if (edgeOutputInfo.isValid() && !edgeOutputInfo.isEofOrFinish(_graph)) {
                NAVI_NODE_LOG(ERROR, "unfinished input must be bound by fork, input: %s", portName.c_str());
                ret = false;
            }
            continue;
        }
    }
    auto inputGroupCount = _internal->_kernelCreator->inputGroupSize();
    // filter out control and resource input group
    for (IndexType index = 0; index < inputGroupCount; index++) {
        if (_internal->_resourceInputGroupIndex == index) {
            continue;
        }
        auto group = _internal->_inputGroups[index];
        PortIndex groupIndex(INVALID_INDEX, index);
        auto groupName = _internal->_kernelCreator->getInputPortName(groupIndex);
        auto groupSize = group->getInputCount();
        if (unlikely(0 == groupSize)) {
            auto edges = domain->getEdgesByPrefix(groupName + GROUP_PORT_SEPERATOR);
            for (auto edge : edges) {
                edge->finishAsDownStream();
            }
            continue;
        }
        for (size_t i = 0; i < groupSize; i++) {
            auto edgeOutputInfo = group->getEdgeOutputInfo(i);
            auto edge = edgeOutputInfo.getEdge();
            auto slotId = edgeOutputInfo.getSlotId();
            auto portName = CommonUtil::getGroupPortName(groupName, i);
            auto peer = domain->getInputPeer(portName);
            if (!peer) {
                if (!group->isEofOrFinish(_port, _graph, i)) {
                    NAVI_NODE_LOG(ERROR,
                             "unfinished input group must be bound by fork, input: %s, index: %lu, maybe need provide "
                             "group index",
                             portName.c_str(),
                             i);
                    ret = false;
                }
                continue;
            }
            _graph->transformEdgeFromInternal(edge)->bindOutputNode(slotId,
                                                                    peer);
        }
    }
    if (!domain->checkBind()) {
        return false;
    }
    if (!ret) {
        return ret;
    }
    setStopPendingWork(this);
    finishPendingWork();
    domain->flushData();
    return true;
}

bool Node::getInput(PortIndex index, EdgeGetType type, StreamData &streamData, bool clear) {
    IndexType inputIndex = 0;
    EdgeOutputInfo inputInfo;
    if (!getInputInfo(index, inputInfo, inputIndex)) {
        return false;
    }
    if (!inputInfo.isValid()) {
        NAVI_NODE_LOG(SCHEDULE1,
                      "input group [%d] index [%d] not connected",
                      index.group(),
                      index.index());
        streamData.hasValue = false;
        streamData.eof = true;
        streamData.data.reset();
        return true;
    }
    auto &inputSnapshot = doFillInput(inputInfo, inputIndex, type);
    streamData.hasValue = inputSnapshot.data.hasValue;
    streamData.eof = inputSnapshot.data.eof;
    streamData.dataPointer = inputSnapshot.data.stealDataPointer();
    if (clear) {
        streamData.data = std::move(inputSnapshot.data.data);
        inputSnapshot.data.data = nullptr;
    } else {
        streamData.data = inputSnapshot.data.data;
    }
    return true;
}

InputSnapshot &Node::doFillInput(const EdgeOutputInfo &info,
                                 IndexType inputIndex,
                                 EdgeGetType type) {
    assert(inputIndex < _inputSnapshotVec.size());
    auto &inputSnapshot = _inputSnapshotVec[inputIndex];
    if (inputSnapshot.cleared) {
        return inputSnapshot;
    }
    auto peek = (EGT_PEEK == type);
    if (peek && inputSnapshot.peeked) {
        return inputSnapshot;
    }
    StreamData streamData;
    streamData.hasValue = info.getData(_graph,
                                       type,
                                       streamData.data,
                                       streamData.dataPointer,
                                       streamData.eof,
                                       this);
    auto dropData = (inputSnapshot.data.eof && !inputSnapshot.data.data &&
                     !inputSnapshot.data.dataPointer);
    if (likely(!dropData && !inputSnapshot.peeked)) {
        inputSnapshot.data = std::move(streamData);
    }
    if (!peek) {
        inputSnapshot.cleared = true;
    }
    inputSnapshot.peeked = true;
    return inputSnapshot;
}

size_t Node::getInputGroupCount() const {
    const auto *kernelDef = _internal->_kernelCreator->def();
    return kernelDef->input_groups().size();
}

bool Node::getInputGroupSize(IndexType group, size_t &size) {
    if (group < _internal->_inputGroups.size()) {
        size = _internal->_inputGroups[group]->getInputCount();
        return true;
    } else {
        return false;
    }
}

bool Node::setInputEof(PortIndex index) {
    IndexType inputIndex = 0;
    EdgeOutputInfo inputInfo;
    if (!getInputInfo(index, inputInfo, inputIndex)) {
        return false;
    }
    auto &inputSnapshot = _inputSnapshotVec[inputIndex];
    inputSnapshot.data.eof = true;
    inputSnapshot.cleared = true;
    if (inputInfo.isValid()) {
        inputInfo.setEof(_graph, false, this);
    }
    return true;
}

bool Node::getOutputGroupSize(IndexType group, size_t &size) {
    if (group < _internal->_outputGroups.size()) {
        size = _internal->_outputGroups[group]->getOutputCount();
        return true;
    } else {
        return false;
    }
}

bool Node::inputConnected(PortIndex index) {
    IndexType inputIndex = 0;
    EdgeOutputInfo inputInfo;
    if (!getInputInfo(index, inputInfo, inputIndex)) {
        return false;
    }
    return inputInfo.isValid();
}

bool Node::outputConnected(PortIndex index) {
    EdgeInternal *edge = nullptr;
    if (!getOutputEdgeInternal(index, edge)) {
        return false;
    }
    return (edge != nullptr);
}

NaviErrorPtr Node::getScopeError() const {
    return _graph->getScopeError(getScope());
}

ErrorHandleStrategy Node::getScopeErrorHandleStrategy() const {
    return _graph->getScopeErrorHandleStrategy(getScope());
}

int64_t Node::getRemainTimeMs() const {
    return _graph->getGraph()->getRemainTimeMs();
}

const TimeoutChecker *Node::getTimeoutChecker() const {
    return _graph->getGraph()->getTimeoutChecker();
}

bool Node::snapshotGroupInput(IndexType group,
                              EdgeGetType type,
                              InputSnapshot *&begin,
                              InputSnapshot *&end)
{
    if (group < _internal->_inputGroups.size()) {
        const auto &inputGroup = _internal->_inputGroups[group];
        auto size = inputGroup->getInputCount();
        auto inputIndexStart = inputGroup->getInputIndexStart();
        auto inputIndex = inputIndexStart;
        for (size_t i = 0; i < size; i++) {
            doFillInput(inputGroup->getEdgeOutputInfo(i), inputIndex, type);
            inputIndex++;
        }
        begin = &_inputSnapshotVec[inputIndexStart];
        end = begin + size;
        return true;
    } else {
        return false;
    }
}

bool Node::setOutput(PortIndex index,
                     const DataPtr &data,
                     Data *dataPointer,
                     bool eof) {
    Edge *edge = nullptr;
    if (!getOutputEdge(index, edge)) {
        delete dataPointer;
        return false;
    }
    if (unlikely(!edge)) {
        delete dataPointer;
        // output not connected
        return true;
    }
    auto ec = edge->setData(this, data, dataPointer, eof);
    if (likely(EC_NONE == ec)) {
        return true;
    }
    if (EC_DATA_OVERRIDE == ec) {
        NAVI_NODE_LOG(ERROR,
                      "output data override, group [%d] index [%d] edge[%p]",
                      index.group(),
                      index.index(),
                      edge);
    } else if (EC_DATA_AFTER_EOF == ec) {
        NAVI_NODE_LOG(ERROR,
                      "output data after eof, group [%d] index [%d] edge "
                      "[%p], new data [%p], new eof [%d], %ld",
                      index.group(),
                      index.index(),
                      edge,
                      data.get(),
                      eof,
                      getComputeId());
    } else {
        NAVI_NODE_LOG(
            ERROR,
            "set output failed, unknown error, group [%d] index [%d] edge[%p]",
            index.group(),
            index.index(),
            edge);
    }
    notifyFinish(ec);
    return false;
}

bool Node::getInputInfo(PortIndex index, EdgeOutputInfo &info, IndexType &inputIndex) const {
    if (_runState && _runState->forkGraph) {
        NAVI_NODE_LOG(ERROR, "input not available after fork");
        return false;
    }
    return _internal->getInputInfo(index, info, inputIndex);
}

bool NodeInternal::getInputInfo(PortIndex index,
                                EdgeOutputInfo &info,
                                IndexType &inputIndex) const {
    if (!index.isGroup()) {
        if (index.index() < _inputs.size()) {
            inputIndex = index.index();
            info = _inputs[index.index()];
            return true;
        } else {
            NAVI_KERNEL_LOG(
                ERROR,
                "invalid input port index [%d], node input size [%lu]",
                index.index(),
                _inputs.size());
            return false;
        }
    } else {
        if (index.group() < _inputGroups.size()) {
            auto group = _inputGroups[index.group()];
            auto inputCount = group->getInputCount();
            if (index.index() < inputCount) {
                inputIndex = group->getInputIndexStart() + index.index();
                info = group->getEdgeOutputInfo(index.index());
                return true;
            } else {
                NAVI_KERNEL_LOG(ERROR,
                                "invalid input port index [%d], group [%d] "
                                "input size [%lu]",
                                index.index(),
                                index.group(),
                                inputCount);
                return false;
            }
        } else {
            NAVI_KERNEL_LOG(
                ERROR,
                "invalid input group index [%d], node input group count [%lu]",
                index.group(),
                _inputGroups.size());
            return false;
        }
    }
    return false;
}

bool Node::getOutputEdgeInternal(PortIndex index, EdgeInternal *&edge) const {
    if (_runState && _runState->forkGraph) {
        NAVI_NODE_LOG(ERROR, "output not available after fork");
        return false;
    }
    return _internal->getOutputEdge(index, edge);
}

bool Node::getOutputEdge(PortIndex index, Edge *&edge) {
    EdgeInternal *edgeInternal = nullptr;
    auto ret = getOutputEdgeInternal(index, edgeInternal);
    if (ret && edgeInternal) {
        edge = _graph->transformEdgeFromInternal(edgeInternal);
    }
    return ret;
}

bool NodeInternal::getOutputEdge(PortIndex index, EdgeInternal *&edge) const {
    if (!index.isGroup()) {
        if (index.index() < _outputs.size()) {
            edge = _outputs[index.index()];
            return true;
        } else {
            NAVI_KERNEL_LOG(
                ERROR,
                "invalid output port index [%d], node output size [%lu]",
                index.index(),
                _outputs.size());
            return false;
        }
    } else {
        if (index.group() < _outputGroups.size()) {
            auto group = _outputGroups[index.group()];
            edge = group->getEdge(index.index());
            return true;
        } else {
            NAVI_KERNEL_LOG(ERROR,
                            "invalid output group index [%d], node output "
                            "group count [%lu]",
                            index.group(),
                            _outputGroups.size());
            return false;
        }
    }
}

Kernel *Node::getKernel() const {
    if (!_runState) {
        return nullptr;
    }
    return _runState->kernel;
}

void Node::setFrozen(bool frozen) {
    _frozen = frozen;
}

bool Node::isFinished() const {
    return SS_FINISH == _scheduleStat;
}

bool Node::getIORange(IoType ioType, const std::string &name, PortIndex &start,
                      PortIndex &end) const
{
    return _internal->getIORange(ioType, name, start, end);
}

bool NodeInternal::getIORange(IoType ioType,
                              const std::string &name,
                              PortIndex &start,
                              PortIndex &end) const {
    PortInfo portInfo;
    if (IOT_INPUT == ioType) {
        portInfo = _kernelCreator->getInputPortInfo(name);
    } else {
        portInfo = _kernelCreator->getOutputPortInfo(name);
    }
    if (!portInfo.isValid()) {
        NAVI_KERNEL_LOG(ERROR,
                        "no %s port named [%s]",
                        CommonUtil::getIoType(ioType),
                        name.c_str());
        return false;
    }
    if (portInfo.isGroup()) {
        auto groupIndex = portInfo.index.group();
        IndexType portCount = 0;
        if (IOT_INPUT == ioType) {
            portCount = _inputGroups[groupIndex]->getInputCount();
        } else {
            portCount = _outputGroups[groupIndex]->getOutputCount();
        }
        start.setGroup(groupIndex);
        start.setIndex(0);
        end.setGroup(groupIndex);
        end.setIndex(portCount);
    } else {
        start.setGroup(INVALID_INDEX);
        start.setIndex(portInfo.index.index());
        end.setGroup(INVALID_INDEX);
        end.setIndex(start.index() + 1);
    }
    return true;
}

bool Node::getInputNode(PortIndex index, std::string &node) const {
    IndexType inputIndex = 0;
    EdgeOutputInfo inputInfo;
    if (!getInputInfo(index, inputInfo, inputIndex)) {
        return false;
    }
    if (!inputInfo.isValid()) {
        node = NAVI_DISCONNECTED_NODE;
        return true;
    }
    node = inputInfo.getInputNodeName(_graph);
    return true;
}

int64_t Node::getScheduleCount() const {
    return _scheduleCount;
}

void Node::updateTraceLevel(const std::string &levelStr) {
    NAVI_NODE_LOG(SCHEDULE1, "update traceLevel[%s]", levelStr.c_str());
    getLogger().logger->updateTraceLevel(levelStr);
    _graph->getParam()->runParams.setTraceLevel(levelStr);
}

bool Node::updateTimeoutMs(int64_t timeoutMs) {
    NAVI_NODE_LOG(SCHEDULE1, "update timeoutMs[%ld]", timeoutMs);
    return _graph->getGraph()->updateTimeoutMs(timeoutMs);
}

void Node::updateThreadLimit(uint32_t threadLimit) {
    NAVI_NODE_LOG(SCHEDULE1, "update threadLimit[%u]", threadLimit);
    auto param = _graph->getParam();
    param->runParams.setThreadLimit(threadLimit);
    param->worker->updateThreadLimit(threadLimit);
}

void Node::updateCollect(bool collectMetric, bool collectPerf) const {
    auto &runParams = _graph->getParam()->runParams;
    runParams.setCollectMetric(collectMetric);
    runParams.setCollectPerf(collectPerf);
}

void Node::collectTrace(std::vector<std::string> &traceVec) const {
    TraceCollector collector;
    SymbolTableDef symbolTableDef;
    _graph->getParam()->graphResult->collectTrace(collector, symbolTableDef);
    collector.format(traceVec);
}

std::shared_ptr<multi_call::GigStreamRpcInfoMap> Node::getRpcInfoMap() const {
    return _graph->getParam()->graphResult->getRpcInfoMap();
}

void Node::setErrorCode(ErrorCode ec, const ComputeScopePtr &computeScope) {
    _graph->getParam()->graphResult->setError(getLogger().logger, ec);
    if (computeScope) {
        computeScope->setErrorCode(ec);
    }
}

void Node::notifyFinish(ErrorCode ec) {
    auto error = _graph->getParam()->graphResult->makeError(getLogger().logger, ec);
    _graph->notifyFinish(this, error);
}

void Node::notifyFinish(const NaviErrorPtr &error) {
    _graph->notifyFinish(this, error);
}

void Node::reportError(const NaviErrorPtr &error) {
    _graph->setScopeError(getScope(), error);
    auto forkNode = _graph->getGraph()->getForkNode();
    if (forkNode) {
        forkNode->reportError(error);
    } else {
        _graph->getParam()->graphResult->updateError(error);
    }
}

std::string Node::getLogPrefix() const {
    NaviObjectLogger logger;
    logger.addPrefix("graph %d %s kernel %s", _graph->getGraphId(), getName().c_str(), getKernelName().c_str());
    return logger.prefix;
}

}
