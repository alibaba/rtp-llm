#include "navi/engine/Edge.h"
#include "navi/engine/Node.h"
#include "navi/log/NaviLogger.h"
#include "navi/engine/LocalSubGraphInternal.h"

namespace navi {

std::ostream &operator<<(std::ostream &os, const EdgeInternal &edge) {
    return os << (void *)&edge << ",input[" << edge.getInputDef()->ShortDebugString() << "]";
}

std::ostream &operator<<(std::ostream &os, const Edge &edge) {
    return os << (void *)&edge << ",input[" << edge.getInputDef()->ShortDebugString() << "]";
}

bool EdgeInternal::init(LocalSubGraphInternal *graph, const NodePortDef &inputDef) {
    const auto &nodeName = inputDef.node_name();
    const auto &portName = inputDef.port_name();
    NodeInternal *inputNode = nullptr;
    if (NAVI_FORK_NODE != nodeName) {
        inputNode = graph->getNode(nodeName, 0);
        if (!inputNode) {
            NAVI_KERNEL_LOG(ERROR, "input node [%s] not found", nodeName.c_str());
            return false;
        }
        if (!inputNode->addOutput(this, portName, _typeId)) {
            return false;
        }
    }
    _inputNode = inputNode;
    _inputDef = &inputDef;
    return true;
}

Edge::Edge(LocalSubGraph *graph, EdgeInternal *internal)
    : _graph(graph)
    , _internal(internal)
    , _overrideIndex(INVALID_OVERRIDE_INDEX)
    , _upStreamIndex(INVALID_INDEX)
    , _edgeBitMapTree(nullptr)
    , _upStream(nullptr)
    , _totalDataCount(0)
{
    auto slotCount = _internal->_outputSlots.size();
    if (likely(slotCount <= 8)) {
        _slotReadyAtomic.init(slotCount);
    } else {
        _edgeBitMapTree = EdgeBitMapTree::createTree(slotCount);
    }
    _outputSlots.reserve(slotCount);
    for (size_t i = 0; i < slotCount; ++i) {
        auto &slotInternal = _internal->_outputSlots[i];
        _outputSlots.emplace_back(&slotInternal);
        if (unlikely(!slotInternal.require)) {
            setOptional(i, true);
        }
    }
    _atomicClear = (slotCount > 1);
}

Edge::~Edge() {
    EdgeBitMapTree::freeTree(_edgeBitMapTree);
    clearDataPointer();
}

EdgeOutputInfo EdgeInternal::addOutput(LocalSubGraphInternal *graph, const EdgeDef &edgeDef) {
    const auto &outputDef = edgeDef.output();
    const auto &selectIndexVec = edgeDef.select_index();
    const auto &nodeName = outputDef.node_name();
    if (NAVI_FORK_NODE == nodeName) {
        IndexType index = _outputSlots.size();
        EdgeOutputInfo edgeOutputInfo(this, index);
        _outputSlots.push_back(OutputSlot::Internal());
        auto &slot = _outputSlots.back();
        slot.outputDef = &outputDef;
        if (!selectIndexVec.empty()) {
            slot.selectIndexSet = &selectIndexVec;
        }
        return edgeOutputInfo;
    }
    auto outputNode = graph->getNode(nodeName, 0);
    if (!outputNode) {
        NAVI_KERNEL_LOG(ERROR, "edge output node [%s] not found", nodeName.c_str());
        return EdgeOutputInfo();
    }
    std::string outputTypeId;
    IndexType index = _outputSlots.size();
    EdgeOutputInfo edgeOutputInfo(this, index);
    _outputSlots.push_back(OutputSlot::Internal());
    auto &slot = _outputSlots.back();
    slot.outputDef = &outputDef;
    if (!selectIndexVec.empty()) {
        slot.selectIndexSet = &selectIndexVec;
    }
    slot.outputNode = outputNode;
    if (!outputNode->addInput(outputDef.port_name(),
                              edgeOutputInfo,
                              slot.outputIndex,
                              outputTypeId)) {
        return EdgeOutputInfo();
    }
    if (_typeId.empty()) {
        _typeId = outputTypeId;
        _typeIdOutputIndex = index;
    } else if (!outputTypeId.empty() && _typeId != outputTypeId) {
        if (INVALID_INDEX != _typeIdOutputIndex) {
            const auto &typeIdSlot = _outputSlots[_typeIdOutputIndex];
            NAVI_KERNEL_LOG(ERROR,
                     "downstream data type mismatch, input kernel[%s], output "
                     "node 1 [node %s kernel %s] port[%s] type[%s], output node 2 "
                     "[node %s kernel %s] port[%s] type[%s]",
                     getInputNode() ? getInputNode()->getKernelName().c_str() : "",
                     typeIdSlot.outputDef->node_name().c_str(),
                     typeIdSlot.outputNode->getKernelName().c_str(),
                     typeIdSlot.outputDef->port_name().c_str(),
                     _typeId.c_str(),
                     outputDef.node_name().c_str(),
                     outputNode->getKernelName().c_str(),
                     outputDef.port_name().c_str(),
                     outputTypeId.c_str());
        } else {
            NAVI_KERNEL_LOG(ERROR,
                     "edge data type mismatch, input kernel[%s] type[%s], "
                     "output [node %s kernel %s] port[%s] type[%s]",
                     getInputNode() ? getInputNode()->getKernelName().c_str() : "",
                     _typeId.c_str(),
                     outputDef.node_name().c_str(),
                     outputNode->getKernelName().c_str(),
                     outputDef.port_name().c_str(),
                     outputTypeId.c_str());
        }
        return EdgeOutputInfo();
    }
    return edgeOutputInfo;
}

void Edge::setOptional(IndexType slotId, bool value) {
    if (likely(!_edgeBitMapTree)) {
        _slotReadyAtomic.setOptional(slotId, value);
    } else {
        _edgeBitMapTree->setOptional(slotId, value);
    }
}

bool EdgeInternal::checkType(const CreatorManager *creatorManager) const {
    if (!_typeId.empty() && !creatorManager->getType(_typeId)) {
        auto inputDef = getInputDef();
        NAVI_KERNEL_LOG(
            ERROR,
            "port data type [%s] not registered, node [%s] port [%s]",
            _typeId.c_str(),
            inputDef->node_name().c_str(),
            inputDef->port_name().c_str());
        return false;
    }
    return true;
}

const NodePortDef *Edge::getInputDef() const {
    if (_upStream) {
        return _upStream->getInputDef();
    } else {
        return _internal->getInputDef();
    }
}

const NodePortDef *Edge::getOutputDef(IndexType slotId) const {
    return _internal->getOutputDef(slotId);
}

void Edge::bindInputNode(const std::pair<Edge *, IndexType> &upStreamOutputInfo) {
    // this is fork node output edge
    auto upStreamEdge = upStreamOutputInfo.first;
    auto upStreamSlotId = upStreamOutputInfo.second;
    auto &slot = upStreamEdge->_outputSlots[upStreamSlotId];
    slot.downStream = this;
    _upStreamIndex = upStreamSlotId;
    if (unlikely(needCollectMetric())) {
        upStreamEdge->setNeedCollectMetric();
    }
    NAVI_MEMORY_BARRIER();
    _upStream = upStreamEdge;
    NAVI_MEMORY_BARRIER();
    if (isAllSlotFinished()) {
        _upStream->setSlotFinish(_upStreamIndex, nullptr);
    }
}

void Edge::bindOutputNode(IndexType slotId, Edge *downStream) {
    // this is fork node input edge
    downStream->_upStream = this;
    downStream->_upStreamIndex = slotId;
    NAVI_MEMORY_BARRIER();
    auto &slot = _outputSlots[slotId];
    slot.downStream = downStream;
    if (downStream->needCollectMetric()) {
        setNeedCollectMetric();
    }
}

void Edge::finishAsDownStream() {
    auto slotCount = _outputSlots.size();
    for (size_t i = 0; i < slotCount; i++) {
        auto &slot = _outputSlots[i];
        setSlotData(nullptr, nullptr, true, false, slot);
    }
    setSlotHasData(true);
}

bool Edge::hasDownStream(IndexType slotId) const {
    return _outputSlots[slotId].downStream != nullptr;
}

bool Edge::hasUpStream() const {
    return _upStream != nullptr;
}

EdgeOutputInfo Edge::getUpStreamOutputInfo() const {
    return EdgeOutputInfo(_upStream ? _upStream->_internal : nullptr,
                          _upStreamIndex);
}

Edge *Edge::getDownStream(IndexType slotId) const {
    return _outputSlots[slotId].downStream;
}

bool Edge::finishInputWork(const Node *callNode) {
    NAVI_EDGE_LOG(SCHEDULE1, "finish input pending work");
    auto slotCount = _outputSlots.size();
    size_t backFinishCount = 0;
    for (size_t i = 0; i < slotCount; i++) {
        auto &slot = _outputSlots[i];
        if (slot.backFinish) {
            backFinishCount++;
        }
        if (slot.downStream) {
            auto finish = (slot.backFinish || slot.isTimeout || slot.forceEof);
            if (!slot.flushed) {
                slot.flushed = true;
                _hasDownStream = true;
                bool hasData = false;
                DataPtr data;
                Data *dataPointer = nullptr;
                bool eof = finish;
                if (slotHasData(i)) {
                    hasData = clearSlotData(i, data, dataPointer, eof);
                }
                if (slot.backFinish) {
                    slot.finishUpStreamClear();
                }
                NAVI_EDGE_LOG(
                    SCHEDULE1,
                    "finish input pending work found downstream "
                    "[%lu], inputNode [%s] inputPort [%s], backFinish [%d] isTimeout [%d] forceEof [%d] "
                    "hasData [%d], data [%p] dataPointer [%p] eof [%d] slot eof [%d]",
                    i,
                    SubGraphBase::getNodeName(_graph, *_internal->getInputDef())
                        .c_str(),
                    _internal->getInputDef()->port_name().c_str(),
                    slot.backFinish,
                    slot.isTimeout,
                    slot.forceEof,
                    hasData,
                    data.get(),
                    dataPointer,
                    eof, slot.eof);
                if (hasData || slot.eof) {
                    slot.downStream->notifyData(data, dataPointer, eof || slot.eof, false, callNode);
                } else if (finish) {
                    slot.downStream->propagateSlotEof(slot.isTimeout, slot.forceEof, callNode);
                }
            } else if (finish) {
                slot.downStream->propagateSlotEof(slot.isTimeout, slot.forceEof, callNode);
            }
            if (likely(!_edgeBitMapTree)) {
                _slotReadyAtomic.clearSlotHasData(i, _atomicClear);
            } else {
                _edgeBitMapTree->clearSlotHasData(i, _atomicClear);
            }
            if (!slot.downStream->finishInputWork(callNode)) {
                return false;
            }
        } else if (slot.backFinish) {
            slot.finishUpStreamClear();
        }
    }
    if (slotCount == backFinishCount) {
        clearDataPointer();
    }
    return true;
}

void Edge::finishOutputWork(IndexType slotId, const Node *callNode) {
    NAVI_EDGE_LOG(SCHEDULE1, "finish output pending work");
    auto &slot = _outputSlots[slotId];
    if (slot.backFinish) {
        slot.finishDownStreamClear();
    }
}

void Edge::notifyInputWork(const Node *callNode) {
    if (_upStream) {
        _upStream->notifyInputWork(callNode);
        return;
    }
    auto node = _graph->transformNodeFromInternal(_internal->_inputNode);
    if (node) {
        if (!node->setOutputPendingWork(callNode ? callNode : node)) {
            NAVI_EDGE_LOG(SCHEDULE2,
                          "notify input pending work failed, node [%s]",
                          _internal->_inputNode->_name.c_str());
            if (_upStream) {
                _upStream->notifyInputWork(callNode);
            }
        } else {
            NAVI_EDGE_LOG(SCHEDULE2,
                          "notify input pending work success, node [%s]",
                          _internal->_inputNode->_name.c_str());
        }
    }
}

void Edge::notifyOutputWork(IndexType slotId, const Node *callNode) {
    auto &slot = _outputSlots[slotId];
    auto node = _graph->transformNodeFromInternal(slot.internal->outputNode);
    if (node) {
        NAVI_EDGE_LOG(SCHEDULE2, "notify output pending work, slot [%d]", slotId);
        if (!node->setInputPendingWork(callNode)) {
            if (slot.downStream) {
                slot.downStream->notifyOutputWork(0, callNode);
            }
        }
    }
}

void Edge::setPendingFlush(IndexType slotId, const Node *callNode) {
    auto &slot = _outputSlots[slotId];
    if (!slot.downStream) {
        return;
    }
    notifyInputWork(callNode);
    notifyOutputWork(slotId, callNode);
}

void Edge::flushData(Node *callNode) {
    NAVI_EDGE_LOG(
        SCHEDULE1,
        "flush data, inputNode [%s] inputPort [%s]",
        SubGraphBase::getNodeName(_graph, *_internal->getInputDef()).c_str(),
        _internal->getInputDef()->port_name().c_str());
    if (_upStream) {
        _upStream->setPendingFlush(_upStreamIndex, callNode);
    } else {
        // disconnected input
        collectSetEvent(true, nullptr, true, callNode);
        notifyData(nullptr, nullptr, true, false, callNode);
    }
}

void Edge::flushAllData(Node *callNode) {
    auto slotCount = _outputSlots.size();
    for (size_t i = 0; i < slotCount; i++) {
        setPendingFlush(i, callNode);
    }
}

bool Edge::checkDataType(const Data *data) const {
    if (data && !_internal->_typeId.empty()) {
        const auto &dataType = data->getTypeId();
        if (_internal->_typeId != dataType) {
            NAVI_EDGE_LOG(ERROR, "edge data type mismatch, expect[%s], actual[%s]",
                     _internal->_typeId.c_str(), dataType.c_str());
            return false;
        }
    }
    return true;
}

void Edge::setEof(const Node *callNode) {
    setData(callNode, nullptr, nullptr, true);
}

void Edge::forceEof(const Node *callNode) {
    auto slotCount = _outputSlots.size();
    for (size_t i = 0; i < slotCount; i++) {
        setOutputSlotEof(i, false, true, false, callNode);
    }
}

ErrorCode Edge::setData(const Node *callNode,
                        const DataPtr &data,
                        Data *dataPointer,
                        bool eof) {
    if (isOverride()) {
        delete dataPointer;
        return EC_NONE;
    }
    return doSetData(callNode, data, dataPointer, eof);
}

ErrorCode Edge::doSetData(const Node *callNode,
                          const DataPtr &data,
                          Data *dataPointer,
                          bool eof) {
    std::unique_ptr<Data> dataPointerUniq(dataPointer);
    if (data && dataPointer) {
        NAVI_EDGE_LOG(ERROR, "got both data and dataPointer, override!");
        return EC_DATA_OVERRIDE;
    }
    assert(!_upStream);
    if (!checkDataType(data.get()) || !checkDataType(dataPointer)) {
        collectSetEvent(false, data ? data.get() : dataPointer, eof, callNode);
        return EC_UNKNOWN;
    }
    if (isEofForSetData()) {
        NAVI_EDGE_LOG(ERROR, "data after eof!");
        collectSetEvent(false, data ? data.get() : dataPointer, eof, callNode);
        return EC_DATA_AFTER_EOF;
    }
    if (!allSlotEmpty()) {
        NAVI_EDGE_LOG(ERROR, "data override!");
        collectSetEvent(false, data ? data.get() : dataPointer, eof, callNode);
        return EC_DATA_OVERRIDE;
    }
    NAVI_EDGE_LOG(SCHEDULE1, "set data [%p] eof [%d], call node: [%p, node %s kernel %s]",
             data.get(), eof, callNode,
             callNode ? callNode->getName().c_str() : "",
             callNode ? callNode->getKernelName().c_str() : "");
    if (isAllSlotFinished()) {
        collectSetEvent(false, data ? data.get() : dataPointer, eof, callNode);
        return EC_NONE;
    }
    collectSetEvent(true, data ? data.get() : dataPointer, eof, callNode);
    notifyData(data, dataPointerUniq.release(), eof, false, callNode);
    return EC_NONE;
}

// for root edge
void Edge::notifyData(const DataPtr &data,
                      Data *dataPointer,
                      bool eof,
                      bool needClone,
                      const Node *callNode) {
    _totalDataCount++;
    auto slotCount = _outputSlots.size();
    needClone = needClone || (slotCount > 1 && (data || dataPointer));
    NaviDisableSignalScope disableScope(slotCount > 1);
    bool used = false;
    clearDataPointer();
    for (size_t i = 0; i < slotCount; i++) {
        auto &slot = _outputSlots[i];
        if (unlikely(isSlotFinished(i))) {
            continue;
        }
        if (unlikely(nullptr != slot.downStream)) {
            slot.flushed = true;
            _hasDownStream = true;
        }
        used = true;
        setSlotData(data, dataPointer, eof, needClone, slot);
    }
    if (!used) {
        DELETE_AND_SET_NULL(dataPointer);
    }
    NAVI_MEMORY_BARRIER();
    setSlotHasData(eof);
    NAVI_MEMORY_BARRIER();
    for (size_t i = 0; i < slotCount; i++) {
        if (unlikely(isSlotFinished(i))) {
            continue;
        }
        auto &slot = _outputSlots[i];
        if (unlikely(nullptr != slot.downStream && slot.flushed)) {
            if (likely(!_edgeBitMapTree)) {
                _slotReadyAtomic.clearSlotHasData(i, _atomicClear);
            } else {
                _edgeBitMapTree->clearSlotHasData(i, _atomicClear);
            }
            DataPtr dataTmp;
            Data *dataPointer = nullptr;
            bool eof = false;
            clearSlotData(i, dataTmp, dataPointer, eof);
            slot.downStream->notifyData(dataTmp, dataPointer, eof, false, callNode);
        } else {
            auto node =
                _graph->transformNodeFromInternal(slot.internal->outputNode);
            if (node) {
                NAVI_EDGE_LOG(SCHEDULE2,
                         "notify with data[%p] eof[%d], idx[%lu]",
                         data.get(),
                         eof,
                         i);
                node->schedule(callNode);
            }
        }
    }
}

// for downstream
bool Edge::getData(EdgeGetType type,
                   IndexType index,
                   DataPtr &data,
                   Data *&dataPointer,
                   bool &eof,
                   const Node *callNode) {
    auto peek = (EGT_PEEK == type);
    auto &slot = _outputSlots[index];
    slot.getCount++;
    eof = isSlotFinished(index);
    if (!slotHasData(index)) {
        if (slot.forceEof || slot.isTimeout || slot.backFinish) {
            eof = true;
            collectGetEvent(false, index, type, nullptr, true, callNode);
            return true;
        }
        NAVI_EDGE_LOG(SCHEDULE1,
                 "get data failed, data not ready, index [%d], output [%s]",
                 index,
                 getOutputName(index).c_str());
        collectGetEvent(false, index, type, nullptr, false, callNode);
        return false;
    }
    if (unlikely(peek)) {
        if (!readSlotData(index, data, dataPointer, eof)) {
            NAVI_EDGE_LOG(ERROR, "get data failed, slot [%d] do not has data", index);
            return false;
        }
        collectGetEvent(true, index, type, data ? data.get() : dataPointer, eof, callNode);
        return true;
    } else {
        if (!clearSlotData(index, data, dataPointer, eof)) {
            if (slot.forceEof || slot.isTimeout || slot.backFinish) {
                eof = true;
            } else {
                NAVI_EDGE_LOG(ERROR, "get data failed, slot [%d] do not has data", index);
                return false;
            }
        }
    }
    if (!clearSlotHasData(index, callNode)) {
        if (dataPointer) {
            delete dataPointer;
            dataPointer = nullptr;
        }
        return false;
    }
    NAVI_EDGE_LOG(SCHEDULE1,
                  "getData success, data [%p], dataPointer [%p], output [%s], "
                  "index [%d], eof [%d]",
                  data.get(),
                  dataPointer,
                  getOutputName(index).c_str(),
                  index,
                  eof);
    collectGetEvent(true, index, type, data ? data.get() : dataPointer, eof, callNode);
    return true;
}

void Edge::propagateSlotEof(bool isTimeout, bool forceEof, const Node *callNode) {
    auto slotCount = _outputSlots.size();
    for (size_t i = 0; i < slotCount; i++) {
        setOutputSlotEof(i, isTimeout, forceEof, true, callNode);
    }
}

// for downstream and timeout
void Edge::setOutputSlotEof(IndexType slotId,
                            bool isTimeout,
                            bool forceEof,
                            bool ignoreInputWork,
                            const Node *callNode) {
    if (isSlotFinished(slotId)) {
        return;
    }
    NAVI_EDGE_LOG(SCHEDULE1, "set finish by node [%s]", callNode ? callNode->getName().c_str() : "");
    auto &slot = _outputSlots[slotId];
    setSlotFinish(slotId, callNode);
    if (isTimeout) {
        slot.isTimeout = true;
    } else if (forceEof) {
        slot.forceEof = true;
    } else {
        slot.backFinish = true;
    }
    if (slot.backFinish) {
        slot.triggerClear();
    }
    if (!ignoreInputWork) {
        notifyInputWork(callNode);
    }
    notifyOutputWork(slotId, callNode);
    if (slot.downStream) {
        slot.downStream->propagateSlotEof(isTimeout, forceEof, callNode);
    }
}

void Edge::setSlotFinish(IndexType slotId, const Node *callNode) {
    bool allSlotFinish = false;
    if (likely(!_edgeBitMapTree)) {
        allSlotFinish = _slotReadyAtomic.setSlotFinish(slotId);
    } else {
        allSlotFinish = _edgeBitMapTree->setSlotFinish(slotId);
    }
    if (!_upStream) {
        if (allSlotFinish) {
            if (!isOverride()) {
                auto node =
                    _graph->transformNodeFromInternal(_internal->_inputNode);
                if (node) {
                    NAVI_EDGE_LOG(SCHEDULE1,
                             "schedule call by node [%s] getData",
                             callNode ? callNode->getName().c_str() : "");
                    node->schedule(callNode);
                }
            }
        }
    } else {
        if (allSlotFinish) {
            // fork output edge
            auto node = _graph->transformNodeFromInternal(_internal->_inputNode);
            if (node) {
                node->schedule(callNode);
            }
        }
    }
}

bool Edge::clearSlotHasData(IndexType slotId, const Node *callNode) {
    bool allDataClear = false;
    bool eofOrFinish = false;
    if (likely(!_edgeBitMapTree)) {
        allDataClear = _slotReadyAtomic.clearSlotHasData(slotId, _atomicClear);
        eofOrFinish = _slotReadyAtomic.isEofOrFinish();
    } else {
        allDataClear = _edgeBitMapTree->clearSlotHasData(slotId, _atomicClear);
        eofOrFinish = _edgeBitMapTree->isEofOrFinish();
    }
    if (_upStream && eofOrFinish) {
        NAVI_EDGE_LOG(SCHEDULE1, "finish fork output edge");
        // fork output edge
        auto node = _graph->transformNodeFromInternal(_internal->_inputNode);
        if (node) {
            node->schedule(callNode);
        }
    }
    if (!eofOrFinish && allDataClear) {
        return scheduleInput(callNode);
    } else {
        return true;
    }
}

bool Edge::scheduleInput(const Node *callNode) {
    if (unlikely(nullptr != _upStream)) {
        return _upStream->scheduleInput(callNode);
    }
    if (likely(!_edgeBitMapTree)) {
        if (_slotReadyAtomic.isEofOrFinish()) {
            return true;
        }
    } else {
        if (_edgeBitMapTree->isEofOrFinish()) {
            return true;
        }
    }
    if (!isOverride()) {
        auto node = _graph->transformNodeFromInternal(_internal->_inputNode);
        if (node) {
            NAVI_EDGE_LOG(SCHEDULE1,
                     "schedule call by node [%s] getData",
                     callNode ? callNode->getName().c_str() : "");
            node->schedule(callNode);
        }
    } else if (!flushOverrideData()) {
        return false;
    }
    return true;
}

std::string EdgeInternal::getOutputName(IndexType index, LocalSubGraph *graph) const {
    assert(index < _outputSlots.size());
    auto outputDef = _outputSlots[index].outputDef;
    return SubGraphBase::getNodeName(graph, *outputDef) + ":" +
           outputDef->port_name();
}

std::string EdgeInternal::getDebugName(IndexType index, LocalSubGraph *graph) const {
    assert(index < _outputSlots.size());
    const NodePortDef *inputDef = getInputDef();
    return SubGraphBase::getNodeName(graph, *inputDef) + ":" +
           inputDef->port_name() + " -> " + getOutputName(index, graph);
}

std::string Edge::getOutputName(IndexType index) const {
    return _internal->getOutputName(index, _graph);
}

bool Edge::overrideData(const std::vector<DataPtr> &datas) {
    if (isOverride()) {
        NAVI_EDGE_LOG(ERROR, "multiple edge data override specified");
        return false;
    }
    _overrideIndex = 0;
    _overrideDatas = datas;
    return flushOverrideData();
}

bool Edge::flushOverrideData() {
    if (!isInputOk()) {
        NAVI_EDGE_LOG(SCHEDULE1, "flush override output data ignored, edge input is not ok");
        return true;
    }
    assert(isOverride());
    auto dataCount = _overrideDatas.size();
    if (_overrideIndex > dataCount) {
        NAVI_EDGE_LOG(ERROR, "override index overflow, dataCount: %lu, index: %u",
                 dataCount, _overrideIndex);
        return false;
    }
    DataPtr data;
    auto index = _overrideIndex++;
    if (index < dataCount) {
        data = _overrideDatas[index];
    }
    auto eof = _overrideIndex >= dataCount;
    if (eof) {
        _overrideDatas.clear();
    }
    NAVI_EDGE_LOG(SCHEDULE1, "override output data, index [%u] data [%p] eof [%d]",
             index, data.get(), eof);
    auto ec = doSetData(nullptr, data, nullptr, eof);
    return (EC_NONE == ec);
}

bool Edge::isOverride() const {
    return INVALID_OVERRIDE_INDEX != _overrideIndex;
}

bool Edge::startUpStreamSchedule(const Node *callNode, bool graphStart) const {
    if (_upStream) {
        return _upStream->startUpStreamSchedule(callNode, graphStart);
    } else {
        auto node = _graph->transformNodeFromInternal(_internal->_inputNode);
        if (node) {
            return node->startSchedule(callNode, false, graphStart);
        }
    }
    return false;
}

void Edge::fillUpstreamState(KernelOutputStateDef *state) const {
    state->set_port(_internal->getInputDef()->port_name());
    if (!_upStream) {
        doFillUpstreamState(state);
    } else {
        _upStream->doFillUpstreamState(state);
    }
}

void Edge::doFillUpstreamState(KernelOutputStateDef *state) const {
    if (likely(!_edgeBitMapTree)) {
        state->set_ready(_slotReadyAtomic.allSlotEmpty());
        state->set_eof(_slotReadyAtomic.isEof());
        state->set_finish(_slotReadyAtomic.isEof());
        state->set_data_count(_totalDataCount);
    } else {
        state->set_ready(_edgeBitMapTree->allSlotEmpty());
        state->set_eof(_edgeBitMapTree->isEof());
        state->set_finish(_edgeBitMapTree->isEof());
        state->set_data_count(_totalDataCount);
    }
}

void Edge::fillDownstreamState(IndexType slotId, KernelInputStateDef *state) const {
    const auto &slot = _outputSlots[slotId];
    if (likely(!_edgeBitMapTree)) {
        state->set_ready(_slotReadyAtomic.slotHasData(slotId));
        state->set_finish(_slotReadyAtomic.isEof(slotId));
        state->set_back_finish(slot.backFinish);
        state->set_optional(_slotReadyAtomic.isOptional(slotId));
    } else {
        state->set_ready(_edgeBitMapTree->slotHasData(slotId));
        state->set_finish(_edgeBitMapTree->isEof(slotId));
        state->set_back_finish(slot.backFinish);
        state->set_optional(_edgeBitMapTree->isOptional(slotId));
    }
    state->set_get_count(slot.getCount);
    state->set_is_timeout(slot.isTimeout);
}

void Edge::setSummaryQuery(GraphId graphId,
                           const std::string &node,
                           const std::string &port,
                           const std::string &query,
                           CreatorManager *creatorManager)
{
    if (!_metric) {
        _metric = std::make_shared<EdgeMetric>(graphId, node, port, query, creatorManager);
        _needCollectMetric = true;
        auto slotCount = _outputSlots.size();
        for (size_t i = 0; i < slotCount; i++) {
            auto outputDef = _internal->_outputSlots[i].outputDef;
            _metric->addSlotName(i,
                                 SubGraphBase::getNodeName(_graph, *outputDef),
                                 outputDef->port_name());
        }
    }
}

void Edge::collectSetEvent(bool success, const Data *data, bool eof, const Node *callNode) const {
    if (!_needCollectMetric) {
        return;
    }
    auto timeNs = CommonUtil::getTimelineTimeNs();
    if (_metric) {
        _metric->reportSetEvent(success, timeNs, data, eof, callNode);
    }
    for (const auto &slot : _outputSlots) {
        if (!slot.downStream) {
            continue;
        }
        slot.downStream->collectSetEvent(success, data, eof, callNode);
    }
}

void Edge::collectGetEvent(
    bool success, IndexType slotId, EdgeGetType type, const Data *data, bool eof, const Node *callNode) const {
    if (_upStream) {
        _upStream->collectGetEvent(success, _upStreamIndex, type, data, eof, callNode);
    }
    if (!_metric) {
        return;
    }
    _metric->reportGetEvent(success, CommonUtil::getTimelineTimeNs(), slotId, type, data, eof, callNode);
}

std::string Edge::getLogPrefix() const {
    auto inputDef = _internal->_inputDef;
    if (!inputDef) {
        return std::string();
    }
    NaviObjectLogger logger;
    logger.addPrefix("%s:%s", inputDef->node_name().c_str(), inputDef->port_name().c_str());
    return logger.prefix;
}

} // namespace navi
