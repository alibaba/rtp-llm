#pragma once

#include "autil/Lock.h"
#include "navi/common.h"
#include "navi/engine/Data.h"
#include "navi/engine/GraphParam.h"
#include "navi/log/NaviLogger.h"
#include "navi/engine/EdgeMetric.h"
#include "navi/engine/EdgeBitMap.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/proto/GraphDef.pb.h"
#include "navi/util/ReadyBitMap.h"

namespace navi {

class Edge;
#define NAVI_EDGE_LOG(level, format, args...)                                                                          \
    {                                                                                                                  \
        static_assert(std::is_same<std::remove_cv<std::remove_pointer_t<decltype(this)>>::type, Edge>::value,          \
                      "type must be Edge");                                                                            \
        NAVI_OBJECT_LOG(this, LOT_EDGE, level, format, ##args);                                                        \
    }

class Node;
struct NodeInternal;
class EdgeOutputInfo;
class LocalSubGraph;
struct LocalSubGraphInternal;
class Edge;

struct OutputSlot {
    struct Internal {
    public:
        Internal()
            : outputDef(nullptr)
            , selectIndexSet(nullptr)
            , outputIndex(INVALID_INDEX)
            , outputNode(nullptr) {}
    public:
        const NodePortDef *outputDef;
        const google::protobuf::RepeatedField<google::protobuf::int32>
            *selectIndexSet;
        PortIndex outputIndex;
        bool require = true;
        NodeInternal *outputNode;
    };
public:
    OutputSlot(Internal *internal_)
        : internal(internal_)
        , outputIndex(internal_->outputIndex)
        , hasData(false)
        , eof(false)
        , needClone(false)
    {
    }
    ~OutputSlot() {
    }
public:
    // fork flush    flushData
    // setOutputEof  setOutputSlotEof
    // timeout       setOutputSlotEof
    // forceEof      setOutputSlotEof
    static constexpr uint16_t CLEAR_STATE_TRIGGER = 1u;
    static constexpr uint16_t CLEAR_STATE_UPSTREAM = (1u << 1);
    static constexpr uint16_t CLEAR_STATE_DOWNSTREAM = (1u << 2);
    static constexpr uint16_t CLEAR_STATE_FINISH = ((1u << 3) - 1u);
public:
    void triggerClear() {
        uint16_t oldState;
        uint16_t newState;
        do {
            oldState = clearDataState;
            if (oldState & CLEAR_STATE_TRIGGER) {
                return;
            }
            newState = (oldState | CLEAR_STATE_TRIGGER);
        } while (!cmpxchg(&clearDataState, oldState, newState));
    }
    void finishUpStreamClear() {
        finishClear(CLEAR_STATE_UPSTREAM);
    }
    void finishDownStreamClear() {
        finishClear(CLEAR_STATE_DOWNSTREAM);
    }
private:
    void finishClear(uint16_t mask) {
        uint16_t oldState;
        uint16_t newState;
        do {
            oldState = clearDataState;
            if (!(oldState & CLEAR_STATE_TRIGGER)) {
                return;
            }
            if (oldState & mask) {
                return;
            }
            newState = (oldState | mask);
        } while (!cmpxchg(&clearDataState, oldState, newState));
        if (CLEAR_STATE_FINISH == newState) {
            NAVI_KERNEL_LOG(SCHEDULE1, "finish clear data [%p]", data.get());
            data.reset();
            hasData = false;
            eof = true;
        }
    }
public:
    Internal *internal;
    DataPtr data;
    Data *dataPointer = nullptr;
    PortIndex outputIndex;
    bool hasData : 1;
    bool eof : 1;
    bool needClone : 1;
    // downStream field
    bool flushed = false;
    bool isTimeout = false;
    bool forceEof = false;
    bool backFinish = false;
    uint16_t clearDataState = 0;
    size_t getCount = 0;
    Edge *downStream = nullptr;
};

struct EdgeInternal {
public:
    EdgeInternal(const EdgeDef &edgeDef, uint32_t id_)
        : _edgeDef(&edgeDef)
        , id(id_)
    {
    }
public:
    const EdgeDef *_edgeDef = nullptr;
    uint32_t id;
    const NodePortDef *_inputDef = nullptr;
    NodeInternal *_inputNode = nullptr;
    NaviVector<OutputSlot::Internal, 4> _outputSlots;
    std::string _typeId;
    IndexType _typeIdOutputIndex = INVALID_INDEX;
public:
    friend std::ostream &operator<<(std::ostream &os, const EdgeInternal &edge);
    bool init(LocalSubGraphInternal *graph,
              const NodePortDef &inputDef);
    bool checkType(const CreatorManager *creatorManager) const;
    EdgeOutputInfo addOutput(LocalSubGraphInternal *graph,
                             const EdgeDef &edgeDef);
    size_t getOutputDegree() const {
        return _outputSlots.size();
    }
    std::string getOutputName(IndexType index, LocalSubGraph *graph) const;
    std::string getDebugName(IndexType index, LocalSubGraph *graph) const;
    const NodePortDef *getInputDef() const {
        return _inputDef;
    }
    const NodePortDef *getOutputDef(IndexType slotId) const {
        return _outputSlots[slotId].outputDef;
    }
    NodeInternal *getInputNode() const {
        return _inputNode;
    }
    const google::protobuf::RepeatedField<google::protobuf::int32> *
    getSelectIndexVec(IndexType slotId) const {
        return _outputSlots[slotId].selectIndexSet;
    }
};

class Edge
{
public:
    Edge(LocalSubGraph *graph, EdgeInternal *internal);
    ~Edge();
private:
    Edge(const Edge &);
    Edge& operator=(const Edge &);

public:
    friend std::ostream &operator<<(std::ostream &os, const Edge &edge);

public:
    LocalSubGraph *getGraph() const {
        return _graph;
    }
    void setOptional(IndexType slotId, bool value);
    const NodePortDef *getInputDef() const;
    const NodePortDef *getOutputDef(IndexType slotId) const;
    void setEof(const Node *callNode);
    void forceEof(const Node *callNode);
    ErrorCode setData(const Node *callNode,
                      const DataPtr &data,
                      Data *dataPointer,
                      bool eof);
    bool getData(EdgeGetType type,
                 IndexType index,
                 DataPtr &data,
                 Data *&dataPointer,
                 bool &eof,
                 const Node *callNode);
    void setOutputSlotEof(IndexType slotId,
                          bool isTimeout,
                          bool forceEof,
                          bool ignoreInputWork,
                          const Node *callNode);
    const NaviVector<OutputSlot, 4> &getOutputSlots() const {
        return _outputSlots;
    }
    void bindInputNode(const std::pair<Edge *, IndexType> &upStreamOutputInfo);
    void bindOutputNode(IndexType slotId, Edge *downStream);
    void finishAsDownStream();
    bool finishInputWork(const Node *callNode);
    void finishOutputWork(IndexType slotId, const Node *callNode);
    void propagateSlotEof(bool isTimeout, bool forceEof, const Node *callNode);
    bool hasDownStream(IndexType slotId) const;
    bool hasUpStream() const;
    void flushData(Node *callNode);
    void flushAllData(Node *callNode);
    std::string getOutputName(IndexType index) const;
    bool overrideData(const std::vector<DataPtr> &datas);
    void setSummaryQuery(GraphId graphId,
                         const std::string &node,
                         const std::string &port,
                         const std::string &query,
                         CreatorManager *creatorManager);
    EdgeOutputInfo getUpStreamOutputInfo() const;
    Edge *getDownStream(IndexType slotId) const;
    bool startUpStreamSchedule(const Node *callNode, bool graphStart) const;
    void fillUpstreamState(KernelOutputStateDef *state) const;
    void fillDownstreamState(IndexType slotId, KernelInputStateDef *state) const;
    const EdgeMetricPtr &getMetric() const {
        return _metric;
    }
    bool needCollectMetric() const {
        return _needCollectMetric;
    }
private:
    bool checkDataType(const Data *data) const;
    ErrorCode doSetData(const Node *callNode,
                        const DataPtr &data,
                        Data *dataPointer,
                        bool eof);
    bool flushOverrideData();
    bool isOverride() const;
    void setNeedCollectMetric() {
        if (_needCollectMetric) {
            return;
        }
        _needCollectMetric = true;
        if (_upStream) {
            _upStream->setNeedCollectMetric();
        }
    }
    void clearDataPointer() {
        bool cleared = false;
        for (auto &slot : _outputSlots) {
            if (!cleared) {
                if (slot.dataPointer) {
                    delete slot.dataPointer;
                    cleared = true;
                }
            }
            slot.dataPointer = nullptr;
        }
    }
    void setSlotData(const DataPtr &data,
                     Data *dataPointer,
                     bool eof,
                     bool needClone,
                     OutputSlot &slot) {
        slot.data = data;
        slot.dataPointer = dataPointer;
        if (eof) {
            slot.eof = eof;
        }
        slot.needClone = needClone;
        slot.hasData = true;
    }
    bool readSlotData(IndexType slotId,
                      DataPtr &data,
                      Data *&dataPointer,
                      bool &eof) {
        auto &slot = _outputSlots[slotId];
        eof = eof || slot.eof;
        auto hasData = slot.hasData;
        if (!hasData) {
            return false;
        }
        data = slot.data;
        dataPointer = nullptr;
        if (slot.needClone && data) {
            auto cloned = data->clone();
            if (cloned) {
                data = std::move(cloned);
            }
        }
        return true;
    }
    bool clearSlotData(IndexType slotId,
                       DataPtr &data,
                       Data *&dataPointer,
                       bool &eof) {
        auto &slot = _outputSlots[slotId];
        eof = eof || slot.eof;
        auto hasData = slot.hasData;
        if (!hasData) {
            return false;
        }
        slot.hasData = false;
        data = std::move(slot.data);
        dataPointer = slot.dataPointer;
        if (slot.needClone) {
            if (dataPointer) {
                auto ptr = dataPointer;
                dataPointer = ptr->pointerClone();
                if (!dataPointer) {
                    NAVI_KERNEL_LOG(ERROR,
                                    "pointerClone return null, dataPointer "
                                    "[%p] type [%s]",
                                    ptr,
                                    typeid(*ptr).name());
                    return false;
                }
            }
        } else {
            slot.dataPointer = nullptr;
        }
        if (slot.needClone && data) {
            auto cloned = data->clone();
            if (cloned) {
                data = std::move(cloned);
            }
        }
        return true;
    }
    void collectSetEvent(bool success, const Data *data, bool eof, const Node *callNode) const;
    void collectGetEvent(
        bool success, IndexType slotId, EdgeGetType type, const Data *data, bool eof, const Node *callNode) const;
public:
    bool isSlotFinished(IndexType index) const {
        if (likely(!_edgeBitMapTree)) {
            return _slotReadyAtomic.isSlotFinished(index);
        } else {
            return _edgeBitMapTree->isSlotFinished(index);
        }
    }
public:
    bool isInputOk() const {
        if (unlikely(isOverride())) {
            return true;
        }
        if (likely(!_hasDownStream)) {
            NAVI_EDGE_LOG(SCHEDULE1, "no downstream");
            if (likely(!_edgeBitMapTree)) {
                if (!_slotReadyAtomic.isInputOk()) {
                    return false;
                }
            } else {
                if (!_edgeBitMapTree->isInputOk()) {
                    return false;
                }
            }
        } else {
            auto slotCount = _outputSlots.size();
            for (size_t i = 0; i < slotCount; i++) {
                auto &slot = _outputSlots[i];
                if (unlikely(nullptr != slot.downStream && slot.flushed)) {
                    NAVI_EDGE_LOG(SCHEDULE1, "slot [%lu] has downstream", i);
                    if (!slot.downStream->isInputOk()) {
                        return false;
                    }
                } else {
                    if (likely(!_edgeBitMapTree)) {
                        if (!_slotReadyAtomic.isInputOk(i)) {
                            return false;
                        }
                    } else {
                        if (!_edgeBitMapTree->isInputOk(i)) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
    bool isOutputOk(IndexType slotId) const {
        if (likely(!_edgeBitMapTree)) {
            return _slotReadyAtomic.isOutputOk(slotId);
        } else {
            return _edgeBitMapTree->isOutputOk(slotId);
        }
    }
    bool isEofOrFinish(IndexType slotId) const {
        if (unlikely(isOverride())) {
            return true;
        }
        if (likely(!_edgeBitMapTree)) {
            return _slotReadyAtomic.isEofOrFinish(slotId);
        } else {
            return _edgeBitMapTree->isEofOrFinish(slotId);
        }
    }
    bool isEofOrFinish() const {
        if (unlikely(isOverride())) {
            return true;
        }
        if (likely(!_edgeBitMapTree)) {
            return _slotReadyAtomic.isEofOrFinish();
        } else {
            return _edgeBitMapTree->isEofOrFinish();
        }
    }
    bool isEofForSetData() const {
        if (likely(!_edgeBitMapTree)) {
            return _slotReadyAtomic.isEof();
        } else {
            return _edgeBitMapTree->isEof();
        }
    }
    bool slotHasData(IndexType index) const {
        if (likely(!_edgeBitMapTree)) {
            return _slotReadyAtomic.slotHasData(index);
        } else {
            return _edgeBitMapTree->slotHasData(index);
        }
    }
    std::string getLogPrefix() const;
private:
    void setSlotHasData(bool eof) {
        if (likely(!_edgeBitMapTree)) {
            _slotReadyAtomic.setSlotHasData(eof);
        } else {
            _edgeBitMapTree->setSlotHasData(eof);
        }
    }
    bool allSlotEmpty() const {
        if (likely(!_edgeBitMapTree)) {
            return _slotReadyAtomic.allSlotEmpty();
        } else {
            return _edgeBitMapTree->allSlotEmpty();
        }
    }
    bool isAllSlotFinished() const {
        if (likely(!_edgeBitMapTree)) {
            return _slotReadyAtomic.isAllSlotFinished();
        } else {
            return _edgeBitMapTree->isAllSlotFinished();
        }
    }
private:
    void notifyInputWork(const Node *callNode);
    void notifyOutputWork(IndexType slotId, const Node *callNode);
    void setPendingFlush(IndexType slotId, const Node *callNode);
    void setSlotFinish(IndexType slotId, const Node *callNode);
    bool clearSlotHasData(IndexType slotId, const Node *callNode);
    bool scheduleInput(const Node *callNode);
    void notifyData(const DataPtr &data,
                    Data *dataPointer,
                    bool eof,
                    bool needClone,
                    const Node *callNode);
    void doFillUpstreamState(KernelOutputStateDef *state) const;
private:
    LocalSubGraph *_graph;
    EdgeInternal *_internal;
    NaviVector<OutputSlot, 4> _outputSlots;
    uint32_t _overrideIndex;
    IndexType _upStreamIndex;
    bool _needCollectMetric = false;
    bool _atomicClear = false;
    bool _hasDownStream = false;
    EdgeBitMapTree *_edgeBitMapTree;
    EdgeBitMap _slotReadyAtomic;
    Edge *_upStream;
    size_t _totalDataCount;
    std::vector<DataPtr> _overrideDatas;
    EdgeMetricPtr _metric;
private:
    static constexpr uint32_t INVALID_OVERRIDE_INDEX = -1;
};

class EdgeOutputInfo {
public:
    EdgeOutputInfo()
        : EdgeOutputInfo(nullptr, INVALID_INDEX)
    {
    }
    EdgeOutputInfo(EdgeInternal *edge,
                   IndexType slotId)
        : _edge(edge)
        , _slotId(slotId)
    {
    }
public:
    bool isValid() const {
        return _edge != nullptr && _slotId != INVALID_INDEX;
    }
    EdgeInternal *getEdge() const {
        return _edge;
    }
    IndexType getSlotId() const {
        return _slotId;
    }
    inline bool getData(LocalSubGraph *graph,
                        EdgeGetType type,
                        DataPtr &data,
                        Data *&dataPointer,
                        bool &eof,
                        const Node *callNode) const {
        assert(isValid());
        return graph->transformEdgeFromInternal(_edge)->getData(
            type, _slotId, data, dataPointer, eof, callNode);
    }
    std::string getDebugName(LocalSubGraph *graph) const {
        assert(isValid());
        return _edge->getDebugName(_slotId, graph);
    }
    const std::string &getInputNodeName(LocalSubGraph *graph) const {
        return SubGraphBase::getNodeName(graph, *_edge->getInputDef());
    }
    const std::string &getInputPortName() const {
        return _edge->getInputDef()->port_name();
    }
    const std::string &getOutputPortName() const {
        return _edge->getOutputDef(_slotId)->port_name();
    }
    inline void setInternlRequire(bool value) {
        _edge->_outputSlots[_slotId].require = value;
    }
    inline NodeInternal *getInputNode() const {
        return _edge->getInputNode();
    }
    inline void setEof(LocalSubGraph *graph,
                       bool isTimeout,
                       const Node *callNode) const {
        graph->transformEdgeFromInternal(_edge)->setOutputSlotEof(_slotId, isTimeout, false, false, callNode);
    }
    inline void setOptional(LocalSubGraph *graph, bool value) const {
        graph->transformEdgeFromInternal(_edge)->setOptional(_slotId, value);
    }
    inline bool isOutputOk(LocalSubGraph *graph) const {
        return graph->transformEdgeFromInternal(_edge)->isOutputOk(_slotId);
    }
    inline bool isEofOrFinish(LocalSubGraph *graph) const {
        return graph->transformEdgeFromInternal(_edge)->isEofOrFinish(_slotId);
    }
    inline bool slotHasData(LocalSubGraph *graph) const {
        return graph->transformEdgeFromInternal(_edge)->slotHasData(_slotId);
    }
    inline void finishOutputWork(LocalSubGraph *graph, const Node *callNode) const {
        return graph->transformEdgeFromInternal(_edge)->finishOutputWork(_slotId, callNode);
    }
    inline bool startUpStreamSchedule(LocalSubGraph *graph,
                                      const Node *callNode,
                                      bool graphStart) const {
        return graph->transformEdgeFromInternal(_edge)->startUpStreamSchedule(
            callNode, graphStart);
    }
    void fillState(LocalSubGraph *graph, KernelInputStateDef *state) const {
        state->set_port(getOutputPortName());
        state->set_input_node(getInputNodeName(graph));
        state->set_input_port(getInputPortName());
        state->set_slot_id(_slotId);
        graph->transformEdgeFromInternal(_edge)->fillDownstreamState(_slotId,
                                                                     state);
    }
    const google::protobuf::RepeatedField<google::protobuf::int32> *
    getSelectIndexVec() const {
        return _edge->getSelectIndexVec(_slotId);
    }
    NodeInternal *getOutputNode() const {
        return _edge->_outputSlots[_slotId].outputNode;
    }

private:
    EdgeInternal *_edge;
    IndexType _slotId;
};

}
