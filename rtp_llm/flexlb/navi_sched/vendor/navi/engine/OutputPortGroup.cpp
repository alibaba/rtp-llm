#include "navi/engine/OutputPortGroup.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/Edge.h"
#include "navi/engine/PartInfo.h"
#include "navi/engine/LocalPortBase.h"
#include "navi/log/NaviLogger.h"

namespace navi {

void OutputPortGroupInternal::postInit() {
    if (_outputs.empty()) {
        return;
    }
    std::sort(_outputs.begin(), _outputs.end(), OutputPortGroupInternal::compare);
    bool hasNotConnected = false;
    IndexType arraySize = _outputs.size();
    for (IndexType index = 0; index < arraySize; index++) {
        auto actual = _outputs[index].first;
        assert(actual >= index);
        if (actual > index) {
            auto inputDef = _outputs[index].second->getInputDef();
            NAVI_KERNEL_LOG(DEBUG,
                            "group index skip, lack index [%d], "
                            "skip to [%d], node [%s], port [%s]",
                            index,
                            actual,
                            inputDef->node_name().c_str(),
                            inputDef->port_name().c_str());
            hasNotConnected = true;
            break;
        }
    }
    if (unlikely(hasNotConnected)) {
        _outputsMapPtr = std::make_shared<NaviUnorderedMap<IndexType, EdgeInternal *>>();
        for (const auto &output : _outputs) {
            _outputCount = std::max(_outputCount, IndexType(output.first + 1));
            _outputsMapPtr->emplace(output);
        }
    } else {
        _outputCount = arraySize;
    }
}

void OutputPortGroupInternal::addOutput(IndexType index, EdgeInternal *edge) {
    assert(edge != nullptr);
    _outputs.emplace_back(index, edge);
}

InlineReadyBitMap OutputPortGroupInternal::createReadyBitMap(const PartInfo &partInfo) {
    assert(0 == _outputs.size());
    auto readyMap = partInfo.createReadyBitMap();
    readyMap.setReady(true);
    readyMap.setOptional(false);
    readyMap.setFinish(false);
    NAVI_KERNEL_LOG(SCHEDULE2, "end post init outputs with part info, group [%p] readyMap [%s]",
                    this, autil::StringUtil::toString(readyMap).c_str());
    return readyMap;
}

size_t OutputPortGroupInternal::getOutputCount() const {
    return _outputCount;
}

bool OutputPortGroupInternal::isOk(LocalPortBase *port, LocalSubGraph *graph) const {
    if (unlikely(nullptr != port)) {
        return port->isOk();
    }
    for (const auto &pair : _outputs) {
        if (!graph->transformEdgeFromInternal(pair.second)->isInputOk()) {
            return false;
        }
    }
    return true;
}

bool OutputPortGroupInternal::isEofOrFinish(LocalPortBase *port,
                                            LocalSubGraph *graph) const {
    if (unlikely(nullptr != port)) {
        return port->isFinish();
    }
    for (const auto &pair : _outputs) {
        if (!graph->transformEdgeFromInternal(pair.second)->isEofOrFinish()) {
            return false;
        }
    }
    return true;
}

bool OutputPortGroupInternal::isEofOrFinish(LocalPortBase *port,
                                            LocalSubGraph *graph,
                                            size_t index) const
{
    if (unlikely(nullptr != port)) {
        return port->isFinish(index);
    }
    if (index < _outputs.size()) {
        return graph->transformEdgeFromInternal(_outputs[index].second)->isEofOrFinish();
    } else {
        return false;
    }
}

EdgeInternal *OutputPortGroupInternal::getEdge(size_t index) const {
    if (unlikely(_outputsMapPtr != nullptr)) {
        auto iter = _outputsMapPtr->find(index);
        if (_outputsMapPtr->end() == iter) {
            return nullptr;
        } else {
            return iter->second;
        }
    } else if (index < _outputs.size()) {
        return _outputs[index].second;
    } else {
        return nullptr;
    }
}

size_t OutputPortGroupInternal::getOutputDegree() const {
    size_t count = 0;
    for (const auto &pair : _outputs) {
        count += pair.second->getOutputDegree();
    }
    return count;
}

bool OutputPortGroupInternal::compare(
    const std::pair<IndexType, EdgeInternal *> &left,
    const std::pair<IndexType, EdgeInternal *> &right) {
    return left.first < right.first;
}

void OutputPortGroupInternal::forceEof(LocalSubGraph *graph,
                                       const Node *callNode) const {
    for (const auto &pair : _outputs) {
        graph->transformEdgeFromInternal(pair.second)->forceEof(callNode);
    }
}

bool OutputPortGroupInternal::finishInputWork(LocalSubGraph *graph, const Node *callNode) {
    for (const auto &pair : _outputs) {
        if (!graph->transformEdgeFromInternal(pair.second)->finishInputWork(callNode)) {
            return false;
        }
    }
    return true;
}

void OutputPortGroupInternal::fillState(
    LocalPortBase *port,
    LocalSubGraph *graph,
    KernelOutputGroupStateDef *state) const {
    state->set_ok(isOk(port, graph));
    state->set_eof(isEofOrFinish(port, graph));
    for (const auto &pair : _outputs) {
        graph->transformEdgeFromInternal(pair.second)
            ->fillUpstreamState(state->add_outputs());
    }
}
}
