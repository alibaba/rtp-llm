#include "navi/engine/InputPortGroup.h"
#include "navi/engine/PartInfo.h"
#include "navi/engine/LocalPortBase.h"

namespace navi {

std::ostream &operator<<(std::ostream &os,
                         const InputPortGroupInternal &inputGroup) {
    os << (void *)&inputGroup << "," << inputGroup.getInputCount();
    return os;
}

extern std::ostream &operator<<(std::ostream &os, const NaviVector<InputPortGroupInternal *, 1> &x) {
    for (auto it = x.begin(); it != x.end(); ++it) {
        if (it != x.begin())
            os << " ";
        autil::StringUtil::toStream(os, *it);
    }
    return os;
}

InputPortGroupInternal::InputPortGroupInternal(InputTypeDef channelType,
                                               InputTypeDef groupType)
    : _type(channelType)
    , _autoFillIndex(true)
    , _validPortGroup(false)
    , _groupType(groupType)
    , _inputIndexStart(INVALID_INDEX) {}

bool InputPortGroupInternal::postInit() {
    if (_inputs.empty()) {
        return true;
    }
    IndexType arraySize = _inputs.size();
    if (!_autoFillIndex) {
        std::sort(_inputs.begin(), _inputs.end(), InputPortGroupInternal::compare);
        for (IndexType index = 0; index < arraySize; index++) {
            auto actual = _inputs[index].first;
            if (actual < index) {
                NAVI_KERNEL_LOG(ERROR,
                        "multiple connection to group index[%d], "
                        "current edge [%s], previous edge [%s]",
                        actual,
                        _inputs[index].second.getDebugName(nullptr).c_str(),
                        _inputs[actual].second.getDebugName(nullptr).c_str());
                return false;
            } else if (actual > index) {
                NAVI_KERNEL_LOG(ERROR,
                        "group index skip,  lack index [%d], skip to "
                        "[%d], edge [%s]",
                        index, actual,
                        _inputs[index].second.getDebugName(nullptr).c_str());
                return false;
            }
        }
    }
    return true;
}

IndexType InputPortGroupInternal::addInput(IndexType index, EdgeOutputInfo &info) {
    NAVI_KERNEL_LOG(SCHEDULE2,
                    "add input with edge[%s],index[%d] group[%s]",
                    autil::StringUtil::toString(*info.getEdge()).c_str(),
                    index,
                    autil::StringUtil::toString(*this).c_str());
    IndexType size = _inputs.size();
    _inputs.emplace_back(index, info);
    if (IT_OPTIONAL & _type) {
        info.setInternlRequire(false);
    }
    if (_autoFillIndex && 0 == index) {
        return size;
    } else {
        _autoFillIndex = false;
        return index;
    }
}

bool InputPortGroupInternal::compare(
    const std::pair<IndexType, EdgeOutputInfo> &left,
    const std::pair<IndexType, EdgeOutputInfo> &right) {
    return left.first < right.first;
}

InlineReadyBitMap InputPortGroupInternal::createReadyBitMap(const PartInfo &partInfo) {
    assert(0 == _inputs.size());
    auto readyMap = partInfo.createReadyBitMap();
    readyMap.setReady(false);
    readyMap.setOptional(IT_OPTIONAL & _type);
    readyMap.setFinish(false);
    NAVI_KERNEL_LOG(
        SCHEDULE2, "end post init inputs with part info, group [%s]", autil::StringUtil::toString(*this).c_str());
    return readyMap;
}

void InputPortGroupInternal::setInputIndexStart(IndexType start) {
    NAVI_KERNEL_LOG(
        SCHEDULE2, "set input index start[%d], group [%s]", start, autil::StringUtil::toString(*this).c_str());
    _inputIndexStart = start;
}

IndexType InputPortGroupInternal::getInputIndexStart() const {
    return _inputIndexStart;
}

size_t InputPortGroupInternal::getInputCount() const {
    return _inputs.size();
}

void InputPortGroupInternal::setEof(LocalSubGraph *graph,
                                    const Node *callNode) const {
    for (const auto &pair : _inputs) {
        pair.second.setEof(graph, false, callNode);
    }
}

void InputPortGroupInternal::setEof(LocalSubGraph *graph,
                                    IndexType index,
                                    const Node *callNode) const {
    _inputs[index].second.setEof(graph, false, callNode);
}

EdgeOutputInfo InputPortGroupInternal::getEdgeOutputInfo(IndexType index) const {
    if (index < _inputs.size()) {
        return _inputs[index].second;
    } else {
        return {};
    }
}

void InputPortGroupInternal::startUpStreamSchedule(LocalSubGraph *graph,
                                                   const Node *callNode,
                                                   bool graphStart) const {
    for (const auto &pair : _inputs) {
        pair.second.startUpStreamSchedule(graph, callNode, graphStart);
    }
}

bool InputPortGroupInternal::startUpStreamSchedule(LocalSubGraph *graph,
                                                   IndexType index,
                                                   const Node *callNode,
                                                   bool graphStart) const {
    return _inputs[index].second.startUpStreamSchedule(
        graph, callNode, graphStart);
}

void InputPortGroupInternal::finishOutputWork(LocalSubGraph *graph, const Node *callNode) {
    for (const auto &pair : _inputs) {
        pair.second.finishOutputWork(graph, callNode);
    }
}

void InputPortGroupInternal::fillState(LocalPortBase *port,
                                       LocalSubGraph *graph,
                                       KernelInputGroupStateDef *state) const {
    state->set_ok(isOk(port, graph));
    state->set_eof(isEofOrFinish(port, graph));
    for (const auto &pair : _inputs) {
        pair.second.fillState(graph, state->add_inputs());
    }
}

bool InputPortGroupInternal::isOk(LocalPortBase *port, LocalSubGraph *graph) const {
    if (unlikely(nullptr != port)) {
        return port->isOk();
    }
    for (const auto &pair : _inputs) {
        if (!pair.second.isOutputOk(graph)) {
            return false;
        }
    }
    return true;
}

bool InputPortGroupInternal::isEofOrFinish(LocalPortBase *port,
                                           LocalSubGraph *graph,
                                           IndexType index) const {
    if (unlikely(nullptr != port)) {
        return port->isFinish(index);
    }
    if (index < _inputs.size()) {
        return _inputs[index].second.isEofOrFinish(graph);
    } else {
        return false;
    }
}

bool InputPortGroupInternal::isEofOrFinish(LocalPortBase *port, LocalSubGraph *graph) const {
    if (unlikely(nullptr != port)) {
        return port->isFinish();
    }
    for (const auto &pair : _inputs) {
        if (!pair.second.isEofOrFinish(graph)) {
            return false;
        }
    }
    return true;
}

bool InputPortGroupInternal::hasReady(LocalPortBase *port, LocalSubGraph *graph) const {
    if (unlikely(nullptr != port)) {
        return port->hasReady();
    }
    for (const auto &pair : _inputs) {
        if (pair.second.slotHasData(graph)) {
            return true;
        }
    }
    return false;
}

}
