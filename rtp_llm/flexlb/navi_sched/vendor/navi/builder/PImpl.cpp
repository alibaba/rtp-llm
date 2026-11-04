#include "navi/builder/PImpl.h"
#include "navi/builder/EImpl.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/builder/GraphDesc.h"
#include "navi/builder/NImpl.h"

namespace navi {

PImpl::PImpl(NImplBase *node, const std::string &port, bool input)
    : _node(node)
    , _nodeName(node->name())
    , _port(port)
    , _input(input)
    , _group(false)
    , _currentIndex(0)
    , _realPort(nullptr, INVALID_INDEX)
{
}

PImpl::~PImpl() {
    for (const auto &pair : _edgeMap) {
        delete pair.second;
    }
}

GraphId PImpl::graphId() const {
    return _node->graphId();
}

NImplBase *PImpl::getNode() const {
    return _node;
}

const std::string &PImpl::getNodeName() const {
    return _node->name();
}

int32_t PImpl::getNodeId() const {
    return _node->getId();
}

const std::string &PImpl::getPort() const {
    return _port;
}

E PImpl::from(IndexType thisIndex, P peer) {
    return peer._impl->to(peer._index, P(this, thisIndex), nullptr);
}

E PImpl::to(IndexType thisIndex, P peer, EImpl *realEdge) {
    auto peerImpl = peer._impl;
    if (_input) {
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                           "error direction, can not call \'to\' on input port, use \'from\' instead, port info: " +
                               autil::StringUtil::toString(P(this, thisIndex)));
    }
    auto peerGraphId = peerImpl->graphId();
    auto key = getEdgeKey(thisIndex, peerGraphId);
    EImpl *e = nullptr;
    auto it = _edgeMap.find(key);
    if (_edgeMap.end() != it) {
        e = it->second;
    } else {
        e = EImpl::createEdge(_node->builder(), peerGraphId, P(this, thisIndex));
        if (!e) {
            AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                               "can't create edge, from: " + autil::StringUtil::toString(P(this, thisIndex)) +
                                   ", to: " + autil::StringUtil::toString(peer));
        }
        _edgeMap.emplace(key, e);
    }
    auto singleE = e->addDownStream(peer, realEdge);
    if (!singleE) {
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                           "can't add down stream edge, from: " + autil::StringUtil::toString(P(this, thisIndex)) +
                               ", to: " + autil::StringUtil::toString(peer));
    }
    E retE = realEdge ? E(realEdge, singleE) : E(e, singleE);
    auto backPressureQueue = _node->builder()->getBackPressureQueueDef();
    if (backPressureQueue) {
        retE.backPressure(backPressureQueue->queue_low(), backPressureQueue->queue_high());
    }
    return retE;
}

E PImpl::fromForkNodeInput(IndexType thisIndex, const std::string &input, IndexType index) {
    auto n = _node->builder()->getParentNode();
    auto other = n.out(input);
    P p(this, thisIndex);
    if (INVALID_INDEX != index) {
        return p.from(other.asGroup().index(index));
    } else {
        return p.from(other);
    }
}

E PImpl::toForkNodeOutput(IndexType thisIndex, const std::string &output, IndexType index) {
    auto n = _node->builder()->getParentNode();
    auto other = n.in(output);
    P p(this, thisIndex);
    if (INVALID_INDEX != index) {
        return other.asGroup().index(index).from(p);
    } else {
        return other.from(p);
    }
}

E PImpl::asGraphOutput(IndexType thisIndex, const std::string &name) {
    auto n = _node->builder()->getOutputNode(name);
    return n.in("").from(P(this, thisIndex));
}

void PImpl::setConnected(IndexType inputIndex, P from, const E &e) {
    if (!_group && (inputIndex != INVALID_INDEX)) {
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                           "invalid port index, port is not in group mode" +
                               autil::StringUtil::toString(P(this, inputIndex)));
    }
    if (INVALID_INDEX == inputIndex) {
        if (_connectedInputSet.empty()) {
            if (e.isValid()) {
                _defaultInputs.emplace_back(std::make_pair(from, e));
            }
        } else {
            AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                               "error, can't use both index and auto index: " +
                                   autil::StringUtil::toString(P(this, inputIndex)));
        }
    } else {
        auto it = _connectedInputSet.find(inputIndex);
        if (_connectedInputSet.end() != it) {
            AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                               "error, double connection to: " + autil::StringUtil::toString(P(this, inputIndex)) +
                                   ", first: " + autil::StringUtil::toString(it->second.first) +
                                   ", second: " + autil::StringUtil::toString(from));
        }
        _connectedInputSet.emplace(inputIndex, std::make_pair(from, e));
    }
}

void PImpl::setRealPort(P real) {
    _realPort = real;
}

P PImpl::getRealPort() const {
    return _realPort;
}

P PImpl::asGroup() {
    if (hasConnected() && !_group) {
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                           "can't switch to group mode, port already connected, " +
                               autil::StringUtil::toString(P(this, INVALID_INDEX)));
    }
    _group = true;
    return P(this, 0);
}

P PImpl::next(IndexType index) {
    if (!_group) {
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                           "not group mode port, call asGroup first, " +
                               autil::StringUtil::toString(P(this, INVALID_INDEX)));
    }
    return P(this, index + 1);
}

P PImpl::index(IndexType index) {
    P p(this, index);
    if (INVALID_INDEX == index) {
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                           "invalid group index: " + autil::StringUtil::toString(index) + ", " +
                               autil::StringUtil::toString(p));
    }
    if (!_group) {
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                           "not group mode port, call asGroup first, " + autil::StringUtil::toString(p));
    }
    return p;
}

P PImpl::autoNext() {
    if (!_group) {
        asGroup();
    }
    return P(this, _currentIndex++);
}

std::vector<std::pair<P, E>> PImpl::getUpStreams() {
    if (!_input) {
        AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                           "can't not collect upstream for output port, " +
                               autil::StringUtil::toString(P(this, INVALID_INDEX)));
    }
    std::vector<std::pair<P, E>> peerPorts;
    for (const auto &pair : _defaultInputs) {
        auto p = pair.first;
        auto realPort = p._impl->getRealPort();
        if (realPort.isValid()) {
            peerPorts.push_back(std::make_pair(realPort, pair.second));
        } else {
            peerPorts.push_back(pair);
        }
    }
    std::map<IndexType, std::pair<P, E>> sortedInput(_connectedInputSet.begin(), _connectedInputSet.end());
    for (const auto &pair : sortedInput) {
        auto p = pair.second.first;
        auto realPort = p._impl->getRealPort();
        if (realPort.isValid()) {
            peerPorts.push_back(std::make_pair(realPort, pair.second.second));
        } else {
            peerPorts.push_back(pair.second);
        }
    }
    return peerPorts;
}

bool PImpl::hasConnected() const {
    return (!_edgeMap.empty()) || (!_connectedInputSet.empty());
}

bool PImpl::isGroup() const {
    return _group;
}

} // namespace navi
