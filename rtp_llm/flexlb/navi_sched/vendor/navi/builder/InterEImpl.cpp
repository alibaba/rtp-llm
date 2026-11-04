#include "navi/builder/InterEImpl.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/builder/PImpl.h"
#include "navi/builder/SingleE.h"
#include "navi/builder/NImplBase.h"

namespace navi {

InterEImpl::InterEImpl() {
}

InterEImpl::~InterEImpl() {
    for (const auto &pair : _edgeMap) {
        delete pair.second;
    }
}

SingleE *InterEImpl::addDownStream(P to, EImpl *realEdge) {
    fillScopeBorder(to);
    auto key = getEdgeKey(to);
    auto it = _edgeMap.find(key);
    if (_edgeMap.end() != it) {
        to._impl->setConnected(to._index, _from, E());
        return it->second;
    }
    const auto &fromNode = _from._impl->getNodeName();
    auto fromNodeId = _from._impl->getNodeId();
    const auto &toNode = to._impl->getNodeName();
    auto toNodeId = to._impl->getNodeId();
    auto toPort = getPortName(to);
    int32_t edgeIndex = -1;
    auto edge = _builder->linkInternal(_fromGraphId,
                                       fromNode,
                                       _fromPort,
                                       fromNodeId,
                                       toNode,
                                       toPort,
                                       toNodeId,
                                       edgeIndex);
    auto singleE = new SingleE(edge, _fromGraphId, edgeIndex);
    _edgeMap.emplace(key, singleE);
    to._impl->setConnected(to._index, _from, E(realEdge ? realEdge : this, singleE));
    return singleE;
}

N InterEImpl::split(const std::string &splitKernel) {
    if (_split) {
        return N(_split);
    }
    auto n = _builder->nodeInner(_fromGraphId, "", NT_FAKE_SPLIT);
    _split = n._impl;
    return n;
}

N InterEImpl::merge(const std::string &mergeKernel) {
    if (_merge) {
        return N(_merge);
    }
    auto n = _builder->nodeInner(_fromGraphId, "", NT_FAKE_MERGE);
    _merge = n._impl;
    return n;
}

void InterEImpl::fillScopeBorder(P to) const {
    auto fromNode = _from._impl->getNode();
    auto toNode = to._impl->getNode();
    if (fromNode->getScope() != toNode->getScope()) {
        fromNode->initScopeBorder(IOT_OUTPUT);
        toNode->initScopeBorder(IOT_INPUT);
    }
}

}

