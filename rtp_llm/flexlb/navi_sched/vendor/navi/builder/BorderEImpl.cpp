#include "navi/builder/BorderEImpl.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/builder/PImpl.h"
#include "navi/builder/NImplBase.h"

namespace navi {

BorderEImpl::BorderEImpl()
    : _fromBorder(nullptr)
    , _toBorder(nullptr)
    , _bpFromBorder(nullptr)
    , _bpToBorder(nullptr)
{
}

BorderEImpl::~BorderEImpl() {
}

SingleE *BorderEImpl::addDownStream(P to, EImpl *realEdge) {
    if (!_fromBorder) {
        return addWithBorder(to);
    } else {
        return addWithoutBorder(to);
    }
}

SingleE *BorderEImpl::addWithBorder(P to) {
    const auto &fromNode = _from._impl->getNodeName();
    auto fromNodeId = _from._impl->getNodeId();
    const auto &toNode = to._impl->getNodeName();
    auto toNodeId = to._impl->getNodeId();
    auto toPort = getPortName(to);
    BorderId fromBorderId(IOT_OUTPUT, _fromGraphId, _toGraphId);
    BorderId toBorderId(IOT_INPUT, _toGraphId, _fromGraphId);
    _fromBorder = _builder->linkBorder(fromBorderId,
                                       fromNode,
                                       _fromPort,
                                       fromNodeId,
                                       toNode,
                                       toPort,
                                       toNodeId);
    _toBorder = _builder->linkBorder(
        toBorderId, fromNode, _fromPort, fromNodeId, toNode, toPort, toNodeId);
    if (!_fromBorder || !_toBorder) {
        return nullptr;
    }
    _fromBorder->set_peer_edge_id(_toBorder->edge_id());
    _toBorder->set_peer_edge_id(_fromBorder->edge_id());
    addSplit();
    return addMerge(to);
}

void BorderEImpl::addSplit() {
    auto nodeName = "split:" + autil::StringUtil::toString(_fromBorder->edge_id()) + ":" +
                    autil::StringUtil::toString(_fromBorder->peer_edge_id());
    _fromBorder->set_node(nodeName);
    auto fromNodeType = _from._impl->getNode()->getNodeType();
    auto fromTerminator = (NT_SCOPE_TERMINATOR == fromNodeType);
    auto nodeType = fromTerminator ? NT_SCOPE_TERMINATOR_SPLIT : NT_SPLIT;
    auto n = _builder->nodeInner(_fromGraphId, nodeName, nodeType);
    n.kernel(DEFAULT_SPLIT_KERNEL);
    _split = n._impl;
    _fromBorder->set_node_id(_split->getId());
    if (!fromTerminator) {
        _split->initScopeBorder(IOT_OUTPUT);
    }
    _from.to(n.in(PORT_KERNEL_INPUT_PORT));
}

SingleE *BorderEImpl::addMerge(P to) {
    auto nodeName = "merge:" + autil::StringUtil::toString(_toBorder->edge_id()) + ":" +
                    autil::StringUtil::toString(_toBorder->peer_edge_id());
    _toBorder->set_node(nodeName);
    auto n = _builder->nodeInner(_toGraphId, nodeName, NT_MERGE);
    n.kernel(DEFAULT_MERGE_KERNEL);
    _merge = n._impl;
    _toBorder->set_node_id(_merge->getId());
    _merge->initScopeBorder(IOT_INPUT);
    auto mergePort = n.out(PORT_KERNEL_OUTPUT_PORT);
    mergePort._impl->setRealPort(_from);
    auto e = mergePort._impl->to(mergePort._index, to, this);
    return e._singleE;
}

SingleE *BorderEImpl::addWithoutBorder(P to) {
    assert(_merge);
    auto from = N(_merge);
    auto mergePort = from.out(PORT_KERNEL_OUTPUT_PORT);
    auto e = mergePort._impl->to(mergePort._index, to, this);
    return e._singleE;
}

N BorderEImpl::split(const std::string &splitKernel) {
    assert(_split);
    auto n = N(_split);
    n.kernel(splitKernel);
    return n;
}

N BorderEImpl::merge(const std::string &mergeKernel) {
    assert(_merge);
    auto n = N(_merge);
    n.kernel(mergeKernel);
    return n;
}

void BorderEImpl::backPressure(int64_t queueLow, int64_t queueHigh) {
    queueHigh = std::max(queueHigh, queueLow);
    queueHigh = std::max(queueHigh, 1l);
    if (!_bpFromBorder && !_bpToBorder) {
        BorderId fromBorderId(IOT_OUTPUT, _toGraphId, _fromGraphId);
        BorderId toBorderId(IOT_INPUT, _fromGraphId, _toGraphId);
        const auto &originEdge = _fromBorder->edge();
        std::string fromNode = "navi.back_pressure." + originEdge.output().node_name();
        const auto &fromPort = originEdge.output().port_name();
        std::string toNode = "navi.back_pressure." + originEdge.input().node_name();
        const auto &toPort = originEdge.input().port_name();
        _bpFromBorder = _builder->linkBorder(fromBorderId, fromNode, fromPort, 0, toNode, toPort, 0);
        _bpToBorder = _builder->linkBorder(toBorderId, fromNode, fromPort, 0, toNode, toPort, 0);
        if (unlikely(!_bpFromBorder || !_bpToBorder)) {
            AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                               "set edge queue size failed, can't create back pressure border, queueLow: " +
                                   autil::StringUtil::toString(queueLow) +
                                   ", queueHigh: " + autil::StringUtil::toString(queueHigh) +
                                   ", fromGraphId: " + autil::StringUtil::toString(_fromGraphId) +
                                   ", toGraphId: " + autil::StringUtil::toString(_toGraphId));
            return;
        }
        _bpFromBorder->set_peer_edge_id(_bpToBorder->edge_id());
        _bpToBorder->set_peer_edge_id(_bpFromBorder->edge_id());
    }
    assert(_fromBorder);
    assert(_toBorder);
    {
        auto queueDef = _bpFromBorder->mutable_queue();
        auto queueInfoDef = queueDef->mutable_queue();
        queueInfoDef->set_queue_low(queueLow);
        queueInfoDef->set_queue_high(queueHigh);

        auto destBorderDef = queueDef->mutable_dest_border();
        destBorderDef->set_graph_id(_toGraphId);
        destBorderDef->set_peer(_fromGraphId);
        destBorderDef->set_io_type(IOT_INPUT);

        queueDef->set_dest_edge_id(_toBorder->edge_id());
    }
    {
        auto queueDef = _bpToBorder->mutable_queue();
        auto queueInfoDef = queueDef->mutable_queue();
        queueInfoDef->set_queue_low(queueLow);
        queueInfoDef->set_queue_high(queueHigh);

        auto destBorderDef = queueDef->mutable_dest_border();
        destBorderDef->set_graph_id(_fromGraphId);
        destBorderDef->set_peer(_toGraphId);
        destBorderDef->set_io_type(IOT_OUTPUT);

        queueDef->set_dest_edge_id(_fromBorder->edge_id());
    }
}

std::ostream &operator<<(std::ostream &os, const BorderEImpl &edge) {
    return os;
}

}

