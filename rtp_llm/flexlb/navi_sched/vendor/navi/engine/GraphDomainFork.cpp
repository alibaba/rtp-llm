#include "navi/engine/GraphDomainFork.h"
#include "navi/engine/Edge.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/Node.h"
#include "navi/util/CommonUtil.h"

namespace navi {

GraphDomainFork::GraphDomainFork(Graph *graph, const BizPtr &biz, Node *node,
                                 NaviPartId partCount, NaviPartId partId,
                                 bool errorAsEof)
    : GraphDomain(graph, GDT_FORK)
    , _biz(biz)
    , _node(node)
    , _partCount(partCount)
    , _partId(partId)
    , _errorAsEof(errorAsEof)
{
    _logger.addPrefix("node [%s]", node->getName().c_str());
}

GraphDomainFork::~GraphDomainFork() {
}

// TODO init border info??
bool GraphDomainFork::postInit() {
    if (!initPortDataType()) {
        NAVI_LOG(ERROR, "init port data type failed");
        return false;
    }
    initPartCount();
    return true;
}

bool GraphDomainFork::run() {
    return true;
}

void GraphDomainFork::notifySend(NaviPartId partId) {
}

void GraphDomainFork::doFinish(ErrorCode ec) {
    if (!_errorAsEof) {
        return;
    }
    NAVI_LOG(DEBUG, "force eof");
    for (const auto &pair : _inputEdgeMap) {
        auto inputEdge = pair.second;
        auto slotInfo = inputEdge->getUpStreamOutputInfo();
        if (slotInfo.isValid()) {
            slotInfo.setEof(_node->getGraph(), false, nullptr);
        }
    }
    for (const auto &pair : _outputEdgeMap) {
        auto edge = pair.second.first;
        auto slotId = pair.second.second;
        auto downStream = edge->getDownStream(slotId);
        if (downStream) {
            downStream->forceEof(_node);
        }
    }
}

bool GraphDomainFork::initPortDataType() {
    auto localGraph = _node->getGraph();
    for (const auto &pair : _portMap) {
        auto port = pair.second;
        if (port->isBackPressurePort()) {
            continue;
        }
        auto ioType = CommonUtil::getReverseIoType(port->getIoType());
        if (!localGraph->bindPortDataType(_node, port, ioType)) {
            return false;
        }
    }
    return true;
}

void GraphDomainFork::initPartCount() {
    auto location = _subGraphDef->mutable_location();
    if (INVALID_NAVI_PART_COUNT == location->part_info().part_count()) {
        location->mutable_part_info()->set_part_count(_partCount);
        location->set_this_part_id(_partId);
    }
    NAVI_LOG(SCHEDULE1, "graph: %d, location: %s", _subGraphDef->graph_id(), location->DebugString().c_str());
}

bool GraphDomainFork::bindGraphEdgeList(Edge *edgeVec, int32_t edgeCount) {
    for (size_t i = 0; i < edgeCount; i++) {
        auto edge = &edgeVec[i];
        auto graph = edge->getGraph();
        auto inputDef = edge->getInputDef();
        if (NAVI_FORK_NODE == SubGraphBase::getNodeName(graph, *inputDef)) {
            _inputEdgeMap.emplace(inputDef->port_name(), edge);
        } else {
            const auto &outputSlots = edge->getOutputSlots();
            auto slotCount = outputSlots.size();
            for (IndexType slotId = 0; slotId < slotCount; slotId++) {
                const auto &slot = outputSlots[slotId];
                auto outputPortDef = slot.internal->outputDef;
                if (NAVI_FORK_NODE !=
                    SubGraphBase::getNodeName(graph, *outputPortDef)) {
                    continue;
                }
                const auto &portName = outputPortDef->port_name();
                auto ret = _outputEdgeMap.emplace(portName,
                                                  std::make_pair(edge, slotId));
                if (!ret.second) {
                    NAVI_LOG(ERROR,
                            "multiple fork output link to the same output port: %s",
                            portName.c_str());
                    return false;
                }
            }
        }
    }
    return true;
}

std::vector<Edge *> GraphDomainFork::getEdgesByPrefix(const std::string &prefix) const {
    std::vector<Edge *> retVec;
    for (const auto &pair : _inputEdgeMap) {
        const auto &name = pair.first;
        if (0 == name.find(prefix)) {
            retVec.push_back(pair.second);
        }
    }
    return retVec;
}

Edge *GraphDomainFork::getInputPeer(const std::string &portName) const {
    auto it = _inputEdgeMap.find(portName);
    if (_inputEdgeMap.end() != it) {
        return it->second;
    } else {
        return nullptr;
    }
}

std::pair<Edge *, IndexType>
GraphDomainFork::getOutputPeer(const std::string &portName) const {
    auto it = _outputEdgeMap.find(portName);
    if (_outputEdgeMap.end() != it) {
        return it->second;
    } else {
        return std::make_pair(nullptr, INVALID_INDEX);
    }
}

void GraphDomainFork::flushData() const {
    for (const auto &pair : _inputEdgeMap) {
        pair.second->flushData(_node);
    }
    for (const auto &pair : _outputEdgeMap) {
        pair.second.first->flushAllData(_node);
    }
}

bool GraphDomainFork::checkBind() const {
    bool ret = true;
    for (const auto &pair : _inputEdgeMap) {
        auto inputEdge = pair.second;
        if (!inputEdge->hasUpStream() && !inputEdge->isEofOrFinish()) {
            NAVI_LOG(
                ERROR,
                "bind input port failed, fork node [%s] kernel [%s] has no input port [%s], maybe need provider group index",
                _node->getName().c_str(),
                _node->getKernelName().c_str(),
                pair.first.c_str());
            ret = false;
        }
    }
    for (const auto &pair : _outputEdgeMap) {
        auto edge = pair.second.first;
        auto slotId = pair.second.second;
        if (!edge->hasDownStream(slotId) && !edge->isSlotFinished(slotId)) {
            NAVI_LOG(ERROR,
                     "bind output port failed, fork node [%s] kernel [%s] has no output port [%s], maybe need provider "
                     "group index",
                     _node->getName().c_str(),
                     _node->getKernelName().c_str(),
                     pair.first.c_str());
            ret = false;
        }
    }
    return ret;
}

BizPtr GraphDomainFork::getBiz() const {
    return _biz;
}

}
