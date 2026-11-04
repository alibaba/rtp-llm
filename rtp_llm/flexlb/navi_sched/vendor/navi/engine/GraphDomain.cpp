#include "navi/engine/PartInfoDefOverride.h"
#include "navi/engine/GraphDomain.h"
#include "navi/engine/Graph.h"
#include "navi/engine/Port.h"
#include "navi/util/ReadyBitMap.h"
#include "navi/util/CommonUtil.h"

namespace navi {

std::ostream& operator<<(std::ostream& os,
                         const GraphDomain &domain)
{
    os << "(" << &domain << ","
       << GraphDomain::getGraphDomainType(domain.getType())
       << ",sub_graphs{";
    autil::StringUtil::toStream(os, domain._graphId);
    return os<< "})";
}

static const char *translateGraphDomainHoldReason(GraphDomainHoldReason reason) {
    switch (reason) {
        case GDHR_DOMAIN_HOLDER:
            return "GDHR_DOMAIN_HOLDER";
        case GDHR_INIT:
            return "GDHR_INIT";
        case GDHR_NOTIFY_FINISH_ASYNC:
            return "GDHR_NOTIFY_FINISH_ASYNC";
        default:
            return "GDHR_UNKNOWN";
    }
}

GraphDomain::GraphDomain(Graph *graph, GraphDomainType type)
    : _graph(graph)
    , _param(_graph->getGraphParam())
    , _graphResult(_param->graphResult)
    , _type(type)
    , _subGraphDef(nullptr)
    , _graphId(INVALID_GRAPH_ID)
    , _counterInfo(nullptr)
    , _terminated(false)
    , _acquireCount(0)
{
    _logger.object = this;
    _logger.logger = _param->logger;
    if (_param->runParams.needLogPrefix()) {
        _logger.addPrefix("domain %s", getGraphDomainType(type));
    }
    acquire(GDHR_INIT);
}

GraphDomain::~GraphDomain() {
    NAVI_LOG(SCHEDULE1, "destruct");
}

void GraphDomain::setSubGraph(SubGraphDef *subGraphDef) {
    _subGraphDef = subGraphDef;
    _graphId = _subGraphDef->graph_id();
}

void GraphDomain::addSubBorder(const SubGraphBorderPtr &border) {
    const auto &borderId = border->getBorderId();
    switch (borderId.ioType) {
    case IOT_INPUT:
        border->setBorderDomainIndex(_inputBorderMap.size());
        _inputBorderMap.emplace(borderId, border);
        if (!border->isLocalBorder()) {
            _remoteInputBorderMap.emplace(borderId, border);
        }
        break;
    case IOT_OUTPUT:
        border->setBorderDomainIndex(_outputBorderMap.size());
        _outputBorderMap.emplace(borderId, border);
        if (!border->isLocalBorder()) {
            _remoteOutputBorderMap.emplace(borderId, border);
        }
        break;
    default:
        assert(false);
    }
    collectPort(border);
}

void GraphDomain::collectPort(const SubGraphBorderPtr &border) {
    const auto &portVec = border->getPortVec();
    for (const auto &port : portVec) {
        _portMap.emplace(port->getPortId(), port);
    }
}

Port *GraphDomain::getPort(PortId portId) const {
    auto it = _portMap.find(portId);
    if (_portMap.end() != it) {
        return it->second;
    } else {
        return nullptr;
    }
}

const NaviUnorderedMap<PortId, Port *> &GraphDomain::getPortMap() const {
    return _portMap;
}

bool GraphDomain::isLocalBorder(GraphId peer) const {
    switch (_type) {
    case GDT_USER:
    case GDT_LOCAL:
    case GDT_FORK:
        return true;
    case GDT_CLIENT:
    case GDT_SERVER:
        return _graphId == peer;
    default:
        assert(false);
        return true;
    }
}

GraphDomainType GraphDomain::getType() const {
    return _type;
}

bool GraphDomain::preInit() {
    return true;
}

bool GraphDomain::postInit() {
    auto location = _subGraphDef->mutable_location();
    if (INVALID_NAVI_PART_COUNT == location->part_info().part_count()) {
        location->mutable_part_info()->set_part_count(1);
    }
    NAVI_LOG(SCHEDULE1, "graph: %d, location: %s", _graphId, location->DebugString().c_str());
    NAVI_LOG(SCHEDULE2, "sub graph def [%s]", _subGraphDef->DebugString().c_str());
    if (!initPeerInfo()) {
        NAVI_LOG(ERROR, "init peer info failed.");
        return false;
    }
    return true;
}

bool GraphDomain::initPeerInfo() {
    auto borderCount = _subGraphDef->borders_size();
    for (int32_t i = 0; i < borderCount; i++) {
        auto border = _subGraphDef->mutable_borders(i);
        auto peerBorder = getPeerBorder(border);
        if (!peerBorder) {
            NAVI_LOG(ERROR, "get peer border failed, border [%s]",
                     border->ShortDebugString().c_str());
            notifyFinish(EC_UNKNOWN, true);
            return false;
        }
        auto peerInfo = border->mutable_peer();
        auto overridePartInfo = PartInfoDefOverride::replacePartInfo(peerInfo);
        if (!overridePartInfo) {
            NAVI_LOG(ERROR, "replace part info failed");
            return false;
        }
        auto partInfo = overridePartInfo->setTlsValue();
        if (!partInfo) {
            NAVI_LOG(ERROR, "set tls value failed");
            return false;
        }
        peerBorder->getPartInfo().fillDef(*partInfo);
        // for test
        overridePartInfo->initCopy(*partInfo);
    }
    return true;
}

SubGraphBorder *GraphDomain::getPeerBorder(const BorderDef *borderDef) const {
    if (borderDef->edges_size() <= 0) {
        NAVI_LOG(ERROR, "border has no edge, border def: %s",
                 borderDef->DebugString().c_str());
        return nullptr;
    }
    const auto &firstPortId = borderDef->edges(0).edge_id();
    auto firstPort = getPort(firstPortId);
    if (!firstPort) {
        NAVI_LOG(ERROR, "port not exist, portId: %d, border def: %s",
                 firstPortId, borderDef->DebugString().c_str());
        return nullptr;
    }
    return firstPort->getPeerBorder();
}

void GraphDomain::notifySend(NaviPartId partId) {
}

void GraphDomain::incMessageCount(NaviPartId partId) {
}

void GraphDomain::decMessageCount(NaviPartId partId) {
}

const char *GraphDomain::getGraphDomainType(GraphDomainType type)
{
#define CASE(t) \
    case t:     \
        return #t

    switch (type) {
        CASE(GDT_USER);
        CASE(GDT_LOCAL);
        CASE(GDT_CLIENT);
        CASE(GDT_SERVER);
        CASE(GDT_FORK);
    default:
        return "UNKNOWN";
    }

#undef CASE
}

void GraphDomain::doFinish(ErrorCode ec) {
}

bool GraphDomain::finished() const {
    return _terminated.load(std::memory_order_relaxed);
}

void GraphDomain::notifyFinish(ErrorCode ec, bool notifyGraph) {
    NaviErrorPtr error;
    if (notifyGraph) {
        error = _graphResult->makeError(_logger.logger, ec);
    }
    notifyFinish(error, notifyGraph);
}

void GraphDomain::notifyFinish(const NaviErrorPtr &error, bool notifyGraph) {
    auto ec = error ? error->ec : EC_NONE;
    NAVI_LOG(SCHEDULE2, "try terminate, ec: %s, notifyGraph: %d, finished [%d]",
             CommonUtil::getErrorString(ec), notifyGraph, finished());
    bool expect = false;
    if (!_terminated.compare_exchange_weak(expect, true)) {
        return;
    }
    NAVI_LOG(SCHEDULE1, "terminate, ec: %s, notifyGraph: %d",
             CommonUtil::getErrorString(ec), notifyGraph);
    if (notifyGraph && error) {
        _param->graphResult->updateError(error);
    }
    doFinish(ec);
    finishGraph(error, notifyGraph);
}

void GraphDomain::finishGraph(const NaviErrorPtr &error, bool notifyGraph) {
    if (notifyGraph) {
        _graph->notifyFinish(error);
    }
    auto worker = release(GDHR_INIT);
    if (worker) {
        worker->decItemCount();
    }
}

GraphParam *GraphDomain::getGraphParam() const {
    return _param;
}

bool GraphDomain::acquire(GraphDomainHoldReason reason) {
    autil::ScopedLock lock(_acquireLock);
    NAVI_LOG(SCHEDULE2, "acquire from [%lu] reason [%s]",
             _acquireCount, translateGraphDomainHoldReason(reason));
    assert(_acquireCount >= 0);
    if (_param) {
        if (0 == _acquireCount) {
            _param->worker->incItemCount();
        }
        _acquireCount++;
        NAVI_LOG(SCHEDULE2, "acquire ok, count: %ld", _acquireCount);
        return true;
    } else {
        NAVI_LOG(SCHEDULE2, "acquire failed");
        return false;
    }
}

NaviWorkerBase *GraphDomain::release(GraphDomainHoldReason reason) {
    NaviWorkerBase *worker = nullptr;
    {
        autil::ScopedLock lock(_acquireLock);
        NAVI_LOG(SCHEDULE2, "release from [%lu] reason [%s]",
             _acquireCount, translateGraphDomainHoldReason(reason));
        assert(_acquireCount > 0);
        assert(_param);
        _acquireCount--;
        if (0 == _acquireCount) {
            worker = _param->worker;
            _param = nullptr;
            NAVI_LOG(SCHEDULE2, "released");
        } else {
            NAVI_LOG(SCHEDULE2, "release, acquire count: %ld", _acquireCount);
        }
    }
    return worker;
}

}
