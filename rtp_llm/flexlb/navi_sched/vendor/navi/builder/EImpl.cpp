#include "navi/builder/EImpl.h"
#include "navi/builder/PImpl.h"
#include "navi/builder/InterEImpl.h"
#include "navi/builder/BorderEImpl.h"
#include "navi/builder/SingleE.h"

namespace navi {

EImpl::EImpl()
    : _builder(nullptr)
    , _fromGraphId(INVALID_GRAPH_ID)
    , _from(nullptr, INVALID_INDEX)
    , _toGraphId(INVALID_GRAPH_ID)
    , _split(nullptr)
    , _merge(nullptr)
{
}

EImpl::~EImpl() {
}

bool EImpl::init(GraphBuilder *builder, GraphId peerGraphId, P from) {
    _builder = builder;
    _fromGraphId = from._impl->graphId();
    _from = from;
    _fromPort = getPortName(from);
    _toGraphId = peerGraphId;
    return true;
}

EImpl *EImpl::createEdge(GraphBuilder *builder, GraphId peerGraphId, P from) {
    std::unique_ptr<EImpl> e;
    if (from._impl->graphId() == peerGraphId) {
        e.reset(new InterEImpl());
    } else {
        e.reset(new BorderEImpl());
    }
    if (!e->init(builder, peerGraphId, from)) {
        return nullptr;
    }
    return e.release();
}

std::string EImpl::getPortName(P p) {
    auto index = p._index;
    const auto &port = p._impl->getPort();
    if (INVALID_INDEX == index) {
        return port;
    }
    return port + GROUP_PORT_SEPERATOR + autil::StringUtil::toString(index);
}

}

