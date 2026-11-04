#include "navi/engine/LocalPortBase.h"
#include "navi/engine/Node.h"
#include "navi/util/ReadyBitMap.h"

namespace navi {

LocalPortBase::LocalPortBase(const NodePortDef &nodePort,
                             const NodePortDef &peerPort,
                             IoType ioType,
                             SubGraphBorder *border)
    : Port(nodePort, peerPort, ioType, PST_DATA, border)
    , _node(nullptr)
{
}

LocalPortBase::~LocalPortBase() {
    InlineReadyBitMap::freeReadyBitMap(_nodeReadyMap);
}

void LocalPortBase::setNode(Node *node, const InlineReadyBitMap &readyBitMap) {
    _node = node;
    _nodeReadyMap = readyBitMap;
    NAVI_LOG(SCHEDULE2, "set node [%p] nodeReadyMap [%s]",
             node, autil::StringUtil::toString(_nodeReadyMap).c_str());
}

void LocalPortBase::scheduleNode() const {
    if (_node) {
        _node->schedule(_node);
    }
}

}
