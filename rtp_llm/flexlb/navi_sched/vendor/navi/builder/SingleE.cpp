#include "navi/builder/SingleE.h"
#include "navi/builder/GraphBuilder.h"

namespace navi {

SingleE::SingleE(EdgeDef *edge, GraphId graphId, int32_t index)
    : _edge(edge)
    , _param(nullptr)
    , _graphId(graphId)
    , _index(index)
{
}

SingleE::~SingleE() {
}

void SingleE::require(bool require, GraphBuilder *builder) {
    if (_edge) {
        _edge->set_require(require);
        if (require) {
            if (!_param) {
                _param = builder->getEdgeParam(_graphId, _index);
            }
        }
        if (_param) {
            _param->set_require(require);
        }
    }
}

bool SingleE::isRequire() const {
    if (_edge) {
        return _edge->require();
    } else {
        return false;
    }
}

void SingleE::timeoutMs(int64_t timeoutMs, GraphBuilder *builder) {
    if (_edge) {
        _edge->set_timeout_ms(timeoutMs);
        if (timeoutMs > 0) {
            if (!_param) {
                _param = builder->getEdgeParam(_graphId, _index);
            }
        }
        if (_param) {
            _param->set_timeout_ms(timeoutMs);
        }
    }
}

void SingleE::addSelectIndex(const std::set<int32_t> &selectIndexSet) {
    if (!_edge) {
        return;
    }
    for (auto index : selectIndexSet) {
        _edge->add_select_index(index);
    }
}

std::ostream &operator<<(std::ostream &os, const SingleE &e) {
    if (!e._edge) {
        return os;
    }
    os << "edge "
       << e._edge->input().node_name() << ":" << e._edge->input().port_name()
       << " -> "
       << e._edge->output().node_name() << ":" << e._edge->output().port_name();
    return os;
}

}

