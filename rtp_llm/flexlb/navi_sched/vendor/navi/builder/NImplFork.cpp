#include "navi/builder/NImplFork.h"
#include "navi/builder/PImpl.h"

namespace navi {

NImplFork::NImplFork() {
}

NImplFork::~NImplFork() {
    for (auto p : _ins) {
        delete p.second;
    }
    for (auto p : _outs) {
        delete p.second;
    }
}

P NImplFork::in(const std::string &port) {
    auto it = _ins.find(port);
    if (it != _ins.end()) {
        return P(it->second, INVALID_INDEX);
    }
    auto impl = new PImpl(this, port, true);
    _ins.emplace(port, impl);
    return P(impl, INVALID_INDEX);
}

P NImplFork::out(const std::string &port) {
    auto it = _outs.find(port);
    if (it != _outs.end()) {
        return P(it->second, INVALID_INDEX);
    }
    auto impl = new PImpl(this, port, false);
    _outs.emplace(port, impl);
    return P(impl, INVALID_INDEX);
}

std::vector<std::pair<P, E>> NImplFork::getInputUpstreams(const std::string &portName) const {
    auto it = _ins.find(portName);
    if (it == _ins.end()) {
        return {};
    }
    return it->second->getUpStreams();
}

}
