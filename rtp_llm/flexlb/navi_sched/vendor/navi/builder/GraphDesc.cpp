#include "navi/builder/GraphDesc.h"
#include "navi/builder/NImpl.h"
#include "navi/builder/PImpl.h"
#include "navi/builder/EImpl.h"
#include "navi/builder/SingleE.h"

namespace navi {

N::N()
    : _impl(nullptr)
{
}

N::N(NImplBase *impl)
    : _impl(impl)
{
}

N::~N() {
}

N &N::kernel(const std::string &name) {
    _impl->kernel(name);
    return *this;
}

N &N::device(const std::string &device) {
    _impl->device(device);
    return *this;
}

N &N::jsonAttrs(const std::string &attrs) {
    _impl->jsonAttrs(attrs);
    return *this;
}

N &N::jsonAttrsFromMap(const std::map<std::string, std::string> &attrs) {
    _impl->jsonAttrsFromMap(attrs);
    return *this;
}

N &N::jsonAttrsFromJsonMap(const autil::legacy::json::JsonMap &attrs) {
    _impl->jsonAttrsFromJsonMap(attrs);
    return *this;
}

N &N::binaryAttrsFromMap(const std::map<std::string, std::string> &binaryAttrs) {
    _impl->binaryAttrsFromMap(binaryAttrs);
    return *this;
}

N &N::attr(const std::string &key, const std::string &value) {
    _impl->attr(key, value);
    return *this;
}

N &N::attrFromCtx(const NaviConfigContext &ctx) {
    _impl->attrFromCtx(ctx);
    return *this;
}

N &N::integerAttr(const std::string &key, int64_t value) {
    _impl->integerAttr(key, value);
    return *this;
}

N &N::integerArrayAttr(const std::vector<int64_t> &attrs) {
    _impl->integerArrayAttr(attrs);
    return *this;
}

N &N::skipConfig(bool skipConfig) {
    _impl->skipConfig(skipConfig);
    return *this;
}

N &N::skipInit(bool skipInit) {
    _impl->skipInit(skipInit);
    return *this;
}

N &N::stopAfterInit(bool stopAfterInit) {
    _impl->stopAfterInit(stopAfterInit);
    return *this;
}

N &N::skipDeleteKernel(bool skipDeleteKernel) {
    _impl->skipDeleteKernel(skipDeleteKernel);
    return *this;
}

bool N::getIntegerAttr(const std::string &key, int64_t &value) const {
    return _impl->getIntegerAttr(key, value);
}

P N::in(const std::string &port) const {
    return _impl->in(port);
}

P N::out(const std::string &port) const {
    return _impl->out(port);
}

void N::dependOn(P p) {
    return _impl->dependOn(p);
}

const std::string &N::name() const {
    return _impl->name();
}

const std::string &N::kernel() const {
    return _impl->kernel();
}

GraphId N::graphId() const {
    return _impl->graphId();
}

NodeType N::getType()const {
    return _impl->getNodeType();
}

int32_t N::getNodeId() const {
    return _impl->getId();
}

std::vector<std::pair<P, E>> N::getInputUpstreams(const std::string &port) const {
    return _impl->getInputUpstreams(port);
}

bool N::operator==(const N &other) const {
    return _impl == other._impl;
}

P::P(PImpl *impl, IndexType index)
    : _impl(impl)
    , _index(index)
{
}

P::P()
    : _impl(nullptr)
    , _index(INVALID_INDEX)
{
}

P::~P() {
}

E P::from(P peer) {
    return _impl->from(_index, peer);
}

E P::to(P peer) {
    return _impl->to(_index, peer, nullptr);
}

E P::fromForkNodeInput(const std::string &input, IndexType index) {
    return _impl->fromForkNodeInput(_index, input, index);
}

E P::toForkNodeOutput(const std::string &output, IndexType index) {
    return _impl->toForkNodeOutput(_index, output, index);
}

E P::asGraphOutput(const std::string &name) {
    return _impl->asGraphOutput(_index, name);
}

P P::asGroup() {
    return _impl->asGroup();
}

P P::next() {
    return _impl->next(_index);
}

P P::index(IndexType index) {
    return _impl->index(index);
}

P P::autoNext() {
    return _impl->autoNext();
}

N P::node() const {
    return N(_impl->getNode());
}


bool P::hasConnected() const {
    return _impl->hasConnected();
}

bool P::isValid() const {
    return nullptr != _impl;
}

bool P::isGroup() const {
    return _impl->isGroup();
}

GraphId P::graphId() const {
    return _impl->graphId();
}

const std::string &P::getNodeName() const {
    return _impl->getNodeName();
}

const std::string &P::getPortName() const {
    return _impl->getPort();
}

IndexType P::getIndex() const {
    return _index;
}

bool P::operator==(const P &other) const {
    return (_impl == other._impl)
        && (_index == other._index);
}

std::ostream &operator<<(std::ostream &os, const P &p) {
    auto impl = p._impl;
    auto realPort = impl->getRealPort();
    if (realPort._impl) {
        return os << realPort;
    }
    os << "graph " << impl->_node->graphId() << " node " << impl->_node->name() << " kernel " << impl->_node->kernel()
       << " port " << impl->_port;
    if (p._index != INVALID_INDEX) {
        os << " index " << p._index;
    }
    return os;
}

E::E(EImpl *e, SingleE *singleE)
    : _e(e)
    , _singleE(singleE)
{
}

E::E()
    : _e(nullptr)
    , _singleE(nullptr)
{
}

E::~E() {
}

N E::split(const std::string &splitKernel) {
    return _e->split(splitKernel);
}

N E::merge(const std::string &mergeKernel) {
    return _e->merge(mergeKernel);
}

E E::require(bool require) {
    _singleE->require(require, _e->getBuilder());
    return *this;
}

E E::backPressure(int64_t queueLow, int64_t queueHigh) {
    _e->backPressure(queueLow, queueHigh);
    return *this;
}

E E::timeoutMs(int64_t timeoutMs) {
    _singleE->timeoutMs(timeoutMs, _e->getBuilder());
    return *this;
}

void E::addSelectIndex(const std::set<int32_t> &selectIndexSet) {
    _singleE->addSelectIndex(selectIndexSet);
}

bool E::isRequire() const {
    return _singleE->isRequire();
}

bool E::isValid() const {
    return nullptr != _e && nullptr != _singleE;
}

bool E::operator==(const E &other) const {
    return _e == other._e && _singleE == other._singleE;
}

std::ostream &operator<<(std::ostream &os, const E &e) {
    if (!e._singleE) {
        return os << "invalid edge";
    }
    return os << *e._singleE;
}

}
