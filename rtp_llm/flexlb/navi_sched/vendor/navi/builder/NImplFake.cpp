#include "navi/builder/NImplFake.h"
#include "navi/builder/PImpl.h"

namespace navi {

NImplFake::NImplFake()
    : _builder(nullptr)
    , _graphId(INVALID_GRAPH_ID)
{
}

NImplFake::~NImplFake() {
}

void NImplFake::init(GraphBuilder *builder,
                     GraphId graphId,
                     const std::string &name,
                     const std::string &kernel,
                     NodeType nodeType)
{
    _builder = builder;
    _graphId = graphId;
    _name = name;
    _kernel = kernel;
    setNodeType(nodeType);
}

void NImplFake::kernel(const std::string &name) {
    _kernel = name;
}

void NImplFake::device(const std::string &device) {
}

void NImplFake::jsonAttrs(const std::string &attrs) {
}

void NImplFake::jsonAttrsFromMap(const std::map<std::string, std::string> &attrs) {
}

void NImplFake::jsonAttrsFromJsonMap(const autil::legacy::json::JsonMap &attrs) {
}

void NImplFake::binaryAttrsFromMap(const std::map<std::string, std::string> &binaryAttrs) {
}

void NImplFake::attr(const std::string &key, const std::string &value) {
}

void NImplFake::attrFromCtx(const NaviConfigContext &ctx) {
}

void NImplFake::integerAttr(const std::string &key, int64_t value) {
}

void NImplFake::integerArrayAttr(const std::vector<int64_t> &attrs) {
}

void NImplFake::skipConfig(bool skipConfig) {
}

void NImplFake::skipInit(bool skipInit) {
}

void NImplFake::stopAfterInit(bool stopAfterInit) {
}

void NImplFake::skipDeleteKernel(bool skipDeleteKernel) {
}

bool NImplFake::getIntegerAttr(const std::string &key, int64_t &value) const {
    return false;
}

P NImplFake::in(const std::string &port) {
    AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                       "error, can't get input port [" + port + "] of " + autil::StringUtil::toString(*this));
}

P NImplFake::out(const std::string &port) {
    AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                       "error, can't get output port [" + port + "] of " + autil::StringUtil::toString(*this));
}

void NImplFake::dependOn(P p) {
    AUTIL_LEGACY_THROW(autil::legacy::ExceptionBase,
                       "error, " + autil::StringUtil::toString(*this) +
                           " can't depend on other node: " + autil::StringUtil::toString(p));
}

GraphBuilder *NImplFake::builder() const {
    return _builder;
}

const std::string &NImplFake::name() const {
    return _name;
}

const std::string &NImplFake::kernel() const {
    return _kernel;
}

GraphId NImplFake::graphId() const {
    return _graphId;
}

std::ostream &operator<<(std::ostream &os, const NImplFake &impl) {
    os << "graphId " << impl._graphId << " node " << impl._name << " type " << NodeType_Name(impl.getNodeType());
    return os;
}

}
