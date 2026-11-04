#pragma once

#include "navi/builder/GraphDesc.h"
#include "navi/proto/GraphDef.pb.h"

namespace navi {

class NodeDef;
class GraphBuilder;

class NImplBase
{
public:
    NImplBase()
        : _nodeType(NT_UNKNOWN)
    {
    }
    virtual ~NImplBase() {
    }
    NImplBase(const NImplBase &) = delete;
    NImplBase &operator=(const NImplBase &) = delete;
public:
    virtual void kernel(const std::string &name) = 0;
    virtual void device(const std::string &device) = 0;
    virtual void jsonAttrs(const std::string &attrs) = 0;
    virtual void jsonAttrsFromMap(const std::map<std::string, std::string> &attrs) = 0;
    virtual void jsonAttrsFromJsonMap(const autil::legacy::json::JsonMap &attrs) = 0;
    virtual void binaryAttrsFromMap(const std::map<std::string, std::string> &binaryAttrs) = 0;
    virtual void attr(const std::string &key, const std::string &value) = 0;
    virtual void attrFromCtx(const NaviConfigContext &ctx) = 0;
    virtual void integerAttr(const std::string &key, int64_t value) = 0;
    virtual void integerArrayAttr(const std::vector<int64_t> &attrs) = 0;
    virtual void skipConfig(bool skipConfig) = 0;
    virtual void skipInit(bool skipInit) = 0;
    virtual void stopAfterInit(bool stopAfterInit) = 0;
    virtual void skipDeleteKernel(bool skipDeleteKernel) = 0;
    virtual bool getIntegerAttr(const std::string &key, int64_t &value) const = 0;
public:
    virtual P in(const std::string &port) = 0;
    virtual P out(const std::string &port) = 0;
    virtual void dependOn(P p) = 0;
public:
    virtual GraphBuilder *builder() const = 0;
    virtual const std::string &name() const = 0;
    virtual const std::string &kernel() const = 0;
    virtual GraphId graphId() const = 0;
public:
    virtual int32_t getScope() const {
        return 0;
    }
    virtual void initScopeBorder(IoType ioType) {
    }
    virtual void setScopeBorder() {
    }
    virtual bool isScopeBorder() const {
        return false;
    }
    virtual std::vector<std::pair<P, E>> getInputUpstreams(const std::string &portName) const {
        return {};
    }
    void setNodeType(NodeType type) {
        _nodeType = type;
    }
    NodeType getNodeType() const {
        return _nodeType;
    }
    void setId(int32_t id) {
        _id = id;
    }
    int32_t getId() const {
        return _id;
    }
private:
    NodeType _nodeType;
    int32_t _id = -1;
};

}

