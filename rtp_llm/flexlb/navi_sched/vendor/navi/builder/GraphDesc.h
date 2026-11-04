#pragma once

#include "navi/common.h"
#include "autil/legacy/json.h"
#include "navi/proto/GraphDef.pb.h"

namespace navi {

class NaviConfigContext;
class NImplBase;
class P;
class E;

class N
{
public:
    N();
private:
    N(NImplBase *impl);
public:
    ~N();
public:
    N &kernel(const std::string &name);
    N &device(const std::string &device);
    N &jsonAttrs(const std::string &attrs);
    N &jsonAttrsFromMap(const std::map<std::string, std::string> &attrs);
    N &jsonAttrsFromJsonMap(const autil::legacy::json::JsonMap &attrs);
    N &binaryAttrsFromMap(const std::map<std::string, std::string> &binaryAttrs);
    N &attr(const std::string &key, const std::string &value);
    N &attrFromCtx(const NaviConfigContext &ctx);
    N &integerAttr(const std::string &key, int64_t value);
    N &integerArrayAttr(const std::vector<int64_t> &attrs);
    N &skipConfig(bool skipConfig);
    N &skipInit(bool skipInit);
    N &stopAfterInit(bool stopAfterInit);
    N &skipDeleteKernel(bool skipDeleteKernel);
    bool getIntegerAttr(const std::string &key, int64_t &value) const;
    bool isValid() const {
        return _impl != nullptr;
    }
public:
    P in(const std::string &port) const;
    P out(const std::string &port) const;
    void dependOn(P p);
public:
    const std::string &name() const;
    const std::string &kernel() const;
    GraphId graphId() const;
    NodeType getType()const;
    int32_t getNodeId() const;
    std::vector<std::pair<P, E>> getInputUpstreams(const std::string &port) const;
public:
    bool operator==(const N &other) const;
private:
    friend class SubGraphBuildInfo;
    friend class BorderEImpl;
    friend class InterEImpl;
    friend class NImpl;
    friend class P;
private:
    NImplBase *_impl;
};

class PImpl;
class E;

class P
{
public:
    P();
private:
    P(PImpl *impl, IndexType index);
public:
    ~P();
public:
    E from(P peer);
    E to(P peer);
    E fromForkNodeInput(const std::string &input, IndexType index = INVALID_INDEX);
    E toForkNodeOutput(const std::string &output, IndexType index = INVALID_INDEX);
    E asGraphOutput(const std::string &name);
public:
    P asGroup();
    P next();
    P index(IndexType index);
    P autoNext();
    N node() const;
    bool hasConnected() const;
    bool isValid() const;
    bool isGroup() const;
    GraphId graphId() const;
    const std::string &getNodeName() const;
    const std::string &getPortName() const;
    IndexType getIndex() const;

public:
    bool operator==(const P &other) const;
private:
    friend class NImpl;
    friend class NImplFake;
    friend class NImplFork;
    friend class PImpl;
    friend class BorderEImpl;
    friend class InterEImpl;
    friend class EImpl;
    friend class SubGraphBuildInfo;
    friend class RunResourceGraphBuilder;
public:
    friend std::ostream &operator<<(std::ostream &os, const P &p);
private:
    PImpl *_impl;
    IndexType _index;
};

class EImpl;
class SingleE;

class E
{
public:
    E();
private:
    E(EImpl *e, SingleE *singleE);
public:
    ~E();
public:
    N split(const std::string &splitKernel);
    N merge(const std::string &mergeKernel);
    E require(bool require);
    E backPressure(int64_t queueLow, int64_t queueHigh);
    E timeoutMs(int64_t timeoutMs);
public:
    bool isRequire() const;
    bool isValid() const;
public:
    bool operator==(const E &other) const;
public:
    friend std::ostream &operator<<(std::ostream &os, const E &e);
private:
    void addSelectIndex(const std::set<int32_t> &selectIndexSet);
    friend class PImpl;
    friend class InterEImpl;
    friend class BorderEImpl;
    friend class ResourceGraphBuilder;
private:
    EImpl *_e;
    SingleE *_singleE;
};

}
