#pragma once

#include "autil/legacy/json.h"
#include "navi/builder/NImplBase.h"
#include "navi/common.h"

namespace navi {

class SubGraphBuildInfo;

class NImpl : public NImplBase
{
public:
    NImpl();
    ~NImpl();
    NImpl(const NImpl &) = delete;
    NImpl &operator=(const NImpl &) = delete;
public:
    void init(NodeDef *def, SubGraphBuildInfo *subGraphBuildInfo);
    void kernel(const std::string &name) override;
    void device(const std::string &device) override;
    void jsonAttrs(const std::string &attrs) override;
    void jsonAttrsFromMap(const std::map<std::string, std::string> &attrs) override;
    void jsonAttrsFromJsonMap(const autil::legacy::json::JsonMap &attrs) override;
    void binaryAttrsFromMap(const std::map<std::string, std::string> &binaryAttrs) override;
    void attr(const std::string &key, const std::string &value) override;
    void attrFromCtx(const NaviConfigContext &ctx) override;
    void integerAttr(const std::string &key, int64_t value) override;
    void integerArrayAttr(const std::vector<int64_t> &attrs) override;
    void skipConfig(bool skipConfig) override;
    void skipInit(bool skipInit) override;
    void stopAfterInit(bool stopAfterInit) override;
    void skipDeleteKernel(bool skipDeleteKernel) override;
    bool getIntegerAttr(const std::string &key, int64_t &value) const override;
public:
    P in(const std::string &port) override;
    P out(const std::string &port) override;
    void dependOn(P p) override;
public:
    GraphBuilder *builder() const override;
    const std::string &name() const override;
    const std::string &kernel() const override;
    GraphId graphId() const override;
public:
    int32_t getScope() const override;
    void initScopeBorder(IoType ioType) override;
    bool isScopeBorder() const override;
    void setScopeBorder() override;
    std::vector<std::pair<P, E>> getInputUpstreams(const std::string &portName) const override;
private:
    void setJsonAttr(const std::string &attrStr);
private:
    friend std::ostream &operator<<(std::ostream &os, const NImpl &impl);
private:
    SubGraphBuildInfo *_subGraphBuildInfo;
    NodeDef *_def;
    NaviUnorderedMap<std::string, PImpl *> _ins;
    NaviUnorderedMap<std::string, PImpl *> _outs;
    NaviUnorderedSet<PImpl *> _dependOns;
    P _scopeTerminatorInput;
};

}

