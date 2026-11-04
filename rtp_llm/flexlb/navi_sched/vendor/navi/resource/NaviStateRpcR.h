#pragma once

#include "navi/engine/Resource.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/rpc_server/NaviRpcServerR.h"

namespace navi {

class NaviStateRpcR : public Resource
{
public:
    NaviStateRpcR();
    ~NaviStateRpcR();
    NaviStateRpcR(const NaviStateRpcR &) = delete;
    NaviStateRpcR &operator=(const NaviStateRpcR &) = delete;
public:
    void def(ResourceDefBuilder &builder) const override;
    bool config(ResourceConfigContext &ctx) override;
    ErrorCode init(ResourceInitContext &ctx) override;
private:
    bool buildGraph(GraphBuilder &builder) const;
public:
    static const std::string RESOURCE_ID;
    static const std::string REQUEST_NAME;
private:
    RESOURCE_DEPEND_DECLARE();
private:
    RESOURCE_DEPEND_ON(NaviRpcServerR, _naviRpcServerR);
};

NAVI_TYPEDEF_PTR(NaviStateRpcR);

}

