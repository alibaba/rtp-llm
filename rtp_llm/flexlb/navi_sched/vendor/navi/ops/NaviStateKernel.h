#pragma once

#include "navi/engine/Kernel.h"
#include "navi/resource/NaviStateR.h"
#include "navi/resource/NaviStateRpcR.h"
#include "navi/rpc_server/NaviArpcRequestData.h"

namespace navi {

class Navi;
class NaviStateResponseDef;

class NaviStateKernel : public Kernel
{
public:
    NaviStateKernel();
    ~NaviStateKernel();
    NaviStateKernel(const NaviStateKernel &) = delete;
    NaviStateKernel &operator=(const NaviStateKernel &) = delete;
public:
    void def(KernelDefBuilder &builder) const override;
    bool config(KernelConfigContext &ctx) override;
    ErrorCode init(KernelInitContext &ctx) override;
    ErrorCode compute(KernelComputeContext &ctx) override;
private:
    void fillVisData(Navi *navi, NaviStateResponseDef *responsePb);
private:
    KERNEL_DEPEND_DECLARE();
private:
    KERNEL_NAMED_DATA(NaviArpcRequestData, _requestData, NaviStateRpcR::REQUEST_NAME);
    KERNEL_DEPEND_ON(NaviStateR, _naviStateR);
};

}
