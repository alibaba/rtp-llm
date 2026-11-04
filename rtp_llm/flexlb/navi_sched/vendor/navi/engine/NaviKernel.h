#pragma once

#include "navi/engine/Kernel.h"
#include "navi/engine/NaviR.h"

namespace navi {

class NaviKernel : public Kernel
{
public:
    NaviKernel();
    ~NaviKernel();
    NaviKernel(const NaviKernel &) = delete;
    NaviKernel &operator=(const NaviKernel &) = delete;
public:
    void def(KernelDefBuilder &builder) const override;
    bool config(KernelConfigContext &ctx) override;
    ErrorCode init(KernelInitContext &ctx) override;
    ErrorCode compute(KernelComputeContext &ctx) override;
private:
    KERNEL_DEPEND_DECLARE();
private:
    KERNEL_DEPEND_ON(NaviR, _naviR);
};

}
