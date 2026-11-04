#pragma once

#include "navi/engine/Kernel.h"

namespace navi {

class IdentityKernel : public navi::Kernel
{
public:
    IdentityKernel();
    ~IdentityKernel();
    IdentityKernel(const IdentityKernel &) = delete;
    IdentityKernel &operator=(const IdentityKernel &) = delete;
public:
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;
private:
    KERNEL_DEPEND_DECLARE();
};

}
