#pragma once

#include "navi/engine/Kernel.h"

namespace navi {

class ResourcePublishKernel : public Kernel
{
public:
    ResourcePublishKernel();
    ~ResourcePublishKernel();
    ResourcePublishKernel(const ResourcePublishKernel &) = delete;
    ResourcePublishKernel &operator=(const ResourcePublishKernel &) = delete;
public:
    void def(KernelDefBuilder &builder) const override;
    bool config(KernelConfigContext &ctx) override;
    ErrorCode init(KernelInitContext &ctx) override;
    ErrorCode compute(KernelComputeContext &ctx) override;
};

}
