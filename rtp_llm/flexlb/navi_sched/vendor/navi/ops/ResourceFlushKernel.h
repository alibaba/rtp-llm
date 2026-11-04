#pragma once

#include "navi/engine/Kernel.h"
#include "navi/ops/ResourceFlushData.h"

namespace navi {

class ResourceFlushKernel : public Kernel
{
public:
    ResourceFlushKernel();
    ~ResourceFlushKernel();
    ResourceFlushKernel(const ResourceFlushKernel &) = delete;
    ResourceFlushKernel &operator=(const ResourceFlushKernel &) = delete;
public:
    void def(KernelDefBuilder &builder) const override;
    bool config(KernelConfigContext &ctx) override;
    ErrorCode init(KernelInitContext &ctx) override;
    ErrorCode compute(KernelComputeContext &ctx) override;
private:
    bool readInput();
    bool createData(KernelComputeContext &ctx);
private:
    ResourceFlushDataPtr _data;
};

}
