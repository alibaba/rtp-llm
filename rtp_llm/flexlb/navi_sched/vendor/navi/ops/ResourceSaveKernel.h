#pragma once

#include "navi/engine/Kernel.h"
#include "navi/resource/ResourceErrorInfoCollectorR.h"

namespace navi {

class ResourceSaveKernel : public Kernel
{
public:
    ResourceSaveKernel();
    ~ResourceSaveKernel();
    ResourceSaveKernel(const ResourceSaveKernel &) = delete;
    ResourceSaveKernel &operator=(const ResourceSaveKernel &) = delete;
public:
    void def(KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    ErrorCode init(KernelInitContext &ctx) override;
    ErrorCode compute(KernelComputeContext &ctx) override;
private:
    KERNEL_DEPEND_DECLARE();
private:
    KERNEL_DEPEND_ON(ResourceErrorInfoCollectorR, _resourceErrorInfoCollectorR);
    int64_t _ignoreProduceError = 0;
    size_t _saveSuccessCount = 0;
    ResourceMap _resourceMap;
};

}
