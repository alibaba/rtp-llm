#pragma once

#include "navi/engine/Kernel.h"

namespace navi {

class PlaceholderKernel : public Kernel
{
public:
    PlaceholderKernel();
    ~PlaceholderKernel();
    PlaceholderKernel(const PlaceholderKernel &) = delete;
    PlaceholderKernel &operator=(const PlaceholderKernel &) = delete;
public:
    void def(KernelDefBuilder &builder) const override;
    ErrorCode init(KernelInitContext &ctx) override;
    ErrorCode compute(KernelComputeContext &ctx) override;
};

}
