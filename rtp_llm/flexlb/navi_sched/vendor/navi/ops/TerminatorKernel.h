#pragma once

#include "navi/engine/Kernel.h"

namespace navi {

class TerminatorKernel : public Kernel
{
public:
    TerminatorKernel();
    ~TerminatorKernel();
private:
    TerminatorKernel(const TerminatorKernel &);
    TerminatorKernel &operator=(const TerminatorKernel &);
public:
    void def(KernelDefBuilder &builder) const override;
    ErrorCode init(KernelInitContext &ctx) override;
    ErrorCode compute(KernelComputeContext &ctx) override;
};

}
