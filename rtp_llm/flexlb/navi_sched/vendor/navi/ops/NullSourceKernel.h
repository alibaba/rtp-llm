#pragma once

#include "navi/engine/Kernel.h"

namespace navi {

class NullSourceKernel : public navi::Kernel
{
public:
    NullSourceKernel();
    ~NullSourceKernel();
    NullSourceKernel(const NullSourceKernel &) = delete;
    NullSourceKernel &operator=(const NullSourceKernel &) = delete;
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;
public:
    static const std::string KERNEL_ID;
    static const std::string OUTPUT_PORT;
};

}
