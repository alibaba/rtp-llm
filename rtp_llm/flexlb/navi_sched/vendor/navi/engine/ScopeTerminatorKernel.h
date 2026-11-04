#pragma once

#include "navi/engine/Kernel.h"

namespace navi {

class ScopeTerminatorKernel : public Kernel
{
public:
    ScopeTerminatorKernel();
    ~ScopeTerminatorKernel();
    ScopeTerminatorKernel(const ScopeTerminatorKernel &) = delete;
    ScopeTerminatorKernel &operator=(const ScopeTerminatorKernel &) = delete;
public:
    void def(KernelDefBuilder &builder) const override final;
    bool config(KernelConfigContext &ctx) override;
    ErrorCode init(KernelInitContext &ctx) override;
public:
    virtual std::string getName() const = 0;
    virtual std::vector<std::string> getOutputs() const;
};

}
