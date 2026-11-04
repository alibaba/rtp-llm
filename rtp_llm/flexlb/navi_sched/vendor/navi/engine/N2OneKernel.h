#pragma once

#include "navi/engine/PipeKernel.h"

namespace navi {

class N2OneKernel : public PipeKernel
{
public:
    N2OneKernel(const std::string &name);
    ~N2OneKernel();
private:
    N2OneKernel(const N2OneKernel &);
    N2OneKernel &operator=(const N2OneKernel &);
public:
    ErrorCode compute(KernelComputeContext &ctx) override;
    virtual ErrorCode collect(const DataPtr &data) = 0;
    virtual ErrorCode finalize(DataPtr &data) = 0;
};

}
