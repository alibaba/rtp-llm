#pragma once

#include "navi/engine/Kernel.h"
#include "navi/engine/PipeKernel.h"

namespace navi {

class MapKernel : public PipeKernel
{
public:
    MapKernel(const std::string &name);
    ~MapKernel();
private:
    MapKernel(const MapKernel &);
    MapKernel &operator=(const MapKernel &);
public:
    ErrorCode compute(KernelComputeContext &ctx) override;
    virtual DataPtr map(const DataPtr &data) = 0;
    virtual void eof();
private:
    std::string _name;
};

}
