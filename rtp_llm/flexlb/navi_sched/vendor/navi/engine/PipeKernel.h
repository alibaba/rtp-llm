#pragma once

#include "navi/engine/Kernel.h"

namespace navi {

class PipeKernel : public Kernel
{
public:
    PipeKernel(const std::string &name);
    ~PipeKernel();
private:
    PipeKernel(const PipeKernel &);
    PipeKernel &operator=(const PipeKernel &);
public:
    void def(KernelDefBuilder &builder) const override final;
public:
    virtual std::vector<StaticResourceBindInfo> dependResources() const;
    virtual std::string inputPort() const;
    virtual std::string inputType() const = 0;
    virtual std::string outputPort() const;
    virtual std::string outputType() const = 0;
private:
    std::string _name;
};

}
