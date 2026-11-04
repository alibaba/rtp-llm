#pragma once

#include "navi/engine/Kernel.h"

namespace navi {

class Port;
class DefaultMergeKernel;
class PartInfo;

class PortMergeKernel : public Kernel
{
public:
    PortMergeKernel();
    ~PortMergeKernel();
private:
    PortMergeKernel(const PortMergeKernel &);
    PortMergeKernel &operator=(const PortMergeKernel &);
public:
    void def(KernelDefBuilder &builder) const override final;
    bool config(KernelConfigContext &ctx) override final;
    ErrorCode init(KernelInitContext &ctx) override final;
public:
    // user override
    ErrorCode compute(KernelComputeContext &ctx) override;
    virtual std::vector<StaticResourceBindInfo> dependResources() const;
    virtual std::string getName() const = 0;
    virtual std::string dataType() const = 0;
    virtual InputTypeDef inputType() const;
    virtual ErrorCode doInit(KernelInitContext &ctx);
    virtual bool doConfig(KernelConfigContext &ctx);
    virtual ErrorCode doCompute(const std::vector<StreamData> &dataVec,
                                StreamData &outputData) = 0;
protected:
    NaviPartId getUsedPartCount() const;
    NaviPartId getUsedPartId(NaviPartId index) const;
    ErrorCode getData(KernelComputeContext &ctx, NaviPartId partId,
                      StreamData &streamData);
private:
    ErrorCode fillInputs(KernelComputeContext &ctx);
private:
    Port *_port;
    size_t _inputGroupSize = 0;
    std::vector<StreamData> _dataVec;
    friend class DefaultMergeKernel;
protected:
    static IndexType INPUT_PORT_GROUP;
    static IndexType OUTPUT_PORT;
};

}
