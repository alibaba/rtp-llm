#pragma once

#include "navi/engine/PortMergeKernel.h"
#include "navi/util/ReadyBitMap.h"

namespace navi {

class InlineReadyBitMap;

class DefaultMergeKernel : public PortMergeKernel
{
public:
    DefaultMergeKernel();
    ~DefaultMergeKernel();
    DefaultMergeKernel(const DefaultMergeKernel &) = delete;
    DefaultMergeKernel &operator=(const DefaultMergeKernel &) = delete;
public:
    std::string getName() const override;
    std::string dataType() const override;
    InputTypeDef inputType() const override;
    ErrorCode doInit(KernelInitContext &ctx) override;
    ErrorCode compute(KernelComputeContext &ctx) override;
    bool isExpensive() const override {
        return false;
    }
    ErrorCode doCompute(const std::vector<StreamData> &dataVec,
                        StreamData &outputData) override;
protected:
    void setFinish(NaviPartId index);
    bool outputEof() const;
protected:
    InlineReadyBitMap _dataReadyMap;
    NaviPartId _loopIndex;
};

}
