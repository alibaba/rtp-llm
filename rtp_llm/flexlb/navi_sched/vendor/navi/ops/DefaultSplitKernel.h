#pragma once

#include "navi/engine/PortSplitKernel.h"

namespace navi {

class DefaultSplitKernel : public PortSplitKernel
{
public:
    DefaultSplitKernel();
    ~DefaultSplitKernel();
    DefaultSplitKernel(const DefaultSplitKernel &) = delete;
    DefaultSplitKernel &operator=(const DefaultSplitKernel &) = delete;
public:
    std::string getName() const override;
    std::string dataType() const override;
    ErrorCode doCompute(const StreamData &streamData,
                        std::vector<DataPtr> &dataVec) override;
    bool isExpensive() const override {
        return false;
    }
};

}
