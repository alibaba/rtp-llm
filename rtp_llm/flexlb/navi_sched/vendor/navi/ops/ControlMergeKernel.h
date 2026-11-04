#pragma once

#include "navi/engine/PortMergeKernel.h"

namespace navi {

class ControlMergeKernel : public PortMergeKernel
{
public:
    ControlMergeKernel();
    ~ControlMergeKernel();
    ControlMergeKernel(const ControlMergeKernel &) = delete;
    ControlMergeKernel &operator=(const ControlMergeKernel &) = delete;
public:
    std::string getName() const override;
    std::string dataType() const override;
    ErrorCode doCompute(const std::vector<StreamData> &dataVec,
                        StreamData &outputData) override;
};

}
