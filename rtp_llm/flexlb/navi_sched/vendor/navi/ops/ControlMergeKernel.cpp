#include "navi/ops/ControlMergeKernel.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

ControlMergeKernel::ControlMergeKernel()
{
}

ControlMergeKernel::~ControlMergeKernel() {
}

std::string ControlMergeKernel::getName() const {
    return CONTROL_MERGE_KERNEL;
}

std::string ControlMergeKernel::dataType() const {
    return "";
}

ErrorCode ControlMergeKernel::doCompute(const std::vector<StreamData> &dataVec,
                                        StreamData &outputData)
{
    for (const auto &streamData : dataVec) {
        if (!streamData.eof) {
            outputData.eof = false;
            return EC_NONE;
        }
    }
    outputData.eof = true;
    return EC_NONE;
}

REGISTER_BUILDIN_KERNEL(ControlMergeKernel);

}

