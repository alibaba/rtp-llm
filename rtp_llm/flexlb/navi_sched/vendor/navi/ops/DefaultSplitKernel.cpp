#include "navi/ops/DefaultSplitKernel.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

DefaultSplitKernel::DefaultSplitKernel()
{
}

DefaultSplitKernel::~DefaultSplitKernel() {
}

std::string DefaultSplitKernel::getName() const {
    return DEFAULT_SPLIT_KERNEL;
}

std::string DefaultSplitKernel::dataType() const {
    return "";
}

ErrorCode DefaultSplitKernel::doCompute(const StreamData &streamData,
                                        std::vector<DataPtr> &dataVec)
{
    NAVI_KERNEL_LOG(SCHEDULE1, "input: %p, size: %lu, eof: %d",
                    streamData.data.get(), dataVec.size(), streamData.eof);
    auto size = dataVec.size();
    for (size_t i = 0; i < size; i++) {
        dataVec[i] = streamData.data;
    }
    return EC_NONE;
}

REGISTER_BUILDIN_KERNEL(DefaultSplitKernel);

}

