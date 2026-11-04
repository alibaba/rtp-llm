#include "navi/engine/MapKernel.h"

namespace navi {

MapKernel::MapKernel(const std::string &name)
    : PipeKernel(name)
{
}

MapKernel::~MapKernel() {
}

ErrorCode MapKernel::compute(KernelComputeContext &ctx) {
    navi::PortIndex inputIndex(0, navi::INVALID_INDEX);
    navi::PortIndex outputIndex(0, navi::INVALID_INDEX);
    bool isEof = false;
    navi::DataPtr data;
    ctx.getInput(inputIndex, data, isEof);
    DataPtr ret;
    if (data) {
        ret = map(data);
        if (!ret) {
            return EC_ABORT;
        }
    }
    if (isEof) {
        eof();
    }
    if (!ctx.setOutput(outputIndex, ret, isEof)) {
        return EC_ABORT;
    }
    return EC_NONE;
}

void MapKernel::eof() {
}

}

