#include "navi/engine/N2OneKernel.h"

namespace navi {

N2OneKernel::N2OneKernel(const std::string &name)
    : PipeKernel(name)
{
}

N2OneKernel::~N2OneKernel() {
}

ErrorCode N2OneKernel::compute(KernelComputeContext &ctx) {
    navi::PortIndex inputIndex(0, navi::INVALID_INDEX);
    navi::PortIndex outputIndex(0, navi::INVALID_INDEX);
    bool eof = false;
    navi::DataPtr data;
    ctx.getInput(inputIndex, data, eof);
    if (data) {
        auto ec = collect(data);
        if (EC_NONE != ec) {
            return ec;
        }
    }
    if (eof) {
        DataPtr data;
        auto ec = finalize(data);
        if (ec != EC_NONE) {
            return ec;
        }
        if (!ctx.setOutput(outputIndex, data, true)) {
            return EC_ABORT;
        }
    }
    return EC_NONE;
}

}

