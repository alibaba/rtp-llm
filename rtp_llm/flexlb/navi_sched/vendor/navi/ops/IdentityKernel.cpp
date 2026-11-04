#include "navi/ops/IdentityKernel.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

IdentityKernel::IdentityKernel() {
}

IdentityKernel::~IdentityKernel() {
}

void IdentityKernel::def(navi::KernelDefBuilder &builder) const {
    builder.name(IDENTITY_KERNEL)
        .input(IDENTITY_KERNEL_INPUT, "")
        .output(IDENTITY_KERNEL_OUTPUT, "");
}

bool IdentityKernel::config(navi::KernelConfigContext &ctx) {
    return true;
}

navi::ErrorCode IdentityKernel::init(navi::KernelInitContext &ctx) {
    return navi::EC_NONE;
}

navi::ErrorCode IdentityKernel::compute(navi::KernelComputeContext &ctx) {
    bool eof = false;
    DataPtr data;
    ctx.getInput(0, data, eof);
    if (!ctx.setOutput(0, data, eof)) {
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

REGISTER_BUILDIN_KERNEL(IdentityKernel);

}
