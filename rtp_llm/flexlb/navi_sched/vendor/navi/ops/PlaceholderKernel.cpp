#include "navi/ops/PlaceholderKernel.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

PlaceholderKernel::PlaceholderKernel() {
}

PlaceholderKernel::~PlaceholderKernel() {
}

void PlaceholderKernel::def(KernelDefBuilder &builder) const {
    builder
        .name(PLACEHOLDER_KERNEL_NAME)
        .output(PLACEHOLDER_OUTPUT_PORT, "");
}

ErrorCode PlaceholderKernel::init(KernelInitContext &ctx) {
    return EC_NONE;
}

ErrorCode PlaceholderKernel::compute(KernelComputeContext &ctx) {
    NAVI_KERNEL_LOG(ERROR, "placeholder must be fed");
    return EC_ABORT;
}

REGISTER_BUILDIN_KERNEL(PlaceholderKernel);

}

