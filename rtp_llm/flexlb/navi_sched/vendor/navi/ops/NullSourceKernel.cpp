#include "navi/ops/NullSourceKernel.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

const std::string NullSourceKernel::KERNEL_ID = "navi.null_source_k";
const std::string NullSourceKernel::OUTPUT_PORT = "navi.null_source_k.out";

NullSourceKernel::NullSourceKernel() {
}

NullSourceKernel::~NullSourceKernel() {
}

void NullSourceKernel::def(navi::KernelDefBuilder &builder) const {
    builder.name(KERNEL_ID)
        .output(OUTPUT_PORT, "");
}

navi::ErrorCode NullSourceKernel::init(navi::KernelInitContext &ctx) {
    return navi::EC_NONE;
}

navi::ErrorCode NullSourceKernel::compute(navi::KernelComputeContext &ctx) {
    NAVI_KERNEL_LOG(TRACE3, "begin output eof");
    if (!ctx.setOutput(0, nullptr, true)) {
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

REGISTER_BUILDIN_KERNEL(NullSourceKernel);

}
