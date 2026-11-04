#include "navi/ops/TerminatorKernel.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi {

TerminatorKernel::TerminatorKernel() {
}

TerminatorKernel::~TerminatorKernel() {
}

ErrorCode TerminatorKernel::init(KernelInitContext &ctx) {
    return EC_NONE;
}

void TerminatorKernel::def(KernelDefBuilder &builder) const {
    builder
        .name(TERMINATOR_KERNEL_NAME)
        .inputGroup(TERMINATOR_INPUT_PORT, "", IT_REQUIRE);
}

ErrorCode TerminatorKernel::compute(KernelComputeContext &ctx) {
    ctx.setIgnoreDeadlock();
    ctx._subGraph->notifyFinish(ctx._node, nullptr);
    ctx.stopSchedule();
    return EC_NONE;
}

REGISTER_BUILDIN_KERNEL(TerminatorKernel);

}
