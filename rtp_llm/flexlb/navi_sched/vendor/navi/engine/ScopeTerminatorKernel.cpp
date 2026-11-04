#include "navi/engine/ScopeTerminatorKernel.h"
#include "navi/engine/LocalSubGraph.h"
#include "navi/engine/Node.h"

namespace navi {

ScopeTerminatorKernel::ScopeTerminatorKernel() {
}

ScopeTerminatorKernel::~ScopeTerminatorKernel() {
}

void ScopeTerminatorKernel::def(KernelDefBuilder &builder) const {
    builder.name(getName())
        .inputGroup(SCOPE_TERMINATOR_INPUT_PORT, "", IT_REQUIRE);
    auto outputs = getOutputs();
    for (const auto &output : outputs) {
        builder.output(output, "");
    }
}

std::vector<std::string> ScopeTerminatorKernel::getOutputs() const {
    return {};
}

bool ScopeTerminatorKernel::config(KernelConfigContext &ctx) {
    return true;
}

ErrorCode ScopeTerminatorKernel::init(KernelInitContext &ctx) {
    return EC_NONE;
}

}
