#include "navi/engine/PipeKernel.h"

namespace navi {

PipeKernel::PipeKernel(const std::string &name)
    : _name(name)
{
}

PipeKernel::~PipeKernel() {
}

void PipeKernel::def(KernelDefBuilder &builder) const {
    builder.name(_name)
        .input(inputPort(), inputType())
        .output(outputPort(), outputType());
    auto infoVec = dependResources();
    for (const auto &info : infoVec) {
        builder.dependOn(info.name, info.require, info.binder);
    }
}

std::vector<StaticResourceBindInfo> PipeKernel::dependResources() const {
    return {};
}

std::string PipeKernel::inputPort() const {
    return MAP_KERNEL_INPUT_PORT;
}

std::string PipeKernel::outputPort() const {
    return MAP_KERNEL_OUTPUT_PORT;
}

}
