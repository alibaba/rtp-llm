#include "navi/log/NaviLogger.h"
#include "navi/engine/Kernel.h"
#include "navi/proto/GraphDef.pb.h"

namespace navi {

Kernel::Kernel()
    : _def(nullptr)
{
}

Kernel::~Kernel() {
}

void Kernel::setNodeDef(const NodeDef *def) {
    _def = def;
}

const std::string &Kernel::getNodeName() const {
    return _def->name();
}

const std::string &Kernel::getKernelName() const {
    return _def->kernel_name();
}

const std::string &Kernel::getDeviceName() const {
    return _def->device();
}

}
