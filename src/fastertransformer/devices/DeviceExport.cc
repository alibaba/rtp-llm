#include "src/fastertransformer/devices/DeviceExport.h"

using namespace fastertransformer;

namespace torch_ext {

void registerDeviceOps(py::module& m) {
    m.def("init_devices", &DeviceFactory::initDevices);
}

} // namespace torch_ext

