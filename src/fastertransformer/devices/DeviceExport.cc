#include "src/fastertransformer/devices/DeviceExport.h"

using namespace fastertransformer;

namespace torch_ext {

DeviceType DeviceExporter::getDeviceType() {
    return device_params_.device_type;
}

int64_t DeviceExporter::getDeviceId() {
    return device_params_.device_id;
}

} // namespace torch_ext

using namespace torch_ext;
