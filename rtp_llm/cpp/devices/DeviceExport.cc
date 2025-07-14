#include "rtp_llm/cpp/devices/DeviceExport.h"

using namespace rtp_llm;

namespace torch_ext {

DeviceType DeviceExporter::getDeviceType() {
    return device_params_.device_type;
}

int64_t DeviceExporter::getDeviceId() {
    return device_params_.device_id;
}

}  // namespace torch_ext

using namespace torch_ext;
