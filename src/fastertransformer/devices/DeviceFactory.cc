#include "src/fastertransformer/devices/DeviceFactory.h"
#include <cassert>

using namespace std;

namespace fastertransformer {

DeviceType getDeviceType(const std::string& device_name) {
    if (device_name == "CPU") {
        return DeviceType::Cpu;
    } else if (device_name == "CUDA") {
        return DeviceType::Cuda;
    } else if (device_name == "YITIAN") {
        return DeviceType::Yitian;
    } else {
        FT_LOG_ERROR("Unknown device type: %s", device_name.c_str());
        abort();
    }
}

unordered_map<DeviceType, function<DeviceBase*()>>& DeviceFactory::getRegistrationMap() {
    static unordered_map<DeviceType, function<DeviceBase*()>> registrationMap;
    return registrationMap;
}

DeviceBase* DeviceFactory::getDevice(DeviceType type, int device_id) {
    auto& registrationMap = getRegistrationMap();
    auto it = registrationMap.find(type);
    if (it == registrationMap.end()) {
        FT_LOG_ERROR("Device type %d is not registered !", static_cast<int>(type));
        abort();
    }
    return it->second();
}

DeviceBase* DeviceFactory::getDefaultDevice() {
    DeviceBase* device = nullptr;
    const std::array<DeviceType, 3> types_to_try = {
        DeviceType::Cuda, DeviceType::Yitian, DeviceType::Cpu};
    for (const auto type : types_to_try) {
        if (getRegistrationMap().find(type) != getRegistrationMap().end()) {
            return getDevice(type);
        }
    }
    FT_LOG_ERROR("No device is registered !");
    abort();
}

void DeviceFactory::registerDevice(DeviceType type, function<DeviceBase*()> creator) {
    auto& registrationMap = getRegistrationMap();
    assert(registrationMap.find(type) == registrationMap.end());
    registrationMap[type] = creator;
}

} // namespace fastertransformer

