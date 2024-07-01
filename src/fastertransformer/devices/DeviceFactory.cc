#include "src/fastertransformer/devices/DeviceFactory.h"
#include <cassert>

using namespace std;

namespace fastertransformer {

DeviceType getDeviceType(const std::string& device_name) {
    if (device_name == "CPU") {
        return DeviceType::Cpu;
    } else if (device_name == "CUDA") {
        return DeviceType::Cuda;
    } else if (device_name == "ROCM") {
        return DeviceType::ROCm;
    } else if (device_name == "YITIAN") {
        return DeviceType::Yitian;
    } else if (device_name == "ARM") {
        return DeviceType::ArmCpu;
    } else {
        FT_LOG_ERROR("Unknown device type: %s", device_name.c_str());
        abort();
    }
}

GlobalDeviceParams DeviceFactory::getDefaultGlobalDeviceParams() {
    GlobalDeviceParams params;
    const std::vector<DeviceType> types_to_try = {
        DeviceType::Cuda, DeviceType::Yitian, DeviceType::ArmCpu, DeviceType::Cpu, DeviceType::ROCm};
    for (const auto type : types_to_try) {
        if (getRegistrationMap().find(type) != getRegistrationMap().end()) {
            FT_LOG_INFO("found device type %d, use as default.", static_cast<int>(type));
            params.device_params.push_back({type, DeviceInitParams{0}});
        } else {
            FT_LOG_INFO("Device type %d is not registered, skip.", static_cast<int>(type));
        }
    }
    if (!params.device_params.size()) {
        FT_LOG_ERROR("FATAL: No device is registered !");
        abort();
    }
    return params;
}

void DeviceFactory::initDevices(const GlobalDeviceParams& global_params) {
    if (getCurrentDevices().size()) {
        FT_LOG_WARNING("Devices are already initialized! will do nothing.");
        return;
    }
    if (!global_params.device_params.size()) {
        FT_LOG_ERROR("No device is specified to init !");
        abort();
    }
    for (const auto& [type, device_params] : global_params.device_params) {
        auto& registrationMap = getRegistrationMap();
        auto it = registrationMap.find(type);
        if (it == registrationMap.end()) {
            FT_LOG_ERROR("Device type %d is not registered !", static_cast<int>(type));
            abort();
        }
        auto device = it->second(device_params);
        getCurrentDevices().push_back(device);
    }
}

unordered_map<DeviceType, DeviceCreatorType>& DeviceFactory::getRegistrationMap() {
    static unordered_map<DeviceType, DeviceCreatorType> registrationMap;
    return registrationMap;
}

vector<DeviceBase*>& DeviceFactory::getCurrentDevices() {
    static vector<DeviceBase *> devices;
    return devices;
}

DeviceBase* DeviceFactory::getDevice(DeviceType type, int device_id) {
    if (!getCurrentDevices().size()) {
        FT_LOG_ERROR("You must explicitly initialize devices before getting device !");
        abort();
    }
    for (const auto device: getCurrentDevices()) {
        const auto& props = device->getDeviceProperties();
        if (props.type == type && props.id == device_id) {
            return device;
        }
    }
    FT_LOG_ERROR("Device type %d with id %d is not found !", static_cast<int>(type), device_id);
    abort();
}

DeviceBase* DeviceFactory::getDefaultDevice() {
    if (!getCurrentDevices().size()) {
        FT_LOG_ERROR("You must explicitly initialize devices before getting device !");
        abort();
    }
    return getCurrentDevices()[0];
}

void DeviceFactory::registerDevice(DeviceType type, DeviceCreatorType creator) {
    auto& registrationMap = getRegistrationMap();
    assert(registrationMap.find(type) == registrationMap.end());
    registrationMap[type] = creator;
}

} // namespace fastertransformer

