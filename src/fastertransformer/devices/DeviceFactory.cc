#include "src/fastertransformer/devices/DeviceFactory.h"
#include <cassert>

using namespace std;

namespace fastertransformer {

unordered_map<DeviceType, function<DeviceBase*()>>& DeviceFactory::getRegistrationMap() {
    static unordered_map<DeviceType, function<DeviceBase*()>> registrationMap;
    return registrationMap;
}

DeviceBase* DeviceFactory::getDevice(DeviceType type, int device_id) {
    auto& registrationMap = getRegistrationMap();
    auto it = registrationMap.find(type);
    assert(it != registrationMap.end());
    return it->second();
}

void DeviceFactory::registerDevice(DeviceType type, function<DeviceBase*()> creator) {
    auto& registrationMap = getRegistrationMap();
    assert(registrationMap.find(type) == registrationMap.end());
    registrationMap[type] = creator;
}

} // namespace fastertransformer

