#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"
#include <unordered_map>

namespace fastertransformer {

enum class DeviceType {
    Cpu  = 0,
    Cuda = 1,
    Yitian = 2,
};

class DeviceFactory {
public:
    static DeviceBase* getDevice(DeviceType type, int device_id = 0);
    static DeviceBase* getDefaultDevice();
    static void registerDevice(DeviceType type, std::function<DeviceBase*()> creator);

private:
    static std::unordered_map<DeviceType, std::function<DeviceBase*()>>& getRegistrationMap();
};

#define RTP_LLM_REGISTER_DEVICE(type) \
    static DeviceBase* type##_device = nullptr; \
    static auto type##_device_reg_creator = []() { \
        DeviceFactory::registerDevice(DeviceType::type, []() { \
            static DeviceBase* type##_device = nullptr; \
            if (type##_device == nullptr) { \
                type##_device = new type##Device(); \
                type##_device->init(); \
            } \
            return type##_device; \
        }); \
        return true; \
    }();

} // namespace fastertransformer
