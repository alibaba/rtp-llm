#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include <unordered_map>
#include <vector>

namespace fastertransformer {

struct GlobalDeviceParams {
    std::vector<std::pair<DeviceType, DeviceInitParams>> device_params;
};

DeviceType getDeviceType(const std::string& device_name);

using DeviceCreatorType = std::function<DeviceBase*(const DeviceInitParams&)>;

class DeviceFactory {
public:
    static void initDevices(const GptInitParameter& params);
    static DeviceBase* getDevice(DeviceType type, int device_id = 0);
    static DeviceBase* getDefaultDevice();
    static void registerDevice(DeviceType type, DeviceCreatorType creator);

private:
    static GlobalDeviceParams getDefaultGlobalDeviceParams();
    static std::unordered_map<DeviceType, DeviceCreatorType>& getRegistrationMap();
    static std::vector<DeviceBase *>& getCurrentDevices();
};

#define RTP_LLM_REGISTER_DEVICE(type)                                   \
    static DeviceBase* type##_device __attribute__((used)) = nullptr;   \
    static auto type##_device_reg_creator = []() {                      \
        DeviceFactory::registerDevice(DeviceType::type, [](const DeviceInitParams& params) { \
            auto device = new type##Device(params);                     \
            device->init();                                             \
            return device;                                              \
        });                                                             \
        return true;                                                    \
    }();

} // namespace fastertransformer
