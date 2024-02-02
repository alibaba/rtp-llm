#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"

namespace fastertransformer {

enum class DeviceType {
    Cpu  = 0,
    Cuda = 1,
};

class DeviceFactory {
public:
    static DeviceBase* getDevice(DeviceType type);
};

} // namespace fastertransformer
