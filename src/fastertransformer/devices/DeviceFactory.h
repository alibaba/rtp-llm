#pragma once

#include "src/fastertransformer/devices/DeviceBase.h"

namespace fastertransformer {

enum class DeviceType {
    CPU  = 0,
    CUDA = 1,
};

class DeviceFactory {
public:
    static DeviceBase* getDevice(DeviceType type);
    static DeviceBase* getCpuDevice();
};

} // namespace fastertransformer
