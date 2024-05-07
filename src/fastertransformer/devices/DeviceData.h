#pragma once
#include <stddef.h>

namespace fastertransformer {

enum class DeviceType {
    Cpu  = 0,
    Cuda = 1,
    Yitian = 2,
    ArmCpu = 3,
};

// hardware-specific device properties, such as op fusion options. should be const.
struct DeviceProperties {
    DeviceType type;
};

struct MemroyStatus {
    size_t used_bytes       = 0;
    size_t free_bytes       = 0;
    size_t allocated_bytes  = 0; // memory allocated via current device
    size_t preserved_bytes  = 0; // memory preserved by current Device object, but not allocated yet
};

// runtime device status, such as available memory.
struct DeviceStatus {
    MemroyStatus device_memory_status;
    MemroyStatus host_memory_status;
    float device_utilization = 0.0f; // percentage of device utilization, 0.0f ~ 100.0f
};

}; // namespace fastertransformer
