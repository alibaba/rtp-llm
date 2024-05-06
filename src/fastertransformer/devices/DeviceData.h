#pragma once
#include <stddef.h>

namespace fastertransformer {

enum class DeviceType {
    Cpu  = 0,
    Cuda = 1,
    Yitian = 2,
};

// hardware-specific device properties, such as op fusion options. should be const.
struct DeviceProperties {
    DeviceType type;
};

// All vars are in bytes.
struct MemroyStatus {
    size_t total_memory;
    size_t free_memory;
    size_t allocated_memory;     // memory allocated via current device
    size_t preserved_memory;     // memory preserved by current Device object, but not allocated yet
};

// runtime device status, such as available memory.
struct DeviceStatus {
    MemroyStatus device_memory_status;
    MemroyStatus host_memory_status;
};

}; // namespace fastertransformer
