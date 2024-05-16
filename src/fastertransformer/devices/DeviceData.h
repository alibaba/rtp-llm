#pragma once
#include <stddef.h>
#include <string>

namespace fastertransformer {

enum class DeviceType {
    Cpu  = 0,
    Cuda = 1,
    Yitian = 2,
    ArmCpu = 3,
};

struct DeviceInitParams {
    size_t device_id       = 0;
    size_t tp_rank         = 0;
    size_t tp_size         = 1;
    // this ip:port pair should be unused, typically provided by gang,
    // to create temporary torch::TcpStore for exchanging communication id.
    // they are only needed when tp_size > 1.
    std::string master_ip  = "";
    int64_t master_port    = 0;
};

// immutable device properties. Can not change since device is initialized.
struct DeviceProperties {
    DeviceType type;
    size_t id = 0;

    /* -- properties related to request construction -- */
    size_t max_batch_size = -1; // -1 indicates no limitation
    bool need_attention_mask = true;

    /* -- distributed properties -- */
    size_t tp_rank = 0;
    size_t tp_size = 1;
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
