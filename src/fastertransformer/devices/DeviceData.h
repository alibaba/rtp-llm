#pragma once
#include <stddef.h>
#include <string>

namespace fastertransformer {

enum class DeviceType {
    Cpu  = 0,
    Cuda = 1,
    Yitian = 2,
    ArmCpu = 3,
    ROCm = 4,
};

struct DeviceInitParams {
    size_t device_id       = 0;
    size_t max_batch_size = 256;

    size_t tp_rank         = 0;
    size_t tp_size         = 1;
    // this ip:port pair should be unused, typically provided by gang,
    // to create temporary torch::TcpStore for exchanging communication id.
    // they are only needed when tp_size > 1.
    std::string master_ip  = "";
    int64_t master_port    = 0;

    // size (bytes) of device memory preallocated and managed by MemoryTracker.
    // negative value means reserving all free memory but remains abs(value) bytes.
    // 0 disables memory reservation
    int64_t device_reserve_memory_bytes = 0;
    int64_t host_reserve_memory_bytes   = 0;
};

// immutable device properties. Can not change since device is initialized.
struct DeviceProperties {
    DeviceType type;
    size_t id = 0;

    /* -- distributed properties -- */
    size_t tp_rank = 0;
    size_t tp_size = 1;

    /* -- device implementation detail -- */
    // These two options are prepared for intel cpu device.
    // xfastertransformer fuses adding residual in their layer implementation.
    bool attn_fuse_add_residual = false;
    bool ffn_fuse_add_residual  = false;
};

struct MemroyStatus {
    size_t used_bytes       = 0;
    size_t free_bytes       = 0;
    size_t available_bytes  = 0; // free + preserved
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
