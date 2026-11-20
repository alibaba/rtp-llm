#pragma once

#if __has_include(<nvml.h>)
#include <nvml.h>
#else
// Minimal NVML declarations for CUDA development images that provide
// libnvidia-ml.so but omit the driver SDK header.
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 32
#define NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE 80

extern "C" {

typedef struct nvmlDevice_st* nvmlDevice_t;

typedef enum nvmlReturn_enum {
    NVML_SUCCESS = 0,
} nvmlReturn_t;

typedef enum nvmlGpuLevel_enum {
    NVML_TOPOLOGY_INTERNAL   = 0,
    NVML_TOPOLOGY_SINGLE     = 10,
    NVML_TOPOLOGY_MULTIPLE   = 20,
    NVML_TOPOLOGY_HOSTBRIDGE = 30,
    NVML_TOPOLOGY_NODE       = 40,
    NVML_TOPOLOGY_SYSTEM     = 50,
} nvmlGpuTopologyLevel_t;

typedef enum nvmlGpuP2PStatus_enum {
    NVML_P2P_STATUS_OK = 0,
} nvmlGpuP2PStatus_t;

typedef enum nvmlGpuP2PCapsIndex_enum {
    NVML_P2P_CAPS_INDEX_NVLINK = 2,
} nvmlGpuP2PCapsIndex_t;

nvmlReturn_t nvmlInit(void);
nvmlReturn_t nvmlShutdown(void);
nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char* pci_bus_id, nvmlDevice_t* device);
nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length);
nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t          device1,
                                    nvmlDevice_t          device2,
                                    nvmlGpuP2PCapsIndex_t p2p_index,
                                    nvmlGpuP2PStatus_t*   p2p_status);
nvmlReturn_t
nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t* path_info);

}  // extern "C"
#endif
