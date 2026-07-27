#include <stddef.h>
#include <stdint.h>

typedef int                CUresult;
typedef int                CUdevice;
typedef void*              CUcontext;
typedef unsigned long long CUmemGenericAllocationHandle;

#define CUDA_SUCCESS 0

CUresult cuInit(unsigned int flags) {
    return flags == 0 ? CUDA_SUCCESS : 1;
}

CUresult cuDeviceGetCount(int* count) {
    if (count == NULL) {
        return 1;
    }
    *count = 3;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    if (device == NULL || ordinal < 0 || ordinal >= 3) {
        return 1;
    }
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice* device) {
    if (device == NULL) {
        return 1;
    }
    *device = 0;
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* context, CUdevice device) {
    if (context == NULL || device < 0 || device >= 3) {
        return 1;
    }
    *context = (void*)(uintptr_t)(device + 1);
    return CUDA_SUCCESS;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle, void* data, int type) {
    static CUmemGenericAllocationHandle next_handle = 0x1000;
    if (handle == NULL || data == NULL || (type != 1 && type != 8)) {
        return 1;
    }
    *handle = next_handle++;
    return CUDA_SUCCESS;
}

CUresult cuMulticastAddDevice(CUmemGenericAllocationHandle handle, CUdevice device) {
    return handle == 0 || device < 0 || device >= 3 ? 1 : CUDA_SUCCESS;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    return handle == 0 ? 1 : CUDA_SUCCESS;
}
